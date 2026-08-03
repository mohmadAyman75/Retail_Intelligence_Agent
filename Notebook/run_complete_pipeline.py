"""Complete 4-camera MTMC ReID pipeline.

1. Processes all 4 CAFE place_05 cameras (17, 18, 19, 20).
2. Performs local tracking for any missing cameras using YOLO + ByteTrack.
3. Applies calibration homography to project foot points.
4. Extracts/reuses OSNet-AIN embeddings and runs MTMC ReID association.
5. Saves global_tracks.csv, reid_mapping.csv, and reid_report.json.
6. Renders global_annotated_*.mp4 videos with:
   - Fixed BLUE foot points (255, 0, 0 in BGR) with white borders as requested.
   - Per-frame Box Deduplication (NMS) to eliminate duplicate highlights on the same person.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "Notebook"
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

import mtmc_reid

TABLES_DIR = PROJECT_ROOT / "Output" / "tables"
VIDEOS_DIR = PROJECT_ROOT / "Output" / "videos"
RAW_DIR = PROJECT_ROOT / "Data" / "raw"
CONFIG_DIR = PROJECT_ROOT / "Data" / "config"
CACHE_DIR = PROJECT_ROOT / "Output" / "cache"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_TRACKS_PATH = TABLES_DIR / "local_tracks.csv"
GLOBAL_TRACKS_PATH = TABLES_DIR / "global_tracks.csv"
REID_MAPPING_PATH = TABLES_DIR / "reid_mapping.csv"
REID_REPORT_PATH = TABLES_DIR / "reid_report.json"
CALIBRATION_PATH = CONFIG_DIR / "camera_calibration.generated.json"
REID_CONFIG_PATH = CONFIG_DIR / "mtmc_reid_config.json"
MODEL_PATH = PROJECT_ROOT / "yolo11m.pt"
if not MODEL_PATH.exists():
    MODEL_PATH = NOTEBOOK_DIR / "yolo11m.pt"
TRACKER_PATH = CONFIG_DIR / "bytetrack_retail.yaml"


def find_raw_videos() -> dict[str, Path]:
    """Find raw videos for place_05 across Data/raw and subdirectories."""
    videos: dict[str, Path] = {}
    for path in RAW_DIR.rglob("*.mp4"):
        if "place_05" in path.stem:
            videos[path.stem] = path
    return videos


def run_local_tracking_if_needed(raw_videos: dict[str, Path]) -> pd.DataFrame:
    """Ensure local_tracks.csv contains tracking data for all 4 cameras."""
    existing_df = pd.DataFrame()
    if LOCAL_TRACKS_PATH.exists():
        existing_df = pd.read_csv(LOCAL_TRACKS_PATH)
        processed_cams = set(existing_df["camera_id"].unique()) if not existing_df.empty else set()
    else:
        processed_cams = set()

    needed_cams = [cam for cam in sorted(raw_videos.keys()) if cam not in processed_cams]

    if not needed_cams:
        print(f"local_tracks.csv already contains all {len(raw_videos)} cameras.")
        return existing_df

    print(f"Running local detection & tracking for missing camera(s): {needed_cams}")
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(MODEL_PATH))

    new_rows = []
    for cam_id in needed_cams:
        video_path = raw_videos[cam_id]
        print(f"  Processing {cam_id} ({video_path})...")
        results = model.track(
            source=str(video_path),
            stream=True,
            persist=False,
            classes=[0],  # person only
            conf=0.45,
            iou=0.45,
            imgsz=960,
            tracker=str(TRACKER_PATH),
            device=device,
            verbose=False,
        )

        for frame_index, result in enumerate(results):
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.int().cpu().tolist()
            confidences = boxes.conf.cpu().numpy()
            fps = float(result.orig_img.shape[0])  # fallback
            timestamp_sec = frame_index / 25.0

            for (x1, y1, x2, y2), track_id, conf in zip(xyxy, track_ids, confidences):
                if conf < 0.25:
                    continue
                foot_x = round((float(x1) + float(x2)) / 2.0, 2)
                foot_y = round(float(y2), 2)
                new_rows.append({
                    "camera_id": cam_id,
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "local_track_id": int(track_id),
                    "is_employee": False,
                    "confidence": round(float(conf), 4),
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2),
                    "foot_x": foot_x,
                    "foot_y": foot_y,
                })

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
    combined.to_csv(LOCAL_TRACKS_PATH, index=False)
    print(f"Updated local_tracks.csv with {len(combined):,} total rows across {combined.camera_id.nunique()} cameras.")
    return combined


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[0])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[0])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def deduplicate_frame_boxes(boxes: list[tuple[int, int, int, int, int, int, int]]) -> list[tuple[int, int, int, int, int, int, int]]:
    """Deduplicate overlapping boxes in the same frame to prevent double highlights."""
    if len(boxes) <= 1:
        return boxes
    sorted_boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept = []
    for candidate in sorted_boxes:
        c_box = (candidate[0], candidate[1], candidate[2], candidate[3])
        c_foot = (candidate[4], candidate[5])
        is_dup = False
        for k in kept:
            k_box = (k[0], k[1], k[2], k[3])
            k_foot = (k[4], k[5])
            iou = compute_iou(c_box, k_box)
            dist = math.hypot(c_foot[0] - k_foot[0], c_foot[1] - k_foot[1])
            if iou > 0.45 or dist < 25.0:
                is_dup = True
                break
        if not is_dup:
            kept.append(candidate)
    return kept


def _global_color(global_id_num: int) -> tuple[int, int, int]:
    """Generate vibrant distinct RGB/BGR color for a global ID."""
    golden = 0.618033988749895
    hue = (global_id_num * golden) % 1.0
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def render_global_videos_with_blue_dots(global_tracks: pd.DataFrame, raw_videos: dict[str, Path]):
    """Render annotated videos with BLUE foot points and box deduplication."""
    print("\nRendering global annotated videos with BLUE foot points...")
    POINT_COLOR_BGR = (255, 0, 0)  # Pure Blue in BGR
    BORDER_COLOR_BGR = (255, 255, 255)  # Pure White

    for camera_id, camera_rows in global_tracks.groupby("camera_id", sort=True):
        source_path = raw_videos.get(str(camera_id))
        if not source_path or not source_path.exists():
            print(f"  Skipping {camera_id}: source video not found.")
            continue

        output_path = VIDEOS_DIR / f"global_annotated_{camera_id}.mp4"
        temporary = VIDEOS_DIR / f"global_annotated_{camera_id}.part.mp4"
        temporary.unlink(missing_ok=True)

        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            continue

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            continue

        rows_by_frame: dict[int, list[tuple[int, int, int, int, int, int, int]]] = {}
        for row in camera_rows.itertuples(index=False):
            frame_idx = int(row.frame_index)
            gid_num = int(re.search(r"\d+", str(row.global_track_id)).group(0)) if re.search(r"\d+", str(row.global_track_id)) else 0
            rows_by_frame.setdefault(frame_idx, []).append(
                (
                    int(round(row.x1)),
                    int(round(row.y1)),
                    int(round(row.x2)),
                    int(round(row.y2)),
                    int(round(row.foot_x)),
                    int(round(row.foot_y)),
                    gid_num,
                )
            )

        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame_boxes = rows_by_frame.pop(frame_index, None)
                if frame_boxes:
                    clean_boxes = deduplicate_frame_boxes(frame_boxes)
                    for x1_raw, y1_raw, x2_raw, y2_raw, fx_raw, fy_raw, gid in clean_boxes:
                        box_color = _global_color(gid)
                        x1 = max(0, min(width - 1, x1_raw))
                        y1 = max(0, min(height - 1, y1_raw))
                        x2 = max(0, min(width - 1, x2_raw))
                        y2 = max(0, min(height - 1, y2_raw))
                        foot_x = max(0, min(width - 1, fx_raw))
                        foot_y = max(0, min(height - 1, fy_raw))

                        # Bounding Box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

                        # BLUE Foot-Point Dot (as requested: blue dot + white border)
                        cv2.circle(frame, (foot_x, foot_y), 9, POINT_COLOR_BGR, -1, cv2.LINE_AA)
                        cv2.circle(frame, (foot_x, foot_y), 11, BORDER_COLOR_BGR, 1, cv2.LINE_AA)

                        # Text Label
                        label = f"GID {gid:06d}"
                        text_y = max(24, y1 - 7)
                        cv2.putText(
                            frame,
                            label,
                            (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            box_color,
                            2,
                            cv2.LINE_AA,
                        )
                writer.write(frame)
                frame_index += 1
        finally:
            capture.release()
            writer.release()

        if frame_index > 0:
            replaced = False
            for _ in range(10):
                try:
                    output_path.unlink(missing_ok=True)
                    temporary.replace(output_path)
                    replaced = True
                    break
                except OSError:
                    time.sleep(0.5)
            if not replaced:
                try:
                    shutil.copy2(temporary, output_path)
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            if output_path.exists():
                print(f"  Rendered {output_path.name} ({output_path.stat().st_size / 1e6:.1f} MB)")


def main():
    raw_videos = find_raw_videos()
    print(f"Found {len(raw_videos)} raw videos for place_05: {sorted(raw_videos.keys())}")

    local_tracks = run_local_tracking_if_needed(raw_videos)

    print("\nLoading calibration and MTMC ReID configuration...")
    calibrations = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    reid_config = json.loads(REID_CONFIG_PATH.read_text(encoding="utf-8"))

    print("Applying homographies to all detections...")
    detections_with_floor = mtmc_reid.apply_homographies(local_tracks, calibrations)

    print("Summarizing tracklets...")
    tracklets = mtmc_reid.summarize_tracklets(detections_with_floor)

    print(f"Extracting/loading embeddings for {len(tracklets):,} tracklets...")
    embedding_config = reid_config["reid"]
    embedding_cache_path = CACHE_DIR / "tracklet_embeddings.npz"

    model, device = mtmc_reid.load_osnet_model(
        PROJECT_ROOT,
        embedding_config["weights_path"],
        embedding_config.get("device", "auto"),
    )

    embeddings, embedding_report = mtmc_reid.extract_tracklet_embeddings(
        detections_with_floor,
        raw_dir=RAW_DIR,
        model=model,
        device=device,
        samples_per_tracklet=embedding_config.get("samples_per_tracklet", 12),
        min_confidence=embedding_config.get("min_confidence", 0.25),
        min_box_area=embedding_config.get("min_box_area", 1024.0),
        batch_size=embedding_config.get("batch_size", 32),
    )

    print("Running cross-camera association...")
    tracklets_with_emb = mtmc_reid.attach_tracklet_embeddings(tracklets, embeddings)
    assoc_params = reid_config.get("association", {})
    assoc_config = mtmc_reid.AssociationConfig(
        **{k: v for k, v in assoc_params.items() if k in mtmc_reid.AssociationConfig.__dataclass_fields__}
    )

    result = mtmc_reid.run_mtmc_association(tracklets_with_emb, config=assoc_config)
    print(f"  Accepted matches: {len(result.accepted_matches):,}")
    print(f"  Unique global identities: {result.mapping.global_track_id.nunique():,}")

    global_tracks = mtmc_reid.apply_global_ids(detections_with_floor, result.mapping)
    global_tracks.to_csv(GLOBAL_TRACKS_PATH, index=False)
    result.mapping.to_csv(REID_MAPPING_PATH, index=False)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cameras": int(local_tracks.camera_id.nunique()),
        "status": "cross_camera_reid",
        "total_tracklets": int(len(tracklets)),
        "global_identities": int(result.mapping.global_track_id.nunique()),
        "accepted_matches": int(len(result.accepted_matches)),
        "rejected_matches": int(len(result.rejected_matches)),
    }
    REID_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved global_tracks.csv, reid_mapping.csv, and reid_report.json.")

    render_global_videos_with_blue_dots(global_tracks, raw_videos)
    print("\nPipeline complete! All 4 cameras processed with BLUE foot-point dots and NMS box deduplication.")


if __name__ == "__main__":
    main()
