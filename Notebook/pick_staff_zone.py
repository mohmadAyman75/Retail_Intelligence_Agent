"""Interactively save one camera-pixel staff polygon to Data/config/staff_zones.json.

Left-click adds a point, right-click removes the most recent point, Enter saves,
and Esc exits without changing the configuration.  The saved coordinates are
image pixels for this camera only; they are not floor-plan coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def resolve_video(raw_dir: Path, camera_id: str) -> Path:
    matches = [
        path for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and path.stem == camera_id
    ]
    if not matches:
        raise FileNotFoundError(f"No raw video for {camera_id} under {raw_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"More than one raw video matches {camera_id}: {matches}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera_id", help="Camera ID, for example CAFE_place_05_camera_17_15min")
    parser.add_argument("--raw-dir", type=Path, default=Path("Data/raw"))
    parser.add_argument("--config", type=Path, default=Path("Data/config/staff_zones.json"))
    args = parser.parse_args()

    video_path = resolve_video(args.raw_dir, args.camera_id)
    capture = cv2.VideoCapture(str(video_path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read the first frame from {video_path}")

    window_name = f"Staff zone: {args.camera_id}"
    points: list[list[int]] = []

    def redraw() -> np.ndarray:
        canvas = frame.copy()
        if points:
            polygon = np.asarray(points, dtype=np.int32)
            cv2.polylines(canvas, [polygon], len(points) >= 3, (0, 0, 255), 2, cv2.LINE_AA)
            for index, point in enumerate(points, start=1):
                cv2.circle(canvas, tuple(point), 5, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(canvas, str(index), (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Left-click: add | Right-click: undo | Enter: save | Esc: cancel", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([int(x), int(y)])
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    while True:
        cv2.imshow(window_name, redraw())
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            print("Cancelled. staff_zones.json was not changed.")
            return 0
        if key in {10, 13}:
            if len(points) < 3:
                print("Select at least three polygon points before saving.")
                continue
            break
    cv2.destroyAllWindows()

    payload = json.loads(args.config.read_text(encoding="utf-8")) if args.config.exists() else []
    if not isinstance(payload, list):
        raise ValueError("staff_zones.json must be a JSON list.")
    payload = [entry for entry in payload if str(entry.get("camera_id")) != args.camera_id]
    payload.append(
        {
            "camera_id": args.camera_id,
            "zone_id": "staff_area",
            "label_ar": "منطقة العمل",
            "kind": "staff",
            "polygon": points,
        }
    )
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(points)} staff-zone points for {args.camera_id} to {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
