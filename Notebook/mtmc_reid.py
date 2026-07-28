"""Strict, calibration-backed helpers for optional multi-camera identity experiments.

This module is deliberately not part of the active camera-local pipeline.  It
only provides explicit utilities for the demo notebook and for offline research
after calibration has been reviewed.  A caller must label any result generated
without a configured calibration as a demo rather than a production identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable
import importlib.util
import math
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as torch_functional


class CalibrationError(ValueError):
    """Raised when a homography is missing, invalid, or an unsafe identity map."""


@dataclass(frozen=True)
class AssociationConfig:
    """Gates and weights for the strict, calibration-backed association path."""

    max_time_gap_sec: float = 12.0
    max_floor_distance: float = 120.0
    max_appearance_distance: float = 0.45
    appearance_weight: float = 0.65
    spatial_weight: float = 0.25
    temporal_weight: float = 0.10
    synchronization_tolerance_sec: float = 0.35
    min_synchronized_samples: int = 3
    max_synchronized_samples: int = 25
    allow_same_camera_nonoverlap: bool = True
    same_camera_max_time_gap_sec: float = 3.0
    same_camera_max_floor_distance: float = 80.0
    same_camera_max_appearance_distance: float = 0.25
    require_same_employee_label: bool = True
    max_assignment_score: float = 0.78
    assignment_window_sec: float = 5.0
    overlap_tolerance_sec: float = 0.0


@dataclass(frozen=True)
class AssociationResult:
    candidates: pd.DataFrame
    accepted_matches: pd.DataFrame
    rejected_matches: pd.DataFrame
    mapping: pd.DataFrame


@dataclass(frozen=True)
class ClusterResult:
    mapping: pd.DataFrame
    accepted_matches: pd.DataFrame
    rejected_matches: pd.DataFrame


def make_tracklet_id(store_id: str, camera_id: str, local_track_id: int | str) -> str:
    """Build a stable ID that is unique only within its source tracklet."""
    return f"{store_id}::{camera_id}::{local_track_id}"


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def _infer_store_id(camera_id: str) -> str:
    import re

    match = re.search(r"(place_\d+)", str(camera_id), flags=re.IGNORECASE)
    return match.group(1).lower() if match else "default_store"


def validate_homography(
    homography: Any,
    *,
    camera_id: str,
    reference_camera: bool = False,
) -> np.ndarray:
    """Validate an invertible 3x3 map and guard unsafe identity placeholders."""
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise CalibrationError(f"Invalid 3x3 homography for {camera_id}.")
    if np.linalg.matrix_rank(matrix) < 3:
        raise CalibrationError(f"Rank-deficient homography for {camera_id}.")
    if np.allclose(matrix, np.eye(3), atol=1e-9) and not reference_camera:
        raise CalibrationError(
            f"Identity homography is allowed only for an explicit reference camera: {camera_id}"
        )
    return matrix


def calibration_mode(entry: dict[str, Any], camera_id: str) -> str:
    """Return the explicit mode used by an output row after validation."""
    matrix = validate_homography(
        entry.get("homography"),
        camera_id=camera_id,
        reference_camera=_as_bool(entry.get("reference_camera", False)),
    )
    return "reference_camera" if np.allclose(matrix, np.eye(3), atol=1e-9) else "configured"


def apply_homographies(
    detections: pd.DataFrame,
    calibrations: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Append floor coordinates without removing or changing input columns."""
    required = {"camera_id", "foot_x", "foot_y"}
    missing = sorted(required.difference(detections.columns))
    if missing:
        raise ValueError(f"Detections are missing required columns: {missing}")

    output = detections.copy()
    output["floor_x"] = np.nan
    output["floor_y"] = np.nan
    output["calibration_mode"] = ""
    for camera_id, camera_rows in output.groupby("camera_id", sort=True):
        entry = calibrations.get(str(camera_id))
        if entry is None:
            raise CalibrationError(f"Missing calibration for {camera_id}.")
        matrix = validate_homography(
            entry.get("homography"),
            camera_id=str(camera_id),
            reference_camera=_as_bool(entry.get("reference_camera", False)),
        )
        points = camera_rows[["foot_x", "foot_y"]].to_numpy(dtype=np.float32)
        if not np.isfinite(points).all():
            raise CalibrationError(f"Non-finite foot point for {camera_id}.")
        mapped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        output.loc[camera_rows.index, ["floor_x", "floor_y"]] = mapped
        output.loc[camera_rows.index, "calibration_mode"] = (
            "reference_camera" if np.allclose(matrix, np.eye(3), atol=1e-9) else "configured"
        )
    return output


def summarize_tracklets(detections: pd.DataFrame) -> pd.DataFrame:
    """Summarize camera-local detections into one row per local tracklet."""
    required = {
        "camera_id",
        "local_track_id",
        "timestamp_sec",
        "floor_x",
        "floor_y",
    }
    missing = sorted(required.difference(detections.columns))
    if missing:
        raise ValueError(f"Cannot summarize tracklets; missing columns: {missing}")
    rows = detections.copy()
    if "store_id" not in rows.columns:
        rows["store_id"] = rows["camera_id"].map(_infer_store_id)
    if "is_employee" not in rows.columns:
        rows["is_employee"] = False
    rows["is_employee"] = rows["is_employee"].map(_as_bool)
    rows = rows.sort_values(["store_id", "camera_id", "local_track_id", "timestamp_sec"])

    summaries: list[dict[str, Any]] = []
    for (store_id, camera_id, local_track_id), group in rows.groupby(
        ["store_id", "camera_id", "local_track_id"], sort=True
    ):
        summaries.append(
            {
                "store_id": str(store_id),
                "camera_id": str(camera_id),
                "local_track_id": int(local_track_id),
                "tracklet_id": make_tracklet_id(store_id, camera_id, local_track_id),
                "start_sec": float(group["timestamp_sec"].min()),
                "end_sec": float(group["timestamp_sec"].max()),
                "start_floor_x": float(group["floor_x"].iloc[0]),
                "start_floor_y": float(group["floor_y"].iloc[0]),
                "end_floor_x": float(group["floor_x"].iloc[-1]),
                "end_floor_y": float(group["floor_y"].iloc[-1]),
                "mean_floor_x": float(group["floor_x"].mean()),
                "mean_floor_y": float(group["floor_y"].mean()),
                "sample_count": int(len(group)),
                "is_employee": bool(group["is_employee"].mode(dropna=False).iloc[0]),
            }
        )
    return pd.DataFrame(summaries)


def attach_tracklet_embeddings(
    tracklets: pd.DataFrame,
    embeddings: dict[str, Iterable[float]],
) -> pd.DataFrame:
    """Attach normalized embedding vectors keyed by `tracklet_id`."""
    output = tracklets.copy()
    attached: list[np.ndarray] = []
    for tracklet_id in output["tracklet_id"].astype(str):
        vector = embeddings.get(tracklet_id)
        if vector is None:
            raise ValueError(f"Missing embedding for {tracklet_id}.")
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(array))
        if array.size == 0 or not np.isfinite(array).all() or norm == 0:
            raise ValueError(f"Invalid embedding for {tracklet_id}.")
        attached.append(array / norm)
    output["embedding"] = attached
    return output


def _cosine_distance(first: Iterable[float], second: Iterable[float]) -> float:
    left = np.asarray(first, dtype=np.float32).reshape(-1)
    right = np.asarray(second, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return float("inf")
    return float(1.0 - np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _time_gap(left: pd.Series, right: pd.Series) -> float:
    return max(
        0.0,
        float(left["start_sec"]) - float(right["end_sec"]),
        float(right["start_sec"]) - float(left["end_sec"]),
    )


def _candidate_score(
    left: pd.Series,
    right: pd.Series,
    config: AssociationConfig,
) -> tuple[float, float, float, float]:
    appearance_distance = _cosine_distance(left["embedding"], right["embedding"])
    floor_distance = float(
        math.hypot(
            float(left["mean_floor_x"]) - float(right["mean_floor_x"]),
            float(left["mean_floor_y"]) - float(right["mean_floor_y"]),
        )
    )
    gap_sec = _time_gap(left, right)
    score = (
        config.appearance_weight * (appearance_distance / config.max_appearance_distance)
        + config.spatial_weight * (floor_distance / config.max_floor_distance)
        + config.temporal_weight * (gap_sec / config.max_time_gap_sec)
    )
    return float(score), appearance_distance, floor_distance, gap_sec


def build_candidate_pairs(
    tracklets: pd.DataFrame,
    config: AssociationConfig,
) -> pd.DataFrame:
    """Return only pairs that pass every strict spatial, temporal, and appearance gate."""
    required = {
        "store_id", "camera_id", "tracklet_id", "start_sec", "end_sec",
        "mean_floor_x", "mean_floor_y", "embedding", "is_employee",
    }
    missing = sorted(required.difference(tracklets.columns))
    if missing:
        raise ValueError(f"Tracklets are missing association columns: {missing}")

    candidates: list[dict[str, Any]] = []
    for store_id, store_tracklets in tracklets.groupby("store_id", sort=True):
        records = list(store_tracklets.sort_values("tracklet_id").iterrows())
        for (_, left), (_, right) in combinations(records, 2):
            same_camera = str(left["camera_id"]) == str(right["camera_id"])
            if same_camera and not config.allow_same_camera_nonoverlap:
                continue
            if config.require_same_employee_label and bool(left["is_employee"]) != bool(right["is_employee"]):
                continue
            score, appearance_distance, floor_distance, gap_sec = _candidate_score(left, right, config)
            max_gap = config.same_camera_max_time_gap_sec if same_camera else config.max_time_gap_sec
            max_floor = config.same_camera_max_floor_distance if same_camera else config.max_floor_distance
            max_appearance = config.same_camera_max_appearance_distance if same_camera else config.max_appearance_distance
            if gap_sec > max_gap or floor_distance > max_floor or appearance_distance > max_appearance:
                continue
            first, second = (left, right) if str(left["tracklet_id"]) < str(right["tracklet_id"]) else (right, left)
            candidates.append(
                {
                    "store_id": str(store_id),
                    "camera_id_a": str(first["camera_id"]),
                    "camera_id_b": str(second["camera_id"]),
                    "tracklet_id_a": str(first["tracklet_id"]),
                    "tracklet_id_b": str(second["tracklet_id"]),
                    "association_time_sec": max(float(first["start_sec"]), float(second["start_sec"])),
                    "score": score,
                    "appearance_distance": appearance_distance,
                    "floor_distance": floor_distance,
                    "time_gap_sec": gap_sec,
                    "spatial_method": "synchronized_median" if gap_sec == 0 else "tracklet_mean",
                    "is_cross_camera": not same_camera,
                }
            )
    columns = [
        "store_id", "camera_id_a", "camera_id_b", "tracklet_id_a", "tracklet_id_b",
        "association_time_sec", "score", "appearance_distance", "floor_distance",
        "time_gap_sec", "spatial_method", "is_cross_camera",
    ]
    return pd.DataFrame(candidates, columns=columns)


def hungarian_assign(
    candidates: pd.DataFrame,
    max_score: float,
    assignment_window_sec: float = 5.0,
) -> pd.DataFrame:
    """Use per-window Hungarian assignment with explicit unmatched dummy costs."""
    required = {
        "store_id", "camera_id_a", "camera_id_b", "tracklet_id_a", "tracklet_id_b",
        "association_time_sec", "score",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Candidates are missing Hungarian columns: {missing}")
    if candidates.empty:
        return candidates.assign(assignment_window=pd.Series(dtype="int64"))
    if assignment_window_sec <= 0:
        raise ValueError("assignment_window_sec must be positive.")

    selected: list[pd.DataFrame] = []
    rows = candidates[candidates["score"].astype(float) <= float(max_score)].copy()
    if rows.empty:
        return rows.assign(assignment_window=pd.Series(dtype="int64"))
    rows["assignment_window"] = np.floor(
        rows["association_time_sec"].astype(float) / assignment_window_sec
    ).astype(int)
    group_columns = ["store_id", "camera_id_a", "camera_id_b", "assignment_window"]
    for _, group in rows.groupby(group_columns, sort=True):
        left_ids = sorted(group["tracklet_id_a"].astype(str).unique())
        right_ids = sorted(group["tracklet_id_b"].astype(str).unique())
        size = len(left_ids) + len(right_ids)
        # A dummy edge costs max_score.  This lets the optimizer leave weak
        # alternatives unmatched instead of forcing a two-way weak pairing.
        costs = np.full((size, size), float(max_score), dtype=np.float64)
        left_index = {tracklet_id: index for index, tracklet_id in enumerate(left_ids)}
        right_index = {tracklet_id: index for index, tracklet_id in enumerate(right_ids)}
        row_lookup: dict[tuple[int, int], int] = {}
        for row_index, candidate in group.iterrows():
            row = left_index[str(candidate["tracklet_id_a"])]
            column = right_index[str(candidate["tracklet_id_b"])]
            cost = float(candidate["score"])
            if cost < costs[row, column]:
                costs[row, column] = cost
                row_lookup[(row, column)] = row_index
        assigned_rows, assigned_columns = linear_sum_assignment(costs)
        picked_indexes = [
            row_lookup[(row, column)]
            for row, column in zip(assigned_rows, assigned_columns)
            if (row, column) in row_lookup and costs[row, column] <= float(max_score)
        ]
        if picked_indexes:
            selected.append(group.loc[picked_indexes])
    if not selected:
        return rows.iloc[0:0].copy()
    return pd.concat(selected, ignore_index=True).sort_values(
        ["store_id", "association_time_sec", "score", "tracklet_id_a", "tracklet_id_b"]
    ).reset_index(drop=True)


def _intervals_overlap(left: pd.Series, right: pd.Series, tolerance_sec: float) -> bool:
    return not (
        float(left["end_sec"]) + tolerance_sec < float(right["start_sec"])
        or float(right["end_sec"]) + tolerance_sec < float(left["start_sec"])
    )


def cluster_tracklets(
    tracklets: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    candidates: pd.DataFrame | None = None,
    cross_camera_conflict_max_gap_sec: float = 0.0,
) -> ClusterResult:
    """Merge selected pairs only when every resulting cross-camera pair is compatible."""
    tracklet_rows = {
        str(row.tracklet_id): row
        for row in tracklets.itertuples(index=False)
    }
    parent = {tracklet_id: tracklet_id for tracklet_id in tracklet_rows}

    def find(tracklet_id: str) -> str:
        while parent[tracklet_id] != tracklet_id:
            parent[tracklet_id] = parent[parent[tracklet_id]]
            tracklet_id = parent[tracklet_id]
        return tracklet_id

    def members(tracklet_id: str) -> set[str]:
        root = find(tracklet_id)
        return {candidate for candidate in parent if find(candidate) == root}

    candidate_pairs: set[frozenset[str]] = set()
    if candidates is not None and not candidates.empty:
        for row in candidates.itertuples(index=False):
            candidate_pairs.add(frozenset((str(row.tracklet_id_a), str(row.tracklet_id_b))))

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for match in matches.sort_values(["score", "tracklet_id_a", "tracklet_id_b"]).to_dict("records"):
        first = str(match["tracklet_id_a"])
        second = str(match["tracklet_id_b"])
        if first not in parent or second not in parent:
            rejected.append({**match, "rejection_reason": "unknown_tracklet"})
            continue
        first_members, second_members = members(first), members(second)
        if first_members == second_members:
            continue
        merged_members = first_members | second_members
        compatible = True
        for left_id, right_id in combinations(sorted(merged_members), 2):
            left = tracklet_rows[left_id]
            right = tracklet_rows[right_id]
            if str(left.camera_id) == str(right.camera_id):
                if _intervals_overlap(pd.Series(left._asdict()), pd.Series(right._asdict()), cross_camera_conflict_max_gap_sec):
                    compatible = False
                    break
            elif candidate_pairs and frozenset((left_id, right_id)) not in candidate_pairs:
                # A chain A-B-C is not evidence for A-C.  Require a compatible
                # direct candidate before declaring all three one person.
                compatible = False
                break
        if not compatible:
            rejected.append({**match, "rejection_reason": "cross_camera_incompatible"})
            continue
        parent[find(second)] = find(first)
        accepted.append(match)

    groups: dict[str, list[str]] = {}
    for tracklet_id in parent:
        groups.setdefault(find(tracklet_id), []).append(tracklet_id)
    global_ids = {
        tracklet_id: f"global_{index:06d}"
        for index, group in enumerate(sorted(groups.values(), key=lambda values: min(values)), start=1)
        for tracklet_id in group
    }
    mapping = tracklets[["store_id", "camera_id", "local_track_id", "tracklet_id"]].copy()
    mapping["global_track_id"] = mapping["tracklet_id"].map(global_ids)
    accepted_frame = pd.DataFrame(accepted, columns=list(matches.columns))
    rejected_columns = [*matches.columns, "rejection_reason"]
    rejected_frame = pd.DataFrame(rejected, columns=rejected_columns)
    return ClusterResult(mapping, accepted_frame, rejected_frame)


def run_mtmc_association(
    tracklets: pd.DataFrame,
    *,
    config: AssociationConfig = AssociationConfig(),
    max_assignment_score: float | None = None,
) -> AssociationResult:
    """Run strict gates, Hungarian assignment, then compatibility-safe clustering."""
    assignment_limit = config.max_assignment_score if max_assignment_score is None else max_assignment_score
    candidates = build_candidate_pairs(tracklets, config)
    assignments = hungarian_assign(candidates, assignment_limit, config.assignment_window_sec)
    clustered = cluster_tracklets(
        tracklets,
        assignments,
        candidates=candidates,
        cross_camera_conflict_max_gap_sec=config.overlap_tolerance_sec,
    )
    return AssociationResult(
        candidates=candidates,
        accepted_matches=clustered.accepted_matches,
        rejected_matches=clustered.rejected_matches,
        mapping=clustered.mapping,
    )


def apply_global_ids(detections: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Append global IDs to a copy of detections, preserving every source column."""
    keys = ["store_id", "camera_id", "local_track_id"]
    output = detections.copy()
    if "store_id" not in output.columns:
        output["store_id"] = output["camera_id"].map(_infer_store_id)
    missing = sorted(set(keys).difference(mapping.columns))
    if missing or "global_track_id" not in mapping.columns:
        raise ValueError("Mapping must include store_id, camera_id, local_track_id, global_track_id.")
    return output.merge(mapping[[*keys, "global_track_id"]], on=keys, how="left", validate="many_to_one")


def load_osnet_model(project_root: Path, weights_path: str | Path, device: str = "auto") -> tuple[torch.nn.Module, torch.device]:
    """Load the vendored OSNet-AIN backbone and its local checkpoint without downloads."""
    project_root = Path(project_root)
    source_path = project_root / "vendor" / "torchreid_osnet" / "osnet_ain.py"
    if not source_path.exists():
        raise FileNotFoundError(f"Vendored OSNet source is missing: {source_path}")
    spec = importlib.util.spec_from_file_location("retail_osnet_ain", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load OSNet source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    resolved_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
    model = module.osnet_ain_x1_0(num_classes=1000, pretrained=False)
    checkpoint_path = project_root / Path(weights_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"OSNet checkpoint is missing: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = model.state_dict()
    compatible_state = {
        str(key).removeprefix("module."): value
        for key, value in state.items()
        if str(key).removeprefix("module.") in model_state
        and model_state[str(key).removeprefix("module.")].shape == value.shape
    }
    if not compatible_state:
        raise RuntimeError("No compatible OSNet checkpoint weights were found.")
    model.load_state_dict(compatible_state, strict=False)
    model.to(resolved_device).eval()
    return model, resolved_device


def resolve_camera_videos(raw_dir: Path, camera_ids: Iterable[str]) -> dict[str, Path]:
    """Resolve camera IDs to unique raw videos, including intentionally nested archives."""
    raw_dir = Path(raw_dir)
    indexed: dict[str, list[Path]] = {}
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            indexed.setdefault(path.stem, []).append(path)
    resolved: dict[str, Path] = {}
    for camera_id in sorted({str(value) for value in camera_ids}):
        matches = indexed.get(camera_id, [])
        if not matches:
            raise FileNotFoundError(f"Raw video for {camera_id} was not found under {raw_dir}.")
        if len(matches) > 1:
            raise RuntimeError(f"More than one raw video matches {camera_id}: {matches}")
        resolved[camera_id] = matches[0]
    return resolved


def extract_tracklet_embeddings(
    detections: pd.DataFrame,
    *,
    raw_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    samples_per_tracklet: int = 12,
    min_confidence: float = 0.25,
    min_box_area: float = 1024.0,
    batch_size: int = 32,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Extract L2-normalized OSNet embeddings from evenly sampled detection crops."""
    required = {"camera_id", "local_track_id", "frame_index", "confidence", "x1", "y1", "x2", "y2"}
    missing = sorted(required.difference(detections.columns))
    if missing:
        raise ValueError(f"Cannot extract embeddings; missing columns: {missing}")
    rows = detections.copy()
    if "store_id" not in rows.columns:
        rows["store_id"] = rows["camera_id"].map(_infer_store_id)
    rows["box_area"] = (rows["x2"] - rows["x1"]) * (rows["y2"] - rows["y1"])
    rows = rows[(rows["confidence"] >= min_confidence) & (rows["box_area"] >= min_box_area)].copy()
    if rows.empty:
        return {}, {"requested_crops": 0, "valid_crops": 0, "embedded_tracklets": 0}
    rows["tracklet_id"] = [
        make_tracklet_id(store, camera, local)
        for store, camera, local in rows[["store_id", "camera_id", "local_track_id"]].itertuples(index=False, name=None)
    ]
    sampled: list[pd.Series] = []
    for _, group in rows.sort_values(["tracklet_id", "frame_index"]).groupby("tracklet_id", sort=True):
        positions = np.linspace(0, len(group) - 1, min(samples_per_tracklet, len(group)), dtype=int)
        sampled.extend(group.iloc[np.unique(positions)].to_dict("records"))
    sample_rows = pd.DataFrame(sampled)
    videos = resolve_camera_videos(raw_dir, sample_rows["camera_id"].unique())
    crops_by_tracklet: dict[str, list[torch.Tensor]] = {}
    valid_crops = 0
    for camera_id, camera_rows in sample_rows.groupby("camera_id", sort=True):
        capture = cv2.VideoCapture(str(videos[str(camera_id)]))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open raw video for {camera_id}.")
        try:
            for row in camera_rows.sort_values("frame_index").itertuples(index=False):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(row.frame_index))
                ok, frame = capture.read()
                if not ok:
                    continue
                height, width = frame.shape[:2]
                x1, y1 = max(0, int(math.floor(row.x1))), max(0, int(math.floor(row.y1)))
                x2, y2 = min(width, int(math.ceil(row.x2))), min(height, int(math.ceil(row.y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (128, 256)), cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(crop).permute(2, 0, 1).float().div(255.0)
                tensor = (tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                crops_by_tracklet.setdefault(str(row.tracklet_id), []).append(tensor)
                valid_crops += 1
        finally:
            capture.release()
    embeddings: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for tracklet_id, tensors in crops_by_tracklet.items():
            outputs: list[torch.Tensor] = []
            for start in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[start:start + batch_size]).to(device)
                outputs.append(torch_functional.normalize(model(batch), p=2, dim=1).cpu())
            vector = torch.cat(outputs).mean(dim=0)
            embeddings[tracklet_id] = torch_functional.normalize(vector, p=2, dim=0).numpy()
    return embeddings, {
        "requested_crops": int(len(sample_rows)),
        "valid_crops": int(valid_crops),
        "embedded_tracklets": int(len(embeddings)),
    }
