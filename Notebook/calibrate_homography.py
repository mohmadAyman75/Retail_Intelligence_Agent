"""Create or validate explicit camera-to-reference homographies.

The active camera-local pipeline uses this file only for zone projection.  This
tool never silently creates an identity matrix: identity is accepted solely for
an explicitly marked reference camera and remains in reference-pixel units.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def validate_homography(entry: dict[str, Any], camera_id: str) -> tuple[np.ndarray, str]:
    matrix = np.asarray(entry.get("homography"), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError(f"Invalid homography for {camera_id}")
    identity = np.allclose(matrix, np.eye(3), atol=1e-9)
    if identity and not bool(entry.get("reference_camera", False)):
        raise ValueError(f"Identity homography is allowed only for explicit reference camera: {camera_id}")
    return matrix, "reference_camera" if identity else "configured"


def calculate_entry(
    camera_id: str,
    source_points: list[list[float]],
    reference_points: list[list[float]],
    *,
    reference_camera_id: str,
    reference_camera: bool = False,
) -> dict[str, Any]:
    """Compute a RANSAC homography from manually reviewed corresponding points."""
    if reference_camera:
        matrix = np.eye(3, dtype=np.float64)
        return {
            "homography": matrix.tolist(),
            "reference_camera": True,
            "reference_camera_id": reference_camera_id,
            "floor_units": "reference_pixels",
            "calibration_method": "explicit_reference_identity",
        }
    source = np.asarray(source_points, dtype=np.float32)
    reference = np.asarray(reference_points, dtype=np.float32)
    if source.shape != reference.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 4:
        raise ValueError(f"{camera_id} needs at least four paired [x, y] points.")
    matrix, inliers = cv2.findHomography(source, reference, cv2.RANSAC, 6.0)
    if matrix is None or inliers is None:
        raise ValueError(f"Could not estimate homography for {camera_id}.")
    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - reference, axis=1)
    inlier_mask = inliers.reshape(-1).astype(bool)
    entry = {
        "homography": matrix.tolist(),
        "reference_camera": False,
        "reference_camera_id": reference_camera_id,
        "floor_units": "reference_pixels",
        "calibration_method": "opencv_findHomography_RANSAC",
        "landmarks": [
            {
                "name": f"point_{index + 1:02d}",
                "source_point": source[index].astype(float).tolist(),
                "reference_point": reference[index].astype(float).tolist(),
                "inlier": bool(inlier_mask[index]),
                "reprojection_error_px": float(errors[index]),
            }
            for index in range(len(source))
        ],
        "validation": {
            "status": "valid" if int(inlier_mask.sum()) >= 4 else "invalid",
            "point_count": int(len(source)),
            "inlier_count": int(inlier_mask.sum()),
            "inlier_ratio": float(inlier_mask.mean()),
            "mean_inlier_reprojection_error_px": float(errors[inlier_mask].mean()) if inlier_mask.any() else None,
            "condition_number": float(np.linalg.cond(matrix)),
            "matrix_rank": int(np.linalg.matrix_rank(matrix)),
            "determinant": float(np.linalg.det(matrix)),
        },
    }
    validate_homography(entry, camera_id)
    return entry


def validate_file(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Calibration JSON must be a non-empty object keyed by camera_id.")
    for camera_id, entry in payload.items():
        _, mode = validate_homography(entry, str(camera_id))
        print(f"{camera_id}: {mode}")
    print(f"Validated {len(payload)} camera calibration(s): {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="Validate an existing generated calibration file.")
    parser.add_argument("--path", type=Path, default=Path("Data/config/camera_calibration.generated.json"))
    parser.add_argument("--points", type=Path, help="JSON file with camera_id, reference_camera_id, source_points, reference_points.")
    args = parser.parse_args()
    if args.validate:
        return validate_file(args.path)
    if args.points is None:
        parser.error("Use --validate or provide --points with reviewed corresponding points.")
    points = json.loads(args.points.read_text(encoding="utf-8"))
    entry = calculate_entry(
        str(points["camera_id"]),
        points.get("source_points", []),
        points.get("reference_points", []),
        reference_camera_id=str(points["reference_camera_id"]),
        reference_camera=bool(points.get("reference_camera", False)),
    )
    existing = json.loads(args.path.read_text(encoding="utf-8")) if args.path.exists() else {}
    existing[str(points["camera_id"])] = entry
    args.path.parent.mkdir(parents=True, exist_ok=True)
    args.path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved reviewed calibration for {points['camera_id']} to {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
