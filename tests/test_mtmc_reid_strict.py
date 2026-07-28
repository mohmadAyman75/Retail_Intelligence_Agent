from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Notebook"))

from mtmc_reid import (  # noqa: E402
    AssociationConfig,
    CalibrationError,
    apply_global_ids,
    apply_homographies,
    attach_tracklet_embeddings,
    hungarian_assign,
    run_mtmc_association,
    summarize_tracklets,
    validate_homography,
)


class StrictMtmcReidTests(unittest.TestCase):
    def test_identity_placeholder_is_rejected(self) -> None:
        with self.assertRaises(CalibrationError):
            validate_homography(np.eye(3), camera_id="cam_b")
        np.testing.assert_allclose(
            validate_homography(np.eye(3), camera_id="cam_a", reference_camera=True),
            np.eye(3),
        )

    def test_hungarian_prefers_one_strong_pair_over_two_weak_ones(self) -> None:
        candidates = pd.DataFrame(
            [
                {"store_id": "s", "camera_id_a": "a", "camera_id_b": "b", "tracklet_id_a": "a1", "tracklet_id_b": "b1", "association_time_sec": 1.0, "score": 0.10},
                {"store_id": "s", "camera_id_a": "a", "camera_id_b": "b", "tracklet_id_a": "a1", "tracklet_id_b": "b2", "association_time_sec": 1.0, "score": 0.70},
                {"store_id": "s", "camera_id_a": "a", "camera_id_b": "b", "tracklet_id_a": "a2", "tracklet_id_b": "b1", "association_time_sec": 1.0, "score": 0.70},
            ]
        )
        assigned = hungarian_assign(candidates, max_score=0.72)
        self.assertEqual(len(assigned), 1)
        self.assertEqual((assigned.iloc[0].tracklet_id_a, assigned.iloc[0].tracklet_id_b), ("a1", "b1"))

    def test_calibrated_association_creates_two_global_ids(self) -> None:
        rows = []
        for (camera_id, local_id), (foot_x, identity_axis) in {
            ("cam_a", 1): (10.0, 0.0),
            ("cam_b", 7): (11.0, 0.0),
            ("cam_a", 2): (200.0, 1.0),
            ("cam_b", 8): (201.0, 1.0),
        }.items():
            for frame_index, timestamp in enumerate((0.0, 1.0, 2.0)):
                rows.append(
                    {
                        "store_id": "place_05",
                        "camera_id": camera_id,
                        "local_track_id": local_id,
                        "frame_index": frame_index,
                        "timestamp_sec": timestamp,
                        "foot_x": foot_x + timestamp,
                        "foot_y": 20.0,
                        "is_employee": False,
                    }
                )
        detections = apply_homographies(
            pd.DataFrame(rows),
            {
                "cam_a": {"homography": np.eye(3), "reference_camera": True},
                "cam_b": {"homography": [[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
            },
        )
        tracklets = summarize_tracklets(detections)
        tracklets = attach_tracklet_embeddings(
            tracklets,
            {
                "place_05::cam_a::1": [1.0, 0.0],
                "place_05::cam_b::7": [0.999, 0.001],
                "place_05::cam_a::2": [0.0, 1.0],
                "place_05::cam_b::8": [0.001, 0.999],
            },
        )
        result = run_mtmc_association(
            tracklets,
            config=AssociationConfig(
                max_floor_distance=10.0,
                max_appearance_distance=0.20,
                min_synchronized_samples=3,
                max_synchronized_samples=5,
            ),
            max_assignment_score=0.8,
        )
        self.assertEqual(len(result.accepted_matches), 2)
        self.assertEqual(result.mapping.global_track_id.nunique(), 2)
        joined = apply_global_ids(detections, result.mapping)
        by_local_id = joined.groupby("local_track_id").global_track_id.first()
        self.assertEqual(by_local_id.loc[1], by_local_id.loc[7])
        self.assertEqual(by_local_id.loc[2], by_local_id.loc[8])


if __name__ == "__main__":
    unittest.main()
