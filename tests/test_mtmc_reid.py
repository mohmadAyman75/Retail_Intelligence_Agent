from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


raise unittest.SkipTest(
    "MTMC/Re-ID was intentionally removed. See test_camera_local_pipeline_contracts.py."
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Notebook"))

from mtmc_reid import (  # noqa: E402
    AssociationConfig,
    CalibrationError,
    apply_global_ids,
    apply_homographies,
    attach_tracklet_embeddings,
    cluster_tracklets,
    hungarian_assign,
    make_tracklet_id,
    run_mtmc_association,
    summarize_tracklets,
    validate_homography,
)


class MTMCReIDTests(unittest.TestCase):
    def test_identity_placeholder_is_rejected(self) -> None:
        with self.assertRaises(CalibrationError):
            validate_homography(np.eye(3), camera_id="cam_b")
        validated = validate_homography(
            np.eye(3), camera_id="cam_a", reference_camera=True
        )
        np.testing.assert_allclose(validated, np.eye(3))

    def test_synchronized_geometry_reid_and_hungarian(self) -> None:
        rows = []
        people = {
            ("cam_a", 1): (10.0, 0.0),
            ("cam_b", 7): (11.0, 0.0),
            ("cam_a", 2): (200.0, 1.0),
            ("cam_b", 8): (201.0, 1.0),
        }
        for (camera_id, local_id), (foot_x, identity_axis) in people.items():
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
        detections = pd.DataFrame(rows)
        calibrated = apply_homographies(
            detections,
            {
                "cam_a": {"homography": np.eye(3), "reference_camera": True},
                "cam_b": {
                    "homography": [[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "reference_camera": False,
                },
            },
        )
        tracklets = summarize_tracklets(calibrated)
        embeddings = {
            make_tracklet_id("place_05", "cam_a", 1): [1.0, 0.0],
            make_tracklet_id("place_05", "cam_b", 7): [0.999, 0.001],
            make_tracklet_id("place_05", "cam_a", 2): [0.0, 1.0],
            make_tracklet_id("place_05", "cam_b", 8): [0.001, 0.999],
        }
        tracklets = attach_tracklet_embeddings(tracklets, embeddings)
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
        self.assertTrue((result.accepted_matches["spatial_method"] == "synchronized_median").all())
        self.assertEqual(result.mapping["global_track_id"].nunique(), 2)
        self.assertTrue(result.mapping["is_cross_camera"].all())

        joined = apply_global_ids(calibrated, result.mapping)
        global_sets = joined.groupby("local_track_id")["global_track_id"].first()
        self.assertEqual(global_sets.loc[1], global_sets.loc[7])
        self.assertEqual(global_sets.loc[2], global_sets.loc[8])
        self.assertNotEqual(global_sets.loc[1], global_sets.loc[2])

    def test_hungarian_does_not_force_two_weak_matches(self) -> None:
        candidates = pd.DataFrame(
            [
                {"store_id": "s", "camera_id_a": "a", "camera_id_b": "b", "tracklet_id_a": "a1", "tracklet_id_b": "b1", "association_time_sec": 1.0, "score": 0.10},
                {"store_id": "s", "camera_id_a": "a", "camera_id_b": "b", "tracklet_id_a": "a1", "tracklet_id_b": "b2", "association_time_sec": 1.0, "score": 0.70},
                {"store_id": "s", "camera_id_a": "a", "camera_id_b": "b", "tracklet_id_a": "a2", "tracklet_id_b": "b1", "association_time_sec": 1.0, "score": 0.70},
            ]
        )
        assigned = hungarian_assign(candidates, max_score=0.72)
        self.assertEqual(len(assigned), 1)
        self.assertEqual(
            (assigned.iloc[0]["tracklet_id_a"], assigned.iloc[0]["tracklet_id_b"]),
            ("a1", "b1"),
        )

    def test_hungarian_allows_disjoint_fragments_in_separate_windows(self) -> None:
        candidates = pd.DataFrame(
            [
                {"store_id": "s", "camera_id_a": "a", "camera_id_b": "b", "tracklet_id_a": "a_long", "tracklet_id_b": "b1", "association_time_sec": 1.0, "score": 0.10},
                {"store_id": "s", "camera_id_a": "a", "camera_id_b": "b", "tracklet_id_a": "a_long", "tracklet_id_b": "b2", "association_time_sec": 6.0, "score": 0.12},
            ]
        )
        assigned = hungarian_assign(
            candidates,
            max_score=0.72,
            assignment_window_sec=5.0,
        )
        self.assertEqual(len(assigned), 2)
        self.assertEqual(set(assigned["assignment_window"]), {0, 1})

    def test_cluster_rejects_incompatible_three_camera_chain(self) -> None:
        tracklets = pd.DataFrame(
            [
                {"store_id": "s", "camera_id": camera, "local_track_id": index, "tracklet_id": name, "start_sec": 0.0, "end_sec": 2.0, "start_floor_x": 0.0, "start_floor_y": 0.0, "end_floor_x": 1.0, "end_floor_y": 0.0, "mean_floor_x": 0.5, "mean_floor_y": 0.0}
                for index, (camera, name) in enumerate((("a", "a1"), ("b", "b1"), ("c", "c1")), start=1)
            ]
        )
        matches = pd.DataFrame(
            [
                {"store_id": "s", "tracklet_id_a": "a1", "tracklet_id_b": "b1", "score": 0.10},
                {"store_id": "s", "tracklet_id_a": "b1", "tracklet_id_b": "c1", "score": 0.20},
            ]
        )
        candidates = matches[["tracklet_id_a", "tracklet_id_b"]].copy()
        result = cluster_tracklets(
            tracklets,
            matches,
            candidates=candidates,
            cross_camera_conflict_max_gap_sec=8.0,
        )
        self.assertEqual(result.mapping["global_track_id"].nunique(), 2)
        self.assertEqual(len(result.accepted_matches), 1)
        self.assertEqual(
            result.rejected_matches.iloc[0]["rejection_reason"],
            "cross_camera_incompatible",
        )


if __name__ == "__main__":
    unittest.main()
