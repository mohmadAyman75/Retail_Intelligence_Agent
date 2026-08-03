from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Notebook"))

from reid_balanced import (  # noqa: E402
    aggregate_local_components,
    annotate_mapping_quality,
    build_local_components,
    cluster_anchor_stages,
    color_for_identity,
    interpolate_display_rows,
    select_anchor_staged_matches,
    select_disjoint_local_stitches,
    select_mutual_best_matches,
    synchronized_cosine_distance,
)


class BalancedReIDTests(unittest.TestCase):
    def test_display_interpolation_fills_only_one_skipped_frame(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "camera_id": "cam_17",
                    "local_track_id": 4,
                    "frame_index": 110,
                    "timestamp_sec": 22.0,
                    "x1": 10.0,
                    "y1": 20.0,
                    "x2": 30.0,
                    "y2": 60.0,
                    "global_track_id": "global_000133",
                },
                {
                    "camera_id": "cam_17",
                    "local_track_id": 4,
                    "frame_index": 112,
                    "timestamp_sec": 22.4,
                    "x1": 14.0,
                    "y1": 22.0,
                    "x2": 34.0,
                    "y2": 62.0,
                    "global_track_id": "global_000133",
                },
                {
                    "camera_id": "cam_17",
                    "local_track_id": 9,
                    "frame_index": 120,
                    "timestamp_sec": 24.0,
                    "x1": 50.0,
                    "y1": 50.0,
                    "x2": 80.0,
                    "y2": 100.0,
                    "global_track_id": "global_000200",
                },
                {
                    "camera_id": "cam_17",
                    "local_track_id": 9,
                    "frame_index": 124,
                    "timestamp_sec": 24.8,
                    "x1": 54.0,
                    "y1": 50.0,
                    "x2": 84.0,
                    "y2": 100.0,
                    "global_track_id": "global_000200",
                },
            ]
        )

        display = interpolate_display_rows(rows, max_frame_gap=2)
        interpolated = display.loc[display["display_interpolated"]]
        self.assertEqual(interpolated["frame_index"].tolist(), [111])
        self.assertEqual(interpolated.iloc[0]["global_track_id"], "global_000133")
        self.assertEqual(interpolated.iloc[0]["x1"], 12.0)
        self.assertFalse(display["frame_index"].isin([121, 122, 123]).any())

    def test_synchronized_distance_uses_one_to_one_temporal_evidence(self) -> None:
        distance, evidence = synchronized_cosine_distance(
            [0.0, 1.0, 2.0],
            np.asarray([[1.0, 0.0], [0.99, 0.01], [1.0, 0.0]]),
            [0.1, 1.1, 2.1],
            np.asarray([[1.0, 0.0], [1.0, 0.0], [0.99, 0.01]]),
            tolerance_sec=0.2,
            min_samples=3,
            max_samples=10,
        )
        self.assertEqual(evidence, 3)
        self.assertLess(distance, 0.001)

        missing_distance, missing_evidence = synchronized_cosine_distance(
            [0.0],
            np.asarray([[1.0, 0.0]]),
            [5.0],
            np.asarray([[1.0, 0.0]]),
            tolerance_sec=0.2,
            min_samples=1,
            max_samples=10,
        )
        self.assertTrue(np.isnan(missing_distance))
        self.assertEqual(missing_evidence, 0)

    def test_mutual_best_rejects_small_margin(self) -> None:
        candidates = pd.DataFrame(
            [
                {"store_id": "s", "camera_id_a": "a", "camera_id_b": "b", "tracklet_id_a": "a1", "tracklet_id_b": "b1", "score": 0.10},
                {"store_id": "s", "camera_id_a": "a", "camera_id_b": "b", "tracklet_id_a": "a1", "tracklet_id_b": "b2", "score": 0.12},
                {"store_id": "s", "camera_id_a": "a", "camera_id_b": "b", "tracklet_id_a": "a2", "tracklet_id_b": "b2", "score": 0.05},
            ]
        )
        accepted, rejected = select_mutual_best_matches(
            candidates, max_score=0.78, min_margin=0.05
        )
        self.assertEqual(
            accepted[["tracklet_id_a", "tracklet_id_b"]].values.tolist(),
            [["a2", "b2"]],
        )
        reasons = set(rejected["rejection_reason"])
        self.assertIn("ambiguous_margin", reasons)
        self.assertIn("not_mutual_best", reasons)

    def test_local_stitching_builds_a_component_before_cross_camera(self) -> None:
        local_candidates = pd.DataFrame(
            [
                {"tracklet_id_a": "a1", "tracklet_id_b": "a2", "score": 0.10},
                {"tracklet_id_a": "a2", "tracklet_id_b": "a3", "score": 0.12},
            ]
        )
        stitches = select_disjoint_local_stitches(local_candidates, max_score=0.78)
        self.assertEqual(len(stitches), 2)
        components = build_local_components(["a1", "a2", "a3", "b1"], stitches)
        self.assertEqual({components[key] for key in ("a1", "a2", "a3")}, {"a1"})

        tracklets = pd.DataFrame(
            [
                {"store_id": "s", "camera_id": "a", "local_track_id": index, "tracklet_id": name, "start_sec": float(index), "end_sec": float(index) + 0.5, "is_employee": False}
                for index, name in enumerate(("a1", "a2", "a3"), start=1)
            ]
        )
        embeddings = {name: np.asarray([1.0, 0.0]) for name in ("a1", "a2", "a3")}
        samples = {
            name: {"times": np.asarray([float(index)]), "vectors": np.asarray([[1.0, 0.0]])}
            for index, name in enumerate(("a1", "a2", "a3"), start=1)
        }
        points = {
            name: pd.DataFrame({"time_sec": [float(index)], "floor_x": [0.0], "floor_y": [0.0]})
            for index, name in enumerate(("a1", "a2", "a3"), start=1)
        }
        pixels = {
            name: pd.DataFrame({"time_sec": [float(index)], "pixel_x": [10.0], "pixel_y": [10.0]})
            for index, name in enumerate(("a1", "a2", "a3"), start=1)
        }
        aggregated, _, _, _ = aggregate_local_components(
            tracklets,
            {name: components[name] for name in ("a1", "a2", "a3")},
            embeddings,
            samples,
            points,
            pixels,
        )
        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated.iloc[0]["member_tracklets"], ("a1", "a2", "a3"))

    def test_identity_color_is_stable_across_cameras(self) -> None:
        self.assertEqual(color_for_identity("global_000133"), color_for_identity("global_000133"))
        self.assertNotEqual(color_for_identity("global_000133"), color_for_identity("global_000134"))

    def test_anchor_stages_reject_wrong_late_camera_edge(self) -> None:
        rules = [
            {"name": "anchor_17_18", "parent_camera": "17", "target_camera": "18", "min_synchronized_samples": 3, "max_score": 0.78, "min_margin": 0.05},
            {"name": "attach_19_to_17", "parent_camera": "17", "target_camera": "19", "min_synchronized_samples": 5, "max_score": 0.78, "min_margin": 0.05},
            {"name": "attach_20_to_19", "parent_camera": "19", "target_camera": "20", "min_synchronized_samples": 5, "max_score": 0.78, "min_margin": 0.05},
        ]
        candidates = pd.DataFrame(
            [
                {"store_id": "s", "camera_id_a": "17", "camera_id_b": "18", "tracklet_id_a": "a17", "tracklet_id_b": "a18", "score": 0.10, "synchronized_samples": 4},
                {"store_id": "s", "camera_id_a": "17", "camera_id_b": "19", "tracklet_id_a": "a17", "tracklet_id_b": "a19", "score": 0.12, "synchronized_samples": 6},
                # Wrong stage edge: Cam20 tries to attach directly to Cam17.
                {"store_id": "s", "camera_id_a": "17", "camera_id_b": "20", "tracklet_id_a": "a17", "tracklet_id_b": "wrong20", "score": 0.01, "synchronized_samples": 9},
                {"store_id": "s", "camera_id_a": "19", "camera_id_b": "20", "tracklet_id_a": "a19", "tracklet_id_b": "a20", "score": 0.14, "synchronized_samples": 6},
            ]
        )
        selected, rejected = select_anchor_staged_matches(candidates, stage_rules=rules)
        self.assertEqual(set(selected["association_stage"]), {"anchor_17_18", "attach_19_to_17", "attach_20_to_19"})
        self.assertTrue(rejected.empty)

        tracklets = pd.DataFrame(
            [
                {"store_id": "s", "camera_id": camera, "local_track_id": index, "tracklet_id": tracklet, "start_sec": 1.0, "end_sec": 2.0}
                for index, (camera, tracklet) in enumerate((("17", "a17"), ("18", "a18"), ("19", "a19"), ("20", "a20"), ("20", "wrong20")), start=1)
            ]
        )
        mapping, accepted, conflicts = cluster_anchor_stages(
            tracklets,
            local_matches=pd.DataFrame(),
            staged_cross_matches=selected,
            stage_rules=rules,
        )
        anchor_id = mapping.loc[mapping["tracklet_id"] == "a17", "global_track_id"].iloc[0]
        self.assertEqual(
            mapping.loc[mapping["tracklet_id"].isin(["a18", "a19", "a20"]), "global_track_id"].tolist(),
            [anchor_id, anchor_id, anchor_id],
        )
        self.assertNotEqual(
            mapping.loc[mapping["tracklet_id"] == "wrong20", "global_track_id"].iloc[0],
            anchor_id,
        )
        self.assertEqual(len(accepted), 3)
        self.assertTrue(conflicts.empty)

    def test_target_component_cannot_be_merged_twice(self) -> None:
        rules = [
            {"name": "attach_19_to_17", "parent_camera": "17", "target_camera": "19", "min_synchronized_samples": 0, "max_score": 0.78, "min_margin": 0.05},
        ]
        tracklets = pd.DataFrame(
            [
                {"store_id": "s", "camera_id": camera, "local_track_id": index, "tracklet_id": tracklet, "start_sec": 1.0, "end_sec": 2.0}
                for index, (camera, tracklet) in enumerate((("17", "a17"), ("17", "b17"), ("19", "a19")), start=1)
            ]
        )
        matches = pd.DataFrame(
            [
                {"tracklet_id_a": "a17", "tracklet_id_b": "a19", "score": 0.10, "association_stage": "attach_19_to_17", "is_cross_camera": True},
                {"tracklet_id_a": "b17", "tracklet_id_b": "a19", "score": 0.20, "association_stage": "attach_19_to_17", "is_cross_camera": True},
            ]
        )
        _, accepted, rejected = cluster_anchor_stages(
            tracklets, local_matches=pd.DataFrame(), staged_cross_matches=matches, stage_rules=rules
        )
        self.assertEqual(len(accepted), 1)
        self.assertIn("target_already_associated", set(rejected["rejection_reason"]))

    def test_mapping_records_stage_and_direct_evidence(self) -> None:
        mapping = pd.DataFrame(
            [
                {"camera_id": "17", "tracklet_id": "a17", "global_track_id": "global_000001"},
                {"camera_id": "19", "tracklet_id": "a19", "global_track_id": "global_000001"},
                {"camera_id": "20", "tracklet_id": "unmatched20", "global_track_id": "global_000002"},
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "tracklet_id_a": "a17",
                    "tracklet_id_b": "a19",
                    "is_cross_camera": True,
                    "synchronized_samples": 6,
                    "match_margin": 0.12,
                    "association_stage": "attach_19_to_17",
                    "evidence_camera_pair": "17<->19",
                    "appearance_distance": 0.21,
                    "floor_distance": 16.0,
                }
            ]
        )
        annotated = annotate_mapping_quality(
            mapping, matches, pd.DataFrame(), target_camera_ids={"19", "20"}
        )
        self.assertEqual(
            annotated.loc[annotated["tracklet_id"] == "unmatched20", "identity_status"].iloc[0],
            "unmatched_target",
        )
        row = annotated.loc[annotated["tracklet_id"] == "a19"].iloc[0]
        self.assertEqual(row["association_stage"], "attach_19_to_17")
        self.assertEqual(row["evidence_camera_pair"], "17<->19")
        self.assertEqual(row["direct_evidence_count"], 6)
        self.assertEqual(row["spatial_distance"], 16.0)


if __name__ == "__main__":
    unittest.main()
