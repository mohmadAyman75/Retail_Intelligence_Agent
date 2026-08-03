from __future__ import annotations

import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "Notebook"


def notebook_code(name: str) -> str:
    notebook = json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


class CameraLocalPipelineContractTests(unittest.TestCase):
    def test_zone_stage_supports_global_track_id(self) -> None:
        source = notebook_code("02_Global_Fusion_and_Zones.ipynb")
        self.assertIn("camera_track_uid", source)
        self.assertIn("global_track_id", source)

    def test_analytics_keeps_the_camera_in_all_track_keys(self) -> None:
        source = notebook_code("03_Retail_Analytics_Agent.ipynb")
        self.assertIn("['store_id', 'camera_id', 'camera_track_uid']", source)
        self.assertIn("'camera_id', 'window_start_sec'", source)

    def test_run_all_uses_the_merged_global_reid_stage(self) -> None:
        source = notebook_code("05_Run_All.ipynb")
        self.assertIn("02_Global_Fusion_and_Zones.ipynb", source)
        self.assertNotIn("02a_Global_ReID.ipynb", source)

    def test_legacy_reid_notebook_cannot_overwrite_active_artifacts(self) -> None:
        source = notebook_code("02a_Global_ReID.ipynb")
        self.assertNotIn("TABLES_DIR / 'global_tracks.csv'", source)
        self.assertNotIn("TABLES_DIR / 'reid_mapping.csv'", source)
        self.assertNotIn("TABLES_DIR / 'reid_report.json'", source)
        self.assertIn("global_tracks_legacy.csv", source)
        self.assertIn("reid_mapping_legacy.csv", source)
        self.assertIn("reid_report_legacy.json", source)

    def test_balanced_reid_contract_is_visible_to_notebook_and_dashboard(self) -> None:
        notebook_source = notebook_code("02_Global_Fusion_and_Zones.ipynb")
        dashboard_source = (PROJECT_ROOT / "Output" / "app" / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("aggregate_local_components", notebook_source)
        self.assertIn("synchronized_cosine_distance", notebook_source)
        self.assertIn("interpolate_display_rows", notebook_source)
        self.assertIn("annotate_mapping_quality", notebook_source)
        self.assertIn("'artifacts': artifacts", notebook_source)
        self.assertIn("validate_reid_manifest", dashboard_source)
        self.assertIn("Artifact hashes are missing", dashboard_source)

    def test_place_05_uses_all_six_camera_pairs_with_balanced_defaults(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "Data" / "config" / "mtmc_reid_config.json").read_text(encoding="utf-8")
        )
        association = config["association"]
        allowed_pairs = {frozenset(pair) for pair in association["allowed_camera_pairs"]}
        cameras = {
            f"CAFE_place_05_camera_{camera_id}_15min"
            for camera_id in (17, 18, 19, 20)
        }
        expected_pairs = {
            frozenset((left, right))
            for left in cameras
            for right in cameras
            if left < right
        }
        self.assertEqual(allowed_pairs, expected_pairs)
        self.assertEqual(config["reid"]["samples_per_tracklet"], 24)
        self.assertEqual(association["same_camera_max_time_gap_sec"], 10.0)
        self.assertEqual(association["min_match_margin"], 0.05)


if __name__ == "__main__":
    unittest.main()
