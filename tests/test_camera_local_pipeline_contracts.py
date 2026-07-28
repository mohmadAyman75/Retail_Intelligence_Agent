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
    def test_zone_stage_uses_camera_local_identity_only(self) -> None:
        source = notebook_code("02_Global_Fusion_and_Zones.ipynb")
        self.assertIn("camera_track_uid", source)
        self.assertNotIn("run_global_fusion", source)
        self.assertNotIn("mtmc_reid", source)
        self.assertNotIn("global_track_id", source)
        self.assertNotIn("global_person_id", source)

    def test_analytics_keeps_the_camera_in_all_track_keys(self) -> None:
        source = notebook_code("03_Retail_Analytics_Agent.ipynb")
        self.assertIn("['store_id', 'camera_id', 'camera_track_uid']", source)
        self.assertIn("'camera_id', 'window_start_sec'", source)
        self.assertNotIn("global_track_id", source)

    def test_run_all_has_no_reid_preflight(self) -> None:
        source = notebook_code("05_Run_All.ipynb")
        self.assertNotIn("setup_reid", source)
        self.assertNotIn("osnet", source.casefold())
        self.assertNotIn("mtmc", source.casefold())


if __name__ == "__main__":
    unittest.main()
