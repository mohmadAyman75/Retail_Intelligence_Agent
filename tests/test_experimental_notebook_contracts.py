from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "Notebook"


def notebook_code(name: str) -> str:
    notebook = json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    ast.parse(code)
    return code


class ExperimentalNotebookContractTests(unittest.TestCase):
    def test_staff_roles_notebook_is_self_contained_and_additive(self) -> None:
        source = notebook_code("01b_Staff_Customer_Zones.ipynb")
        self.assertIn("staff_zones.json", source)
        self.assertIn("cv2.pointPolygonTest", source)
        self.assertIn("track_roles.csv", source)
        self.assertIn("staff_customer_summary.csv", source)
        self.assertIn("roles_{camera_id}.mp4", source)
        self.assertNotIn("import mtmc_reid", source)
        self.assertNotIn("local_tracks.to_csv", source)

    def test_global_demo_notebook_is_self_contained_and_explicitly_demo(self) -> None:
        source = notebook_code("02b_Global_Identity_Demo_Mode.ipynb")
        self.assertIn("DEMO_SIMILARITY_THRESHOLD = 0.75", source)
        self.assertIn("demo_appearance_only", source)
        self.assertIn("global_tracks_demo.csv", source)
        self.assertIn("is_demo_mode", source)
        self.assertIn("linear_sum_assignment", source)
        self.assertIn("build_global_mapping", source)
        self.assertNotIn("import mtmc_reid", source)
        self.assertNotIn("from mtmc_reid", source)

    def test_support_scripts_compile(self) -> None:
        for name in ("mtmc_reid.py", "calibrate_homography.py", "pick_staff_zone.py"):
            ast.parse((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
