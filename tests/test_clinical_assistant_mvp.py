import json
import tempfile
import unittest
from pathlib import Path

from clinical_assistant.modules.case_manager import CaseManager
from clinical_assistant.modules.inference_service import InferenceService


class ClinicalAssistantMvpTests(unittest.TestCase):
    def test_case_manager_lists_supported_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a_case.npy").write_text("x", encoding="utf-8")
            (root / "b_case.nii.gz").write_text("x", encoding="utf-8")
            (root / "ignore.txt").write_text("x", encoding="utf-8")

            cases = CaseManager(root).list_cases()

            self.assertEqual(2, len(cases))
            self.assertEqual(["a_case.npy", "b_case.nii.gz"], [p.name for p in cases])

    def test_inference_service_generates_json_artifact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_path = root / "demo_case.npy"
            case_path.write_text("x", encoding="utf-8")

            output_root = root / "out"
            service = InferenceService(output_root)
            result = service.run_case(case_path, "T_staging")

            artifact = Path(result.output_json)
            self.assertTrue(artifact.exists())

            payload = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual("demo_case", payload["case_id"])
            self.assertEqual("T_staging", payload["task"])
            self.assertIn("confidence", payload)


if __name__ == "__main__":
    unittest.main()

