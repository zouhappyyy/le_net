import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class InferenceResult:
    case_id: str
    task: str
    confidence: float
    status: str
    output_json: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "case_id": self.case_id,
            "task": self.task,
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "output_json": self.output_json,
        }


class InferenceService:
    """Mock inference service for fast workflow validation before model integration."""

    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run_case(self, case_path: Path, task_name: str) -> InferenceResult:
        case_path = Path(case_path)
        case_id = case_path.stem.replace(".nii", "")

        # Simulate runtime so the GUI behavior is closer to real inference.
        time.sleep(0.3)

        confidence = self._estimate_confidence(case_id, task_name)
        status = "needs_review" if confidence < 0.75 else "auto_draft_ready"

        payload = {
            "case_id": case_id,
            "task": task_name,
            "confidence": round(confidence, 4),
            "status": status,
            "recommendation": self._build_recommendation(task_name, confidence),
        }

        out_path = self.output_root / f"{case_id}_{task_name}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return InferenceResult(
            case_id=case_id,
            task=task_name,
            confidence=confidence,
            status=status,
            output_json=str(out_path),
        )

    @staticmethod
    def _estimate_confidence(case_id: str, task_name: str) -> float:
        seed = sum(ord(ch) for ch in (case_id + task_name))
        return 0.6 + (seed % 35) / 100.0

    @staticmethod
    def _build_recommendation(task_name: str, confidence: float) -> str:
        if task_name == "T_staging":
            return "T-stage proposal generated from lesion extent and boundary cues."
        if task_name == "rt_targeting":
            return "GTV auto-contour generated; physician contour refinement required."
        if task_name == "response_eval":
            return "Longitudinal volume trend prepared for treatment response review."
        return "Task completed."

