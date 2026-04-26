import json
from pathlib import Path


class ResultViewer:
    """Loads generated result artifacts for display in the UI."""

    @staticmethod
    def load_result_text(result_json_path: str) -> str:
        path = Path(result_json_path)
        if not path.exists():
            return f"Result file not found: {result_json_path}"

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return path.read_text(encoding="utf-8")

        lines = [
            f"Case: {payload.get('case_id', 'unknown')}",
            f"Task: {payload.get('task', 'unknown')}",
            f"Confidence: {payload.get('confidence', 'unknown')}",
            f"Status: {payload.get('status', 'unknown')}",
            f"Recommendation: {payload.get('recommendation', 'n/a')}",
            "",
            f"Artifact: {path}",
        ]
        return "\n".join(lines)

