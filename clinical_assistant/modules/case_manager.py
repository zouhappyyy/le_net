from pathlib import Path
from typing import List


SUPPORTED_SUFFIXES = (".npy", ".nii", ".nii.gz")


class CaseManager:
    """Discovers available study files under a selected data root."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)

    def list_cases(self) -> List[Path]:
        if not self.data_root.exists():
            return []

        files = []
        for suffix in SUPPORTED_SUFFIXES:
            files.extend(self.data_root.glob(f"*{suffix}"))

        # Deduplicate and keep deterministic display order.
        unique_files = sorted({f.resolve() for f in files}, key=lambda p: p.name.lower())
        return unique_files

