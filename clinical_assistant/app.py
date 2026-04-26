import json
import traceback
from dataclasses import dataclass
from pathlib import Path

from PyQt5 import QtCore

from clinical_assistant.modules.case_manager import CaseManager
from clinical_assistant.modules.inference_service import InferenceService
from clinical_assistant.modules.result_viewer import ResultViewer
from clinical_assistant.ui.main_window import MainWindow


@dataclass
class AppConfig:
    data_root: str
    output_root: str


def load_config(config_path: Path) -> AppConfig:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return AppConfig(
        data_root=payload.get("data_root", ""),
        output_root=payload.get("output_root", "clinical_outputs"),
    )


class InferenceWorker(QtCore.QThread):
    completed = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, inference_service: InferenceService, case_path: str, task_name: str):
        super().__init__()
        self.inference_service = inference_service
        self.case_path = case_path
        self.task_name = task_name

    def run(self):
        try:
            result = self.inference_service.run_case(Path(self.case_path), self.task_name)
            self.completed.emit(result.to_dict())
        except Exception as exc:  # pylint: disable=broad-except
            details = "\n".join([str(exc), traceback.format_exc()])
            self.failed.emit(details)


class ClinicalAssistantController(QtCore.QObject):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.window = MainWindow()
        self.window.data_root_input.setText(config.data_root)

        self.case_manager = CaseManager(Path(config.data_root))
        self.inference_service = InferenceService(Path(config.output_root))
        self.result_viewer = ResultViewer()

        self._worker = None
        self._connect_signals()
        self.refresh_cases(config.data_root)

    def _connect_signals(self):
        self.window.data_root_changed.connect(self.refresh_cases)
        self.window.run_requested.connect(self.run_inference)

    @QtCore.pyqtSlot(str)
    def refresh_cases(self, data_root: str):
        if not data_root:
            self.window.append_log("Please select a case folder.")
            return

        self.case_manager = CaseManager(Path(data_root))
        cases = self.case_manager.list_cases()
        self.window.set_cases(cases)

    @QtCore.pyqtSlot(str, str)
    def run_inference(self, case_path: str, task_name: str):
        if self._worker is not None and self._worker.isRunning():
            self.window.append_log("Inference already running. Please wait.")
            return

        self.window.append_log(f"Running {task_name} for {Path(case_path).name} ...")
        self._worker = InferenceWorker(self.inference_service, case_path, task_name)
        self._worker.completed.connect(self._on_inference_completed)
        self._worker.failed.connect(self._on_inference_failed)
        self._worker.start()

    @QtCore.pyqtSlot(dict)
    def _on_inference_completed(self, payload: dict):
        result_text = self.result_viewer.load_result_text(payload["output_json"])
        self.window.set_result_text(result_text)
        self.window.append_log(
            f"Done: case={payload['case_id']} task={payload['task']} confidence={payload['confidence']}"
        )

    @QtCore.pyqtSlot(str)
    def _on_inference_failed(self, details: str):
        self.window.append_log("Inference failed. See details in preview panel.")
        self.window.set_result_text(details)

    def show(self):
        self.window.show()

