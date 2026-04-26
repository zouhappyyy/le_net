import sys
from pathlib import Path

from PyQt5 import QtWidgets

from clinical_assistant.app import ClinicalAssistantController, load_config


def main():
    config_path = Path(__file__).resolve().parent / "config" / "default.json"
    config = load_config(config_path)

    app = QtWidgets.QApplication(sys.argv)
    controller = ClinicalAssistantController(config)
    controller.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

