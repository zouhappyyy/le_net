from PyQt5 import QtCore, QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    run_requested = QtCore.pyqtSignal(str, str)
    data_root_changed = QtCore.pyqtSignal(str)

    TASK_OPTIONS = [
        ("T staging", "T_staging"),
        ("RT target generation", "rt_targeting"),
        ("Response evaluation", "response_eval"),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clinical Assistant MVP")
        self.resize(1200, 760)

        self.case_list = QtWidgets.QListWidget()
        self.case_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        self.result_text = QtWidgets.QPlainTextEdit()
        self.result_text.setReadOnly(True)

        self.task_combo = QtWidgets.QComboBox()
        for label, value in self.TASK_OPTIONS:
            self.task_combo.addItem(label, userData=value)

        self.data_root_input = QtWidgets.QLineEdit()
        self.data_root_btn = QtWidgets.QPushButton("Browse")
        self.refresh_btn = QtWidgets.QPushButton("Refresh Cases")
        self.run_btn = QtWidgets.QPushButton("Run")

        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(500)

        self._build_layout()
        self._connect_events()

    def _build_layout(self):
        root_widget = QtWidgets.QWidget()
        self.setCentralWidget(root_widget)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Horizontal)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.addWidget(QtWidgets.QLabel("Cases"))
        left_layout.addWidget(self.case_list)

        center_panel = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center_panel)
        center_layout.addWidget(QtWidgets.QLabel("Result Preview"))
        center_layout.addWidget(self.result_text)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.addWidget(QtWidgets.QLabel("Data Root"))

        data_root_row = QtWidgets.QHBoxLayout()
        data_root_row.addWidget(self.data_root_input)
        data_root_row.addWidget(self.data_root_btn)
        right_layout.addLayout(data_root_row)

        right_layout.addWidget(self.refresh_btn)
        right_layout.addWidget(QtWidgets.QLabel("Clinical Task"))
        right_layout.addWidget(self.task_combo)
        right_layout.addWidget(self.run_btn)
        right_layout.addWidget(QtWidgets.QLabel("Logs"))
        right_layout.addWidget(self.log_text)
        right_layout.addStretch(1)

        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 550, 350])

        root_layout = QtWidgets.QVBoxLayout(root_widget)
        root_layout.addWidget(splitter)

    def _connect_events(self):
        self.refresh_btn.clicked.connect(self._emit_data_root_changed)
        self.data_root_btn.clicked.connect(self._pick_data_root)
        self.run_btn.clicked.connect(self._emit_run_requested)

    def _pick_data_root(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select case folder")
        if path:
            self.data_root_input.setText(path)
            self.data_root_changed.emit(path)

    def _emit_data_root_changed(self):
        self.data_root_changed.emit(self.data_root_input.text().strip())

    def _emit_run_requested(self):
        case_item = self.case_list.currentItem()
        if case_item is None:
            self.append_log("No case selected.")
            return
        case_path = case_item.data(QtCore.Qt.UserRole)
        task_value = self.task_combo.currentData()
        self.run_requested.emit(case_path, task_value)

    def set_cases(self, case_paths):
        self.case_list.clear()
        for path in case_paths:
            item = QtWidgets.QListWidgetItem(path.name)
            item.setData(QtCore.Qt.UserRole, str(path))
            self.case_list.addItem(item)
        self.append_log(f"Loaded {len(case_paths)} cases.")

    def set_result_text(self, text: str):
        self.result_text.setPlainText(text)

    def append_log(self, text: str):
        self.log_text.appendPlainText(text)

