# Clinical Assistant MVP (PyQt)

This is a minimal desktop prototype to validate the workflow from case selection to task execution and physician review.

## Features

- Case discovery from a local folder (`.npy`, `.nii`, `.nii.gz`)
- Three clinical task placeholders:
  - `T_staging`
  - `rt_targeting`
  - `response_eval`
- Non-blocking execution with background worker thread
- Structured JSON artifact output per case and task
- Result preview + operation log panel

## Project Layout

- `clinical_assistant/main.py`: app entrypoint
- `clinical_assistant/app.py`: controller and worker orchestration
- `clinical_assistant/ui/main_window.py`: UI layout and events
- `clinical_assistant/modules/case_manager.py`: case discovery
- `clinical_assistant/modules/inference_service.py`: mock task execution backend
- `clinical_assistant/modules/result_viewer.py`: preview formatting

## Install

```bash
pip install -r requirements-pyqt-mvp.txt
```

## Run

```bash
python -m clinical_assistant.main
```

## Quick Validation

1. Open the app.
2. Click `Browse` and select a folder with sample files (`.npy`/`.nii`/`.nii.gz`).
3. Pick a case from the left panel.
4. Select a task from the right panel.
5. Click `Run` and verify output in the center panel.

## Integration Notes

- Replace the mock implementation in `InferenceService.run_case` with your real inference pipeline.
- Keep the output JSON contract stable so UI and downstream reporting remain compatible.

