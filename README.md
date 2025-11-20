# Unattended Detection - README

Brief overview
- This folder contains scripts to detect abandoned luggage and evaluate models on video datasets using YOLO (Ultralytics).
- Key outputs: annotated videos (`annotated_videos/`), screenshots (`screenshots/`), and evaluation CSV (`metrics/unattended_results_summary.csv`).

Quick start / prerequisites
- Python 3.12
- Install required packages (example):
  pip install ultralytics opencv-python numpy matplotlib transformers
- torch is optional but recommended for faster inference time
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

Primary scripts
- [unattended_eval.py](unattended_eval.py) — Batch evaluation runner and CSV summary generator. See [`run_evaluation_on_video`](unattended_eval.py), [`load_cvat_annotations`](unattended_eval.py), [`write_summary_to_csv`](unattended_eval.py), and [`main`](unattended_eval.py).
- [unattended.py](unattended.py) — Real-time/recording abandoned-luggage detection demo. Uses movement helpers like [`calculate_distance_squared`](unattended.py) and [`has_movement`](unattended.py).
- [unattended_ft.py](unattended_ft.py) — Fine-tuning / alternate demo variant. Uses the same movement helpers in [`unattended_ft.py`](unattended_ft.py).
- [main.py](main.py) — Unused, with the algo from original paper. See [`calculate_distance_squared`](main.py) and [`has_movement`](main.py).
- [get_YOLO_detections.py](get_YOLO_detections.py) — Single-frame YOLO detection example and manual drawing of detections.
- [logic_code.py](logic_code.py) — (helper / shared logic used across demos; inspect file for specifics).
- [depth.py](depth.py), [depth_ft.py](depth_ft.py) — depth algo code (unfinished).

Configuration highlights (in unattended_eval.py)
- [`EVALUATION_MODE`](unattended_eval.py) — toggle batch evaluation on/off.
- [`VIDEO_FOLDER`](unattended_eval.py) and [`ANNOTATIONS_FOLDER`](unattended_eval.py) — where videos and CVAT XMLs are read from.
- [`MODELS_TO_EVALUATE`](unattended_eval.py) and [`VIDEOS_TO_PROCESS`](unattended_eval.py) — model and video lists used by the batch runner.
- [`IOU_THRESHOLD`](unattended_eval.py) and [`RESULTS_CSV_PATH`](unattended_eval.py) — evaluation thresholds and output CSV path.

How to run
- Run batch evaluation:
  python unattended_eval.py
  (Ensure [`EVALUATION_MODE`](unattended_eval.py) is True and `test_videos/` + `test_annotations/` are populated.)
- Run a demo / visualizer:
  python unattended.py
  or
  python unattended_ft.py
  or
  python main.py
- Run single-frame detection:
  python get_YOLO_detections.py

Inputs & outputs
- Inputs:
  - Video files: [test_videos/](test_videos/)
  - CVAT annotation XMLs: [test_annotations/](test_annotations/)
  - Model weights: [weights/](weights/)
- Outputs:
  - Annotated videos: [annotated_videos/](annotated_videos/)
  - Screenshots: [screenshots/](screenshots/)
  - Evaluation summary: [metrics/unattended_results_summary.csv](metrics/unattended_results_summary.csv)

Useful helpers to inspect
- [`run_evaluation_on_video`](unattended_eval.py) — core evaluation and annotation loop.
- [`load_cvat_annotations`](unattended_eval.py) — parses CVAT XML per video.
- [`write_summary_to_csv`](unattended_eval.py) / [`write_frame_data_to_csv`](unattended_eval.py) — CSV output helpers.
- Movement helpers: [`calculate_distance_squared`](main.py), [`has_movement`](main.py) (also defined in `unattended.py` / `unattended_ft.py` / `unattended_eval.py`).

Notes and tips
- Ensure FPS and video read succeed (scripts default to fallbacks like 30 FPS).
- Adjust model paths in the scripts (e.g., in [unattended_eval.py](unattended_eval.py) `MODELS_TO_EVALUATE`) to point to the correct files under [weights/](weights/).
- If CVAT XMLs are missing size information, evaluation will fall back to disabling spatial scaling for GT boxes.

Files referenced in this README
- [unattended_eval.py](unattended_eval.py)
- [unattended.py](unattended.py)
- [unattended_ft.py](unattended_ft.py)
- [main.py](main.py)
- [get_YOLO_detections.py](get_YOLO_detections.py)
- [logic_code.py](logic_code.py)
- [depth.py](depth.py)
- [depth_ft.py](depth_ft.py)
- [weights/](weights/)
- [test_videos/](test_videos/)
- [test_annotations/](test_annotations/)
- [annotated_videos/](annotated_videos/)
- [screenshots/](screenshots/)
- [metrics/unattended_results_summary.csv](metrics/unattended_results_summary.csv)

License / attribution
- See repository root README.md for general project info and license.
