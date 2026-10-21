# AgentCam

AgentCam V1 is a reproducible experimental framework for camera-degraded dashcam object detection. The V1 contribution is a rule-based agentic perception router around object detectors, not a learned router, VLM/LLM reasoning, or heavy image restoration.

Development target: RTX PRO 6000 for offline experiments. Jetson AGX Orin deployment is intentionally left as a later validation stage.

## Dataset Assumptions

- Google Drive file ID: `1ROUN1MkJ2l36KLBsEf_-BNF81xFD2uWe`
- Expected archive: `data/raw/data_eda_most25.zip`
- Input annotations are assumed to be YOLO format.
- The raw archive is assumed to contain only original `train` and `test` splits.
- V1 recognizes degradation types from paths/filenames:
  - `blur`, `blurry` -> `blur`
  - `overexposure`, `exposure`, `over_exposure`, `glare`, `saturated`, `sunlight` -> `overexposure`
  - anything else -> `unknown`
- Clean/normal data is not assumed unless the data audit discovers it.

Label policy:

- Existing empty `.txt` label file: labeled negative frame.
- Missing `.txt` label file: unlabeled frame.
- Unlabeled frames may be used for degradation score statistics and latency measurement.
- Unlabeled frames are excluded from supervised training and mAP evaluation.

Split policy:

- `train_core`: sampled from original train for fine-tuning.
- `val_from_train`: sampled from original train for checkpoint selection and router threshold calibration.
- `test`: original test, untouched for final reporting only.
- If every train image has an inferred `sequence_id`, splitting is sequence-level. Otherwise the code uses image-level stratified splitting by degradation type and reports that limitation in `artifacts/data_report.md`.

## Setup

Run commands from the repository root (`~/agentcam`). Source modules live directly under this root, for example `data/`, `degradation/`, `detectors/`, `evaluation/`, `routing/`, and `workers/`.

Use `python3` if your environment does not provide a `python` executable.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Download

```bash
bash scripts/00_download_data.sh
```

The script uses:

```bash
gdown --fuzzy "https://drive.google.com/file/d/1ROUN1MkJ2l36KLBsEf_-BNF81xFD2uWe/view" -O data/raw/data_eda_most25.zip
```

If Google Drive permissions, quota, or authentication block `gdown`, manually download the file and place it at `data/raw/data_eda_most25.zip`.

## Prepare Data

```bash
python3 -m data_utils.prepare_yolo_train_test \
  --raw data/raw/data_eda_most25.zip \
  --out data/processed/agentcam \
  --train_val_ratio 0.85 \
  --allow_unlabeled \
  --seed 42
```

Outputs:

- `data/processed/agentcam/manifest.csv`
- `data/processed/agentcam/data_trainval.yaml`
- `data/processed/agentcam/data_test.yaml`
- `artifacts/data_report.md`

Validate labels:

```bash
python3 -m data_utils.validate_yolo \
  --data data/processed/agentcam \
  --out artifacts/validation_report.md
```

Compute degradation scores:

```bash
python3 -m degradation.compute_scores \
  --data data/processed/agentcam \
  --out artifacts/scores/degradation_scores.csv
```

Normalization is fit on `train_core` only, then applied to `val_from_train` and `test`.

## Training And Baselines

YOLOv8 fine-tuning from COCO-pretrained weights:

```bash
bash scripts/10_train_yolo.sh \
  --data data/processed/agentcam/data_trainval.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640
```

RT-DETR fine-tuning or verifier baseline:

```bash
bash scripts/11_train_rtdetr.sh \
  --data data/processed/agentcam/data_trainval.yaml \
  --model rtdetr-l.pt \
  --epochs 50 \
  --imgsz 640
```

Detector baseline runner examples:

```bash
python3 -m evaluation.run_detector \
  --data data/processed/agentcam \
  --split test \
  --detector yolo \
  --weights artifacts/runs/yolov8_agentcam/weights/best.pt \
  --out artifacts/predictions/yolo_test.json

python3 -m evaluation.run_detector \
  --data data/processed/agentcam \
  --split test \
  --detector yolo \
  --enhancement exposure \
  --weights artifacts/runs/yolov8_agentcam/weights/best.pt \
  --out artifacts/predictions/yolo_exposure_test.json

python3 -m evaluation.run_detector \
  --data data/processed/agentcam \
  --split test \
  --detector yolo_plus_rtdetr \
  --weights artifacts/runs/yolov8_agentcam/weights/best.pt \
  --rtdetr_weights weights/rtdetr_agentcam.pt \
  --out artifacts/predictions/yolo_plus_rtdetr_test.json
```

Convert labeled test frames to COCO and evaluate:

```bash
python3 -m data_utils.yolo_to_coco \
  --data data/processed/agentcam \
  --split test \
  --out artifacts/coco/test_gt.json

python3 -m evaluation.coco_eval \
  --gt artifacts/coco/test_gt.json \
  --pred artifacts/predictions/yolo_test.json \
  --out artifacts/metrics/yolo_test.json
```

Evaluation protocol notes:

- `run_detector` defaults follow the frozen offline protocol: `imgsz=640`, `conf=0.001` (candidate retention, not a deployment threshold), `iou=0.7`, `max_det=300`, batch 1, with warm-up before the timed loop. Per-frame latency records include the ultralytics preprocess/inference/postprocess decomposition.
- For COCO-pretrained weights pass `--class_map coco_car_truck` so native COCO classes (2 car, 7 truck) are remapped to dataset classes (0, 1) and prediction is restricted to them.
- `coco_eval` reports macro mAP (mean of 101-point interpolated per-class AP), a `per_class` block, and a `by_degradation` block over every degradation label present. If the GT JSON lacks degradation labels, pass `--images_meta <csv>` (e.g. an `evaluated_images.csv` with a `manual_degradation` column).

Select a deployment confidence threshold from saved predictions (never on the test split):

```bash
python3 -m evaluation.operating_point \
  --gt artifacts/coco/val_gt.json \
  --pred artifacts/predictions/yolo_val.json \
  --objective f1 \
  --out artifacts/metrics/yolo_operating_point.json \
  --curve_out artifacts/metrics/yolo_pr_curve.csv
```

`--objective recall_at_precision --min_precision 0.8` selects the highest-recall threshold that keeps precision at or above the floor.

## Rule-Based Router

Router thresholds live in `configs/router_rule.yaml`. Quantile calibration uses `val_from_train` scores only.

```bash
python3 -m routing.run_router \
  --data data/processed/agentcam \
  --scores artifacts/scores/degradation_scores.csv \
  --yolo_weights artifacts/runs/yolov8_agentcam/weights/best.pt \
  --rtdetr_weights weights/rtdetr_agentcam.pt \
  --config configs/router_rule.yaml \
  --split test \
  --out artifacts/predictions/agentcam_router_test.json \
  --log artifacts/logs/router_test.jsonl
```

If RT-DETR weights are not available:

```bash
python3 -m routing.run_router \
  --data data/processed/agentcam \
  --scores artifacts/scores/degradation_scores.csv \
  --yolo_weights artifacts/runs/yolov8_agentcam/weights/best.pt \
  --rtdetr_weights weights/rtdetr_agentcam.pt \
  --config configs/router_rule.yaml \
  --split test \
  --out artifacts/predictions/agentcam_router_test.json \
  --log artifacts/logs/router_test.jsonl \
  --disable_rtdetr
```

Each JSONL row logs action, degradation scores, YOLO confidence statistics, predictions, and latency fields.

## Offline Analysis And Summaries

Oracle upper-bound analysis:

```bash
python3 -m routing.oracle \
  --gt artifacts/coco/test_gt.json \
  --candidate YOLO_ONLY=artifacts/predictions/yolo_test.json \
  --candidate RTDETR_ONLY=artifacts/predictions/rtdetr_test.json \
  --candidate EXPOSURE_ENHANCE_THEN_YOLO=artifacts/predictions/yolo_exposure_test.json \
  --candidate SHARPEN_THEN_YOLO=artifacts/predictions/yolo_sharpen_test.json \
  --candidate YOLO_PLUS_RTDETR_ALWAYS=artifacts/predictions/yolo_plus_rtdetr_test.json \
  --out artifacts/analysis/oracle.json
```

Generate tables and lightweight SVG figure artifacts:

```bash
python3 -m evaluation.summarize \
  --metrics YOLO=artifacts/metrics/yolo_test.json AgentCam=artifacts/metrics/agentcam_router_test.json \
  --router_log artifacts/logs/router_test.jsonl \
  --scores artifacts/scores/degradation_scores.csv \
  --out_dir artifacts/tables \
  --figures_dir artifacts/figures
```

## Dry Runs

All project CLIs and shell scripts accept `--dry_run` and `--limit N`. Dry-runs avoid expensive model loading/training where possible while still validating argument flow and output shape.

Examples:

```bash
bash scripts/00_download_data.sh --dry_run
bash scripts/10_train_yolo.sh --dry_run
python3 -m routing.run_router \
  --data data/processed/agentcam \
  --scores artifacts/scores/degradation_scores.csv \
  --yolo_weights missing.pt \
  --rtdetr_weights missing_rtdetr.pt \
  --config configs/router_rule.yaml \
  --split test \
  --out artifacts/predictions/router_dry_run.json \
  --log artifacts/logs/router_dry_run.jsonl \
  --dry_run \
  --limit 5
```

## Limitations

- No performance claims are made by this repository alone; metrics require the actual dataset and completed detector runs.
- The router is intentionally rule-based in V1.
- Enhancement workers are lightweight only and do not perform diffusion, GAN, or heavy restoration.
- Test labels are never used for router threshold calibration.
