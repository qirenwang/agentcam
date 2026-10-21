# YOLO26n 10-Image val_from_train Inference Smoke

## Scope

Run exactly one YOLO26n batch inference call over exactly 10 fixed `val_from_train` images using local `weights/yolo26n.pt`, CUDA, and native COCO class filtering for `[2, 7]`.

This is a pipeline smoke test, not detector evaluation.

## Active Model Policy

- YOLO26n: primary fast detector
- YOLO12n: optional later ablation
- RT-DETR-L: later robust verifier
- YOLOv8: retired and excluded from active AgentCam experiments

YOLOv8 was not loaded or used. YOLO12n and RT-DETR-L were also not loaded or used in this task.

## Fixed Selected Images

This subset is balanced for smoke coverage because it contains 5 pure blur and 5 pure overexposure validation images from `val_from_train`.

Blur:

1. `202410311226102600164.png`
2. `202410311226102600251.png`
3. `202410311226102600257.png`
4. `202410311226102600267.png`
5. `202410311226102600274.png`

Overexposure:

6. `202410311719239750046.png`
7. `202410311719239750052.png`
8. `202410311719239750130.png`
9. `202410311719239750228.png`
10. `202410311719239750261.png`

Selection validation:

- exactly 10 rows
- exactly 5 blur and 5 overexposure rows
- all `val_from_train`
- no test rows
- no duplicate filenames
- all image paths existed
- all label paths existed
- selected order matched the fixed order

Label files were checked for existence only.

## Exact Commands

```bash
source ag1_venv/bin/activate && git status --short
source ag1_venv/bin/activate && git diff --stat
source ag1_venv/bin/activate && which python && python --version && python -m pip --version
source ag1_venv/bin/activate && python - <<'PY'
# confirmed torch/CUDA/Ultralytics without loading a detector model
PY
test -f weights/yolo26n.pt
ls -lh weights/yolo26n.pt
sha256sum weights/yolo26n.pt
source ag1_venv/bin/activate && python - <<'PY'
# validated fixed image list and wrote selected_images.csv/json
PY
source ag1_venv/bin/activate && python - <<'PY'
# made exactly one YOLO26n model.predict call over 10 images and wrote outputs
PY
find weights -maxdepth 2 -type f | sort
find runs -maxdepth 3 -type f 2>/dev/null | sort | head -100 || true
git status --short
git diff --stat
git status --ignored -s | head -140
```

## Inference Configuration

- backend: direct Ultralytics
- model: YOLO26n
- weight: `weights/yolo26n.pt`
- weight SHA256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- device: `0`
- conf: `0.25`
- classes: `[2, 7]`
- save: `False`
- verbose: `False`
- stream: `False`
- predict-call count: `1`
- input image count: `10`
- warm-up inference: none

The fixed confidence value was only a smoke-test filter and was not tuned.

## Input/Result Mapping

Results were mapped to selected rows strictly by input list order:

```python
for selected_row, result in zip(selected_rows, results):
    ...
```

`result.path` was retained only as diagnostic metadata because Ultralytics can report synthetic paths such as `image0.jpg` for list inputs.

## Output Schemas

`predictions_raw.json` contains:

- run metadata
- one entry for each of the 10 selected images
- diagnostic `result_path_reported_by_ultralytics`
- original image shape
- Ultralytics speed dictionary
- predictions with native COCO class ID/name, confidence, and xyxy box

`predictions_dataset_remapped.csv` columns:

- selection_index
- image_filename
- image_path
- processed_split
- manual_degradation
- coco_class_id
- coco_class_name
- dataset_class_id
- dataset_class_name
- confidence
- x1, y1, x2, y2

`per_image_summary.csv` includes one row per selected image, including zero-prediction images.

`latency_summary.json` includes aggregate and per-degradation mean/min/max timing fields.

## Prediction-Count Summary

- total native car/truck predictions: 7
- total remapped predictions: 7
- predictions by degradation:
  - blur: 2
  - overexposure: 5
- car count: 6
- truck count: 1
- zero-prediction images: 6
- native class IDs present: [2, 7]
- dataset class IDs present: [0, 1]

No correctness or accuracy metric was computed.

## Latency Summary

- batch_wall_clock_ms: 549.0749389864504
- batch_wall_clock_ms_per_image: 54.90749389864504
- aggregate preprocess_ms mean/min/max: 0.9054364170879126 / 0.9054364170879126 / 0.9054364170879126
- aggregate inference_ms mean/min/max: 2.6644404977560043 / 2.6644404977560043 / 2.6644404977560043
- aggregate postprocess_ms mean/min/max: 1.757379900664091 / 1.757379900664091 / 1.757379900664091
- aggregate total_ultralytics_reported_ms mean/min/max: 5.327256815508008 / 5.327256815508008 / 5.327256815508008

This is a no-warm-up smoke measurement, not a benchmark.

## Artifact Paths

- `artifacts/yolo26n_10_image_val_smoke/selected_images.csv`
- `artifacts/yolo26n_10_image_val_smoke/selected_images.json`
- `artifacts/yolo26n_10_image_val_smoke/predictions_raw.json`
- `artifacts/yolo26n_10_image_val_smoke/predictions_dataset_remapped.csv`
- `artifacts/yolo26n_10_image_val_smoke/per_image_summary.csv`
- `artifacts/yolo26n_10_image_val_smoke/latency_summary.json`
- `artifacts/yolo26n_10_image_val_smoke/yolo26n_10_image_val_smoke.md`
- `artifacts/yolo26n_10_image_val_smoke/overlays/`
- `artifacts/yolo26n_10_image_val_smoke.md`

## Safety Constraints

- no commit
- no training or fine-tuning
- no YOLOv8 load or use
- no YOLO12n load or use
- no RT-DETR-L load or use
- no detector evaluation
- no router evaluation
- no AP/mAP/precision/recall/F1/confusion matrix/IoU
- no false positive/false negative/correctness calculation
- no threshold tuning
- no test split inference
- no additional weight download
- no package install/uninstall/upgrade
- no dataset image, label, manifest, YAML, or manual annotation modification
- no ground-truth label parsing or comparison

## Unresolved Issues

- No blocker for the YOLO26n 10-image smoke.
- This did not evaluate detection quality.

## Next Recommended Action

Proceed to an RT-DETR-L one-image `val_from_train` smoke test only after explicit approval. Keep YOLOv8 excluded from active AgentCam baselines and reports.
