# YOLO 10-Image Validation Smoke

## Purpose

Run a smoke-only YOLO inference pass on exactly 10 `val_from_train` images using local `weights/yolov8n.pt`, CUDA, direct Ultralytics, and class filtering for COCO car/truck classes `[2, 7]`.

No evaluation metrics were computed.

## Previous Failure And Rerun Approval

The first 10-image attempt selected the correct images and made one inference call, but post-processing failed because Ultralytics returned synthetic `result.path` values such as `image0.jpg` for list input.

Explicit approval was granted for exactly one additional batch inference call on the same selected images. The rerun fixed only result mapping:

```python
for selected_row, result in zip(selected_rows, results):
    ...
```

`result.path` is now recorded only as diagnostic metadata.

## Selected Image Policy

- source manifest artifact: `artifacts/yolo_10_image_val_smoke/selected_images.csv`
- processed split: `val_from_train`
- selected categories:
  - 5 rows where `manual_degradation == blur`
  - 5 rows where `manual_degradation == overexposure`
- deterministic ordering: sorted by image filename, first 5 per category
- test images allowed: false
- selected image paths and label paths were confirmed to exist

## Selected Images

- blur: `202410311226102600164.png`
- blur: `202410311226102600251.png`
- blur: `202410311226102600257.png`
- blur: `202410311226102600267.png`
- blur: `202410311226102600274.png`
- overexposure: `202410311719239750046.png`
- overexposure: `202410311719239750052.png`
- overexposure: `202410311719239750130.png`
- overexposure: `202410311719239750228.png`
- overexposure: `202410311719239750261.png`

## Command Sequence

```bash
source ag1_venv/bin/activate && which python && python --version
source ag1_venv/bin/activate && python - <<'PY'
# verified torch/CUDA, ultralytics, local weights/yolov8n.pt, and COCO class names
PY
source ag1_venv/bin/activate && python - <<'PY'
# validated selected_images.csv: 10 rows, 5 blur, 5 overexposure, val_from_train only
PY
source ag1_venv/bin/activate && python - <<'PY'
# made exactly one additional direct Ultralytics batch model.predict call;
# mapped results with zip(selected_rows, results)
PY
find weights -maxdepth 2 -type f | sort
find runs -maxdepth 3 -type f 2>/dev/null | sort | head -80 || true
git status --short
git diff --stat
git status --ignored -s | head -120
```

## Inference Method

- backend: direct Ultralytics
- model object count: 1
- weight: `weights/yolov8n.pt`
- device: cuda:0
- conf: 0.25
- classes: [2, 7]
- stream: false
- save: false
- verbose: false
- additional `model.predict` call count for rerun: 1
- source image count: 10
- post-processing join: `zip(selected_rows, results)`

## Output Schema

Raw JSON includes:

- image_filename
- image_path
- result_path_reported_by_ultralytics
- processed_split
- manual_degradation
- native COCO class ID/name
- confidence
- xyxy box
- speed dictionary

Remapped CSV includes:

- image_filename
- processed_split
- manual_degradation
- coco_class_id
- coco_class_name
- dataset_class_id
- dataset_class_name
- confidence
- x1, y1, x2, y2

Per-image summary includes:

- image_filename
- processed_split
- manual_degradation
- num_predictions_car_truck
- num_car_predictions
- num_truck_predictions
- preprocess_ms
- inference_ms
- postprocess_ms
- total_reported_ms
- result_path_reported_by_ultralytics

Latency summary includes mean/min/max for available timing fields.

## Result Summary

- result: PASS
- selected image count: 10
- selected degradation counts: blur=5, overexposure=5
- test image used: false
- prediction count after car/truck filtering: 4
- prediction count by degradation group: blur=2, overexposure=2
- remapped dataset class IDs present: [0]
- remapped dataset classes present: car
- overlays saved: 10
- extra weights downloaded: false
- `runs/` artifacts appeared: false

## Artifact Paths

- `artifacts/yolo_10_image_val_smoke/selected_images.csv`
- `artifacts/yolo_10_image_val_smoke/selected_images.json`
- `artifacts/yolo_10_image_val_smoke/predictions_raw.json`
- `artifacts/yolo_10_image_val_smoke/predictions_dataset_remapped.csv`
- `artifacts/yolo_10_image_val_smoke/per_image_summary.csv`
- `artifacts/yolo_10_image_val_smoke/latency_summary.json`
- `artifacts/yolo_10_image_val_smoke/yolo_10_image_val_smoke.md`
- `artifacts/yolo_10_image_val_smoke/overlays/`
- `artifacts/yolo_10_image_val_smoke.md`

## Safety Constraints Satisfied

- no commit
- no training
- no detector evaluation
- no router evaluation
- no AP, mAP, precision, recall, or F1
- no threshold tuning
- no test split inference
- no extra weight download
- no dataset image, label, manifest, YAML, or manual annotation modification

## Next Recommended Action

Proceed to a small validation detector-output pipeline dry-run only after explicit approval. Keep that step separate from evaluation and mAP reporting.
