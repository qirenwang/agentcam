# YOLO26n One-Image Validation Inference Smoke

## Scope And Purpose

Run exactly one YOLO26n inference call on exactly one `val_from_train` image using local `weights/yolo26n.pt`, CUDA, and native COCO class filtering for `[2, 7]`.

Purpose:

- validate one real YOLO26n inference on CUDA
- validate result parsing
- validate COCO car/truck filtering
- validate AgentCam class remapping output schema
- validate latency and artifact logging

This was not detector evaluation and produced no accuracy metrics.

## Active Model Policy

- YOLO26n: primary fast detector
- YOLO12n: optional later ablation detector
- RT-DETR-L: later robust verifier
- YOLOv8: retired from active AgentCam experiments

YOLOv8 was excluded because it is no longer part of the active baseline/routing pipeline. Historical YOLOv8 weights and artifacts were not deleted or modified, but YOLOv8 was not loaded or used.

## Selected Image Policy

- manifest: `data/processed/agentcam/manifest_with_manual_degradation.csv`
- accepted processed split column: `processed_split` or `split`
- selected split: `val_from_train`
- selected degradation: `blur`
- deterministic choice: first pure-blur validation image after sorting by image filename
- selected image: `202410311226102600164.png`
- test split used: false
- image path existed: true
- label path existed: true
- labels parsed or compared: false

## Selected Image

- image_filename: `202410311226102600164.png`
- image_path: `data/processed/agentcam/_raw_extracted/data_eda_most25/data/training-data/images/202410311226102600164.png`
- processed_split: `val_from_train`
- source_split: `training-data`
- manual_degradation: `blur`
- label_path: `data/processed/agentcam/_raw_extracted/data_eda_most25/data/training-data/labels/202410311226102600164.txt`
- include_primary_eval: `true`
- selected_metadata: `artifacts/yolo26n_one_image_smoke/selected_image.json`

## Exact Commands Executed

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
cat data/processed/agentcam/data_trainval.yaml
cat data/processed/agentcam/data_test.yaml
source ag1_venv/bin/activate && python - <<'PY'
# selected exactly one validation blur image and wrote selected_image.json
PY
source ag1_venv/bin/activate && python - <<'PY'
# instantiated only YOLO(weights/yolo26n.pt), ran exactly one predict call,
# synchronized CUDA before/after timing, parsed/remapped outputs, wrote artifacts
PY
find weights -maxdepth 2 -type f | sort
find runs -maxdepth 3 -type f 2>/dev/null | sort | head -100 || true
```

## Exact Inference Configuration

- backend: direct Ultralytics
- model: YOLO26n
- weight: `weights/yolo26n.pt`
- weight SHA256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- device: `0` / CUDA
- conf: `0.25`
- classes: `[2, 7]`
- save: `False`
- verbose: `False`
- inference_call_count: `1`
- input_image_count: `1`
- warm-up inference: none

The fixed confidence value was only a smoke-test display/filter value and was not tuned.

## Output Schema

`predictions_raw.json` includes:

- run metadata
- selected image metadata
- diagnostic `result_path_reported_by_ultralytics`
- original image shape
- Ultralytics speed dictionary
- wall-clock predict time in milliseconds
- native predictions with COCO class ID/name, confidence, and xyxy boxes

`predictions_dataset_remapped.csv` columns:

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

Mapping:

- COCO `2 / car` -> dataset `0 / car`
- COCO `7 / truck` -> dataset `1 / truck`

## Result Summary

- result: PASS
- native prediction count: 0
- remapped prediction count: 0
- native COCO class IDs present: []
- dataset class IDs present: []
- overlay saved: true
- extra weights downloaded: false
- `runs/` artifacts appeared: false

Zero detections are allowed and do not indicate failure for this smoke test.

## Latency Summary

- preprocess_ms: 1.0891547426581383
- inference_ms: 22.554995957762003
- postprocess_ms: 0.2919510006904602
- total_ultralytics_reported_ms: 23.9361017011106
- predict_wall_clock_ms: 260.52292669191957
- note: one-image, no-warm-up smoke measurement; not an evaluation benchmark

## Safety Constraints

- no commit
- no training or fine-tuning
- no YOLOv8 load or use
- no YOLO12n load or use
- no RT-DETR-L load or use
- no detector evaluation
- no router evaluation
- no AP/mAP/precision/recall/F1/confusion matrix
- no threshold tuning
- no test split inference
- no additional weight download
- no package install/uninstall/upgrade
- no dataset image, label, manifest, YAML, or manual annotation modification
- no ground-truth box parsing or comparison

## Artifact Paths

- `artifacts/yolo26n_one_image_smoke/selected_image.json`
- `artifacts/yolo26n_one_image_smoke/predictions_raw.json`
- `artifacts/yolo26n_one_image_smoke/predictions_dataset_remapped.csv`
- `artifacts/yolo26n_one_image_smoke/latency_summary.json`
- `artifacts/yolo26n_one_image_smoke/prediction_overlay.jpg`
- `artifacts/yolo26n_one_image_smoke/yolo26n_one_image_smoke.md`
- `artifacts/yolo26n_one_image_smoke.md`
- `tasks/yolo26n_one_image_smoke.md`

## Unresolved Issues

- No unresolved blocker for one-image YOLO26n smoke.
- This did not evaluate detection quality.

## Next Recommended Action

Proceed to a YOLO26n 10-image `val_from_train` smoke test only after explicit approval. Keep YOLOv8 excluded from active AgentCam baselines, routing workers, result tables, figures, latency comparisons, and recommendations.
