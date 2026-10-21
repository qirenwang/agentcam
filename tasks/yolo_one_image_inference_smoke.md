# YOLO One-Image Inference Smoke

## Purpose

Run exactly one YOLO inference call on exactly one `val_from_train` image using the local `weights/yolov8n.pt`, with CUDA enabled and class filtering limited to COCO car/truck classes `[2, 7]`.

This was a smoke test only, not evaluation.

## Selected Image

- image_path: data/processed/agentcam/_raw_extracted/data_eda_most25/data/training-data/images/202410311226102600164.png
- image_filename: 202410311226102600164.png
- processed_split: val_from_train
- source_split: training-data
- manual_degradation: blur
- label_path: data/processed/agentcam/_raw_extracted/data_eda_most25/data/training-data/labels/202410311226102600164.txt
- label_path_exists: true
- selected_metadata: artifacts/yolo_one_image_smoke/selected_image.json

## Command Sequence

```bash
source ag1_venv/bin/activate && which python && python --version
source ag1_venv/bin/activate && python - <<'PY'
# confirmed CUDA, ultralytics, local yolov8n.pt, and COCO class names
PY
cat data/processed/agentcam/data_trainval.yaml
cat data/processed/agentcam/data_test.yaml
source ag1_venv/bin/activate && python - <<'PY'
# selected one val_from_train blur/overexposure image from manifest_with_manual_degradation.csv
PY
sed -n '1,260p' detectors/yolo_worker.py
rg -n "class YOLOWorker|def __init__|def predict|classes|model\\.predict" detectors/yolo_worker.py
source ag1_venv/bin/activate && python - <<'PY'
# exactly one model.predict call on the selected image
PY
find weights -maxdepth 2 -type f | sort
find runs -maxdepth 3 -type f 2>/dev/null | sort | head -80 || true
```

## Inference Method

- method: direct Ultralytics
- reason: project `YOLOWorker.predict` supports one-image prediction but does not expose the requested `classes=[2, 7]` filter; direct Ultralytics satisfied the class-filter requirement without code changes.
- model: weights/yolov8n.pt
- device: cuda:0
- conf: 0.25
- classes: [2, 7]
- inference_call_count: 1
- source_image_count: 1
- test_split_used: false

## Output Schema

Raw JSON includes:

- image path and filename
- processed/source split
- manual degradation
- label path existence
- weight/backend/device/conf/classes
- Ultralytics speed values
- native COCO class ID/name
- confidence
- xyxy box

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

Mapping:

- COCO 2 car -> dataset 0 car
- COCO 7 truck -> dataset 1 truck

## Result Summary

- prediction_count_before_car_truck_filter: unavailable because the single inference call was constrained to classes [2, 7]
- prediction_count_after_car_truck_filter: 1
- remapped_prediction_count: 1
- remapped_dataset_class_ids_present: [0]
- observed remapped prediction: dataset class 0 / car
- visualization_saved: true
- overlay_path: artifacts/yolo_one_image_smoke/prediction_overlay.jpg

## Artifact Paths

- artifacts/yolo_one_image_smoke/selected_image.json
- artifacts/yolo_one_image_smoke/predictions_raw.json
- artifacts/yolo_one_image_smoke/predictions_dataset_remapped.csv
- artifacts/yolo_one_image_smoke/yolo_one_image_smoke.md
- artifacts/yolo_one_image_smoke/prediction_overlay.jpg
- artifacts/yolo_one_image_smoke.md

## Safety Constraints Satisfied

- no commit
- no training
- no detector evaluation
- no router evaluation
- no mAP computation or reporting
- no threshold tuning
- no test split inference
- no extra weight download
- no dataset image/label/manifest/YAML/manual annotation modification

## Next Recommended Step

Proceed to a 10-image validation smoke test only after explicit approval. Keep that step separate from evaluation and mAP reporting.
