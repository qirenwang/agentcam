# RT-DETR-L One-Image Smoke

## Scope and Purpose
Run a single-image RT-DETR-L CUDA inference smoke test on the fixed AgentCam val_from_train blur image. The purpose is pipeline validation: model load, one predict call, result parsing, COCO car/truck filtering, AgentCam class remapping, latency logging, overlay generation, and schema compatibility with YOLO26n smoke artifacts.

## Active Detector Roles
- YOLO26n is the primary fast detector but was not loaded in this task.
- RT-DETR-L is the robust verifier and was the only detector loaded.
- YOLO12n is optional later ablation and was not loaded.
- YOLOv8 is retired and was not loaded or used.

## Fixed Same-Image Selection Policy
Selected `202410311226102600164.png` from `manifest_with_manual_degradation.csv` because it is the fixed image used by the YOLO26n one-image smoke comparison. It validated as `processed_split=val_from_train` and `manual_degradation=blur` with image and label files present.

## Commands Executed
- `cd ~/agentcam; source ag1_venv/bin/activate; git status --short`
- `cd ~/agentcam; source ag1_venv/bin/activate; git diff --stat`
- `cd ~/agentcam; source ag1_venv/bin/activate; which python; python --version; python -m pip --version`
- `cd ~/agentcam; source ag1_venv/bin/activate; python - <<'PY'  # CUDA/package check without detector load`
- `cd ~/agentcam; source ag1_venv/bin/activate; test -f weights/rtdetr-l.pt; ls -lh weights/rtdetr-l.pt; sha256sum weights/rtdetr-l.pt`
- `cd ~/agentcam; source ag1_venv/bin/activate; cat data/processed/agentcam/data_trainval.yaml`
- `cd ~/agentcam; source ag1_venv/bin/activate; cat data/processed/agentcam/data_test.yaml`
- `cd ~/agentcam; source ag1_venv/bin/activate; python - <<'PY'  # validate fixed selected image and write selected_image.json`
- `cd ~/agentcam; source ag1_venv/bin/activate; sed -n '1,300p' detectors/rtdetr_worker.py`
- `cd ~/agentcam; source ag1_venv/bin/activate; python - <<'PY'  # exactly one RTDETR predict call and artifact write`
- `cd ~/agentcam; source ag1_venv/bin/activate; python - <<'PY'  # schema/report/task generation without detector load`

## Inference Backend
Direct Ultralytics `RTDETR` was used. `detectors/rtdetr_worker.py` can instantiate and predict, but it does not expose the required native `classes=[2, 7]` control or latency metadata for this smoke test.

## Exact Inference Configuration
- Model class: `ultralytics.RTDETR`
- Weight: `weights/rtdetr-l.pt`
- Device: `0`
- Conf: `0.25`
- Classes: `[2, 7]`
- Save: `False`
- Verbose: `False`
- Inference call count: 1
- Input image count: 1

## Output Schemas
- `predictions_raw.json`: run metadata, selected image metadata, Ultralytics diagnostic path, original shape, speed dictionary, wall-clock timing, and native COCO prediction records.
- `predictions_dataset_remapped.csv`: one normalized record per retained car/truck detection with detector name/family, native COCO class, AgentCam dataset class, confidence, and xyxy box.
- `latency_summary.json`: one-image timing fields and smoke-test note.
- `schema_compatibility.json`: schema-only comparison with YOLO26n smoke artifacts.

## Result Summary
- Native car/truck predictions: 4
- Remapped predictions: 4
- Native COCO class IDs present: [2, 7]
- Dataset class IDs present: [0, 1]

## Latency Summary
- preprocess_ms: 1.0229600593447685
- inference_ms: 10.666386689990759
- postprocess_ms: 5.437018815428019
- total_ultralytics_reported_ms: 17.126365564763546
- predict_wall_clock_ms: 841.0799019038677

This is a no-warm-up smoke measurement, not a benchmark.

## Schema Compatibility Findings
Both artifacts use the same image filename: True. The raw prediction fields overlap on ['confidence', 'native_coco_class_id', 'native_coco_class_name', 'xyxy']. Both can be normalized into the common detector output record: True.

## Artifact Paths
- `artifacts/rtdetr_l_one_image_smoke/selected_image.json`
- `artifacts/rtdetr_l_one_image_smoke/predictions_raw.json`
- `artifacts/rtdetr_l_one_image_smoke/predictions_dataset_remapped.csv`
- `artifacts/rtdetr_l_one_image_smoke/latency_summary.json`
- `artifacts/rtdetr_l_one_image_smoke/prediction_overlay.jpg`
- `artifacts/rtdetr_l_one_image_smoke/schema_compatibility.json`
- `artifacts/rtdetr_l_one_image_smoke.md`
- `tasks/rtdetr_l_one_image_smoke.md`

## Safety Constraints
No training, fine-tuning, detector evaluation, router evaluation, AP/mAP, precision/recall/F1, IoU, detection correctness comparison, threshold tuning, package installation, dataset modification, weight download, or commit was performed.

## Unresolved Issues
None for this smoke-test scope.

## Next Recommended Action
Proceed to an RT-DETR-L 10-image val_from_train smoke test using the same balanced 10-image subset as YOLO26n.
