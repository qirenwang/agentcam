# YOLO Weight Acquisition and Model-Load Smoke

## Purpose

Acquire exactly one allowed YOLO weight, `yolov8n.pt`, save it under `weights/yolov8n.pt`, and verify that the local model can load and move to CUDA without running inference, training, evaluation, export, benchmark, router evaluation, threshold tuning, or mAP reporting.

## Commands Run

```bash
source ag1_venv/bin/activate && which python && python --version && python -m pip --version
source ag1_venv/bin/activate && python - <<'PY'
# torch/CUDA/Ultralytics import confirmation
PY
cat data/processed/agentcam/data_trainval.yaml
cat data/processed/agentcam/data_test.yaml
mkdir -p weights && find weights -maxdepth 2 -type f | sort || true
source ag1_venv/bin/activate && python - <<'PY'
# download exactly yolov8n.pt via Ultralytics if absent
PY
ls -lh weights/yolov8n.pt
sha256sum weights/yolov8n.pt
source ag1_venv/bin/activate && python - <<'PY'
# load local model and move to CUDA; no inference
PY
source ag1_venv/bin/activate && python - <<'PY'
# check COCO class 2 and 7 names; no inference
PY
sed -n '1,220p' detectors/yolo_worker.py
rg -n "class YOLOWorker|def __init__|def predict|model\\.predict|YOLO\\(" detectors/yolo_worker.py
source ag1_venv/bin/activate && python - <<'PY'
# instantiate YOLOWorker with local weight and device=cuda; no predict
PY
stat -c '%n %s bytes' weights/yolov8n.pt
find weights -maxdepth 2 -type f | sort
```

## Artifact Paths

- `weights/yolov8n.pt`
- `artifacts/yolo_weight_model_load_smoke.md`
- `tasks/yolo_weight_model_load_smoke.md`

## Result

- weight downloaded: true
- final weight path: `weights/yolov8n.pt`
- weight size: 6549796 bytes
- weight SHA256: f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36
- torch: 2.11.0+cu128
- torch CUDA build: 12.8
- CUDA available: true
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- ultralytics: 8.4.70
- model-load smoke: PASS
- model moved to CUDA: PASS
- model task: detect
- COCO class mapping smoke: PASS
  - class 2: car
  - class 7: truck
- project YOLO worker constructor smoke: PASS

No inference, training, detector evaluation, router evaluation, threshold tuning, mAP reporting, RT-DETR weight download, or extra YOLO weight download was performed.

## Remaining Blockers

- No detector inference smoke has been run yet.
- No detector evaluation has been run.
- No mAP has been computed or reported.

## Next Recommended Action

Proceed to a one-image YOLO inference smoke test only after explicit approval. Use the local `weights/yolov8n.pt`; keep it separate from detector evaluation and mAP reporting.
