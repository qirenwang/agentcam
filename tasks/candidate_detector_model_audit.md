# Candidate Detector Model Availability and Weight-Load Audit

## Purpose

Audit whether modern candidate detectors can be used for AgentCam without running inference or evaluation:

- YOLO26n as the primary fast detector candidate
- YOLO12n as alternate fast detector candidate
- RT-DETR-L as robust verifier candidate

## Commands Run

```bash
source ag1_venv/bin/activate && which python && python --version && python -m pip --version
source ag1_venv/bin/activate && python - <<'PY'
# confirm torch, CUDA, GPU, Ultralytics, and weights directory
PY
mkdir -p weights && find weights -maxdepth 2 -type f | sort
source ag1_venv/bin/activate && python - <<'PY'
# load/acquire only yolo26n.pt, yolo12n.pt, and rtdetr-l.pt;
# move each model to CUDA and record metadata; no inference
PY
source ag1_venv/bin/activate && python - <<'PY'
# summarize candidate_detector_model_audit_raw.json
PY
sha256sum weights/yolo26n.pt weights/yolo12n.pt weights/rtdetr-l.pt weights/yolov8n.pt
ls -lh weights/yolo26n.pt weights/yolo12n.pt weights/rtdetr-l.pt weights/yolov8n.pt
sed -n '1,180p' detectors/yolo_worker.py && sed -n '1,180p' detectors/rtdetr_worker.py
```

## Package And Version Environment

- Python: /home/qiren/agentcam/ag1_venv/bin/python
- Python version: 3.12.12
- torch: 2.11.0+cu128
- torch CUDA build: 12.8
- CUDA available: true
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- ultralytics: 8.4.70

No package installation, uninstall, or upgrade was performed.

## Candidates Tested

### YOLO26n

- model name: yolo26n.pt
- local path: weights/yolo26n.pt
- downloaded/acquired: yes
- size: 5544453 bytes
- SHA256: 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
- load status: PASS
- CUDA move status: PASS
- task: detect
- names length: 80
- class 2: car
- class 7: truck
- worker compatibility: `YOLOWorker` can load the weight; downstream class filtering/remapping remains external.

### YOLO12n

- model name: yolo12n.pt
- local path: weights/yolo12n.pt
- downloaded/acquired: yes
- size: 5595063 bytes
- SHA256: 419ff3dca37d69bacc93a50fa0c186a1c6f9fe62fae0f108b0872829689e9ca6
- load status: PASS
- CUDA move status: PASS
- task: detect
- names length: 80
- class 2: car
- class 7: truck
- worker compatibility: `YOLOWorker` can load the weight; downstream class filtering/remapping remains external.

### RT-DETR-L

- model name: rtdetr-l.pt
- local path: weights/rtdetr-l.pt
- downloaded/acquired: yes
- size: 66511432 bytes
- SHA256: 6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f
- load status: PASS
- CUDA move status: PASS
- task: detect
- names length: 80
- class 2: car
- class 7: truck
- worker compatibility: `RTDETRWorker` can load the weight if the local path is supplied; downstream class filtering/remapping remains external.

## Results

- all candidates loaded: true
- all candidates moved to CUDA: true
- all candidates expose COCO class 2 as car and class 7 as truck: true
- extra weights beyond the allowed list appeared: false
- inference run: false
- training/evaluation/router/mAP run: false

## Blockers

- No blocker for model-load smoke.
- Before using these candidates in an AgentCam detector-output pipeline, native COCO outputs must still be filtered/remapped:
  - COCO 2 car -> dataset 0 car
  - COCO 7 truck -> dataset 1 truck

## Next Recommended Action

Run a one-image YOLO26n smoke test with local `weights/yolo26n.pt` only after explicit approval.
