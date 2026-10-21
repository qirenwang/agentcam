# Full val_from_train Pretrained Baseline

## Purpose
Move beyond smoke testing by first completing RT-DETR-L 10-image parity smoke and then evaluating YOLO26n and RT-DETR-L on the full labeled `val_from_train` split.

## Commands Run
- `cd ~/agentcam; source ag1_venv/bin/activate; git status --short`
- `cd ~/agentcam; source ag1_venv/bin/activate; git diff --stat`
- `cd ~/agentcam; source ag1_venv/bin/activate; which python; python --version; python -m pip --version`
- `cd ~/agentcam; source ag1_venv/bin/activate; python - <<'PY'  # inspect evaluation utilities, selected smoke artifacts, and val split counts`
- `cd ~/agentcam; source ag1_venv/bin/activate; python - <<'PY'  # Stage A RT-DETR-L 10-image parity smoke`
- `cd ~/agentcam; source ag1_venv/bin/activate; python - <<'PY'  # Stage B full val_from_train baseline evaluation`

## Stage A
RT-DETR-L ran exactly one batch predict call on the existing YOLO26n fixed 10-image selected CSV/JSON. Outputs normalized into the common detector schema. No metrics were computed in Stage A. Result: PASS.

## Stage B Configuration
- Full validation image count: 96
- Ground-truth object count: 497
- Image size: 640
- Confidence threshold: 0.001
- IoU/NMS threshold: 0.7
- Batch size: 1
- Device: CUDA 0
- Precision mode: FP32
- Remapping: COCO 2 car -> dataset 0 car; COCO 7 truck -> dataset 1 truck

## Results
- yolo26n: mAP50=0.307777, mAP50-95=0.204769, precision=0.057114, recall=0.692153, predictions=6023, FN@IoU50=153
- rtdetr_l: mAP50=0.609854, mAP50-95=0.380662, precision=0.046294, recall=0.887324, predictions=9526, FN@IoU50=56

## Artifacts
- `artifacts/rtdetr_l_10_image_val_smoke/`
- `artifacts/rtdetr_l_10_image_val_smoke.md`
- `artifacts/baselines/val_from_train/`
- `tasks/full_val_pretrained_baseline.md`

## Safety Constraints
No training, fine-tuning, test images, AgentCam router, oracle evaluation, threshold tuning, extra weight downloads, YOLOv8 load, YOLO12n load, or commit was performed.

## Exclusions
No evaluated images were excluded.

## Next Step
Use these baseline artifacts to decide whether to run a controlled threshold/calibration study on val_from_train only or proceed to router design without touching the test split.
