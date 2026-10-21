# CUDA/PyTorch Remediation

## Problem

The existing virtual environment had torch `2.12.0+cu130`, built for CUDA 13.0. The workstation driver is `575.57.08`, with `nvidia-smi` reporting CUDA 12.9 support. Torch saw one CUDA device but `torch.cuda.is_available()` returned false because the driver was too old for the CUDA 13.0 torch build.

## Root Cause

The Python virtual environment contained a CUDA 13.0 PyTorch build, while the system NVIDIA driver remained on a CUDA 12.x-compatible branch. The task explicitly ruled out a system driver upgrade, so the remediation was to install CUDA 12.8 PyTorch wheels inside the venv.

## Commands Executed

```bash
source ag1_venv/bin/activate && which python && python --version && which pip || true && python -m pip --version
mkdir -p artifacts/env && source ag1_venv/bin/activate && python -m pip freeze > artifacts/env/pip_freeze_before_cuda_remediation.txt
source ag1_venv/bin/activate && python - <<'PY'
# pre-remediation package/CUDA status
PY
source ag1_venv/bin/activate && python -m pip index versions torch --index-url https://download.pytorch.org/whl/cu128 || true
source ag1_venv/bin/activate && python -m pip index versions torchvision --index-url https://download.pytorch.org/whl/cu128 || true
source ag1_venv/bin/activate && python -m pip uninstall -y torch torchvision torchaudio
source ag1_venv/bin/activate && python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
source ag1_venv/bin/activate && python -m pip install pandas
source ag1_venv/bin/activate && python -m pip check
source ag1_venv/bin/activate && python -m pip freeze > artifacts/env/pip_freeze_after_cuda_remediation.txt
source ag1_venv/bin/activate && python - <<'PY'
# CUDA tensor smoke test
PY
source ag1_venv/bin/activate && python - <<'PY'
# import smoke tests
PY
source ag1_venv/bin/activate && python - <<'PY'
# ultralytics YOLO/RTDETR class import only
PY
source ag1_venv/bin/activate && python -m evaluation.run_detector --help
source ag1_venv/bin/activate && python -m routing.run_router --help
source ag1_venv/bin/activate && python -m data_utils.validate_yolo --data data/processed/agentcam --out artifacts/data_report_after_cuda_remediation.md
```

## Packages Changed

Removed:

- torch 2.12.0
- torchvision 0.27.0
- torchaudio: not installed, skipped

Installed:

- torch 2.11.0+cu128
- torchvision 0.26.0+cu128
- pandas 3.0.3
- CUDA 12.8 torch runtime packages
- triton 3.6.0

## Validation Results

- `python -m pip check`: PASS
- torch after remediation: 2.11.0+cu128
- torch CUDA build after remediation: 12.8
- `torch.cuda.is_available()`: true
- CUDA device count: 1
- GPU from torch: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- CUDA tensor smoke result: 64.0
- ultralytics import: PASS
- pycocotools import: PASS
- pandas import: PASS
- `evaluation.run_detector --help`: PASS
- `routing.run_router --help`: PASS
- validate_yolo after remediation: PASS, errors=0

No detector weights were downloaded and no YOLO or RT-DETR model was instantiated.

## Remaining Blockers

- No local detector weight files are present under `weights/`.
- Detector training/inference has not been run and still requires an explicit next command.
- Some orphaned CUDA 13 package names remain in the venv, but torch is now CUDA 12.8, CUDA smoke passes, and `python -m pip check` is clean.

## Next Recommended Action

Proceed to YOLO baseline dry-run setup. Actual training or inference should remain gated behind an explicit command and local weights or explicit approval for any detector-weight download.
