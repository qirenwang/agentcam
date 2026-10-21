# Detector Dependency and CUDA Audit

## Commands Run

```bash
source ag1_venv/bin/activate && which python && python --version && which pip && pip --version
source ag1_venv/bin/activate && python -m pip --version
source ag1_venv/bin/activate && python -m pip list
source ag1_venv/bin/activate && python - <<'PY'
# checked importlib availability for torch, torchvision, torchaudio, ultralytics,
# cv2, PIL, numpy, yaml, pycocotools, onnx, onnxruntime, tensorrt,
# transformers, timm, scipy, pandas, matplotlib
PY
nvidia-smi || true
nvcc --version || true
source ag1_venv/bin/activate && python - <<'PY'
# checked torch version, torch CUDA availability, CUDA version, and device count
PY
ls -lh
find . -maxdepth 3 -type f \( -name 'requirements*.txt' -o -name 'pyproject.toml' -o -name 'environment*.yml' -o -name '*.yaml' -o -name '*.yml' \) | sort
sed -n '1,240p' requirements.txt || true
sed -n '1,260p' pyproject.toml || true
cat data/processed/agentcam/data_trainval.yaml
cat data/processed/agentcam/data_test.yaml
ls -lh research/annotations/
ls -lh artifacts/ | head -80
ls -lh artifacts/scores/ || true
test -f research/annotations/manual_degradation_labels.csv && echo "manual labels exist"
test -f data/processed/agentcam/manifest_with_manual_degradation.csv && echo "enriched manifest exists"
source ag1_venv/bin/activate && python -m evaluation.run_detector --help || true
source ag1_venv/bin/activate && python -m routing.run_router --help || true
source ag1_venv/bin/activate && python - <<'PY'
# import smoke tests for detector/router modules
PY
source ag1_venv/bin/activate && python - <<'PY'
# imported ultralytics.YOLO and ultralytics.RTDETR without instantiating models
PY
nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
find weights -maxdepth 2 -type f -printf '%p\t%s bytes\n' | sort
```

## Observed Environment

- Python path: /home/qiren/agentcam/ag1_venv/bin/python
- Python version: Python 3.12.12
- `which pip` after activation: /usr/bin/pip
- `pip --version` after activation: system pip for Python 3.10
- `python -m pip --version`: venv pip 25.0.1 for Python 3.12

Use `python -m pip` for future package operations in this environment.

## Installed Packages

Installed and relevant:

- torch 2.12.0+cu130
- torchvision 0.27.0
- ultralytics 8.4.70
- opencv-python 4.13.0.92
- pillow 12.2.0
- numpy 2.4.6
- PyYAML 6.0.3
- pycocotools 2.0.11
- scipy 1.17.1
- matplotlib 3.11.0

Missing:

- torchaudio
- onnx
- onnxruntime
- tensorrt
- transformers
- timm
- pandas

## CUDA/GPU Status

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Driver: 575.57.08
- nvidia-smi CUDA support: 12.9
- GPU memory: 97887 MiB
- nvcc: CUDA 12.8, V12.8.93
- torch CUDA build: 13.0
- torch CUDA available: false
- torch CUDA device count: 1

Blocker: PyTorch reports the driver is too old for the installed CUDA 13.0 torch build, so GPU execution is not currently usable through torch.

## Detector Import Status

- `evaluation.run_detector --help`: PASS
- `routing.run_router --help`: PASS
- `detectors.yolo_worker`: IMPORT_OK
- `detectors.rtdetr_worker`: IMPORT_OK
- `evaluation.run_detector`: IMPORT_OK
- `routing.run_router`: IMPORT_OK
- `ultralytics.YOLO`: IMPORT_OK
- `ultralytics.RTDETR`: IMPORT_OK

No model was instantiated and no weights were loaded.

## Dataset Metadata Status

- `data_trainval.yaml` names: [car, truck]
- `data_trainval.yaml` train: images/train_core
- `data_trainval.yaml` val: images/val_from_train
- `data_trainval.yaml` test: images/test
- `data_test.yaml` names: [car, truck]
- `data_test.yaml` train: images/test
- `data_test.yaml` val: images/test
- `data_test.yaml` test: images/test
- manual degradation labels: present
- enriched manifest: present

## Blocker List

- CUDA is not usable from torch because torch is built for CUDA 13.0 and the installed driver reports CUDA 12.9 support.
- No local detector weight files were found under `weights/`.
- `pandas` is missing for result-table analysis.
- `onnx`, `onnxruntime`, and `tensorrt` are missing for export/runtime workflows.
- `transformers` and `timm` are missing for alternative non-Ultralytics RT-DETR workflows.
- Shell `pip` points outside the venv.

## Recommended Next Step

Resolve the torch/CUDA driver compatibility issue before GPU detector training or evaluation. After that, either provide local weights or explicitly approve any model-weight download. For package work, use `/home/qiren/agentcam/ag1_venv/bin/python -m pip`.

## Conclusion

PARTIAL readiness:

- YOLO training: PARTIAL
- YOLO evaluation: PARTIAL
- RT-DETR evaluation: PARTIAL
- result analysis: PARTIAL
