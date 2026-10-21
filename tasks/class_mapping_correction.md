# Class Mapping Correction

## Problem

The processed YOLO YAML files used placeholder class names:

- class_0
- class_1

The raw COCO-style annotation metadata defines the actual object classes:

- 0 -> car
- 1 -> truck

Detector evaluation should not proceed while YAML class names disagree with raw metadata.

## Root Cause

The existing data-preparation YAML writer inferred only the number of classes from YOLO label IDs, then generated placeholder names with `class_names_from_labels`. It did not read `categories` from the raw `annotations.json` files.

## Implementation

- Added raw category extraction in `data_utils/prepare_yolo_train_test.py`.
- Validates COCO category IDs and names:
  - unique category IDs
  - unique category names
  - non-negative integer IDs
  - train/test category mapping agreement
- Validates YOLO label class IDs are covered by the extracted raw category mapping.
- Preserves placeholder fallback for datasets with no category metadata, with an explicit runtime warning.
- Updated `data_utils/manifest.py` with reusable YOLO class-ID scanning.
- Added focused tests in `tests/test_class_mapping.py`.
- Corrected only the `names` field in:
  - `data/processed/agentcam/data_trainval.yaml`
  - `data/processed/agentcam/data_test.yaml`

## Validation

Commands run:

```bash
git status --short
git diff --stat
/home/qiren/agentcam/ag1_venv/bin/python - <<'PY'
# inspected raw ZIP annotation metadata and YOLO label class IDs
PY
/home/qiren/agentcam/ag1_venv/bin/python - <<'PY'
# captured YAML contents, manifest checksums, and image/label inventory digests
PY
source ag1_venv/bin/activate && python -m unittest tests.test_class_mapping -v
source ag1_venv/bin/activate && python -m unittest discover -v
source ag1_venv/bin/activate && python -m compileall data_utils degradation detectors evaluation routing workers tests
source ag1_venv/bin/activate && python -m data_utils.validate_yolo --data data/processed/agentcam --out artifacts/data_report_after_class_mapping.md
```

Results:

- focused class-mapping tests: PASS, 9 tests
- full unittest discovery: PASS, 26 tests
- compileall: PASS
- validate_yolo: PASS, errors=0
- manifest.csv unchanged: true
- manifest_with_manual_degradation.csv unchanged: true
- image inventory unchanged: true
- label inventory unchanged: true

## Files Modified

- `data_utils/manifest.py`
- `data_utils/prepare_yolo_train_test.py`
- `tests/test_class_mapping.py`
- `data/processed/agentcam/data_trainval.yaml`
- `data/processed/agentcam/data_test.yaml`
- `artifacts/data_report_after_class_mapping.md`
- `artifacts/class_mapping_correction.md`
- `tasks/class_mapping_correction.md`

## Remaining Next Step

The dataset metadata is ready for detector dependency/CUDA audit before any detector training or inference.
