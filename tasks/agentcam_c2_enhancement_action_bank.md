# AgentCam C2 Enhancement Action Bank

Run directory: `artifacts/agentcam_c2_enhancement_action_bank/20260621T011459Z`

Source C1 run: `artifacts/agentcam_qwen_vlm_degradation_audit/20260620T024704Z`

Stage C2 status: `PASSED_WITH_WARNINGS`

## Scope

- The scientific development pool is all 640 `training-data` images.
- The prior `train_core=544` and `val_from_train=96` values are provenance only.
- `testing-data` was untouched.
- C1 VLM inference was not repeated.
- No detector inference, training, fine-tuning, best-of-bank, or package change was performed.
- Manual labels were retained only for reference summaries and did not affect actions.

## C1 Integrity

- Authoritative input: `audit/qwen_vlm_degradation_audit.jsonl`
- JSONL records: 640
- Unique `image_id` values: 640
- `train_core` provenance rows: 544
- `val_from_train` provenance rows: 96
- Test rows: 0
- Parse failures: 0
- Schema failures: 0
- Consistency failures: 0

## Frozen Mapping

- `clear -> no_enhancement`
- `blur -> blur_enhancement`
- `overexposure -> overexposure_enhancement`
- `underexposure -> underexposure_enhancement`
- `blur_overexposure -> overexposure_then_blur_enhancement`
- `unknown -> defer_no_automatic_enhancement`

`mixed_sequential_or_best_of_bank` is deprecated and not active in C2.

## Canonical Counts

- `blur`: 177
- `overexposure`: 306
- `underexposure`: 1
- `blur_overexposure`: 109
- `clear`: 0
- `unknown`: 47

## Smoke Results

Selected seven deterministic smoke images by lexicographic filename and VLM class. The `clear` class had zero VLM examples and was not backfilled from manual labels.

All seven selected images processed successfully. Source hashes were unchanged and no source image was overwritten.

## Tests

- `"$PY" -m unittest discover -v`: passed, 35 tests.
- `"$PY" -m compileall data_utils degradation detectors evaluation routing workers tests vlm`: passed.

## Warning

The inherited C1 warning remains: an earlier dependency command invoked `/usr/bin/pip`. C2 did not attempt to repair or alter the environment.
