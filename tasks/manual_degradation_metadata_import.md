# Manual Degradation Metadata Import

## Scope

Implemented a metadata-only importer and audit path for manual degradation labels. The task does not train models, run detector inference, run router evaluation, tune thresholds, report mAP, or modify raw images/labels.

## Design

- `data_utils.import_manual_degradation` parses the manual name list, preserves `source_split` and raw category text, normalizes degradation categories, and detects duplicates, conflicts, cross-split overlaps, unmatched annotations, and unlisted manifest rows.
- Matching is by exact filename stem plus source split. `training-data` annotations can match only `train_core` or `val_from_train`; `testing-data` annotations can match only `test`.
- The original `data/processed/agentcam/manifest.csv` is not modified. The enriched output is written to `data/processed/agentcam/manifest_with_manual_degradation.csv`.
- ZIP inspection uses `zipfile` in memory only and does not extract the archive.
- Score alignment joins existing degradation scores to manual labels by filename stem and processed split. It reports distributions only and does not classify images from scores.

## Normalization Policy

- `Blur` or `blur` -> `blur`
- `exporsure`, `exposure`, or `overexposure` -> `overexposure`
- `blur&overexposure` -> `blur_overexposure`
- `underexposure` -> `underexposure`
- `Clear` or `clear` -> `clear`

## Primary Evaluation Policy

- `blur`: include in primary evaluation
- `overexposure`: include in primary evaluation
- `blur_overexposure`: secondary combined degradation only
- `underexposure`: out of scope for primary evaluation
- `clear`: retained but out of scope for primary degradation evaluation
- `conflict`, `unmatched`, or `unknown`: excluded from primary evaluation

## Commands Executed

```bash
git status --short
git diff --stat
git diff -- data_utils/manifest.py tests/test_data_pipeline.py
/home/qiren/agentcam/ag1_venv/bin/python - <<'PY'
# manifest inspection with csv
PY
/home/qiren/agentcam/ag1_venv/bin/python - <<'PY'
# raw ZIP inspection with zipfile/json
PY
source ag1_venv/bin/activate
python -m unittest tests.test_manual_degradation_import -v
python -m unittest discover -v
python -m compileall data_utils degradation detectors evaluation routing workers tests
python -m data_utils.import_manual_degradation \
  --annotations research/annotations/manual_degradation_names.txt \
  --manifest data/processed/agentcam/manifest.csv \
  --raw data/raw/data_eda_most25.zip \
  --scores artifacts/scores/degradation_scores.csv \
  --labels-out research/annotations/manual_degradation_labels.csv \
  --manifest-out data/processed/agentcam/manifest_with_manual_degradation.csv \
  --report artifacts/manual_degradation_audit.md \
  --score-report artifacts/manual_degradation_score_audit.md \
  --strict
sed -n '1,360p' artifacts/manual_degradation_audit.md
sed -n '1,360p' artifacts/manual_degradation_score_audit.md
sed -n '1,240p' artifacts/class_mapping_audit.md
```

## Files Generated

- `research/annotations/manual_degradation_labels.csv`
- `data/processed/agentcam/manifest_with_manual_degradation.csv`
- `artifacts/manual_degradation_duplicates.csv`
- `artifacts/manual_degradation_conflicts.csv`
- `artifacts/manual_degradation_cross_split.csv`
- `artifacts/manual_degradation_unmatched.csv`
- `artifacts/manual_degradation_unlisted.csv`
- `artifacts/manual_degradation_audit.md`
- `artifacts/manual_degradation_score_audit.md`
- `artifacts/scores/degradation_scores_with_manual_labels.csv`
- `artifacts/class_mapping_audit.md`

## Test Results

- Focused tests: `10` tests passed.
- Full unittest discovery: `17` tests passed.
- Compileall: passed.

## Audit Results

- Manual annotation source file was not present at `research/annotations/manual_degradation_names.txt`.
- Strict import exited non-zero after writing diagnostics and reports.
- Parsed manual records: `0`
- Matched manual records: `0`
- Unlisted manifest rows: `800`
- Enriched manifest rows: `800`
- Score rows joined: `800`
- Unmatched score rows: `800`
- Raw ZIP object classes: `0 -> car`, `1 -> truck`
- Current YAML class names: `class_0`, `class_1`

## Unresolved Issues

- The required manual source text file is missing, so the canonical manual label CSV is header-only and the enriched manifest contains `unknown` manual degradation values with `unlisted` status.
- The strict manual degradation audit failed by design because required manual annotations were unavailable.
- YAML class names remain placeholders and should be fixed before detector evaluation.

## Next Recommended Action

Add the real `research/annotations/manual_degradation_names.txt` file without changing its contents, then rerun the strict import CLI. After the strict audit passes, fix YAML class-name generation from raw annotation metadata before detector baselines.
