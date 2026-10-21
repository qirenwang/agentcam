# AgentCam V1 Task Status

Status legend: `done` means implemented in the repository; `runtime-required` means the code path exists but requires the dataset, model weights, or optional dependencies to execute on real data.

| Task | Status | Notes |
| --- | --- | --- |
| T00 repository setup | done | Root repository is used as the AgentCam project root. |
| T01 download script | done | `scripts/00_download_data.sh` uses `gdown --fuzzy` and prints manual fallback instructions. |
| T02 train/test YOLO preparation | done, runtime-required | `data_utils.prepare_yolo_train_test` creates manifest, processed splits, YAMLs, and data report. |
| T03 YOLO label validation | done, runtime-required | `data_utils.validate_yolo` validates images, labels, duplicates, and split counts. |
| T04 degradation scoring | done, runtime-required | `degradation.compute_scores` fits robust normalization on `train_core` only. |
| T05 detector workers | done, runtime-required | YOLO and RT-DETR workers wrap Ultralytics when installed; dry-run avoids model loading. |
| T06 baselines/evaluation | done, runtime-required | Training shell scripts, detector runner, COCO conversion, evaluator, and summarizer exist. |
| T07 lightweight enhancement | done | Pillow CLAHE-like contrast/gamma and unsharp mask workers. |
| T08 rule-based router | done, runtime-required | Router supports calibrated score thresholds from `val_from_train` and JSONL logging. |
| T09 fusion logic | done | Class-aware replacement and NMS utilities implemented. |
| T10 latency measurement | done | Batch-1 timing helpers and summary statistics implemented. |
| T11 offline oracle analysis | done, runtime-required | Candidate utility analysis is available for labeled test predictions. |
| T12 tables/figures | done, runtime-required | CSV summary scaffolding exists; figures require generated run artifacts. |
| T13 dry-run and tests | done | Python CLIs accept `--limit` and `--dry_run`; unit tests cover required core logic. |
