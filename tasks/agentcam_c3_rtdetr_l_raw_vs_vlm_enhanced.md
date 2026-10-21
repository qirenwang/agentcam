# AgentCam C3 RT-DETR-L Raw vs VLM-Guided Enhanced

Run directory: `artifacts/agentcam_c3_rtdetr_l_raw_vs_vlm_enhanced/20260621T022849Z`
Status: `PASSED_WITH_WARNINGS`

Raw full inference count: 640
Enhanced paired inference count: 593
Deferred unknown count: 47

Primary delta: `{"FN_iou_0.50": 32, "mAP50": -0.004166658824523828, "mAP50-95": -0.006836200157921413, "max_det_saturation_count": -1, "per_class": {"car": {"AP50": 0.010319851007853376, "AP50-95": 0.0005910240969281833, "FN_iou_0.50": 28, "precision_conf_0.001_iou_0.50": -0.0012238447840024896, "recall_conf_0.001_iou_0.50": -0.010328292143120588}, "truck": {"AP50": -0.01865316865690103, "AP50-95": -0.014263424412771009, "FN_iou_0.50": 4, "precision_conf_0.001_iou_0.50": 0.0005507042688084433, "recall_conf_0.001_iou_0.50": -0.00936768149882905}}, "precision_conf_0.001_iou_0.50": -0.0005595039227620086, "prediction_count": 24, "prediction_count_mean": 0.040472175379420605, "prediction_count_median": 0, "recall_conf_0.001_iou_0.50": -0.010197578075207048}`

Primary aggregate result: RT-DETR-L worsened on mAP50, mAP50-95, recall, precision, and FN in the paired 593-image comparison.

Tests:

- `"$PY" -m unittest discover -v`: passed, 48 tests.
- `"$PY" -m compileall data_utils degradation detectors evaluation routing workers tests vlm scripts`: passed.

No VLM inference, training/fine-tuning, package change, threshold tuning, enhancement tuning, or best-of-bank selection occurred.
Recommendation: Do not tune parameters in this run; recommend a separately preregistered action-bank revision stage.
