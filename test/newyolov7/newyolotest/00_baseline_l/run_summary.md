# Run Summary - baseline_export

## Decision

- decision: blocker
- reason: Exception: Dataset not found.
- stage_id: 1.3.1
- stage_name: baseline_export
- train_type: baseline_export
- current_run: runs/newyolov7/newyolotest/00_baseline_l

## Artifact Check

| Artifact | Status | Path |
| --- | --- | --- |
| best.pt | missing | weights/best.pt |
| last.pt | missing | weights/last.pt |
| results.csv | ok | results.csv |
| loss_detail.csv | ok | loss_detail.csv |
| stage_result.yaml | ok | stage_result.yaml |

## Metric Summary

| Metric | Value |
| --- | ---: |
| primary_mAP | None |
| mAP50 | None |
| precision |  |
| recall |  |
| GFLOPs | None |
| GFLOPs_delta_percent | None |

## Stability And Risk

- hard_fail: True
- soft_fail: False
- failed_category: train
- export_status: skip

## Next Action

- next_stage: 
- carry_flags: {}
- rollback_flags: {}
- code_area: 
