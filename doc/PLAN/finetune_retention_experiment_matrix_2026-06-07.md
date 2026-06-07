# Finetune Retention Experiment Matrix 2026-06-07

## Goal

The target is not plain fine-tuning. Plain fine-tuning usually improves the tuning-data distribution and degrades the original/base distribution. This experiment searches for a setting that:

1. improves the finetune validation set,
2. keeps the base/original validation set close to the original model,
3. saves `best.pt` using a metric that sees both validation sets.

Use `--best-val-set combined` when `data.yaml` has both finetune and base validation sets. If a named base validation set must dominate selection, use that name instead.

## Design Reference

The current design comes from the 1.3.7 fine-tuning/continual-learning plan and the YOLOv7 custom design spec:

- Replay Buffer first
- Replay + class/objectness distillation second
- Replay + class/objectness + regression distillation third
- Freeze/BatchNorm policy as drift control
- Pseudo label as a separate optional stage, not required for the first retention sweep

This maps to:

- `E0`: plain short fine-tune baseline
- `E1`: replay-only retention
- `E2`: replay + cls/objectness distillation
- `E3`: replay + cls/reg distillation

## Current Completion Check

Ready enough for controlled experiments:

- `finetune.py` creates class mapping checks, replay manifests, merged train yaml, dataset manifest, and train commands.
- `train.py` supports multiple validation sets and `--best-val-set`.
- `train.py` supports `--bn-policy`, `--freeze`, and teacher distillation flags.
- `tools/run_finetune_sweep.py` now creates and optionally executes a retention sweep.

Still limited:

- Replay ratio is implemented as selected replay images concatenated to the train list, not guaranteed per-batch sampling.
- Distillation is an MVP MSE-based implementation, not a full paper-faithful YOLO LwF implementation.
- Pseudo labels are separate tools and are not part of the first sweep.
- `tools/evaluate_forgetting.py` is still coarse. Use `results_detail.txt` and `scenario_metrics.csv` for per-validation judgement.

## Keep/Drop Rule

Use the original model evaluation as the baseline.

- Base retention: `base_val_after / base_val_before * 100`
- Finetune gain: `finetune_val_after - finetune_val_before`

Recommended first pass:

- keep candidate if base retention is at least `95%`
- prefer the highest finetune-val `mAP50-95` among candidates
- reject if finetune-val improves but base-val collapses
- reject `last.pt` if it is worse than `best.pt`

## Core Sweep

`tools/run_finetune_sweep.py --preset core` creates these experiments:

| ID | Purpose |
| --- | --- |
| `e00_no_replay_short` | short plain fine-tune baseline |
| `e01_no_replay_bn_eval` | isolate BN-stat freezing |
| `e02_no_replay_freeze_neck` | limit trainable layers without replay |
| `e10_replay_005_bn_eval` | minimal replay |
| `e11_replay_010_bn_eval` | primary small-replay candidate |
| `e12_replay_030_bn_eval` | original replay level |
| `e20_replay_010_cls_distill` | small replay + cls/objectness distill |
| `e21_replay_030_cls_distill` | original replay + cls/objectness distill |
| `e30_replay_010_cls_reg_light` | small replay + light reg distill |
| `e31_replay_010_freeze_neck_cls` | conservative retention candidate |

Use `--preset full` only after the core sweep shows the likely direction.

## Recommended Data YAML

Use named validation sets so reports are easy to read:

```yaml
train: /workspace/datasets/Congress-1/file/list/train.txt
val:
  - name: congress_valid
    path: /workspace/datasets/Congress-1/file/list/valid.txt
  - name: base_valid
    path: /workspace/datasets/Congress-1/file/list/valil_falldown.txt
nc: <same_nc>
names: <same_names>
```

If both sets are same classes, `--base-data` and `--data` can point to the same yaml for the first sweep. If the base replay source should be a broader old dataset, set `--base-data` to that old dataset yaml while keeping the same class mapping.

## Command: Prepare Only

This creates the sweep plan and dry-run artifacts, but does not train:

```bash
python tools/run_finetune_sweep.py \
  --weights /workspace/datasets/svms/file/SIAV2_Detector_YOLOV7_SafeEnv_V7.0.0_FP32_260223.pt \
  --base-data /workspace/datasets/svms/file/data/data.yaml \
  --data /workspace/datasets/svms/file/data/data.yaml \
  --hyp /workspace/datasets/svms/file/data/hyp.scratch.custom.yaml \
  --epochs 8 \
  --img 640 \
  --batch 64 \
  --workers 8 \
  --project runs/fine \
  --prefix congress_retention_core \
  --preset core \
  --launcher distributed \
  --nproc-per-node 4 \
  --master-port 9527 \
  --best-val-set combined \
  --save-best-only
```

Outputs:

- `runs/fine/congress_retention_core_sweep/sweep_plan.yaml`
- `runs/fine/congress_retention_core_sweep/commands.sh`
- one dry-run folder per experiment

## Command: Execute Sweep

Add `--execute` to run all core experiments sequentially:

```bash
python tools/run_finetune_sweep.py \
  --weights /workspace/datasets/svms/file/SIAV2_Detector_YOLOV7_SafeEnv_V7.0.0_FP32_260223.pt \
  --base-data /workspace/datasets/svms/file/data/data.yaml \
  --data /workspace/datasets/svms/file/data/data.yaml \
  --hyp /workspace/datasets/svms/file/data/hyp.scratch.custom.yaml \
  --epochs 8 \
  --img 640 \
  --batch 64 \
  --workers 8 \
  --project runs/fine \
  --prefix congress_retention_core \
  --preset core \
  --launcher distributed \
  --nproc-per-node 4 \
  --master-port 9527 \
  --best-val-set combined \
  --save-best-only \
  --execute
```

## Result Reading

For each experiment, inspect:

- `weights/best.pt`
- `results_detail.txt`
- `scenario_metrics.csv`
- `stage_result.yaml`
- `replay_manifest.json`

Primary decision should use `best.pt`, not `last.pt`.

If the best candidate still drops the base validation too much, next actions are:

1. lower LR with a dedicated finetune hyp,
2. keep `--bn-policy eval`,
3. prefer replay `0.05` or `0.10` over `0.30`,
4. use cls distillation before reg distillation,
5. use `neck_lower` or `backbone` freeze only if finetune gain remains acceptable.
