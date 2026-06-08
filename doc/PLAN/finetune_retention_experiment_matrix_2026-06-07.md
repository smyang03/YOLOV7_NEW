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
- `tools/run_finetune_sweep.py` now creates and optionally executes retention and pseudo-label sweeps.
- `tools/prepare_pseudo_old_labels.py` creates a shadow dataset with teacher pseudo old labels merged into finetune labels.
- `tools/summarize_finetune_retention.py` summarizes base retention and finetune gain from `results_detail.txt`.

Still limited:

- Replay ratio is implemented as selected replay images concatenated to the train list, not guaranteed per-batch sampling.
- Distillation is an MVP MSE-based implementation, not a full paper-faithful YOLO LwF implementation.
- Pseudo old-label completion now has a tool and sweep preset, but it still depends on teacher pseudo quality.
- Per-batch replay ratio is still not guaranteed. Use `--replay-ratio-source finetune` or `--replay-count` to make the selected replay count explicit.

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

## Pseudo Old-Label Sweep

`tools/run_finetune_sweep.py --preset pseudo` creates these experiments:

| ID | Purpose |
| --- | --- |
| `p00_pseudo_only` | teacher pseudo old-label completion without replay |
| `p01_pseudo_replay005` | pseudo labels with replay `0.05` |
| `p02_pseudo_replay010` | primary pseudo + replay candidate |
| `p03_pseudo_replay010_cls` | pseudo + replay + cls/objectness distill |
| `p04_pseudo_replay010_freeze_cls` | conservative pseudo + replay + freeze + cls distill |
| `p05_pseudo_replay030` | pseudo labels with original replay level |

The pseudo stage writes:

- `pseudo_old_labels/pseudo_old_data.yaml`
- `pseudo_old_labels/train_pseudo_old_labels.txt`
- `pseudo_old_labels/pseudo_old_label_manifest.json`
- `pseudo_old_labels/images/`
- `pseudo_old_labels/labels/`

The original dataset is not modified.

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

## Command: Pseudo 1-Epoch Smoke

Use this first to check that pseudo-label preparation, finetune dry-run, DDP train launch, and multi-validation work end to end:

```bash
python tools/run_finetune_sweep.py \
  --weights /workspace/datasets/svms/file/SIAV2_Detector_YOLOV7_SafeEnv_V7.0.0_FP32_260223.pt \
  --teacher-weights /workspace/datasets/svms/file/SIAV2_Detector_YOLOV7_SafeEnv_V7.0.0_FP32_260223.pt \
  --base-data /workspace/datasets/svms/file/data/data.yaml \
  --data /workspace/datasets/svms/file/data/data.yaml \
  --hyp /workspace/datasets/svms/file/data/hyp.scratch.custom.yaml \
  --epochs 1 \
  --img 640 \
  --batch 64 \
  --workers 8 \
  --project runs/fine \
  --prefix congress_pseudo_smoke \
  --preset pseudo \
  --launcher distributed \
  --nproc-per-node 4 \
  --master-port 9527 \
  --best-val-set combined \
  --replay-ratio-source finetune \
  --pseudo-conf 0.6 \
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

## Result Summary Command

```bash
python tools/summarize_finetune_retention.py \
  --baseline runs/fine/base_eval_or_original_run \
  --runs runs/fine/congress_pseudo_smoke_* \
  --finetune-scenario congress_valid \
  --base-scenario base_valid \
  --select-scenario combined \
  --output runs/fine/congress_pseudo_retention_summary.csv \
  --json-output runs/fine/congress_pseudo_retention_summary.json
```

## One-Command Full Suite

This runs core and pseudo experiments, continues after individual failures, and writes the final summary:

```bash
python tools/run_finetune_sweep.py \
  --weights /workspace/datasets/svms/file/SIAV2_Detector_YOLOV7_SafeEnv_V7.0.0_FP32_260223.pt \
  --teacher-weights /workspace/datasets/svms/file/SIAV2_Detector_YOLOV7_SafeEnv_V7.0.0_FP32_260223.pt \
  --base-data /workspace/datasets/svms/file/data/data.yaml \
  --data /workspace/datasets/svms/file/data/data.yaml \
  --hyp /workspace/datasets/svms/file/data/hyp.scratch.custom.yaml \
  --epochs 1 \
  --img 640 \
  --batch 64 \
  --workers 8 \
  --project runs/fine \
  --prefix congress_all_smoke \
  --preset core_pseudo \
  --launcher distributed \
  --nproc-per-node 4 \
  --master-port 9527 \
  --best-val-set combined \
  --replay-ratio-source finetune \
  --pseudo-conf 0.6 \
  --save-best-only \
  --execute \
  --summarize \
  --continue-on-error \
  --baseline-run runs/fine/base_eval_or_original_run \
  --finetune-scenario congress_valid \
  --base-scenario base_valid \
  --summary-select-scenario combined
```

Final outputs:

- `runs/fine/congress_all_smoke_sweep/sweep_plan.yaml`
- `runs/fine/congress_all_smoke_sweep/commands.sh`
- `runs/fine/congress_all_smoke_sweep/sweep_run_status.json`
- `runs/fine/congress_all_smoke_sweep/retention_summary.csv`
- `runs/fine/congress_all_smoke_sweep/retention_summary.json`

For copy/paste command variants, use `doc/PLAN/finetune_command_guide_2026-06-08.md`.
