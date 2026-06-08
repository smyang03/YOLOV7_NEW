# Finetuning Recent Paper Review 2026-06-08

## Conclusion

The current finetuning pipeline is directionally correct, but it is still a first-order continual-learning baseline:

- replay images are selected and concatenated,
- distillation is raw output MSE,
- pseudo-label generation exists as a separate tool but is not central to the retention sweep,
- freeze policy is coarse.

Recent object-detection continual-learning papers point to four improvements worth considering:

1. fill missing old-class annotations with teacher pseudo labels,
2. improve replay quality and replay ratio control,
3. replace raw MSE distillation with confidence/IoU-filtered YOLO-aware distillation,
4. make parameter updates more selective than the current coarse freeze policies.

The strongest near-term implementation path is not generative replay or architecture replacement. It is:

`pseudo old-label completion -> per-batch/balanced replay -> safer YOLO distillation -> better retention reporting`.

## Sources Reviewed

Primary sources:

- Teach YOLO to Remember: A Self-Distillation Approach for Continual Object Detection, arXiv 2503.04688, 2025-03-06  
  https://arxiv.org/abs/2503.04688
- Replay Consolidation with Label Propagation for Continual Object Detection, arXiv 2409.05650, 2024-09-09  
  https://arxiv.org/abs/2409.05650
- Bridge Past and Future: Overcoming Information Asymmetry in Incremental Object Detection, ECCV 2024 / arXiv 2407.11499  
  https://arxiv.org/abs/2407.11499
- IOR: Inversed Objects Replay for Incremental Object Detection, arXiv 2406.04829, 2024-06-07  
  https://arxiv.org/abs/2406.04829
- SDDGR: Stable Diffusion-based Deep Generative Replay for Class Incremental Object Detection, CVPR 2024 / arXiv 2402.17323  
  https://arxiv.org/abs/2402.17323
- Re-examining Distillation For Continual Object Detection, BMVC 2022 / arXiv 2204.01407  
  https://arxiv.org/abs/2204.01407
- Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation, CVPR 2022 / arXiv 2204.02136  
  https://arxiv.org/abs/2204.02136
- YOLO-IOD: Towards Real Time Incremental Object Detection, AAAI 2026 / arXiv 2512.22973  
  https://arxiv.org/abs/2512.22973
- Informativeness-Aware Layer Freezing and Sample Replay for Efficient Online Continual Object Detection, KTSDE 2026  
  https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART003330250

## Paper Findings Mapped To Our Problem

### 1. Missing annotations are a core cause, not a side issue

Several papers describe the same failure mode we observed: when new/finetune images contain old objects without labels, the detector learns those old objects as background. This is directly relevant to YOLO finetuning.

Relevant papers:

- YOLO LwF paper discusses the missing annotation problem for CLOD.
- BPF focuses on inconsistent foreground/background definitions across stages.
- YOLO-IOD explicitly calls out foreground-background confusion.
- SDDGR uses pseudo-labeling for old objects in new-task images.
- RCLPOD improves replay memory labels through label propagation.

Current repo state:

- `tools/generate_pseudo_labels.py` and `tools/merge_labels.py` exist.
- `finetune.py` only writes skip manifests; it does not make pseudo-label completion part of the main sweep.

Recommendation:

- Add a controlled pseudo old-label completion stage before replay/distill sweeps.
- Keep finetune GT labels authoritative.
- Add teacher boxes only for old/base classes missing from the finetune labels.
- Use conservative thresholds first: confidence `0.6~0.7`, dedupe IoU `0.8`, NMS IoU `0.45`.
- Log added boxes per class and per split.

Priority: high.

### 2. Replay quality matters more than just replay size

RCLPOD and YOLO LwF both emphasize class-balanced replay. The 2026 informativeness-aware work goes further: replay should prioritize samples that are hard but under-trained, using sample usage frequency and EMA loss. IOR and SDDGR cover cases where old images are unavailable, but those are more expensive paths.

Current repo state:

- `ReplayBufferBuilder` picks rare-class-heavy samples from base train.
- `finetune.py` concatenates finetune images and replay images.
- There is no per-batch replay ratio guarantee.
- There is no loss/usage based replay priority.

Recommendation:

- Add explicit replay count semantics:
  - `--replay-ratio-source base|finetune`
  - `--replay-count N`
- Add a balanced replay list builder that targets a uniform class distribution, not only rare-score sorting.
- Add optional weighted replay sampling or a mixed batch sampler so replay ratio is visible during training.
- Later, add a `loss_ema` replay manifest update after each experiment.

Priority: high for ratio semantics and balanced list; medium for loss-EMA replay.

### 3. Current MSE distillation is too naive

The repo currently applies MSE over raw prediction slices. Recent YOLO/one-stage papers repeatedly warn that dense regression outputs are noisy. YOLO LwF proposes weighting distillation by teacher confidence and spatial agreement. ERD and SID also focus on selecting valuable response locations rather than distilling every dense output uniformly. Re-examining Distillation also warns that wrong overconfident teacher predictions can hurt learning.

Current repo state:

- `utils/continual_loss.py` supports schedules and guards shape mismatch.
- Classification/objectness distillation is MSE.
- Regression distillation is MSE on positions where `t[..., 4] >= conf_thres`.
- There is no decoded-box IoU weighting.
- There is no adaptive Huber or KL/CE mode.

YOLOv7 note:

- YOLO LwF is written around YOLOv8-style output. YOLOv7 is anchor-based and does not expose the same DFL distribution. Directly copying the paper loss is not appropriate.

Recommendation:

- Add `--distill-mode mse|selected_mse|huber`.
- Implement `selected_mse` first:
  - select teacher locations by objectness/class confidence,
  - optionally keep top-k teacher predictions per image after NMS,
  - distill classification/objectness only for selected locations,
  - distill box only when decoded teacher and student boxes are spatially aligned enough,
  - keep raw MSE mode as backward-compatible default.
- Add adaptive Huber for cls/objectness as a safer second step.
- Do not enable reg distill by default until a selected/filtered mode exists.

Priority: high.

### 4. Parameter-efficient updates are relevant

YOLO-IOD identifies parameter interference and proposes selecting important kernels. The 2026 informativeness-aware paper similarly combines layer freezing with sample replay. This supports our current use of `--bn-policy eval` and `--freeze`, but suggests that the current policies are too coarse.

Current repo state:

- `finetune.py` maps `none`, `backbone`, `partial`, `neck_lower` to fixed layer counts.
- The sweep tests BN eval and freeze combinations.

Recommendation:

- Add more explicit freeze policies:
  - `head_only`
  - `neck_head`
  - `last_n`
- Log trainable/frozen parameter count and resolved module names.
- Treat kernel-level selection as research-grade and defer until the simpler freeze policies are tested.

Priority: medium.

### 5. Generative replay is useful only if old data is unavailable

SDDGR and IOR are relevant when old class images are missing or cannot be stored. They are more complex than needed if we already have base data or can keep a small replay buffer.

Current repo state:

- Base data is available in the current workflow.
- The user wants practical retention experiments now.

Recommendation:

- Do not implement Stable Diffusion replay now.
- Keep it as a fallback if old data retention/storage becomes impossible.
- If needed later, start with object crop replay/copy-paste before full generative replay.

Priority: low.

## Proposed Implementation Backlog

### P0: Evaluation and experiment safety

1. Add a parser for `results_detail.txt` to produce per-val and per-class retention CSV.
2. Add a post-sweep summarizer:
   - finetune validation best mAP50-95,
   - base validation best mAP50-95,
   - base retention percent,
   - old-class worst drop,
   - selected checkpoint path.
3. Keep `--best-val-set combined` as the default in sweep runner.

Reason:

- Without this, we can run many experiments but still judge them manually and inconsistently.

### P1: Pseudo old-label completion

1. Add `--pseudo-old-labels on|off` to `finetune.py` or the sweep tool.
2. Generate teacher pseudo labels on finetune train images.
3. Merge only old/base classes not already covered by GT.
4. Write a new data yaml pointing to merged labels.
5. Log added pseudo counts per class.

Expected effect:

- Reduces old objects being trained as background.
- Directly addresses the failure mode cited by YOLO LwF, BPF, SDDGR, RCLPOD, and YOLO-IOD.

### P2: Replay ratio and replay distribution

1. Add `--replay-ratio-source base|finetune`.
2. Add `--replay-count`.
3. Add uniform target class distribution replay selection.
4. Add manifest fields:
   - target count,
   - actual replay/finetune ratio,
   - per-class replay counts,
   - duplicated image count.

Expected effect:

- Makes experiments reproducible and interpretable.
- Prevents replay `0.3` from meaning different things depending on base dataset size.

### P3: Safer YOLOv7 distillation

1. Keep existing MSE as `--distill-mode mse`.
2. Add `--distill-mode selected_mse`.
3. Use teacher confidence/top-k/NMS selection before applying distill.
4. Add Huber option after selected MSE is stable.
5. Log selected distill locations per batch/epoch.

Expected effect:

- Avoids forcing the student to copy dense noisy teacher outputs.
- Should reduce the type of late-epoch retention collapse seen in the previous run.

### P4: Selective training policies

1. Add freeze policies with explicit module semantics.
2. Record trainable/frozen parameter counts.
3. Test `head_only`, `neck_head`, `neck_lower`, `none`.

Expected effect:

- Reduces parameter interference while preserving enough plasticity for tuning data.

## Recommended Next Sweep After P0/P1

Use short epochs first:

| ID | pseudo old labels | replay | BN | freeze | distill |
| --- | --- | --- | --- | --- | --- |
| `p0_plain` | off | 0.0 | train | none | off |
| `p1_pseudo_only` | on | 0.0 | eval | none | off |
| `p2_pseudo_replay005` | on | 0.05 | eval | none | off |
| `p3_pseudo_replay010` | on | 0.10 | eval | none | off |
| `p4_pseudo_replay010_cls` | on | 0.10 | eval | none | selected cls |
| `p5_pseudo_replay010_freeze` | on | 0.10 | eval | neck_lower | selected cls |

Run for `6~8` epochs first. Select by combined best and confirm with base validation retention.

## Decision

Implementable now:

- per-val/per-class retention summarizer,
- pseudo old-label completion stage,
- explicit replay count/ratio semantics,
- selected/high-confidence distillation mode,
- better freeze logging.

Defer:

- diffusion replay,
- detector inversion replay,
- kernel-level selection,
- cross-stage asymmetric distillation.

The most useful immediate correction is pseudo old-label completion. It directly targets the condition where old objects in finetune images are mislabeled as background, which is likely one of the main reasons base validation drops during naive finetuning.

## Implementation Update 2026-06-08

Implemented in this pass:

- `tools/prepare_pseudo_old_labels.py`
  - creates a shadow `images/labels` dataset under the experiment directory,
  - generates teacher pseudo labels when `--weights` is provided,
  - can also consume an existing pseudo-label directory,
  - merges GT and pseudo labels without modifying the original dataset,
  - writes `pseudo_old_data.yaml` for training.
- `tools/run_finetune_sweep.py`
  - adds `--preset pseudo`,
  - adds `--preset core_pseudo`,
  - runs pseudo preparation before `finetune.py --dry-run` when an experiment requires it,
  - adds `--replay-ratio-source` and `--replay-count` passthrough.
- `finetune.py`
  - supports `--replay-ratio-source base|finetune`,
  - supports `--replay-count`,
  - records requested replay count in the manifest and CSV.
- `tools/summarize_finetune_retention.py`
  - reads `results_detail.txt`,
  - selects a best epoch by `combined` or named validation set,
  - reports finetune gain, base retention percent, and worst base-class drop.

Not implemented yet:

- selected/high-confidence distillation mode,
- teacher high-confidence background ignore inside the YOLO objectness loss,
- objectness-loss hyp variant generation in the sweep runner.
