# CrowdHuman Training Sequence Comparison - 2026-05-29

## Scope

- Source run: `runs/newyolov7_0527`
- Dataset: `data/crowdhuman.yaml`
- Model family: L only
- Stages: `00` to `08`
- Epochs: 300
- Batch: global 512, 8 GPUs, per-rank 64
- Validation mode: `--notest`, so reported mAP is final-epoch validation only, not best-over-epochs.
- Sequence mode: later stages use the previous kept stage weight as `start_weight`; this is sequential comparison, not fully isolated ablation.

## Final Stage Metrics

| Stage | Change | Decision | mAP50-95 | mAP50 | P | R | Delta mAP50-95 vs baseline | Train Loss | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 00 | baseline | keep_candidate | 0.46658 | 0.73603 | 0.84854 | 0.68384 | 0.00000 | 0.08111 | baseline reference |
| 01 | phase training | drop | 0.04532 | 0.08874 | 0.35976 | 0.10042 | -0.42127 | 0.14296 | phase2/3 collapse |
| 02 | decoupled head | keep_candidate | 0.46858 | 0.73411 | 0.86039 | 0.67470 | +0.00200 | 0.07803 | best mAP50-95 |
| 03 | WIoU v3 | keep_candidate | 0.46833 | 0.73270 | 0.85985 | 0.67627 | +0.00175 | 0.07419 | close to stage 02, lower loss |
| 04 | TAL + VFL | keep_candidate | 0.45862 | 0.71718 | 0.86087 | 0.67032 | -0.00796 | 0.10715 | regression |
| 05 | core cumulative | keep_candidate | 0.45412 | 0.71219 | 0.85651 | 0.67030 | -0.01246 | 0.10047 | cumulative regression |
| 06 | CCTV pixel aug | keep_candidate | 0.46490 | 0.72815 | 0.85691 | 0.67271 | -0.00168 | 0.08218 | near baseline |
| 07 | patch paste / hard negative | keep_candidate | 0.46475 | 0.72708 | 0.85480 | 0.67357 | -0.00183 | 0.08069 | near baseline |
| 08 | weighted sampler | drop | 0.00000 | 0.00000 | 0.00000 | 0.00000 | -0.46658 | NaN | loss NaN from epoch 88 |

## Findings

1. Best candidate is Stage 02 `head_decoupled`.
   - mAP50-95 improved from `0.46658` to `0.46858` (`+0.00200`).
   - Precision improved to `0.86039`, but recall decreased from `0.68384` to `0.67470`.
   - GFLOPs stayed at `103.17`, so current report does not show compute cost increase.

2. Stage 03 `wiou_v3` is also viable.
   - mAP50-95 is `0.46833`, nearly identical to Stage 02.
   - Train loss is the lowest among stable stages (`0.07419`), and validation loss is also low.
   - Because the sequence carries weights, Stage 03 was not a pure isolated WIoU-only ablation; it followed the kept Stage 02 weight.

3. Stage 01 phase training should not be adopted as currently configured.
   - Phase split was `200/60/40`.
   - Phase1 was stable, but phase2/phase3 changed to `640x384` rectangular mode.
   - Positive count dropped sharply from roughly `860k` in phase1 to `123k` at phase2 start and `60k` by final phase3.
   - Final mAP50-95 collapsed to `0.04532`.

4. Stage 04 and Stage 05 are not attractive for CrowdHuman in this run.
   - TAL+VFL reduced mAP50-95 by `-0.00796`.
   - Cumulative core combination reduced mAP50-95 by `-0.01246`.
   - These are within the runner's soft threshold but worse than baseline and worse than Stage 02/03.

5. Stage 06 and Stage 07 are neutral/slightly negative.
   - CCTV pixel augmentation: `-0.00168`.
   - Patch paste / hard negative: `-0.00183`.
   - They may still be useful for robustness, but this run does not show primary mAP improvement.

6. Stage 08 weighted sampler is invalid as a performance result.
   - Loss becomes NaN at epoch `88`.
   - Final metrics are all zero.
   - `sampler_stats.csv` shows both classes sampled in balanced counts, so the failure is likely not class omission.
   - More likely cause: weighted sampling repeatedly selects dense CrowdHuman images, causing unstable OTA assignment / objectness dynamics under high global batch.

## Recommendation

- Carry forward: Stage 02 first.
- Re-test candidate: Stage 03 as an isolated run from Stage 00 baseline weight, not only after Stage 02.
- Do not carry: Stage 01, Stage 08.
- Hold / optional: Stage 06 and Stage 07 only if robustness to CCTV artifacts matters more than small mAP loss.
- Rework before reuse: Stage 08 needs sampler caps before another 300 epoch run.

## Next Experiments

1. Run isolated comparison from Stage 00 weight:
   - `02_head_decoupled`
   - `03_wiou_v3`
   - `02+03` combined

2. Re-test phase training with safer settings:
   - Disable `640x384` rectangular shrink for CrowdHuman, or use a less aggressive phase2/3 size.
   - Keep phase3 short and validate before/after the phase boundary.

3. Rework weighted sampler:
   - Add per-image repeat cap.
   - Add max-label-density cap or down-weight very dense images.
   - Add non-finite loss guard to stop early and report the first NaN epoch.
   - Start with 20 to 50 epochs before any full 300 epoch run.
