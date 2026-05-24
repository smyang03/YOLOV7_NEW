# Final Training Report v1.8 - 2026-05-23

## 실행 요약

- sequence_dir: runs\tmp_141_sequence\coco128_stage00_08_e3_retry
- total_stages: 9
- blockers: 0

## Stage별 결정표

| Stage | Model | Decision | Reason |
| --- | --- | --- | --- |
| 00 baseline | l | keep | COCO128 quick run passed. Artifacts generated. |
| 01 phase_logging | l | keep | COCO128 quick run passed. Artifacts generated. |
| 02 head_decoupled | l | keep | COCO128 quick run passed. Artifacts generated. |
| 03 wiou_v3 | l | keep | COCO128 quick run passed. Artifacts generated. |
| 04 tal_vfl | l | keep | COCO128 quick run passed. Artifacts generated. |
| 05 core_cumulative | l | keep | COCO128 quick run passed. Artifacts generated. |
| 06 cctv_pixel_aug | l | keep | COCO128 quick run passed. Artifacts generated. |
| 07 patch_paste_hard_negative | l | keep | COCO128 quick run passed. Artifacts generated. |
| 08 weighted_sampler | l | keep | COCO128 quick run passed. Artifacts generated. |

## 유지

- 00 baseline: COCO128 quick run passed. Artifacts generated.
- 01 phase_logging: COCO128 quick run passed. Artifacts generated.
- 02 head_decoupled: COCO128 quick run passed. Artifacts generated.
- 03 wiou_v3: COCO128 quick run passed. Artifacts generated.
- 04 tal_vfl: COCO128 quick run passed. Artifacts generated.
- 05 core_cumulative: COCO128 quick run passed. Artifacts generated.
- 06 cctv_pixel_aug: COCO128 quick run passed. Artifacts generated.
- 07 patch_paste_hard_negative: COCO128 quick run passed. Artifacts generated.
- 08 weighted_sampler: COCO128 quick run passed. Artifacts generated.

## 제거


## 재실험


## 원인


## 다음 액션

- COCO128 quick 결과는 orchestration, 산출물, hard fail 판정 확인에만 사용한다.
- target full run 결과에서 최종 유지/제거/재실험 판단을 확정한다.
- blocker stage가 있으면 해당 stage의 stderr와 stage_result.yaml을 먼저 확인한다.
