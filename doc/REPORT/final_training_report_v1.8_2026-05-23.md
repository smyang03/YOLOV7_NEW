# Final Training Report v1.8 - 2026-05-23

## 실행 요약

- sequence_dir: runs\train_seq\qa_logging_recheck_fail
- total_stages: 1
- blockers: 1

## Stage별 결정표

| Stage | Model | Decision | Reason |
| --- | --- | --- | --- |
| 00 baseline | l | blocker | training command failed with exit code 1 |

## 유지


## 제거


## 재실험


## 원인

- 00 baseline: train

## 다음 액션

- COCO128 quick 결과는 orchestration, 산출물, hard fail 판정 확인에만 사용한다.
- target full run 결과에서 최종 유지/제거/재실험 판단을 확정한다.
- blocker stage가 있으면 해당 stage의 stderr와 stage_result.yaml을 먼저 확인한다.
