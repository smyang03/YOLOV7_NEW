# v1.8 학습 실행 플랜 재검토 리포트

## 문서 정보

- 작성일: 2026-05-22
- 대상 문서: `doc/PLAN/training_execution_plan_v1.8.md`
- 추가 문서: `doc/PLAN/training_report_format_v1.8.md`
- 검토 관점: 개발 완료 후 연속 stage 학습, 문제 원인 추적, 최종 리포트 판정 가능성

## 1. 재검토 결론

v1.8의 stage 순서는 적합하다. Baseline에서 시작해 phase/logging, core model/loss, augmentation/sampler, W6 구조, optional, fine-tuning으로 넘어가는 흐름은 문제 원인을 분리하기에 맞다.

다만 기존 v1.8 문서는 “최종 리포트에 포함할 항목”은 있었지만, 실제로 내가 결과를 받아 판단하기 위한 리포트 판정 규격이 부족했다. 이번 재검토에서 리포트 생성 방식, delta 계산 기준, stage별 decision 값, 실패 원인 분류, 최종 판단 형식을 별도 문서로 보강했다.

## 2. 발견한 부족점

1. Stage별 `keep/drop/retry/blocker` 판정 규칙이 명확하지 않았다.
2. `baseline 대비`, `직전 stage 대비`, `이전 최고 stage 대비` 비교 기준이 분리되어 있지 않았다.
3. Stage별 요약 리포트와 최종 리포트의 역할이 나뉘어 있지 않았다.
4. 실패 원인을 data/loader/loss/assignment/architecture/export/runtime/augmentation/finetune으로 분류하는 기준이 없었다.
5. 최종적으로 내가 어떤 형태로 “유지/제거/재실험/원인/다음 액션”을 줄지 명시되지 않았다.

## 3. 반영한 수정

- `doc/PLAN/training_report_format_v1.8.md`를 신규 작성했다.
- `doc/PLAN/training_execution_plan_v1.8.md`에 리포트 규격 문서 링크를 추가했다.
- 공통 산출물에 `stage_summary.md`, `sequence_summary.md`를 추가했다.
- 최종 리포트 구조에 3단계 리포트 체계와 decision 값을 추가했다.
- 최종 판단 항목을 `유지`, `제거`, `재실험`, `원인`, `다음 액션`으로 고정했다.
- 최초 빠른 검증은 `data/coco128.yaml` 기반 COCO128 quick run으로 수행하도록 추가했다.

## 4. 최종 리포트 운영 방식

학습 서버 실행 후에는 각 stage마다 `stage_summary.md`를 먼저 만든다. 이 파일은 해당 stage의 즉시 판정을 담당한다.

전체 sequence가 끝나면 `sequence_summary.md`로 stage 흐름을 합친다. 이때 각 stage는 baseline, previous success, best previous 세 기준으로 delta를 계산한다.

마지막으로 `doc/REPORT/final_training_report_v1.8_YYYY-MM-DD.md`를 작성한다. 이 리포트는 단순 성능표가 아니라 최종 의사결정 문서다.

## 5. 내가 최종 리포트에서 판단할 내용

최종 결과를 받으면 아래 순서로 분석한다.

1. hard fail stage를 먼저 분리한다.
2. mAP 상승이 비용 증가를 정당화하는지 본다.
3. L 모델은 속도형 역할을 유지하는지 본다.
4. W6 모델은 small AP/rare recall 개선이 충분한지 본다.
5. 기능별로 `keep`, `drop`, `retry_tune`, `defer`를 결정한다.
6. 최종 full run 후보를 1~2개로 줄인다.
7. 다음 코드 수정 또는 재실험 액션을 지정한다.

## 6. 현재 판단

v1.8은 이제 학습 실행 문서로 사용 가능하다. 리포트 규격까지 추가했기 때문에, 개발 완료 후 stage 결과만 제대로 저장되면 어떤 기능이 성능을 올렸고 어떤 기능이 비용이나 문제를 만들었는지 추적할 수 있다.

최초 실행은 COCO128 quick run으로 `Stage 00 -> Stage 01 -> Stage 02`를 먼저 확인한다. 이 결과는 최종 성능 판단이 아니라 orchestration, 산출물 생성, 리포트 판정 검증에만 사용한다.

다음 실제 개발 문서 기준 시작점은 여전히 `1.3.1-P1`이다. v1.8은 모든 개발이 완료된 뒤 학습 서버에서 연속 실행할 운영 플랜으로 사용한다.
