# v1.3 개발 계획 재정비 리포트

## 문서 정보

- 작성일: 2026-05-22
- 기준 리포트: `doc/REPORT/ai_perspective_yolov7_improvement_analysis_2026-05-22.md`
- 대상 문서: `doc/PLAN/development_plan_v1.3.md`, `doc/PLAN/development_requirements_1.3.1_*.md` ~ `doc/PLAN/development_requirements_1.3.7_*.md`

## 1. 정비 목적

AI 관점 분석 리포트에서 확인한 위험을 개발 문서에 반영했다. 핵심은 개발 문서 위치를 `doc/PLAN/`으로 통일하고, 각 차수를 실패 원인 추적이 가능한 순서로 재정렬하는 것이다.

## 2. 변경 내용

1. `doc/dev/`에 있던 1.3.1~1.3.7 코드레벨 개발 요구서를 `doc/PLAN/`으로 이동했다.
2. `doc/PLAN/development_plan_v1.3.md`를 리포트 기반 실행 계획서로 재작성했다.
3. 상위 plan에 문서 인덱스, 공통 개발 원칙, 전체 실행 순서, 공통 중단 조건, 현재 착수 위치를 추가했다.
4. 1.3.1~1.3.7 코드레벨 요구서 상단에 `리포트 기반 정비 기준`을 추가했다.
5. 1.3.3 core model/loss 문서의 검증 순서를 `Decoupled Head 단독 -> WIoU 단독 -> TAL+VFL 단독 -> 누적 적용`으로 정렬했다.

## 3. 현재 개발 순서

현재 코드 개발 시작점은 `1.3.1-P1`이다.

우선 개발 항목:
- CLI alias 정리
- 일반 YOLO layout `images -> labels` 매핑 복구
- label cache hash/version invalidation 복구
- `persistent_workers`와 Close Mosaic 충돌 방지

`1.3.1-P1`과 `1.3.1-P2`가 끝나기 전에는 loss, head, augmentation, W6 구조 변경을 시작하지 않는다.

## 4. 남은 주의점

과거 `doc/REPORT/*` 파일에는 당시 기준의 `doc/dev/` 경로가 기록되어 있을 수 있다. 현재 개발 문서의 기준 위치는 `doc/PLAN/`이다. 새로 작성하거나 수정하는 개발 문서는 모두 `doc/PLAN/`에 저장한다.

## 5. 결론

v1.3 개발 문서는 이제 리포트 기반의 실행 순서와 문서 위치 정책을 반영한다. 다음 작업은 문서 추가 작성이 아니라 `doc/PLAN/development_requirements_1.3.1_baseline_export.md`의 `1.3.1-P1` 구현이다.
