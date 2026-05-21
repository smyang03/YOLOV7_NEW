# 1.3.1-1.3.7 코드레벨 요구서 적합성 점검

- 점검일: 2026-05-21
- 대상 문서: `doc/dev/development_requirements_1.3.1_*.md` ~ `doc/dev/development_requirements_1.3.7_*.md`
- 기준 계획: `doc/PLAN/development_plan_v1.3.md`
- 점검 범위: 현재 코드 구조와 계획 차수 간 정합성
- 제외: C++ 후처리, TensorRT runtime, TensorRT engine build, 별도 추론 서버

## 결론

`doc/dev`의 1.3.1~1.3.7 요구서는 차수별 순서와 큰 범위가 계획서와 맞다. 다만 실제 코드 기준으로 바로 구현 시 혼동될 수 있는 세부 항목이 있어 각 요구서에 보강을 반영했다. 현재 문서 상태는 코드 개발 착수용으로 사용 가능하다.

## 차수별 점검 결과

| 차수 | 판정 | 반영한 개선점 |
| --- | --- | --- |
| 1.3.1 | 적합, 보강 필요 | validation checksum 산출 도구가 누락되어 `tools/dataset_manifest.py`, `dataset_manifest.json` 요구사항을 추가했다. 학습 `--img`와 export `--img H W` 의미 차이를 명시했다. |
| 1.3.2 | 적합, 보강 필요 | `train.py`/`train_aux.py` 통합 시 중복 로직 방지를 위해 `utils/train_common.py` 요구사항을 추가했다. `--aux auto`와 GFLOPs delta 기록 기준을 추가했다. |
| 1.3.3 | 적합, 보강 필요 | `Detect`/`IDetect`/`IAuxDetect` 계열별 decoupled head 처리, bias init, fuse/export 경로를 명시했다. WIoU/VFL 공통 helper와 CUDA device 정합성 기준을 보강했다. |
| 1.3.4 | 적합, 일부 예시 수정 | `--image-weights False`는 argparse 형식상 잘못된 예시라 제거했다. aug hook 좌표계, cache 비저장 정책, dataloader 통합 지점을 명시했다. |
| 1.3.5 | 보강 후 적합 | 기존 W6는 P3/P4/P5/P6 4-level 구조이므로 P2 추가 시 5-level이 되어야 한다. 문서의 4-level 표현을 5-level, stride `4,8,16,32,64` 기준으로 수정했다. |
| 1.3.6 | 적합, 결정 기록 보강 | optional 실험은 기본값 승격/보류 판단이 필요하므로 `doc/REPORT/optional_decision_*.md` 산출물을 추가했다. |
| 1.3.7 | 적합, 데이터 이력 보강 | finetune/replay/pseudo 병합 데이터 manifest/hash와 Python-only pseudo label 생성 기준을 추가했다. |

## 현재 코드 기준 주요 주의점

- `train.py`, `train_aux.py`, `test.py`, `export.py`는 아직 신규 플래그 대부분을 갖고 있지 않다. 문서는 개발 요구서이며 구현 완료 상태를 의미하지 않는다.
- 현재 학습 스크립트의 `--img-size`는 기존 YOLOv7 방식의 train/test scalar size다. 학습 명령에서 `--img 1280 736`을 H/W로 쓰면 안 된다.
- `export.py`, `tools/verify_export.py`, `tools/profile_model.py` 계열은 `--img H W`를 input shape으로 사용하도록 요구서에 구분했다.
- 현재 W6 cfg는 `IAuxDetect`에 P3/P4/P5/P6 main 4개와 aux 4개 feature를 전달한다. P2 추가 시 main 5개와 aux 5개 처리가 필요하다.
- `utils/datasets.py`의 label cache hash 검증과 `persistent_workers=True` 고정은 1.3.1/1.3.2에서 먼저 보정해야 한다.

## 다음 단계

1. `1.3.1` 구현 전 `doc/dev/development_requirements_1.3.1_baseline_export.md`를 기준으로 P1 blocker를 처리한다.
2. 각 차수 개발 완료 후 `doc/REPORT/stage_result_1.3.x_YYYY-MM-DD.md` 또는 각 요구서에 정의한 산출물을 저장한다.
3. 후속 차수는 직전 차수의 `stage_result.yaml`, `profile.json`, `export_check.json`이 존재할 때만 시작한다.
