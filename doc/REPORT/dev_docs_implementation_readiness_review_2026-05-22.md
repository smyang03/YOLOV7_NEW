# doc/dev 개발 착수성 재검토

- 작성일: 2026-05-22
- 기준 문서: `doc/dev/development_requirements_1.3.1_*.md` ~ `doc/dev/development_requirements_1.3.7_*.md`
- 관점: 실제 개발자가 지금 코드 구현을 시작한다고 가정한 착수성 점검
- 현재 코드 기준: 신규 플래그와 신규 도구 대부분은 아직 미구현

## 결론

`doc/dev` 문서들은 기능 요구 수준으로는 충분하다. 다만 실제 개발 착수 관점에서는 각 차수가 너무 큰 단위로 보일 수 있어, 이번 점검에서 모든 dev 문서에 `개발 착수 분리 기준`을 추가했다. 이제 각 차수는 PR 단위와 완료 기준이 명확하다.

첫 개발은 `1.3.1-P1`부터 시작하는 것이 맞다. 데이터 경로, cache, dataloader, CLI alias가 안정화되기 전에는 export/profile/phase 작업으로 넘어가면 실패 원인 추적이 어렵다.

## 즉시 개발 순서

1. `1.3.1-P1`: CLI alias, `images -> labels`, cache invalidation, persistent worker
2. `1.3.1-P2`: `best.pt`, `test.test()` 4-return, `opt.yaml`, `results.txt`
3. `1.3.1-P3`: raw ONNX export, profile, verify export
4. `1.3.1-P4`: dataset/metric manifest와 baseline report
5. `1.3.2-P1`: Phase boundary 모듈과 dry-run test

`1.3.2` 이후는 `1.3.1` 산출물인 `best.pt`, `profile.json`, `export_check.json`, `output_contract.json`, `dataset_manifest.json`이 있어야 시작한다.

## 차수별 개발자 관점 보강

### 1.3.1

문제:
- 요구사항이 많아 첫 PR 범위가 커질 수 있었다.
- export/profile 도구를 먼저 만들면 dataloader/checkpoint 문제가 원인을 흐릴 수 있다.

보강:
- `1.3.1-P1`~`P4` PR 단위 추가
- 데이터/체크포인트 안정화 후 export/profile 도구로 넘어가도록 명시

### 1.3.2

문제:
- `train.py`와 `train_aux.py` 통합은 대형 변경이라 바로 통합하면 회귀 추적이 어렵다.

보강:
- `utils/phase.py`, `utils/train_logger.py`, dataloader rebuild, `utils/train_common.py`, `train_aux.py` wrapper화를 순차 PR로 분리
- `train_aux.py`는 바로 삭제/대체하지 않고 W6 AUX smoke 후 wrapper화하도록 명시

### 1.3.3

문제:
- Head/Loss/Assign 기능이 서로 얽혀 있어 구현 순서와 검증 순서가 혼동될 수 있었다.

보강:
- `1.3.3-P1`~`P5`로 loss option, WIoU, TAL/VFL, Decoupled Head, 누적 적용을 분리
- 검증 run은 반드시 A1/A2/A3 단독으로 남기도록 명시

### 1.3.4

문제:
- CCTV augmentation 목록이 넓어서 첫 구현 범위가 과도했다.

보강:
- 최소 pixel aug 세트를 SpiderWeb, ToGray, CLAHE, blur로 제한
- LensFlare, RandomSunFlare, Helmet paste, MixUp, Rolling Shutter는 옵션 후순위로 분리
- hard negative mining과 scenario metric을 별도 PR로 분리

### 1.3.5

문제:
- P2와 SCDown을 하나의 cfg로 처리하면 route/channel 오류 원인을 분리하기 어렵다.
- 문서 예시가 P2+SCDown인데 cfg가 `yolov7-w6-p2.yaml`로 되어 있어 혼동 여지가 있었다.

보강:
- `yolov7-w6-scdown.yaml`, `yolov7-w6-p2.yaml`, `yolov7-w6-p2-scdown.yaml`로 training/deploy cfg 분리
- D1/D2/D3와 P1/P2/P3/P4 PR 단위를 명시
- P2+SCDown 예시는 `yolov7-w6-p2-scdown.yaml`로 수정

### 1.3.6

문제:
- optional 실험은 구현하면 기본 경로처럼 굳어질 위험이 있다.

보강:
- `1.3.6-P0`에서 optional 진입 사유 문서를 먼저 쓰도록 추가
- optional flag validation을 먼저 넣고, 둘 이상 켜지면 실행 전 실패하도록 명시

### 1.3.7

문제:
- `--distill-alpha 0.2:0.5` 같은 schedule 문자열은 argparse 처리 기준이 없었다.
- finetune loop를 복사하면 1.3.2의 공통 학습 정책과 갈라질 수 있다.

보강:
- `parse_float_or_schedule()` helper 기준 추가
- `finetune.py`가 학습 loop를 복사하지 않고 1.3.2 공통 helper를 재사용하도록 명시
- Replay only, cls distill, cls+reg distill을 PR 단위로 분리

## 개발 전 남은 확인

- `requirements.txt`에서 `onnx` 주석 처리 여부를 먼저 정리해야 한다.
- `tools/`에는 현재 notebook만 있고 Python 도구가 없으므로 1.3.1-P3/P4에서 신규 스크립트 생성이 필요하다.
- `utils/datasets.py`의 `persistent_workers=True` 고정과 label cache hash 주석 처리는 1.3.1-P1 blocker다.
- `train.py`와 `train_aux.py`의 `best.pt` 기본 분기 누락 가능성, `test.test()` 3-value unpack은 1.3.1-P2 blocker다.
- `export.py`는 현재 opset 12 고정이므로 1.3.1-P3에서 `--opset 16`, `--nms-mode none`을 먼저 맞춘다.

## 최종 착수 판단

문서 추가 보강은 충분하다. 실제 개발은 `doc/dev/development_requirements_1.3.1_baseline_export.md`의 `1.3.1-P1`부터 시작한다. `1.3.1` 전체를 한 번에 구현하지 말고, P1/P2/P3/P4 순서로 작게 끝내는 것이 원인 추적에 가장 안전하다.
