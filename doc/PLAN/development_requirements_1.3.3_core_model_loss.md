# 1.3.3 Code-Level Development Requirements

## 공통 예외 사항 - 원 코드 유지 개발

- 본 차수의 새 기능은 기본값으로 비활성화한다. 플래그를 켜지 않으면 기존 YOLOv7 학습, 평가, export 동작이 유지되어야 한다.
- 기존 함수/클래스는 버그 수정, 호환성 보강, 공통 helper 호출 연결에 한해서만 직접 수정한다.
- 신규 기능은 가능한 `utils/*`, `models/*`의 새 helper/class/wrapper로 분리하고, 기존 entrypoint는 기존 CLI와 출력 경로를 유지한다.
- `train.py`, `train_aux.py`, `test.py`, `export.py`는 기존 옵션명을 삭제하지 않는다. alias를 추가할 때도 기존 `dest`와 결과 파일명을 바꾸지 않는다.
- `train_aux.py`는 즉시 삭제하거나 대체하지 않는다. 공통 helper를 먼저 만들고 AUX/W6 smoke 검증 후 얇은 wrapper로 축소한다.

## 1.3.3.1 코드 구현 상세

이 세부 항목은 기존 `utils/loss.py`, `utils/loss_aux.py`, `models/yolo.py` 구조를 유지하면서 head/loss/assignment를 플래그 기반으로 추가하는 내부 설계를 고정한다.

### 대상 파일과 클래스

| 파일 | 클래스/함수 | 구현 방식 |
| --- | --- | --- |
| `models/yolo.py` | `Detect`, `IDetect`, `IAuxDetect` | 기존 output contract를 유지한다. Decoupled Head는 새 class로 추가하거나 기존 head 내부 branch로 추가하되 raw output shape는 동일해야 한다. |
| `models/yolo.py` | `Model.__init__` stride build | `IAuxDetect`는 현재 `self.forward(... )[:4]` 형태가 있으므로 향후 5-level 대응 시 `m.nl` 기준으로 slicing해야 한다. |
| `utils/loss.py` | `ComputeLoss`, `ComputeLossOTA`, `ComputeLossAuxOTA` | `--loss-box`, `--assign`, `--loss-cls`에 따라 CIoU/WIoU, SimOTA/TAL, BCE/VFL을 선택한다. |
| `utils/loss_aux.py` | OTA matching block | `matching_matrix`는 CPU로 강제하지 않는다. `cost.device` 또는 prediction device를 따른다. |
| `utils/wiou.py` | `WIoUState`, `wiou_loss()` | WIoU running mean과 dynamic focusing을 분리한다. dynamic weight 계산은 `.detach()`를 사용한다. |
| `utils/tal.py` | `TaskAlignedAssigner` | `topk`, `alpha`, `beta`를 hyp에서 읽고 positive count를 반환한다. |
| `utils/vfl.py` 또는 `utils/loss_components.py` | `varifocal_loss()` | TAL positive의 IoU-aware target을 사용한다. TAL 없이 VFL을 호출하면 실패 처리한다. |

### argparse 구현 규칙

`train.py`, `train_aux.py`에 동일하게 추가한다.

```python
parser.add_argument('--head', choices=['coupled', 'decoupled'], default='coupled')
parser.add_argument('--loss-box', choices=['ciou', 'wiou_v3'], default='ciou')
parser.add_argument('--assign', choices=['simota', 'tal'], default='simota')
parser.add_argument('--loss-cls', choices=['bce', 'vfl'], default='bce')
```

검증 제약:
- `--loss-cls vfl`이면 `--assign tal`이 아니면 parser 이후 validation에서 실패 처리한다.
- `--head decoupled`는 처음에는 실제 cfg가 사용하는 Detect 계열부터 지원한다.

### WIoU state 저장

checkpoint extra field에 아래를 저장한다.

```python
ckpt['loss_state'] = {
    'wiou': wiou_state.state_dict() if wiou_state else None,
    'assign': opt.assign,
    'loss_box': opt.loss_box,
    'loss_cls': opt.loss_cls,
}
```

resume 시 `loss_state['wiou']`가 있으면 복원하고, 없으면 새로 초기화한다. state 누락은 crash가 아니라 warning이어야 한다.

### loss_detail 확장

`loss_detail.csv`에는 최소 아래 컬럼을 추가한다.

```text
epoch,phase,box_loss,cls_loss,obj_loss,aux_loss,total_loss,
positive_count,assigner,loss_box,loss_cls,head
```

### 검증 순서

1. 기존 `coupled + ciou + simota + bce` 결과가 변하지 않는지 smoke
2. `--head decoupled` 단독
3. `--loss-box wiou_v3` 단독
4. `--assign tal --loss-cls vfl` 단독
5. 전체 누적

각 단계마다 `export_check.json`과 `profile.json`을 남긴다.

## 리포트 기반 정비 기준

- 문서 위치 기준: 본 코드레벨 개발 요구서는 `doc/PLAN/`에 둔다.
- 기준 리포트: `doc/REPORT/ai_perspective_yolov7_improvement_analysis_2026-05-22.md`
- Head, WIoU, TAL/VFL은 동시에 처음 켜지 않는다.
- 검증 run은 `Decoupled Head 단독 -> WIoU 단독 -> TAL+VFL 단독 -> 누적 적용` 순서로 남긴다.
- 각 run은 `loss_detail.csv`, positive count, `profile.json`, `export_check.json`을 비교 가능하게 저장한다.

- 기준 계획: `doc/PLAN/development_plan_v1.3.md`
- 대상 차수: `1.3.3 Core Model / Loss 분리 적용`
- 선행 조건: `1.3.2` Phase/DataLoader/logging 통과
- 목적: Head, box loss, assignment/cls loss를 독립 플래그로 구현하고 한 단계씩 켜서 원인을 분리한다.

## 1. 범위

포함:
- `--head coupled|decoupled`
- `--loss-box ciou|wiou_v3`
- `--assign simota|tal`
- `--loss-cls bce|vfl`
- CIoU/BCE/SimOTA fallback 유지
- WIoU state checkpoint/resume 저장
- Phase별 loss weight schedule 적용
- 클래스 불균형 대응 `cls_pw`, 소형 객체 IoU weight 기록
- AUX loss device 정합성 보정
- 각 하위 단계별 `stage_result.yaml`, `profile.json`, `export_check.json`

제외:
- CCTV augmentation, sampler
- W6 P2 Anchor, SCDown
- L AUX 성능형 옵션
- FCOS P2
- C++ 후처리, TensorRT runtime, 추론 서버

## 2. 하위 단계

| 단계 | 목적 | 실행 플래그 | 비교 기준 |
| --- | --- | --- | --- |
| `1.3.3-A1` | Head만 변경 | `--head decoupled --loss-box ciou --assign simota --loss-cls bce` | 1.3.2 |
| `1.3.3-A2` | Box loss만 변경 | `--head coupled --loss-box wiou_v3 --assign simota --loss-cls bce` | 1.3.2 |
| `1.3.3-A3` | TAL+VFL 변경 | `--head coupled --loss-box ciou --assign tal --loss-cls vfl` | 1.3.2 |
| `1.3.3-A` | 누적 적용 | `--head decoupled --loss-box wiou_v3 --assign tal --loss-cls vfl` | A1/A2/A3 통과 후 |

운영 기본값은 누적 적용이지만, 개발 검증은 A1, A2, A3 단독 결과를 먼저 남긴다.

## 3. 파일 단위 개발 요구사항

| 파일 | 구분 | 요구사항 |
| --- | --- | --- |
| `train.py` | 수정 | 신규 플래그를 파싱하고 loss/head 선택값을 `opt.yaml`, `stage_result.yaml`에 저장한다. |
| `models/yolo.py` | 수정 | `DecoupledDetect` 또는 동등 모듈을 추가한다. raw output shape은 기존 Detect와 동일하게 유지한다. |
| `models/common.py` | 확인/최소 수정 | decoupled head에 필요한 공통 block만 추가한다. Backbone 구조는 변경하지 않는다. |
| `utils/loss.py` | 수정 | CIoU/BCE/SimOTA와 WIoU/TAL/VFL을 플래그 기반으로 선택한다. |
| `utils/loss_aux.py` | 수정 | AUX 경로도 같은 옵션을 지원하고 matching tensor device mismatch를 제거한다. |
| `utils/loss_components.py` | 신규 | WIoU, VFL, 공통 IoU/target helper를 `loss.py`와 `loss_aux.py`가 공유한다. |
| `utils/wiou.py` | 신규 | WIoU v3 계산과 running state를 분리한다. dynamic weight 계산에는 `.detach()`를 적용한다. |
| `utils/tal.py` | 신규 | TAL assigner를 구현한다. `topk`, `alpha`, `beta`는 hyp에서 읽는다. |
| `data/hyp_phase*.yaml` | 수정 | `loss_box`, `loss_cls`, `assign`, `tal_topk`, `tal_alpha`, `tal_beta`, `wiou_momentum`, `cls_pw`, `small_iou_weight` key를 추가한다. |
| `tools/check_loss_smoke.py` | 신규 | dummy batch로 forward/loss/backward를 수행하고 NaN/Inf와 device mismatch를 검사한다. |

## 4. CLI 요구사항

```bash
python train.py --cfg cfg/training/yolov7.yaml --data data/custom_example.yaml --hyp data/hyp_phase1.yaml --epochs 1 --img 640 --batch 16 --aux auto --head decoupled --loss-box ciou --assign simota --loss-cls bce --name head_only_smoke
```

금지 조합:
- `--loss-cls vfl --assign simota`
- `--assign tal --loss-cls bce`는 실험 가능하지만 기본 경로가 아니다.
- `--loss-cls vfl`은 TAL positive와 IoU-aware target이 없으면 실패 처리한다.

## 5. 구현 계약

### 5.1 Decoupled Head

- 입력 feature level 수와 stride는 기존 Detect와 동일해야 한다.
- inference output은 `[batch, total_boxes, nc + 5]`를 유지한다.
- ONNX raw export에서 추가 tuple이나 dict를 반환하지 않는다.
- 기존 coupled head weight 로딩은 strict 실패가 아닌 partial load로 처리하고 누락 key를 로그에 남긴다.
- `Detect`, `IDetect`, `IAuxDetect` 계열의 training/inference/fuse 경로를 각각 고려한다.
- `parse_model()`에서 decoupled module을 인식하고 channel 계산, save list, bias initialization이 기존 Detect 계열과 같은 기준으로 동작해야 한다.
- deploy cfg와 training cfg의 마지막 head module 이름이 다르면 export 시 명확한 변환 또는 오류 메시지를 제공한다.

### 5.2 WIoU v3

- WIoU state는 `state_dict` 또는 checkpoint extra field에 저장한다.
- resume 시 running state가 복원되어야 한다.
- dynamic focusing weight는 gradient graph에서 분리한다.
- NaN/Inf 발생 시 해당 epoch를 실패 처리하고 fallback 로그를 남긴다.

### 5.3 TAL + VFL

- TAL positive 결정 후 VFL target을 만든다.
- anchor head의 obj BCE는 유지한다.
- VFL은 cls branch에만 적용한다.
- TAL matching tensor는 prediction tensor와 같은 device에 둔다.
- `matching_matrix`, `topk_idxs`, foreground mask는 CPU/CUDA가 섞이지 않아야 한다.
- target 수가 0개인 batch에서도 loss가 0 tensor로 정상 반환되어야 한다.

### 5.4 Loss Weight Schedule

PhaseState를 기준으로 아래 값을 적용한다.

| 구간 | 조건 | `lambda_main` | `lambda_free` | `lambda_aux` |
| --- | --- | --- | --- | --- |
| warmup | Phase 1 내부 첫 30 epoch | 1.0 | 0.1 | 0.25 |
| Phase 1 | warmup 이후 | 1.0 | 0.5 | 0.25 |
| Phase 2 | rect finetune | 1.0 | 0.5 | 0.0 |
| Phase 3 | close mosaic | 1.0 | 0.5 | 0.0 |

`lambda_aux=0.0`은 loss 비활성 의미다. 파라미터 freeze를 적용할 경우 별도 로그에 `requires_grad` 변경 여부를 남긴다.

### 5.5 Class Imbalance / Small Object Weight

- `cls_pw`는 기본 scalar로 처리한다.
- per-class positive weight를 지원할 경우 key는 `cls_pw_per_class`로 분리하고 shape `[nc]`를 검증한다.
- 희귀 클래스 weight는 data yaml의 class id와 label cache 통계 기준으로 계산한다.
- 소형 객체 IoU weight는 `iou_weight = 1.0 + (1.0 - normalize_area)` 형태를 기본으로 하되, max clamp 값을 hyp에 둔다.
- weight 적용 전후 loss scale을 `loss_detail.csv`에 기록한다.

## 6. 산출물

각 하위 단계 run directory에 아래 파일을 남긴다.
- `stage_result.yaml`
- `loss_detail.csv`
- `profile.json`
- `export_check.json`
- `fallback.log`

`stage_result.yaml` 필수 필드:
- `stage`
- `head`
- `loss_box`
- `assign`
- `loss_cls`
- `baseline_profile`
- `current_profile`
- `gflops_delta_percent`
- `best_map_50_95`
- `nan_inf_detected`
- `fallback_used`
- `lambda_aux`
- `lambda_free`
- `cls_pw_mode`
- `small_iou_weight_enabled`
- `export_passed`
- `status`

## 7. 통과 기준

1. A1, A2, A3 단독 smoke가 각각 완료된다.
2. 누적 적용 A smoke가 완료된다.
3. 모든 단계에서 NaN/Inf loss가 발생하지 않는다.
4. GFLOPs 증가율이 L/W6 각각 baseline 대비 10% 미만이다.
5. `tools/verify_export.py` 비교가 통과한다.
6. WIoU checkpoint resume 후 loss state가 유지된다.
7. TAL/VFL 적용 시 positive 수와 loss scale이 `loss_detail.csv`에 기록된다.
8. `Detect`/`IDetect`/`IAuxDetect` 계열 중 현재 cfg가 사용하는 head 타입에서 forward, loss, export가 모두 통과한다.
9. warmup/Phase 1/2/3의 `lambda_aux`, `lambda_free` 값이 epoch boundary test에서 확인된다.
10. `cls_pw`와 small object IoU weight가 켜진 경우 loss scale이 NaN/Inf 없이 기록된다.

## 8. 구현 순서

1. CLI 플래그와 hyp key 추가
2. `tools/check_loss_smoke.py` 작성
3. Decoupled Head 단독 구현과 A1 smoke/export 검증
4. WIoU 단독 구현과 state 저장/resume 검증
5. TAL assigner 구현
6. VFL 구현 및 TAL 연동
7. Detect/IDetect/IAuxDetect head 타입별 smoke
8. A1, A2, A3 단독 smoke
9. 누적 A smoke, profile, export 검증

## 9. 리스크 및 주의사항

- 여러 기법을 동시에 넣고 첫 검증을 시작하지 않는다.
- L 모델은 P2, AUX, Neck 변경 없이 Head/Loss만 다룬다.
- W6 AUX 경로의 CPU/CUDA tensor 혼합은 P1 blocker로 처리한다.
- fallback은 조용히 동작하지 않고 반드시 로그와 `stage_result.yaml`에 남긴다.

## 10. 개발 착수 분리 기준

하위 단계와 PR 단위는 일치해야 한다. 기능 구현 순서가 다르더라도 검증 run은 반드시 A1, A2, A3 단독으로 남긴다.

| PR | 범위 | 완료 기준 |
| --- | --- | --- |
| `1.3.3-P1` | loss option parser, hyp key, `tools/check_loss_smoke.py` | 기존 CIoU/BCE/SimOTA 결과가 변하지 않음 |
| `1.3.3-P2` | Decoupled Head | A1 smoke, raw ONNX export 통과 |
| `1.3.3-P3` | WIoU v3와 checkpoint/resume state | A2 smoke, resume smoke 통과 |
| `1.3.3-P4` | TAL assigner와 VFL | A3 smoke, positive count/loss scale 기록 |
| `1.3.3-P5` | 누적 A 적용 및 fallback | A smoke, profile/export/fallback log 검증 |

`P2`의 Decoupled Head는 `Detect`, `IDetect`, `IAuxDetect` 중 실제 cfg가 쓰는 module부터 구현한다. 사용하지 않는 head 타입까지 한 PR에서 모두 완성하려고 하지 않는다.

## 11. 1.3.3 구현 반영 현황

### 11.1 코드 반영 파일

| 영역 | 파일 | 반영 내용 |
| --- | --- | --- |
| CLI/학습 연결 | `train.py`, `train_aux.py` | `--head`, `--loss-box`, `--assign`, `--loss-cls` 추가, `vfl + simota` 금지 검증, hyp/opt/checkpoint/stage result 저장 |
| Head | `models/yolo.py` | `DecoupledDetect`, `DecoupledAuxDetect` 추가. 기본 cfg 파일은 유지하고 `--head decoupled` 실행 시 메모리상의 마지막 head만 치환 |
| WIoU | `utils/wiou.py`, `utils/loss_components.py` | WIoU running mean state, checkpoint 저장/복원 helper 추가 |
| TAL/VFL | `utils/tal.py`, `utils/loss_components.py`, `utils/loss.py`, `utils/loss_aux.py` | TAL top-k assignment cost와 VFL classification loss 추가. 기존 SimOTA/BCE는 기본값 유지 |
| AUX device | `utils/loss_aux.py` | `matching_matrix` CPU 강제 생성 제거 |
| 로그 | `utils/train_logger.py` | `positive_count`, `assigner`, `loss_box`, `loss_cls`, `head` 컬럼 기록 |
| 검증 도구 | `tools/check_loss_smoke.py` | dummy batch forward/loss/backward smoke 추가 |

재검토 반영:
- WIoU running mean은 `torch.is_grad_enabled()`일 때만 갱신한다. validation loss 계산은 state를 바꾸지 않아야 한다.
- Decoupled checkpoint yaml에는 `head_base_module`을 저장한다. 이후 `--head coupled` 요청 시 원래 `IDetect`, `Detect`, `IAuxDetect`로 복원 가능해야 한다.
- `tools/check_loss_smoke.py --empty-targets`로 target 없는 batch를 검증한다.

### 11.2 현재 통과한 smoke

```bash
python tools/check_loss_smoke.py --cfg cfg/training/yolov7.yaml --device cpu --img 64 --batch 1 --head coupled --loss-box ciou --assign simota --loss-cls bce
python tools/check_loss_smoke.py --cfg cfg/training/yolov7.yaml --device cpu --img 64 --batch 1 --head coupled --loss-box wiou_v3 --assign simota --loss-cls bce
python tools/check_loss_smoke.py --cfg cfg/training/yolov7.yaml --device cpu --img 64 --batch 1 --head coupled --loss-box ciou --assign tal --loss-cls vfl
python tools/check_loss_smoke.py --cfg cfg/training/yolov7.yaml --device cpu --img 64 --batch 1 --head decoupled --loss-box wiou_v3 --assign tal --loss-cls vfl
python tools/check_loss_smoke.py --cfg cfg/training/yolov7-w6.yaml --device cpu --img 128 --batch 1 --head decoupled --loss-box wiou_v3 --assign tal --loss-cls vfl
python tools/check_loss_smoke.py --cfg cfg/training/yolov7.yaml --device cpu --img 64 --batch 1 --head decoupled --loss-box wiou_v3 --assign tal --loss-cls vfl --empty-targets
python tools/check_loss_smoke.py --cfg cfg/training/yolov7-w6.yaml --device cpu --img 128 --batch 1 --head decoupled --loss-box wiou_v3 --assign tal --loss-cls vfl --empty-targets
```

W6 smoke는 `img=64, batch=1`에서 BatchNorm 입력이 `1x1`이 되어 실패하므로 `img=128` 이상으로 확인한다. 이는 구현 오류가 아니라 CPU dummy smoke 크기 조건이다.

### 11.3 다음 학습 검증 순서

1. A1: `--head decoupled --loss-box ciou --assign simota --loss-cls bce`
2. A2: `--head coupled --loss-box wiou_v3 --assign simota --loss-cls bce`
3. A3: `--head coupled --loss-box ciou --assign tal --loss-cls vfl`
4. A: `--head decoupled --loss-box wiou_v3 --assign tal --loss-cls vfl`

각 run은 `loss_detail.csv`의 `positive_count`, `assigner`, `loss_box`, `loss_cls`, `head`를 먼저 비교한다. COCO128 1차 테스트에서는 mAP보다 NaN/Inf, positive count 급변, loss scale 폭주 여부를 우선 판정한다.
