# 1.3.5 Code-Level Development Requirements

## 공통 예외 사항 - 원 코드 유지 개발

- 본 차수의 새 기능은 기본값으로 비활성화한다. 플래그를 켜지 않으면 기존 YOLOv7 학습, 평가, export 동작이 유지되어야 한다.
- 기존 함수/클래스는 버그 수정, 호환성 보강, 공통 helper 호출 연결에 한해서만 직접 수정한다.
- 신규 기능은 가능한 `utils/*`, `models/*`의 새 helper/class/wrapper로 분리하고, 기존 entrypoint는 기존 CLI와 출력 경로를 유지한다.
- `train.py`, `train_aux.py`, `test.py`, `export.py`는 기존 옵션명을 삭제하지 않는다. alias를 추가할 때도 기존 `dest`와 결과 파일명을 바꾸지 않는다.
- `train_aux.py`는 즉시 삭제하거나 대체하지 않는다. 공통 helper를 먼저 만들고 AUX/W6 smoke 검증 후 얇은 wrapper로 축소한다.

## 1.3.5.1 코드 구현 상세

이 세부 항목은 현재 `cfg/training/yolov7-w6.yaml`, `models/yolo.py`, `models/common.py` 기준으로 W6 P2 Anchor와 SCDown을 구현할 때 필요한 내부 설계를 고정한다.

### 대상 파일과 구현 위치

| 파일 | 위치 | 구현 방식 |
| --- | --- | --- |
| `models/common.py` | 신규 `SCDown` class | `Conv` 기반 downsampling block으로 구현한다. 표준 PyTorch 연산만 사용한다. |
| `models/yolo.py` | `Model.__init__` stride build | `IAuxDetect` stride 계산은 `self.forward(... )[:m.nl]` 또는 main output 수 기준으로 동작해야 한다. 현재처럼 `[:4]` 하드코딩을 유지하면 P2 포함 5-level에서 실패한다. |
| `models/yolo.py` | `IAuxDetect.__init__` | `self.nl = len(anchors)` 기준으로 `ch[:self.nl]`, `ch[self.nl:]`를 main/aux로 분리한다. P2 적용 시 main 5개 + aux 5개 입력을 받는다. |
| `cfg/training/yolov7-w6-scdown.yaml` | 신규 cfg | SCDown only 실험. P2 anchor는 추가하지 않는다. |
| `cfg/training/yolov7-w6-p2.yaml` | 신규 cfg | P2 Anchor only 실험. SCDown은 적용하지 않는다. |
| `cfg/training/yolov7-w6-p2-scdown.yaml` | 신규 cfg | P2 + SCDown 누적 실험. |
| `tools/estimate_nms_cost.py` | 신규 CLI | output box 수, per-level grid, Python NMS ms를 추정한다. |

### SCDown class 설계

`SCDown`은 최소 형태로 시작한다.

```python
class SCDown(nn.Module):
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k, s)

    def forward(self, x):
        return self.cv2(self.cv1(x))
```

구현 시 `parse_model()`의 module list에 `SCDown`이 포함되어야 한다. `Conv`와 같은 방식으로 `c1`, `c2` channel 계산이 가능해야 한다.

### P2 Anchor cfg 규칙

W6 P2는 5-level detection이다.

```yaml
anchors:
  - [...]  # P2/4
  - [...]  # P3/8
  - [...]  # P4/16
  - [...]  # P5/32
  - [...]  # P6/64
```

마지막 `IAuxDetect` 입력은 main 5개 + aux 5개 총 10개 feature index를 받아야 한다. 기존 W6의 P3/P4/P5/P6 4-level 구조를 보존하고 P2만 추가한다.

### argparse 구현 규칙

```python
parser.add_argument('--p2-head', choices=['none', 'anchor'], default='none')
parser.add_argument('--neck-mod', choices=['none', 'scdown'], default='none')
```

validation 규칙:
- L 모델 cfg에서 `--p2-head anchor` 또는 `--neck-mod scdown`이면 실패 처리한다.
- `--p2-head anchor`를 켰는데 cfg 파일명이 `p2` 계열이 아니면 warning 또는 실패 처리한다.
- `--neck-mod scdown`을 켰는데 cfg가 scdown 계열이 아니면 warning 또는 실패 처리한다.

### 검증 명령

```bash
python models/yolo.py --cfg cfg/training/yolov7-w6-scdown.yaml
python models/yolo.py --cfg cfg/training/yolov7-w6-p2.yaml
python models/yolo.py --cfg cfg/training/yolov7-w6-p2-scdown.yaml
python tools/estimate_nms_cost.py --cfg cfg/training/yolov7-w6-p2.yaml --img 1280 736
```

필수 확인:
- stride가 `[4, 8, 16, 32, 64]`
- anchors shape가 `[5, na, 2]`
- IAuxDetect main/aux feature 수가 각각 5
- deploy cfg model build 통과
- ONNX/ONNX Runtime 비교와 C++/TensorRT runtime 코드는 본 차수 필수 검증에서 제외
- GFLOPs 증가율 10% 미만

## 리포트 기반 정비 기준

- 문서 위치 기준: 본 코드레벨 개발 요구서는 `doc/PLAN/`에 둔다.
- 기준 리포트: `doc/REPORT/ai_perspective_yolov7_improvement_analysis_2026-05-22.md`
- 본 차수는 YOLOv7-W6 전용이다. YOLOv7-L 구조는 변경하지 않는다.
- SCDown only, P2 Anchor only, P2 Anchor + SCDown을 분리해 검증한다.
- P2 적용 후 output box 수, memory, Python NMS 비용을 반드시 기록한다.

- 기준 계획: `doc/PLAN/development_plan_v1.3.md`
- 대상 차수: `1.3.5 W6 구조 확장`
- 선행 조건: `1.3.4` augmentation/sampler 검증 통과
- 목적: YOLOv7-W6에만 P2 Anchor Head와 SCDown을 적용해 소형 객체 성능을 검증한다.

## 1. 범위

포함:
- W6 P2 Anchor Head
- W6 SCDown
- output box 수 계산
- Python NMS 비용 추정
- memory 사용량 측정
- deploy cfg model build 검증

제외:
- YOLOv7-L 구조 변경
- FCOS P2
- PSA, GELAN
- ONNX Runtime 비교
- C++ 후처리, TensorRT runtime, 추론 서버

## 2. 파일 단위 개발 요구사항

| 파일 | 구분 | 요구사항 |
| --- | --- | --- |
| `models/common.py` | 수정 | `SCDown` block을 추가한다. Conv/BN/activation 조합은 표준 PyTorch 연산만 사용한다. |
| `models/yolo.py` | 수정 | W6 Detect/IAuxDetect가 P2 포함 5개 level을 처리한다. stride `4, 8, 16, 32, 64`를 지원한다. |
| `cfg/training/yolov7-w6-scdown.yaml` | 신규 | SCDown 단독 실험용 W6 training cfg. |
| `cfg/training/yolov7-w6-p2.yaml` | 신규 | P2 Anchor 단독 실험용 W6 training cfg. |
| `cfg/training/yolov7-w6-p2-scdown.yaml` | 신규 | P2 Anchor + SCDown 누적 실험용 W6 training cfg. |
| `cfg/deploy/yolov7-w6-scdown.yaml` | 신규 | SCDown 단독 deploy cfg. |
| `cfg/deploy/yolov7-w6-p2.yaml` | 신규 | P2 Anchor 단독 deploy cfg. |
| `cfg/deploy/yolov7-w6-p2-scdown.yaml` | 신규 | P2 Anchor + SCDown deploy cfg. |
| `utils/autoanchor.py` | 확인/최소 수정 | 5개 detection layer anchor check를 지원한다. |
| `tools/profile_model.py` | 수정 | `total_boxes`, `output_shapes`, `activation_memory_mb`를 기록한다. |
| `tools/estimate_nms_cost.py` | 신규 | PyTorch/Python 기준 NMS 입력 box 수와 소요 시간을 추정한다. |
| `tools/check_output_contract.py` | 신규 | Detect/IAuxDetect stride, anchor shape, output box 수 계약을 JSON으로 기록한다. |
| `train.py` | 수정 | `--p2-head none|anchor`, `--neck-mod none|scdown` 플래그를 저장하고 cfg 선택을 검증한다. |
| `train_aux.py` | 수정 | `train.py`와 동일한 구조 플래그를 저장하고 W6/P2/SCDown cfg 선택을 검증한다. |

## 2.1 하위 단계

| 단계 | 목적 | 실행 플래그 |
| --- | --- | --- |
| `1.3.5-D1` | SCDown 단독 | `--p2-head none --neck-mod scdown` |
| `1.3.5-D2` | P2 Anchor 단독 | `--p2-head anchor --neck-mod none` |
| `1.3.5-D3` | P2 Anchor + SCDown 누적 | `--p2-head anchor --neck-mod scdown` |

D1/D2를 먼저 확인한 뒤 D3를 full W6 후보로 기록한다.

## 3. CLI 요구사항

W6 P2 + SCDown smoke:

```bash
python train.py --cfg cfg/training/yolov7-w6-p2-scdown.yaml --data data/custom_example.yaml --hyp data/hyp_phase1.yaml --epochs 1 --img 1280 --batch 4 --head decoupled --loss-box wiou_v3 --assign tal --loss-cls vfl --aux on --p2-head anchor --neck-mod scdown --name w6_p2_scdown_smoke
```

Full phase run에서는 1.3.2 정책에 따라 `--rect-size-w6 1280 736 --grad-accumulate 4`를 함께 사용한다.

Profile:

```bash
python tools/profile_model.py --weights runs/train/w6_p2_scdown_smoke/weights/best.pt --cfg cfg/training/yolov7-w6-p2-scdown.yaml --img 1280 736 --output runs/train/w6_p2_scdown_smoke/profile.json
```

NMS cost estimate:

```bash
python tools/estimate_nms_cost.py --weights runs/train/w6_p2_scdown_smoke/weights/best.pt --img 1280 736 --output runs/train/w6_p2_scdown_smoke/nms_cost.json
```

Output contract:

```bash
python tools/check_output_contract.py --cfg cfg/training/yolov7-w6-p2-scdown.yaml --img 640 640 --expect-levels 5 --expect-strides 4 8 16 32 64 --output runs/train/w6_p2_scdown_smoke/output_contract.json
```

## 4. 구조 계약

### 4.1 P2 Anchor Head

- W6 전용이다. L 모델에서 `--p2-head anchor`를 주면 에러를 낸다.
- 기존 W6의 P3/P4/P5/P6를 유지하고 P2를 추가한다. 기본 output level은 P2/P3/P4/P5/P6 5개다.
- P2 stride는 4로 기록되어야 한다.
- P2 path는 upsample + Conv 2회 구조를 기본으로 한다.
- raw output은 기존 anchor detect 형식 `[batch, total_boxes, nc + 5]`를 유지한다.
- anchor 수는 level별 동일 개수 유지가 기본이다.
- anchor yaml은 5개 level을 가져야 하며 `anchors` 길이와 head input level 수가 일치하지 않으면 모델 생성 단계에서 실패한다.
- 현재 `IAuxDetect` stride build에 4-level 하드코딩이 있으면 `m.nl` 기반으로 수정한다. P2 적용 후 main 5개 + aux 5개 feature를 처리해야 한다.

### 4.2 SCDown

- W6 neck/downsample 경로에만 적용한다.
- Backbone 구조는 변경하지 않는다.
- `--neck-mod none`으로 기존 W6 구조를 즉시 복원할 수 있어야 한다.
- deploy cfg/model build가 실패하는 custom op를 사용하지 않는다.
- D1 단독 실험에서 SCDown의 GFLOPs/memory/deploy cfg 영향을 먼저 기록한다.

## 5. 산출물

필수 산출물:
- `profile.json`
- `nms_cost.json`
- `output_contract.json`
- `stage_result.yaml`
- `results.csv`

`profile.json` 추가 필드:
- `baseline_gflops`
- `current_gflops`
- `gflops_delta_percent`
- `total_boxes`
- `output_shapes`
- `activation_memory_mb`
- `max_cuda_memory_mb`
- `small_AP`
- `small_recall`
- `rare_recall`

`nms_cost.json` 필수 필드:
- `input_shape`
- `total_boxes`
- `conf_thres`
- `iou_thres`
- `mean_nms_ms`
- `p95_nms_ms`
- `device`
- `status`

`stage_result.yaml` 추가 필드:
- `sub_stage`
- `p2_head`
- `neck_mod`
- `baseline_run`
- `current_run`
- `output_contract_json`
- `nms_cost_json`
- `memory_peak_mb`
- `decision`

## 6. 통과 기준

1. W6 P2/SCDown smoke 학습이 완료된다.
2. small AP 또는 small recall이 1.3.4 W6 기준보다 개선된다.
3. GFLOPs 증가율이 W6 baseline 대비 10% 미만이다.
4. `total_boxes`와 Python NMS 비용이 report에 기록된다.
5. max CUDA memory가 학습 서버 허용 범위 안에 있다.
6. deploy cfg model build가 통과한다.
7. L 모델 구조는 변경되지 않는다.
8. D1, D2, D3 결과가 분리 저장되어 P2와 SCDown 효과를 구분할 수 있다.

## 7. 구현 순서

1. `SCDown` block 추가
2. W6 P2 cfg 작성
3. `models/yolo.py` 5-level Detect/IAuxDetect 검증
4. autoanchor 5-level 확인
5. profile output 확장
6. NMS cost estimate 도구 작성
7. D1 SCDown 단독 smoke/profile/deploy cfg build 검증
8. D2 P2 Anchor 단독 smoke/profile/deploy cfg build 검증
9. D3 P2+SCDown 누적 smoke/profile/deploy cfg build 검증
10. full W6 실험 후 `stage_result.yaml` 저장

## 8. 리스크 및 주의사항

- P2 추가는 GFLOPs보다 output box 수 증가가 더 큰 리스크다.
- P2 Anchor가 부족할 때만 FCOS P2를 1.3.6에서 검토한다.
- C++ NMS나 runtime deploy 코드는 본 차수에서 작성하지 않는다.
- L 속도형 모델의 구조 경량 원칙은 깨지 않는다.

## 9. 개발 착수 분리 기준

W6 구조 변경은 cfg route/channel 오류가 자주 나는 영역이므로 cfg를 실험별로 분리한다.

| PR | 범위 | 완료 기준 |
| --- | --- | --- |
| `1.3.5-P1` | `SCDown` block + SCDown 단독 cfg | D1 model build/profile 통과 |
| `1.3.5-P2` | P2 Anchor cfg + 5-level Detect/IAuxDetect | D2 model build/profile/output contract 통과 |
| `1.3.5-P3` | P2 + SCDown 누적 cfg | D3 model build/profile/output contract 통과 |
| `1.3.5-P4` | `tools/estimate_nms_cost.py`, output/memory report | `nms_cost.json`, `profile.json`, `output_contract.json` 확장 필드 생성 |

`cfg/training/yolov7-w6-p2.yaml` 하나로 D1/D2/D3를 모두 처리하지 않는다. 각 실험의 route diff를 작게 유지해야 원인 추적이 가능하다.

## 10. 구현 반영 상태 (2026-05-22)

구현 완료:
- `models/common.py`: `SCDown` 추가.
- `models/yolo.py`: `base:` thin YAML 확장 로더, W6 P2 anchor 주입, SCDown 주입, 5-level `IAuxDetect`/`DecoupledAuxDetect` 입력 검증, repo-root 기준 cfg resolve 적용.
- `cfg/training/`, `cfg/deploy/`: `yolov7-w6-scdown.yaml`, `yolov7-w6-p2.yaml`, `yolov7-w6-p2-scdown.yaml` 추가.
- `train.py`, `train_aux.py`: `--p2-head`, `--neck-mod` 추가 및 cfg/flag 검증.
- `tools/profile_model.py`: GFLOPs delta, per-level box 수, output shape, activation memory, stride/anchor 계약 필드 기록.
- `tools/estimate_nms_cost.py`: Python NMS 입력 box 수와 평균/p95 ms 기록.
- `tools/check_output_contract.py`: P2 5-level stride, anchor, output box 계약 검증 JSON 생성.

검증 결과:
- `python -m py_compile models/yolo.py models/common.py utils/model_options.py tools/profile_model.py tools/estimate_nms_cost.py tools/check_output_contract.py train.py train_aux.py` 통과.
- 기본 W6 training cfg: `105.5 GFLOPs`.
- P2 단독 training cfg: `108.1 GFLOPs`, 기준 대비 약 `+2.46%`.
- P2+SCDown training cfg: `110.4 GFLOPs`, 기준 대비 약 `+4.66%`.
- P2 deploy cfg: model build 통과, `stride=[4,8,16,32,64]`, `anchors_shape=[5,3,2]`.
- `runs/tmp_135_profile/p2_scdown_output_contract.json`: `status=pass`, `actual_total_boxes=102300`.
- `runs/tmp_135_profile/p2_scdown_profile.json`: `gflops_delta_percent=4.663865023696678`, `activation_memory_mb=66.34140014648438`.
- `runs/tmp_135_profile/p2_scdown_nms_cost.json`: `1280x768`, `total_boxes=245520`, CPU `candidate_ratio=0.001`, `mean_nms_ms=2.90165`.

남은 검증:
- COCO128 빠른 학습 smoke는 학습 서버에서 1.8 실행 계획에 맞춰 수행한다.
- 실제 CCTV 데이터 full W6 학습 전에는 D1, D2, D3를 분리 실행해 small AP/recall, rare recall, memory peak를 비교한다.
- `1280x736` 입력은 현재 W6 최대 stride 64 기준 검증 경로에서 `1280x768`로 보정될 수 있으므로, 결과 리포트에는 실제 입력 shape를 함께 기록한다.
