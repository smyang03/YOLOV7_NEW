# 1.4.1 Anchor-Free Code-Level Development Requirements

- 기준 브랜치: `anchor-free`
- 작성 기준일: 2026-05-23
- 선행 문서: `doc/PLAN/development_requirements_1.3.6_optional_experiments.md`
- 핵심 원칙: 기존 YOLOv7 anchor 기반 코드는 유지하고, anchor-free는 명시 플래그와 전용 cfg에서만 동작시킨다.

## 1.4.1 목적

1.3.6에서 FCOS P2는 Python raw/decode 검증까지만 허용된 decode-only 후보였다. 1.4.1은 이 후보를 학습 가능한 anchor-free 실험으로 승격하기 위한 코드레벨 개발 요구서다.

Anchor-free 적용은 헤드만 바꾸는 작업이 아니다. 현재 `Detect`, `IDetect`, `IAuxDetect`는 anchor 수(`na`), anchor grid, anchor matching loss, YOLO NMS 입력 계약에 묶여 있다. 따라서 head 출력, target assign, loss, validation decode, export/profile/report가 함께 바뀌어야 한다.

## 1.4.1 적용 범위

### 포함

- YOLOv7-L anchor-free 실험 경로
- YOLOv7-W6 P2 anchor-free 실험 경로
- W6 full anchor-free 후보 경로
- Python 학습, 평가, raw ONNX export 검증
- stage runner/report에 anchor-free metric 기록

### 제외

- C++ 추론 코드
- TensorRT plugin/runtime postprocess
- main 브랜치와 `anchor-free` 브랜치 merge/rebase/cherry-pick
- 기존 anchor 기반 baseline 제거
- backbone 구조 교체

## 1.4.1 개발 방향

1. `anchor` 기본값을 유지한다.
2. `fcos`는 새 head class로 추가한다.
3. 기존 `Detect` 계열을 직접 anchor-free로 변형하지 않는다.
4. L과 W6를 동시에 전면 교체하지 않는다.
5. W6는 먼저 P2 FCOS hybrid로 검증한다.
6. full anchor-free는 L P3/P4/P5, W6 P2/P3/P4/P5/P6 순서로 확장한다.

## 1.4.1 단계

| 단계 | 목적 | 모델 | 내용 |
| --- | --- | --- | --- |
| `1.4.1-P0` | 현재 anchor 기준 고정 | L/W6 | baseline output/loss/export/report contract 기록 |
| `1.4.1-P1` | FCOS head smoke | L | `FCOSDetect` build, synthetic decode, no-train smoke |
| `1.4.1-P2` | W6 P2 hybrid | W6 | anchor head 유지 + P2 FCOS 보조 branch 학습 |
| `1.4.1-P3` | L full FCOS | L | P3/P4/P5 anchor-free 학습 |
| `1.4.1-P4` | W6 full FCOS | W6 | P2/P3/P4/P5/P6 anchor-free 학습 |
| `1.4.1-P5` | stage/report 통합 | L/W6 | runner stage, metric, export/profile 비교 |

## 1.4.1.1 코드 구현 상세

### 현재 코드 상태

| 파일 | 현재 상태 | anchor-free 적용 시 문제 |
| --- | --- | --- |
| `models/yolo.py` | `Detect`, `IDetect`, `IAuxDetect`, `DecoupledDetect`가 anchor 기반 | `na`, `anchors`, `anchor_grid`, `check_anchor_order()` 의존 |
| `utils/fcos.py` | `decode_fcos_raw()`, `fcos_contract()`만 존재 | 학습 target assign/loss 없음 |
| `utils/model_options.py` | `--p2-head fcos`를 decode-only로 차단 | 학습 가능 cfg/flag validator 필요 |
| `utils/loss.py` | `ComputeLoss`, `ComputeLossOTA`가 anchor matching 사용 | FCOS point assignment/ltrb/centerness loss 필요 |
| `utils/loss_aux.py` | AUX OTA도 anchor 기반 | AUX anchor-free 별도 설계 필요 |
| `train.py` | `--head coupled|decoupled`, `--p2-head none|anchor|fcos` | `--det-head`, FCOS loss 선택 필요 |
| `train_aux.py` | AUX loss 고정 | FCOS AUX는 1차에서 차단 또는 전용 loss 필요 |
| `test.py` | `model()` 출력이 YOLO `[xywh,obj,cls...]`라고 가정 | FCOS raw를 `[xyxy,conf,cls]`로 decode 후 NMS 필요 |
| `export.py` | YOLO Detect output 중심 | FCOS raw output 이름/shape 계약 필요 |
| `tools/run_training_sequence.py` | stage 00~13만 있음 | anchor-free stage 추가 필요 |

### 신규 CLI

`train.py`, `train_aux.py`, `test.py`, `export.py`, `tools/profile_model.py`에 같은 의미의 옵션을 둔다.

```python
parser.add_argument('--det-head', choices=['anchor', 'fcos', 'hybrid'], default='anchor')
parser.add_argument('--anchor-free-levels',
                    choices=['p2', 'p3p4p5', 'p2p3p4p5p6'],
                    default='p3p4p5')
parser.add_argument('--lambda-free', type=float, default=1.0)
parser.add_argument('--fcos-center-radius', type=float, default=1.5)
parser.add_argument('--fcos-score-mode',
                    choices=['sqrt_cls_centerness', 'mul_cls_centerness'],
                    default='sqrt_cls_centerness')
parser.add_argument('--fcos-loss-box', choices=['giou', 'ciou'], default='giou')
```

호환 규칙:

- `--det-head anchor`: 기존 경로 그대로 사용
- `--det-head fcos`: anchor head를 FCOS head로 교체
- `--det-head hybrid`: anchor output과 FCOS output을 동시에 학습/평가
- 기존 `--p2-head fcos`는 deprecated alias로 유지하되 내부적으로 `--det-head hybrid --anchor-free-levels p2`로 해석한다.

### `models/yolo.py`

신규 class:

```python
class FCOSDetect(nn.Module):
    stride = None
    export = False

    def __init__(self, nc=80, ch=(), levels=None):
        self.nc = nc
        self.no = nc + 5
        self.nl = len(ch)
        self.grid = [torch.zeros(1)] * self.nl
        self.m = nn.ModuleList(nn.Conv2d(c, self.no, 1) for c in ch)
```

forward 계약:

- training: `List[Tensor]`, 각 tensor shape는 `[B, nc+5, H, W]`
- inference: `(decoded, raw)` 또는 export 모드에서는 raw list
- raw channel 순서: `[l, t, r, b, centerness, cls0..clsN]`
- box 거리값은 학습 loss에서 positive로 유도하고 inference decode에서는 `relu()` 또는 `softplus()`를 적용한다.

수정 위치:

- `parse_model()`에 `FCOSDetect`를 detect 계열로 등록한다.
- `Model.__init__` stride build에서 `FCOSDetect`는 `check_anchor_order()`와 `m.anchors /= ...`를 호출하지 않는다.
- `_initialize_biases()`와 별도로 `_initialize_fcos_biases()`를 추가한다.
- `forward_once(profile=True)`의 detect-like 판단에 `FCOSDetect`를 포함한다.
- `_apply_head_override()`는 anchor/decoupled만 담당하고, FCOS 교체는 별도 `_apply_det_head_override(det_head, levels)`로 분리한다.

주의:

- `Detect`, `IDetect`, `IAuxDetect`의 output shape는 절대 변경하지 않는다.
- anchor-free cfg에서만 마지막 head를 `FCOSDetect`로 바꾼다.
- W6 P2 hybrid는 기존 `IAuxDetect`를 바로 제거하지 않는다.

### `cfg/experiments`

신규 cfg 후보:

| 파일 | 목적 |
| --- | --- |
| `cfg/experiments/yolov7-l-fcos.yaml` | L P3/P4/P5 full FCOS |
| `cfg/experiments/yolov7-w6-p2-fcos-hybrid.yaml` | W6 anchor head + P2 FCOS branch |
| `cfg/experiments/yolov7-w6-fcos.yaml` | W6 P2/P3/P4/P5/P6 full FCOS |

YAML metadata schema:

```yaml
det_head: fcos
anchor_free_levels: p3p4p5
fcos:
  center_radius: 1.5
  score_mode: sqrt_cls_centerness
  loss_box: giou
  lambda_free: 1.0
```

검증 규칙:

- `det_head=fcos`이면 마지막 head는 `FCOSDetect`여야 한다.
- `det_head=hybrid`이면 anchor head와 FCOS branch가 모두 존재해야 한다.
- L은 `p3p4p5`만 1차 허용한다.
- W6는 `p2` 또는 `p2p3p4p5p6`만 허용한다.
- decode-only cfg인 `yolov7-w6-fcos-p2-decode.yaml`은 학습에 계속 차단한다.

### `utils/model_options.py`

추가 함수:

```python
def validate_anchor_free_options(opt, parser=None):
    ...
```

검증 항목:

- `--det-head fcos`와 `--head decoupled` 동시 사용 금지
- `--det-head hybrid`는 W6 P2부터 허용
- `--anchor-free-levels p2`는 W6 cfg에서만 허용
- `--lambda-free`는 `0 < value <= 2.0`
- `--fcos-center-radius`는 `0.5 <= value <= 3.0`
- `--p2-head fcos` alias 사용 시 warning 출력
- `fcos_decode_only: true` cfg는 train/test/export에서 차단

### `utils/fcos.py`

현재 decode helper를 보존하고 학습 helper를 추가한다.

신규 함수:

```python
def fcos_targets(targets, feature_shapes, strides, img_size, num_classes,
                 center_radius=1.5, level_ranges=None):
    ...

def compute_centerness_targets(ltrb_targets):
    ...

def fcos_losses(raw_outputs, targets, strides, img_size, hyp, opt):
    ...

def decode_fcos_outputs(raw_outputs, strides, img_size, conf_thres, topk, score_mode):
    ...
```

target assign 요구:

- 입력 target은 기존 YOLO 형식 `[image, class, x, y, w, h]` normalized 기준을 사용한다.
- 각 feature point center가 GT box 내부에 있어야 positive가 된다.
- center sampling radius를 적용한다.
- level별 object size range를 둔다.
- 한 point가 여러 GT에 매칭되면 면적이 가장 작은 GT를 선택한다.
- positive count가 0이면 loss는 0 tensor로 안전하게 반환한다.

loss 구성:

- box loss: GIoU 또는 CIoU
- cls loss: BCE 또는 Focal BCE
- centerness loss: BCE
- total: `box * hyp['box'] + cls * hyp['cls'] + ctr * hyp.get('ctr', 1.0)`

디버그 통계:

- `fcos_positive_count`
- `fcos_positive_per_level`
- `fcos_ctr_mean`
- `fcos_loss_box`
- `fcos_loss_cls`
- `fcos_loss_ctr`

### `utils/loss.py`

신규 class:

```python
class ComputeLossFCOS:
    def __init__(self, model, hyp=None, opt=None):
        ...

    def __call__(self, predictions, targets, imgs=None):
        ...
```

hybrid용 class:

```python
class ComputeLossHybrid:
    def __init__(self, anchor_loss, fcos_loss, lambda_free=1.0):
        ...

    def __call__(self, predictions, targets, imgs=None):
        ...
```

계약:

- anchor loss의 `loss_items` 3개 계약은 유지한다.
- FCOS loss는 내부 상세값을 logger/report로 별도 기록한다.
- hybrid는 기존 anchor loss scale이 깨지지 않도록 `loss = anchor_loss + lambda_free * fcos_loss`로 합산한다.
- `ComputeLossOTA`는 기본 경로에서 수정하지 않는다.

### `utils/loss_aux.py`

1차 정책:

- AUX full anchor-free는 1.4.1-P2까지 차단한다.
- `train_aux.py --det-head fcos`는 명시 오류를 낸다.
- `train_aux.py --det-head hybrid --anchor-free-levels p2`만 W6 P2 실험에서 허용한다.

후속 정책:

- AUX main/aux 양쪽에 FCOS branch를 붙일지, main만 붙일지는 P2 hybrid 결과 보고 결정한다.

### `train.py`

수정 항목:

- argparse에 신규 anchor-free 옵션 추가
- `ensure_structure_option_defaults()` 이후 `validate_anchor_free_options()` 호출
- model 생성 시 `Model(..., head=opt.head, det_head=opt.det_head, anchor_free_levels=...)` 형태로 확장
- loss 선택 로직 수정

```python
if opt.det_head == 'anchor':
    compute_loss = ComputeLossOTA(model) if hyp.get('loss_ota', 1) == 1 else ComputeLoss(model)
elif opt.det_head == 'fcos':
    compute_loss = ComputeLossFCOS(model, hyp=hyp, opt=opt)
elif opt.det_head == 'hybrid':
    anchor_loss = ComputeLossOTA(model) if hyp.get('loss_ota', 1) == 1 else ComputeLoss(model)
    fcos_loss = ComputeLossFCOS(model, hyp=hyp, opt=opt)
    compute_loss = ComputeLossHybrid(anchor_loss, fcos_loss, opt.lambda_free)
```

logging:

- progress log에 `det_head`, `anchor_free_levels`, `fcos_positive_count` 추가
- `loss_detail.csv`에 FCOS 상세 loss column 추가
- `debug_trace.log`에는 stage별 FCOS assign summary 기록

### `train_aux.py`

수정 항목:

- argparse 옵션은 `train.py`와 동일하게 둔다.
- `det_head=fcos` full mode는 차단한다.
- `det_head=hybrid + p2`만 허용하고, 허용 전까지는 `parser.error()`로 명확히 실패한다.
- AUX loss가 FCOS raw를 anchor loss에 넘기지 않도록 output 분기 검증을 둔다.

### `test.py`

신규 helper:

```python
def normalize_model_predictions(out, train_out, model, img_size, opt):
    ...
```

계약:

- anchor: 기존 `non_max_suppression(out, ...)`
- fcos: `decode_fcos_outputs()` 결과를 image별 `[xyxy, conf, cls]` list로 반환
- hybrid: anchor prediction과 fcos decoded prediction을 같은 `[xyxy, conf, cls]`로 맞춘 뒤 concat하고 NMS 수행

주의:

- `non_max_suppression()` 입력 계약을 무리하게 바꾸지 않는다.
- FCOS decoded output은 이미 `xyxy`이므로 YOLO `xywh`처럼 처리하면 안 된다.
- source tag가 필요하면 report용 metadata로만 관리하고 metric 계산 tensor에는 넣지 않는다.

### `utils/general.py`

원칙:

- 기존 `non_max_suppression()` 계약을 유지한다.
- 필요한 경우 별도 helper를 추가한다.

신규 helper 후보:

```python
def nms_xyxy_predictions(predictions, conf_thres, iou_thres, max_det=300):
    ...

def merge_anchor_fcos_predictions(anchor_pred, fcos_pred):
    ...
```

### `export.py`

요구:

- `--det-head anchor`: 기존 export 그대로
- `--det-head fcos`: raw FCOS outputs export
- `--det-head hybrid`: anchor output + fcos raw output을 둘 다 export
- ONNX output name에 `fcos_p2`, `fcos_p3` 같은 level명을 포함한다.

주의:

- TensorRT 32배수 입력 제약은 유지한다.
- 1.4.1에서는 TensorRT plugin이나 C++ postprocess를 만들지 않는다.
- Python `tools/verify_export.py`에서 raw output shape와 decoded box 수만 검증한다.

### `tools/verify_export.py`

추가 검증:

- FCOS raw output shape: `[B, nc+5, H, W]`
- channel 수가 `nc + 5`인지 확인
- PyTorch raw와 ONNX raw의 max abs diff 기록
- decode 후 box count가 NaN 없이 생성되는지 확인

출력:

```json
{
  "det_head": "fcos",
  "anchor_free_levels": "p3p4p5",
  "fcos_raw_shapes": [[1, 7, 80, 80], [1, 7, 40, 40], [1, 7, 20, 20]],
  "onnx_max_abs_diff": 0.00001,
  "decoded_box_count": [120]
}
```

### `tools/profile_model.py`

추가 metric:

- `det_head`
- `anchor_free_levels`
- `fcos_candidate_count`
- `fcos_raw_shapes`
- `decoded_candidate_count`
- `python_decode_ms`
- `python_nms_ms`

기준:

- 기존 L/W6 baseline GFLOPs 대비 증가율은 10% 미만을 목표로 한다.
- FCOS는 anchor 수가 없어 head conv parameter는 줄 수 있지만 P2 candidate 수 때문에 decode/NMS 시간이 증가할 수 있다.

### `tools/run_training_sequence.py`

1.4.1 신규 stage 후보:

```python
StageSpec('14', 'w6_p2_fcos_hybrid', {'det-head': 'hybrid', 'anchor-free-levels': 'p2'}, families='w6')
StageSpec('15', 'l_fcos_full', {'det-head': 'fcos', 'anchor-free-levels': 'p3p4p5'}, families='l')
StageSpec('16', 'w6_fcos_full', {'det-head': 'fcos', 'anchor-free-levels': 'p2p3p4p5p6'}, families='w6')
```

운영 규칙:

- stage 14는 W6 P2 anchor 결과가 있어야 진입한다.
- stage 15는 L baseline, L decoupled/core cumulative와 비교한다.
- stage 16은 W6 P2 hybrid가 성능/속도 기준을 통과한 뒤에만 진입한다.
- 실패 시 anchor baseline weight를 fallback으로 둔다.

### report schema

`stage_result.yaml` 추가 필드:

```yaml
det_head: anchor|fcos|hybrid
anchor_free_levels: p2|p3p4p5|p2p3p4p5p6
fcos_positive_count:
fcos_positive_per_level:
fcos_loss_box:
fcos_loss_cls:
fcos_loss_ctr:
python_decode_ms:
candidate_count_delta_percent:
```

최종 report 비교 기준:

- person AP / head AP
- small AP
- recall
- FP per image
- FN per image
- GFLOPs delta
- Python inference/decode/NMS latency
- NaN/Inf 발생 여부
- positive count 안정성
- export raw diff

## 장점

- anchor 크기 튜닝 의존도가 줄어든다.
- CrowdHuman처럼 사람/머리 크기 변화가 큰 데이터에서 point 기반 positive assign을 실험할 수 있다.
- W6 P2와 결합하면 작은 head/object recall 개선 가능성이 있다.
- anchor 개수만큼 channel을 늘리지 않아 head parameter가 줄 수 있다.
- anchor 기반 baseline과 비교하면 성능 저하 원인이 head/loss/assign 중 어디인지 분리하기 쉽다.

## 단점

- head만 바꾸면 학습되지 않는다. loss, assign, test decode가 모두 필요하다.
- P2를 켜면 후보 point 수가 크게 늘어 decode/NMS 시간이 증가할 수 있다.
- centerness와 class score calibration이 anchor obj score와 다르다.
- hybrid는 anchor/FCOS 중복 detection이 증가할 수 있다.
- ONNX export raw output은 가능하지만 TensorRT runtime 후처리는 별도 작업이다.

## 리스크와 대응

| 리스크 | 영향 | 대응 |
| --- | --- | --- |
| positive count 0 | loss 0 또는 NaN | positive guard, per-level count log |
| 너무 많은 positive | FP 증가, 학습 불안정 | center radius, level range, topk 제한 |
| box distance 음수 | decode box 오류 | `relu` 또는 `softplus` 적용 |
| hybrid 중복 검출 | FP/image 증가 | NMS iou/score calibration, source별 분석 |
| mAP 급락 | anchor-free 설계 실패 | stage별 fallback weight 유지 |
| latency 증가 | 실사용 부적합 | `python_decode_ms`, `python_nms_ms` stage report 필수화 |
| export 불일치 | 배포 검증 실패 | raw ONNX diff 먼저 검증, decoded 검증은 Python에서 분리 |
| AUX 경로 충돌 | W6/AUX 학습 실패 | full AUX FCOS는 차단하고 P2 hybrid부터 제한 허용 |

## 검증 명령

문법 검증:

```bash
python -m py_compile models/yolo.py utils/fcos.py utils/loss.py utils/loss_aux.py utils/model_options.py train.py train_aux.py test.py export.py tools/profile_model.py tools/verify_export.py tools/run_training_sequence.py
```

decode smoke:

```bash
python tools/decode_fcos_outputs.py --allow-synthetic --img 128 128 --nc 2 --stride 4 --output runs/tmp_141_fcos_decode/fcos_decode_check.json --require-pass
```

L FCOS smoke:

```bash
python train.py --cfg cfg/experiments/yolov7-l-fcos.yaml --data data/coco128.yaml --epochs 1 --batch-size 2 --img-size 640 640 --det-head fcos --anchor-free-levels p3p4p5 --debug-log error --name smoke_141_l_fcos
```

W6 P2 hybrid smoke:

```bash
python train.py --cfg cfg/experiments/yolov7-w6-p2-fcos-hybrid.yaml --data data/coco128.yaml --epochs 1 --batch-size 2 --img-size 640 640 --det-head hybrid --anchor-free-levels p2 --debug-log error --name smoke_141_w6_p2_fcos
```

stage runner 후보:

```bash
python tools/run_training_sequence.py --plan doc/PLAN/training_execution_plan_v1.8.md --data data/coco128.yaml --dataset-profile coco128_quick --model-family l,w6 --output runs/train_seq/anchor_free_141 --start-stage 14 --end-stage 16 --epochs 3 --batch-size 2 --img 640 640 --workers 2 --debug-log error --console-log stderr
```

## 합격 기준

- 기존 `--det-head anchor` 학습/평가/export 결과가 기존과 동일하게 동작한다.
- FCOS cfg model build가 통과한다.
- synthetic FCOS decode가 통과한다.
- COCO128 1 epoch smoke에서 NaN/Inf가 없다.
- `stage_result.yaml`에 FCOS loss/positive/decode metric이 기록된다.
- raw ONNX export에서 FCOS output shape가 검증된다.
- L/W6 baseline 대비 GFLOPs 증가가 10% 미만이다.
- P2 hybrid는 head/person recall 개선 가능성이 없으면 full FCOS로 확장하지 않는다.

## 재검토 결과

- 기존 anchor 경로 보존 원칙을 명시했다.
- 1.3.6 decode-only FCOS와 1.4.1 trainable FCOS의 차이를 분리했다.
- `models/yolo.py`, `utils/fcos.py`, loss, train/test/export/runner까지 영향 범위를 포함했다.
- L과 W6를 동시에 전면 교체하지 않고 단계별 실험으로 나누었다.
- C++/TensorRT runtime은 제외 범위로 고정했다.
- 장점, 단점, 리스크, 검증 명령, 합격 기준을 문서에 포함했다.
