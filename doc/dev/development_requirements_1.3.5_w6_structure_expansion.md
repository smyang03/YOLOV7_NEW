# 1.3.5 Code-Level Development Requirements

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
- ONNX raw export 및 ONNX Runtime 비교

제외:
- YOLOv7-L 구조 변경
- FCOS P2
- PSA, GELAN
- C++ 후처리, TensorRT runtime, 추론 서버

## 2. 파일 단위 개발 요구사항

| 파일 | 구분 | 요구사항 |
| --- | --- | --- |
| `models/common.py` | 수정 | `SCDown` block을 추가한다. Conv/BN/activation 조합은 ONNX export 가능한 연산만 사용한다. |
| `models/yolo.py` | 수정 | W6 Detect/IAuxDetect가 P2 포함 5개 level을 처리한다. stride `4, 8, 16, 32, 64`를 지원한다. |
| `cfg/training/yolov7-w6-scdown.yaml` | 신규 | SCDown 단독 실험용 W6 training cfg. |
| `cfg/training/yolov7-w6-p2.yaml` | 신규 | P2 Anchor 단독 실험용 W6 training cfg. |
| `cfg/training/yolov7-w6-p2-scdown.yaml` | 신규 | P2 Anchor + SCDown 누적 실험용 W6 training cfg. |
| `cfg/deploy/yolov7-w6-scdown.yaml` | 신규 | SCDown 단독 export cfg. |
| `cfg/deploy/yolov7-w6-p2.yaml` | 신규 | P2 Anchor 단독 export cfg. |
| `cfg/deploy/yolov7-w6-p2-scdown.yaml` | 신규 | P2 Anchor + SCDown export cfg. |
| `utils/autoanchor.py` | 확인/최소 수정 | 5개 detection layer anchor check를 지원한다. |
| `tools/profile_model.py` | 수정 | `total_boxes`, `output_shapes`, `activation_memory_mb`를 기록한다. |
| `tools/estimate_nms_cost.py` | 신규 | PyTorch/Python 기준 NMS 입력 box 수와 소요 시간을 추정한다. |
| `train.py` | 수정 | `--p2-head none|anchor`, `--neck-mod none|scdown` 플래그를 저장하고 cfg 선택을 검증한다. |

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
- ONNX export가 실패하는 custom op를 사용하지 않는다.
- D1 단독 실험에서 SCDown의 GFLOPs/memory/export 영향을 먼저 기록한다.

## 5. 산출물

필수 산출물:
- `profile.json`
- `nms_cost.json`
- `export_check.json`
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
6. ONNX raw export와 ONNX Runtime 비교가 통과한다.
7. L 모델 구조는 변경되지 않는다.
8. D1, D2, D3 결과가 분리 저장되어 P2와 SCDown 효과를 구분할 수 있다.

## 7. 구현 순서

1. `SCDown` block 추가
2. W6 P2 cfg 작성
3. `models/yolo.py` 5-level Detect/IAuxDetect 검증
4. autoanchor 5-level 확인
5. profile output 확장
6. NMS cost estimate 도구 작성
7. D1 SCDown 단독 smoke/profile/export 검증
8. D2 P2 Anchor 단독 smoke/profile/export 검증
9. D3 P2+SCDown 누적 smoke/profile/export 검증
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
| `1.3.5-P1` | `SCDown` block + SCDown 단독 cfg | D1 model build/profile/export 통과 |
| `1.3.5-P2` | P2 Anchor cfg + 5-level Detect/IAuxDetect | D2 model build/profile/export 통과 |
| `1.3.5-P3` | P2 + SCDown 누적 cfg | D3 model build/profile/export 통과 |
| `1.3.5-P4` | `tools/estimate_nms_cost.py`, output/memory report | `nms_cost.json`, `profile.json` 확장 필드 생성 |

`cfg/training/yolov7-w6-p2.yaml` 하나로 D1/D2/D3를 모두 처리하지 않는다. 각 실험의 route diff를 작게 유지해야 원인 추적이 가능하다.
