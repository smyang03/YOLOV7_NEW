# 1.3.6 Code-Level Development Requirements

## 1.3.6.1 코드 구현 상세

이 세부 항목은 optional 실험이 기본 경로에 섞이지 않도록 flag validation, cfg 분리, decode 검증 도구를 고정한다.

### 대상 파일과 구현 위치

| 파일 | 위치 | 구현 방식 |
| --- | --- | --- |
| `train.py` | argparse validation | optional flag가 동시에 둘 이상 켜지면 실패 처리한다. |
| `models/common.py` | PSA/GELAN 후보 block | 기본 import/parse만 가능하게 하되 실제 사용은 cfg와 flag가 맞을 때만 허용한다. |
| `models/yolo.py` | FCOS 후보 head | 기본 Detect 경로와 output contract를 섞지 않는다. FCOS는 별도 raw output으로 둔다. |
| `tools/decode_fcos_outputs.py` | 신규 CLI | FCOS raw output을 Python에서 decode하고 score 결합을 검증한다. |
| `doc/REPORT/optional_decision_*.md` | 사전 리포트 | optional 진입 사유와 목표 metric 부족분을 기록한다. |

### argparse 구현 규칙

```python
parser.add_argument('--aux', choices=['auto', 'on', 'off'], default='auto')
parser.add_argument('--psa-level', choices=['none', 'p5', 'p4p5', 'p3p4p5'], default='none')
parser.add_argument('--p2-head', choices=['none', 'anchor', 'fcos'], default='none')
parser.add_argument('--neck-mod', choices=['none', 'scdown', 'psa', 'gelan'], default='none')
```

optional validation:
- `--neck-mod psa`이면 `--psa-level p5`만 1차 허용한다.
- `--p2-head fcos`는 W6에서만 허용한다.
- `--neck-mod gelan`은 W6 일부 neck cfg에서만 허용한다.
- PSA, FCOS, GELAN은 동시에 켜지 않는다.

### optional_decision schema

```yaml
date:
baseline_stage:
target_model: "l|w6"
missing_metric:
current_value:
target_value:
remaining_gflops_budget_percent:
requested_experiment:
expected_gain:
stop_condition:
```

이 파일이 없으면 optional 실험 실행을 막는다.

### FCOS P2 decode 계약

FCOS P2는 본 차수에서 Python raw/decode까지만 검증한다.

필수 출력:
- `fcos_decode_check.json`
- raw output shape
- decoded box count
- score 결합 방식
- anchor output과 NMS input 결합 여부

C++ postprocess, TensorRT plugin, runtime deploy는 이 문서 범위가 아니다.

### 검증 명령

```bash
python train.py --cfg cfg/training/yolov7.yaml --data data/coco128.yaml --epochs 1 --aux on --name smoke_1361_l_aux
python train.py --cfg cfg/training/yolov7-w6-p2.yaml --data data/coco128.yaml --epochs 1 --p2-head fcos --name smoke_1361_fcos
python tools/decode_fcos_outputs.py --weights runs/train/.../weights/best.pt --data data/coco128.yaml
```

필수 확인:
- optional decision report 존재
- optional flag 동시 적용 차단
- export 실패 시 optional drop
- 효과 미미 시 기본 off 유지

## 리포트 기반 정비 기준

- 문서 위치 기준: 본 코드레벨 개발 요구서는 `doc/PLAN/`에 둔다.
- 기준 리포트: `doc/REPORT/ai_perspective_yolov7_improvement_analysis_2026-05-22.md`
- Optional 실험은 기본 개발 흐름이 아니다.
- 1.3.1~1.3.5 산출물이 안정화되고 목표 metric이 부족할 때만 시작한다.
- optional 구조는 동시에 둘 이상 켜지 않으며, 진입 전 `doc/REPORT/optional_decision_*.md`를 먼저 작성한다.

- 기준 계획: `doc/PLAN/development_plan_v1.3.md`
- 대상 차수: `1.3.6 Optional / 후순위 실험`
- 선행 조건: `1.3.5`까지 필수 구성이 안정화되었고 개선 목표가 아직 부족한 경우
- 목적: 필수 경로가 안정화된 뒤 선택 실험을 하나씩 분리 적용한다.

## 1. 범위

포함 후보:
- L AUX on 성능형 옵션
- PSA P5
- FCOS P2
- GELAN 일부 block

제외:
- 필수 기본값으로 자동 활성화
- 여러 optional 구조 동시 적용
- C++ 후처리, TensorRT runtime, 추론 서버

## 2. 진입 조건

아래 중 하나 이상을 만족할 때만 1.3.6을 시작한다.
- 1.3.3/1.3.4/1.3.5 후에도 primary mAP 목표 미달
- W6 small AP/recall 목표 미달
- GFLOPs 증가 여유가 10% 예산 안에 남아 있음
- ONNX export 기준선이 안정적임

## 3. 실험 단위

| 단계 | 목적 | 실행 플래그 | 대상 |
| --- | --- | --- | --- |
| `1.3.6-C1` | L AUX 성능형 | `--aux on` | L |
| `1.3.6-C2` | PSA P5 | `--neck-mod psa --psa-level p5` | W6 우선 |
| `1.3.6-C3` | FCOS P2 | `--p2-head fcos` | W6 |
| `1.3.6-C4` | GELAN | `--neck-mod gelan` | W6 우선 |

각 실험은 직전 성공 stage를 기준으로 단독 적용한다.

## 4. 파일 단위 개발 요구사항

| 파일 | 구분 | 요구사항 |
| --- | --- | --- |
| `train.py` | 수정 | optional flag 조합을 검증한다. 동시에 둘 이상 켜면 실패 처리한다. |
| `models/yolo.py` | 수정 | AUX, FCOS P2, GELAN cfg를 parse할 수 있게 한다. |
| `models/common.py` | 수정 | PSA/GELAN block을 추가하되 ONNX export 가능한 연산만 사용한다. |
| `utils/loss.py` | 수정 | FCOS P2 사용 시 anchor-free box/cls/centerness loss를 분리한다. |
| `utils/fcos.py` | 신규 | FCOS P2 target assign, decode, output shape helper를 구현한다. |
| `tools/decode_fcos_outputs.py` | 신규 | FCOS raw output을 Python에서 decode하고 anchor output과 score 결합을 검증한다. |
| `cfg/experiments/*.yaml` | 신규 | optional 실험별 cfg를 분리한다. 기존 baseline cfg를 덮어쓰지 않는다. |
| `tools/profile_model.py` | 수정 | optional 실험별 GFLOPs delta와 output shape을 기록한다. |
| `tools/verify_export.py` | 수정 | FCOS raw output이 추가되는 경우 output key/shape 비교를 지원한다. |
| `doc/REPORT/optional_decision_*.md` | 신규 | optional 실험 진입 사유, 결과, 기본값 승격/보류 결정을 기록한다. |

## 5. 실험별 계약

### 5.1 L AUX on

- L 기본값은 AUX off다.
- `--aux on`은 성능형 옵션일 뿐 기본 모델 구조가 아니다.
- train loop는 AUX branch를 자동 감지해야 하며 W6 전용 하드코딩을 사용하지 않는다.
- inference/export에서는 main head raw output만 기본으로 사용한다.

### 5.2 PSA P5

- W6 P5 neck 일부에만 적용한다.
- L 모델에는 기본 적용하지 않는다.
- GFLOPs와 activation memory 증가를 반드시 기록한다.
- 적용 순서는 P5 단독, 필요 시 P4 추가, 마지막으로 P3 추가다.
- `--psa-level p5|p4p5|p3p4p5`로 명시하고 동시 전체 적용을 기본값으로 두지 않는다.

### 5.3 FCOS P2

- W6 P2 Anchor로도 small recall이 부족할 때만 진행한다.
- anchor output과 별도 raw output을 명확히 분리한다.
- Python 평가와 ONNX Runtime 비교까지만 구현한다.
- C++ postprocess는 본 차수에서 제외한다.
- Python decode에서 centerness와 obj score 결합 방식을 명시하고 `fcos_decode_check.json`으로 저장한다.

### 5.4 GELAN

- 전체 backbone 교체가 아니라 neck 일부 block 단독 교체로 제한한다.
- route/channel mismatch가 발생하면 즉시 중단한다.
- export 실패 시 실험 실패로 기록한다.

## 6. 산출물

각 optional 실험 run directory에 아래 파일을 남긴다.
- `stage_result.yaml`
- `profile.json`
- `export_check.json`
- `optional_ablation.csv`
- `fcos_decode_check.json`
- `doc/REPORT/optional_decision_YYYY-MM-DD.md`
- `failure_reason.txt` 또는 `fallback.log`

`optional_ablation.csv` 컬럼:
- `stage`
- `option`
- `baseline_run`
- `current_run`
- `primary_map`
- `small_ap`
- `rare_recall`
- `gflops_delta_percent`
- `total_boxes`
- `export_passed`
- `decision`
- `reason`

## 7. 통과 기준

1. optional 실험은 한 번에 하나만 활성화된다.
2. GFLOPs 증가율이 10% 미만이다.
3. ONNX export와 ONNX Runtime 비교가 통과한다.
4. primary mAP 또는 목표 scenario metric이 개선된다.
5. output shape/postprocess 복잡도가 report에 기록된다.
6. 효과가 미미하면 기본값으로 승격하지 않는다.
7. 기본값 승격 또는 보류 결정이 `doc/REPORT/optional_decision_*.md`에 기록된다.
8. PSA는 P5 단독 결과 없이 P3/P4/P5 동시 적용으로 진행하지 않는다.

## 8. 중단 기준

- NaN/Inf loss 발생
- primary mAP 2 percentage points 이상 하락
- GFLOPs 증가율 10% 이상
- ONNX export 실패
- output shape이 후속 Python 검증 도구에서 처리 불가
- FCOS P2 postprocess 복잡도가 과도함

## 9. 구현 순서

1. optional flag validation 추가
2. 진입 조건을 `optional_decision_*.md` 초안에 기록
3. L AUX on 실험
4. PSA P5 실험
5. FCOS P2 실험
6. GELAN 실험
7. 각 실험별 ablation report 작성
8. 기본값 승격/보류 결정 기록

## 10. 리스크 및 주의사항

- 1.3.6은 필수 개발 경로가 아니다.
- optional 결과가 좋더라도 기본값 승격은 별도 문서에서 결정한다.
- FCOS P2는 raw output과 Python 검증까지만 다룬다.
- runtime deploy 요구사항은 별도 차수로 분리한다.

## 11. 개발 착수 분리 기준

1.3.6은 앞 단계가 부족할 때만 들어가는 선택 차수다. 개발자는 진입 조건 문서 없이 optional 코드를 작성하지 않는다.

| PR | 범위 | 완료 기준 |
| --- | --- | --- |
| `1.3.6-P0` | `optional_decision_*.md` 진입 사유 작성 | 목표 미달 metric과 남은 GFLOPs 여유가 기록됨 |
| `1.3.6-P1` | L AUX on | L 성능형 smoke와 off 기본값 유지 확인 |
| `1.3.6-P2` | PSA P5 단독 | `--psa-level p5`만 통과, P4/P3는 미적용 |
| `1.3.6-P3` | FCOS P2 Python raw/decode | `fcos_decode_check.json` 생성 |
| `1.3.6-P4` | GELAN neck 일부 단독 | route/channel/export 통과 |

optional flag validation은 `P1` 전에 먼저 넣는다. 둘 이상의 optional 구조가 동시에 켜지면 실행 전 실패해야 한다.
