# 1.3.7 Code-Level Development Requirements

- 기준 계획: `doc/PLAN/development_plan_v1.3.md`
- 대상 차수: `1.3.7 Fine-tuning / Continual Learning`
- 선행 조건: scratch 학습 기준선과 선택 구조가 확정됨
- 목적: 기존 클래스 망각을 억제하면서 신규/부분 데이터로 파인튜닝하는 파이프라인을 만든다.

## 1. 범위

포함:
- `finetune.py`
- Replay Buffer
- Pseudo Label 생성/병합
- YOLO LwF A/B
- Replay only / Replay + LwF cls / Replay + LwF cls+reg 단계 분리
- 파인튜닝 전용 hyp/data 예시
- 기존 클래스와 파인튜닝 대상 클래스의 metric 분리

제외:
- 최종 inference 구조 변경
- teacher model을 최종 export 그래프에 포함
- C++ 후처리, TensorRT runtime, 추론 서버

## 2. 파일 단위 개발 요구사항

| 파일 | 구분 | 요구사항 |
| --- | --- | --- |
| `finetune.py` | 신규 | `train.py` 공통 학습 함수를 재사용하고 replay/pseudo/distill 옵션을 추가한다. |
| `utils/continual_loss.py` | 신규 | LwF distillation loss를 구현한다. teacher forward는 `torch.no_grad()`와 eval mode로 실행한다. |
| `utils/replay_buffer.py` | 신규 | hard case replay sample 선택, manifest 생성, class balance 통계를 담당한다. |
| `utils/pseudo_label.py` | 신규 | teacher 추론 결과를 YOLO label로 저장하고 GT와 병합한다. |
| `tools/generate_pseudo_labels.py` | 신규 | pseudo label 생성 CLI. |
| `tools/merge_labels.py` | 신규 | GT label과 pseudo label 병합 CLI. |
| `data/hyp_finetune.yaml` | 신규 | 낮은 LR, mosaic off, copy-paste off 등 파인튜닝 hyp. |
| `data/finetune_example.yaml` | 신규 | 파인튜닝 데이터 yaml 예시. 클래스 수량은 예시로만 둔다. |
| `tools/evaluate_forgetting.py` | 신규 | old/new/held-out class metric을 분리 비교한다. |
| `tools/dataset_manifest.py` | 수정/재사용 | finetune/replay/pseudo 병합 데이터의 manifest와 hash를 생성한다. |
| `tools/check_class_mapping.py` | 신규 | scratch/finetune/replay/pseudo data yaml의 `nc`, `names`, class id mapping 일치를 검사한다. |

## 3. CLI 요구사항

Pseudo label 생성:

```bash
python tools/generate_pseudo_labels.py --weights runs/train/final_scratch/weights/best.pt --source finetune_data/images --data data/finetune_example.yaml --conf-thres 0.5 --iou-thres 0.45 --output finetune_data/pseudo_labels
```

Label 병합:

```bash
python tools/merge_labels.py --gt-labels finetune_data/labels --pseudo-labels finetune_data/pseudo_labels --output finetune_data/merged_labels --dedupe-iou 0.8
```

Fine-tuning:

```bash
python finetune.py --weights runs/train/final_scratch/weights/best.pt --teacher-weights runs/train/final_scratch/weights/best.pt --data data/finetune_example.yaml --replay-buffer data/replay_buffer --replay-ratio 0.3 --hyp data/hyp_finetune.yaml --epochs 100 --img 640 --batch 32 --freeze neck_lower --distill-alpha 0.5 --distill-beta 0.3 --distill-conf-thres 0.5 --name finetune_v1
```

`--distill-alpha`와 `--distill-beta`는 scalar 또는 `start:end` schedule 문자열을 허용한다. 구현 시 argparse type은 별도 `parse_float_or_schedule()` helper로 처리한다.

## 3.1 하위 단계

| 단계 | 목적 | 실행 옵션 |
| --- | --- | --- |
| `1.3.7-E1` | Replay only | `--replay-ratio 0.3 --distill-alpha 0.0 --distill-beta 0.0` |
| `1.3.7-E2` | Replay + cls distill | `--distill-alpha 0.2:0.5 --distill-beta 0.0` |
| `1.3.7-E3` | Replay + cls/reg distill | `--distill-alpha 0.2:0.5 --distill-beta 0.1:0.3 --distill-conf-thres 0.5` |

E1을 먼저 통과한 뒤 forgetting이 남을 때 E2/E3를 진행한다.

## 4. Replay Buffer 계약

- replay sample은 기존 데이터와 동일한 YOLO image/label 구조로 저장한다.
- class별 수량은 설정값이며 하드코딩하지 않는다.
- 기본 `replay-ratio`는 `0.3`이다.
- hard case 기준은 낮은 confidence, false negative 후보, 소형 객체, 야간/역광/IR 조건을 우선한다.
- `replay_manifest.json`에 source path, class count, hash, selection reason을 저장한다.
- replay dataset과 finetune dataset의 class id 매핑이 다르면 즉시 실패한다.
- replay buffer를 재생성할 때마다 이전 manifest와 새 manifest를 비교해 데이터 리빌드 이력을 남긴다.
- replay와 finetune batch 비율은 기본 30:70이며, 실제 sampling count를 `finetune_results.csv`에 기록한다.

## 5. Pseudo Label 계약

- teacher는 기존 scratch best weight를 사용한다.
- pseudo label은 confidence threshold 미만이면 저장하지 않는다.
- GT와 IoU `0.8` 이상 중복되는 pseudo box는 제거한다.
- 파인튜닝 대상 클래스 GT는 pseudo label보다 우선한다.
- 기존/미포함 클래스 보존 목적의 pseudo label은 별도 source marker를 남긴다.
- merge 결과는 YOLO label 포맷 `class x y w h`를 유지한다.
- Python model forward와 기존 NMS만 사용한다. C++/TensorRT runtime은 사용하지 않는다.

## 6. LwF 계약

- teacher model은 학습되지 않는다.
- teacher forward는 training step 안에서만 사용한다.
- final checkpoint와 export에는 student model만 저장한다.
- distillation loss는 old class에 우선 적용한다.
- `distill-alpha`: cls/objectness distill weight
- `distill-beta`: box/regression distill weight
- `distill-alpha` 기본 스케줄은 `0.2 -> 0.5`이며, 기존 클래스 망각이 심하면 report 판단 후 `0.8`까지 올릴 수 있다.
- `distill-beta` 기본 스케줄은 `0.1 -> 0.3`이다.
- cls distillation은 전체 예측 위치에 적용한다.
- reg distillation은 teacher confidence가 `distill_conf_thres` 이상인 박스에만 적용한다.
- teacher/student output shape이 다르면 distill을 skip하지 말고 명확히 실패 처리한다.
- 1.3.2의 공통 학습 helper가 없으면 `finetune.py`에서 학습 루프를 복사하지 않고 먼저 공통 helper를 정리한다.

## 6.1 Freeze / BatchNorm 정책

- scratch teacher는 freeze하고 `eval()` 상태로 둔다.
- student는 `--freeze neck_lower` 기본값을 지원한다.
- 기본 trainable 범위는 upper neck + head 전체다.
- BatchNorm은 데이터 분포가 크게 다른 경우 통계 갱신 여부를 `--bn-policy train|eval`로 명시한다.
- BN policy, freeze layer list, trainable parameter count를 `stage_result.yaml`에 기록한다.

## 7. 파인튜닝 hyp 기본값

`data/hyp_finetune.yaml` 기본 방향:
- `lr0: 0.0005`
- `lrf: 0.01`
- `warmup_epochs: 3`
- `mosaic: 0.0`
- `mixup: 0.0`
- `copy_paste: 0.0`
- `hsv_v: 0.3`
- `scale: 0.3`

## 8. 산출물

필수 산출물:
- `finetune_results.csv`
- `forgetting_report.yaml`
- `replay_manifest.json`
- `pseudo_label_manifest.json`
- `merge_report.json`
- `stage_result.yaml`
- `dataset_manifest.json`
- `class_mapping_check.json`

`forgetting_report.yaml` 필수 필드:
- `scratch_baseline`
- `finetune_run`
- `new_class_map`
- `old_class_map`
- `heldout_class_map`
- `old_class_drop_percent`
- `new_class_retention_percent`
- `replay_ratio`
- `distill_alpha`
- `distill_beta`
- `bn_policy`
- `freeze_policy`
- `sub_stage`
- `status`

## 9. 통과 기준

1. pseudo label 생성과 병합이 완료된다.
2. replay manifest가 생성되고 class id mapping이 일치한다.
3. finetune smoke가 완료된다.
4. teacher가 최종 checkpoint/export graph에 포함되지 않는다.
5. 파인튜닝 대상 클래스 mAP가 scratch 기준 90% 이상이다.
6. 기존/미포함 클래스 mAP가 scratch 기준 95% 이상이다.
7. 전체 mAP가 scratch 기준 93% 이상이다.
8. ONNX export와 ONNX Runtime 비교가 기존 경로로 통과한다.
9. finetune/replay/pseudo 병합 데이터의 manifest hash가 저장된다.
10. `class_mapping_check.json`에서 scratch/finetune/replay/pseudo class id mapping 일치가 확인된다.
11. E1/E2/E3 하위 단계 결과가 분리 저장된다.

## 10. 구현 순서

1. `data/hyp_finetune.yaml`, `data/finetune_example.yaml` 작성
2. class mapping 검사 도구 작성
3. pseudo label 생성 도구 작성
4. label merge 도구 작성
5. replay buffer module 작성
6. continual loss module 작성
7. `finetune.py` 작성
8. forgetting 평가 도구 작성
9. E1/E2/E3 순서로 smoke run 후 `stage_result.yaml` 저장

## 11. 리스크 및 주의사항

- 클래스별 수량은 운영 데이터에 따라 달라지는 예시값이다.
- replay buffer는 데이터 리빌드 정책과 manifest/hash 기준을 반드시 따른다.
- teacher가 student와 같은 구조가 아니면 distill output alignment를 먼저 명시한다.
- pseudo label 품질이 낮으면 기존 클래스 보존보다 오염이 커질 수 있으므로 threshold와 merge report를 남긴다.

## 12. 개발 착수 분리 기준

finetune은 학습 루프 복제 위험이 크므로 1.3.2의 공통 helper가 준비된 뒤 시작한다.

| PR | 범위 | 완료 기준 |
| --- | --- | --- |
| `1.3.7-P1` | data/hyp 예시, class mapping 검사 | `class_mapping_check.json` 생성 |
| `1.3.7-P2` | pseudo label 생성/병합 | `pseudo_label_manifest.json`, `merge_report.json` 생성 |
| `1.3.7-P3` | replay buffer | `replay_manifest.json`, replay sampling smoke 통과 |
| `1.3.7-P4` | Replay only finetune | E1 smoke와 forgetting report 생성 |
| `1.3.7-P5` | cls distillation | E2 smoke, alpha schedule 로그 확인 |
| `1.3.7-P6` | cls+reg distillation | E3 smoke, confidence-filtered reg distill 확인 |

`finetune.py`에서 `train.py`의 학습 loop를 복사해 분기시키면 이후 Phase/logging/export 정책이 갈라진다. 공통 helper가 부족하면 먼저 1.3.2 쪽을 보강한다.
