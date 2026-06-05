# 파인튜닝 튜닝/실험 가능성 검토 2026-06-05

## 결론

현재 상태는 **E1 Replay-only dry-run 및 배선 검증은 가능**하다. 실제 `finetune.py`가 산출물 생성, class mapping 검사, replay manifest 생성, `train.py` 호출 명령 조립까지 수행한다.

다만 **성능 튜닝/비교 실험을 신뢰성 있게 반복할 수준은 아직 부족**하다. 원인은 학습 실행 자체보다 실험 판정에 필요한 replay 비율 제어, per-class forgetting 평가, distillation 로그/수식 정합성이 약한 데 있다.

권장 판정:

- 가능: E1 dry-run, class mapping 검사, replay manifest smoke, `train.py` 호출 명령 생성
- 제한적 가능: E1 `--epochs 1` smoke, 실제 weight/data가 준비된 경우
- 보류: E2/E3 distillation 성능 비교, 최종 keep/drop 판단

## 확인한 명령

```bash
python -m py_compile finetune.py utils/class_mapping.py utils/pseudo_label.py utils/replay_buffer.py utils/continual_loss.py tools/check_class_mapping.py tools/generate_pseudo_labels.py tools/merge_labels.py tools/build_replay_buffer.py tools/evaluate_forgetting.py tools/dataset_manifest.py
```

결과: 통과.

```bash
python finetune.py --weights runs/train/final_scratch/weights/best.pt --base-data data/finetune_example.yaml --data data/finetune_example.yaml --replay-ratio 0.0 --hyp data/hyp_finetune.yaml --epochs 1 --img 640 --batch 2 --freeze none --distill-alpha 0.0 --distill-beta 0.0 --dry-run --project runs/finetune_review --name dry_run --exist-ok
```

결과: 통과. `runs/finetune_review/dry_run/` 아래 산출물이 생성됨.

생성 산출물:

- `class_mapping_check.json`
- `dataset_manifest.json`
- `finetune_data.yaml`
- `finetune_results.csv`
- `forgetting_report.yaml`
- `merge_report.json`
- `pseudo_label_manifest.json`
- `replay_manifest.json`
- `stage_result.yaml`

```bash
python finetune.py --weights runs/train/final_scratch/weights/best.pt --teacher-weights runs/train/final_scratch/weights/best.pt --base-data data/finetune_example.yaml --data data/finetune_example.yaml --replay-ratio 0.0 --hyp data/hyp_finetune.yaml --epochs 1 --img 640 --batch 2 --freeze none --distill-alpha 0.2:0.5 --distill-beta 0.0 --dry-run --project runs/finetune_review --name dry_run_distill --exist-ok
```

결과: 통과. `train_command`에 `--teacher-weights`, `--distill-alpha`, `--distill-beta`, `--distill-conf-thres`가 포함됨.

```bash
python finetune.py --weights runs/train/final_scratch/weights/best.pt --base-data data/finetune_example.yaml --data data/finetune_example.yaml --replay-ratio 0.0 --hyp data/hyp_finetune.yaml --epochs 1 --img 640 --batch 2 --freeze none --distill-alpha 0.1 --distill-beta 0.0 --dry-run --project runs/finetune_review --name dry_run_distill_missing_teacher --exist-ok
```

결과: 기대대로 실패. `--teacher-weights is required when distill alpha/beta is non-zero`.

## 가능한 부분

### 1. E1 실험 진입점

`finetune.py`는 별도 학습 loop를 복제하지 않고 `train.py`를 호출한다. 이 구조는 기존 YOLOv7 동작 보존 원칙과 맞다.

현재 E1 기준 명령은 동작 가능하다.

```bash
python finetune.py --weights <scratch_best.pt> --base-data <base.yaml> --data <finetune.yaml> --replay-ratio 0.3 --hyp data/hyp_finetune.yaml --epochs 1 --img 640 --batch 16 --freeze none --distill-alpha 0.0 --distill-beta 0.0 --project runs/finetune --name e1_replay_smoke
```

단, 실제 학습 smoke에는 `<scratch_best.pt>`, `<base.yaml>`, `<finetune.yaml>`이 실제 파일이어야 한다.

### 2. class mapping guard

`utils/class_mapping.py`는 base/finetune yaml의 `nc`, `names`, index 이동, 기존 클래스 누락을 잡는다. 클래스 index가 바뀌는 파인튜닝에서는 최소 안전장치로 쓸 수 있다.

### 3. distill schedule guard

`utils/continual_loss.py`의 `parse_float_or_schedule()`은 `0.2:0.5` 형식을 처리한다. `finetune.py`도 distill weight가 0보다 크면 teacher weight를 요구한다.

## 부족한 부분

### P1. replay ratio가 실험 제어 변수로 부정확하다

문서상 `replay-ratio 0.3`은 batch 30:70처럼 읽힌다. 실제 구현은 train image list에 replay image를 concat한다.

문제:

- 실제 batch마다 replay 30%가 보장되지 않는다.
- base dataset 크기 기준으로 replay 수가 정해져 finetune dataset이 작으면 replay가 과도해질 수 있다.
- `finetune_results.csv`는 이미지 수만 기록하고 실제 batch sampling 비율은 기록하지 않는다.

필요 조치:

- E1 smoke 전에는 `replay-ratio` 의미를 "base train image 수 기준 replay 선택 비율"로 문서화한다.
- 튜닝 실험 전에는 finetune/replay target count 또는 실제 sampler 비율을 명시한다.

### P1. forgetting 평가가 keep/drop 판단에 부족하다

`tools/evaluate_forgetting.py`는 전체 metric 하나를 scratch/finetune 사이에서 비교한다. 문서의 old/new/held-out class mAP 분리 기준을 만족하지 못한다.

문제:

- 기존 클래스가 무너졌는지 별도 판단할 수 없다.
- 신규/파인튜닝 대상 클래스 학습 부족도 별도 판단할 수 없다.
- `status: pass`가 metric 파일이 없어도 retention이 `null`이면 pass가 될 수 있다.

필요 조치:

- 최소한 per-class AP를 JSON/YAML로 저장하는 평가 산출물을 확정한다.
- `evaluate_forgetting.py`는 metric 파일이 없으면 `warn` 또는 `fail`로 처리한다.

### P1. pseudo label은 실험 파이프라인에 자동 연결되어 있지 않다

`finetune.py`는 pseudo 생성/병합을 수행하지 않고 skip manifest를 만든다. 별도 CLI는 있지만 E1/E2/E3 중 어디서 필수인지 명확하지 않다.

필요 조치:

- E1 Replay-only: pseudo skip 허용.
- Pseudo 실험: `tools/generate_pseudo_labels.py` -> `tools/merge_labels.py` -> merged label data yaml 생성 순서를 별도 단계로 고정.

### P2. distillation 구현은 MVP다

문서상 LwF는 KLDiv/SmoothL1 기반으로 설명되어 있다. 현재 `utils/continual_loss.py`는 MSE 기반이다.

실험 영향:

- "논문식 YOLO LwF" 실험이라고 부르기에는 부족하다.
- E2/E3는 기능 smoke 수준으로 먼저 봐야 하며, 성능 비교 결과를 일반화하면 안 된다.

### P2. distill 로그가 부족하다

`train.py`에서 `distill_items`는 계산되지만 별도 CSV/YAML 로그로 저장되는 경로가 명확하지 않다.

필요 조치:

- epoch별 `distill_total`, `distill_cls`, `distill_reg`, `alpha`, `beta`를 기록한다.
- `stage_result.yaml`에 teacher path, distill mode, output shape check 결과를 남긴다.

### P2. freeze/BN 실험 정보가 부족하다

현재 `stage_result.yaml`은 `freeze_policy`, `bn_policy`, `train_command`를 남긴다. 그러나 trainable/frozen parameter count와 실제 freeze layer list는 부족하다.

필요 조치:

- `stage_result.yaml`에 resolved freeze index, trainable parameter count, frozen parameter count를 추가한다.

## 실험 착수 기준

### 지금 가능한 최소 실험

1. base/finetune yaml class mapping 검사
2. E1 dry-run
3. E1 `--epochs 1` smoke
4. 전체 mAP 기준의 매우 거친 retention 확인

### 아직 보강 후 해야 하는 실험

1. replay ratio sweep: `0.1`, `0.3`, `0.5`
2. BN policy A/B: `train`, `eval`
3. freeze policy A/B: `none`, `backbone`, `partial`, `neck_lower`
4. distill alpha schedule: `0.2:0.5`, `0.5`, `0.5:0.8`
5. pseudo label threshold: `0.5`, `0.6`, `0.7`

위 sweep은 per-class forgetting 평가가 붙은 뒤 진행해야 한다.

## 최종 판정

현재 코드는 **실험 프레임의 골격은 있다**. 하지만 **튜닝 결과를 신뢰하고 의사결정할 수준은 E1 smoke까지만**이다.

바로 다음 개발은 모델 구조가 아니라 아래 순서가 맞다.

1. `evaluate_forgetting.py` per-class old/new/held-out 평가 보강
2. replay 선택 수와 실제 학습 비율 기록 보강
3. `stage_result.yaml`에 freeze/trainable count와 distill 로그 추가
4. E1 Replay-only 1 epoch smoke
5. E1 통과 후 E2/E3 distill 기능 smoke
