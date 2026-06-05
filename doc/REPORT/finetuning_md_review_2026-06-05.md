# 파인튜닝 MD 검토 2026-06-05

## 대상

- `doc/PLAN/development_requirements_1.3.7_finetuning_continual_learning.md`
- `doc/BASIC/YOLOv7_Custom_Design_Spec_v1.3.md` 11장
- `doc/PLAN/training_execution_plan_v1.8.md` Stage 13
- 현재 구현 파일: `finetune.py`, `utils/class_mapping.py`, `utils/replay_buffer.py`, `utils/pseudo_label.py`, `utils/continual_loss.py`, `tools/*finetune*`, `data/hyp_finetune.yaml`

## 결론

파인튜닝 방향은 현재 상황에서 진행 가치가 있다. 모델 구조 개선을 더 밀기보다 `Replay only -> cls distill -> cls+reg distill` 순서로 기존 클래스 망각을 통제하는 계획은 타당하다.

다만 문서는 아직 설계 요구와 실제 구현 상태가 섞여 있다. 바로 실험 기준으로 쓰기 전에 아래 항목을 정리해야 한다.

## 주요 발견

### P1. Replay ratio가 문서상 batch ratio처럼 쓰였지만 실제 구현은 이미지 리스트 병합이다

- 문서: replay와 finetune batch 비율 기본 30:70, 실제 sampling count 기록.
- 현재 `finetune.py`: finetune train 이미지 리스트 뒤에 replay 이미지를 단순 concat한다.
- 현재 `ReplayBufferBuilder`: `replay_ratio <= 1`이면 base dataset 길이 기준으로 replay image 수를 뽑는다.

이 상태에서는 실제 학습 batch가 30:70으로 유지되지 않는다. finetune dataset이 작고 base dataset이 크면 replay가 과도하게 많아질 수 있다.

정리 방향:
- MVP 문서 기준을 "이미지 리스트 병합 비율"로 낮추거나,
- 별도 sampler/ConcatDataset을 구현해 진짜 batch ratio를 보장한다고 명시한다.

### P1. forgetting 평가 문서와 실제 도구 수준이 다르다

- 문서: 기존 클래스, 파인튜닝 대상 클래스, held-out class metric 분리.
- 현재 `tools/evaluate_forgetting.py`: 전체 result metric 하나를 scratch/finetune 사이에서 비교한다.

이 상태로는 "기존/미포함 클래스 mAP 95% 이상" 같은 통과 기준을 검증할 수 없다.

정리 방향:
- 첫 실험은 overall retention만 보는 smoke로 분리한다.
- 실제 keep/drop 판단 전에는 per-class AP 입력 또는 `test.py` per-class 결과를 읽는 forgetting 평가를 보강한다.

### P1. pseudo label 단계가 필수인지 선택인지 문서가 흔들린다

- 문서 통과 기준은 pseudo label 생성과 병합 완료를 요구한다.
- 현재 `finetune.py`는 pseudo/merge를 내부 실행하지 않고, 실행하지 않은 경우 `status: skip` 산출물을 만든다.
- `tools/generate_pseudo_labels.py`, `tools/merge_labels.py`는 별도 CLI로 존재한다.

정리 방향:
- E1 Replay only에서는 pseudo label을 `skip` 허용으로 둔다.
- P2 또는 E2 이후에서 pseudo label 생성/병합을 필수로 올린다.
- `finetune.py` 내부 자동 실행인지, 외부 tool 선실행인지 하나로 고정한다.

### P2. distillation loss 수식이 설계서와 실제 구현이 다르다

- BASIC 설계서: `KLDiv(Student cls, Teacher cls)`, `SmoothL1(Student box, Teacher box)`.
- 현재 `utils/continual_loss.py`: cls/objectness와 box 모두 MSE 기반이다.

정리 방향:
- 현재 구현을 MVP로 인정하면 문서를 MSE 기반으로 수정한다.
- 논문식 YOLO LwF를 목표로 유지하면 구현을 KL/SmoothL1 또는 명시한 수식으로 보강한다.

### P2. class mapping 검증 범위가 문서보다 좁다

- 문서: scratch/finetune/replay/pseudo data yaml의 `nc`, `names`, class id mapping 일치 확인.
- 현재 `tools/check_class_mapping.py`와 `finetune.py`: base data와 finetune data 중심 검증.

정리 방향:
- replay/pseudo가 별도 yaml을 갖는지 먼저 결정한다.
- 별도 yaml이 없다면 문서 표현을 base/finetune 중심으로 낮춘다.
- 별도 yaml을 쓴다면 check tool 인자를 확장한다.

### P2. freeze/BN 기록 요구가 현재 산출물보다 강하다

- 문서: BN policy, freeze layer list, trainable parameter count를 `stage_result.yaml`에 기록.
- 현재 `stage_result.yaml`: `freeze_policy`, `bn_policy`, `train_command` 위주.
- trainable parameter count는 별도 기록되지 않는다.

정리 방향:
- 착수 전 `stage_result.yaml`에 resolved freeze layers와 trainable/frozen parameter count를 추가한다.

### P2. `tools/dataset_manifest.py --help`가 가벼운 명령인데 Torch import를 탄다

- 확인 결과 `python tools/dataset_manifest.py --help` 실행 중 `utils.datasets` import로 Torch 로딩이 발생했고, 현재 Windows 환경에서 `shm.dll` 로딩 실패가 났다.
- manifest/hash 도구는 데이터 검증용이므로 `--help`와 경량 manifest 작업은 Torch 없이 동작하는 편이 낫다.

정리 방향:
- `img2label_paths`, `img_formats`만 필요한 경우 경량 helper로 분리하거나 lazy import한다.

### P3. 실행 명령은 문서마다 다르다

- `doc/PLAN/development_requirements_1.3.7...`의 명령은 현재 CLI에 비교적 가깝다.
- `doc/BASIC/YOLOv7_Custom_Design_Spec_v1.3.md`의 예시는 `tools/` 경로와 `--base-data`가 빠져 있어 그대로 쓰면 실패한다.

정리 방향:
- 실제 실행 기준은 `doc/PLAN/development_requirements_1.3.7...`로 고정한다.
- BASIC 설계서는 개념 문서로 두고 실행 명령은 PLAN 문서로 링크한다.

## 진행 권장 순서

1. E1 Replay only 기준으로 문서와 코드 기준을 낮춰 정리한다.
2. `dataset_manifest.py` 경량화와 `stage_result.yaml` freeze/trainable count 기록을 먼저 고친다.
3. 작은 데이터로 `finetune.py --dry-run`을 통과시킨다.
4. E1 `--epochs 1` smoke를 돌린다.
5. per-class forgetting 평가가 준비된 뒤 E2/E3 distillation을 켠다.

## 현재 바로 사용할 기준 명령

```bash
python finetune.py --weights runs/train/final_scratch/weights/best.pt --base-data data/base.yaml --data data/finetune_example.yaml --replay-ratio 0.3 --hyp data/hyp_finetune.yaml --epochs 1 --img 640 --batch 16 --freeze none --distill-alpha 0.0 --distill-beta 0.0 --dry-run
```

E1은 distill off가 기본이다. `--teacher-weights`는 E2/E3에서 distill을 켤 때 필수로 둔다.
