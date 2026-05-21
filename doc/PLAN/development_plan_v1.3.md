# YOLOv7 Custom Development Plan v1.3

- 기준 요구서: `doc/BASIC/YOLOv7_Custom_Design_Spec_v1.3.md`
- 기준 리포트: `doc/REPORT/ai_perspective_yolov7_improvement_analysis_2026-05-22.md`
- 작성일: 2026-05-21
- 정비일: 2026-05-22
- 문서 위치 기준: 개발 계획과 코드레벨 개발 요구서는 모두 `doc/PLAN/`에 둔다.

## 0. 세부 번호 체계

각 개발 차수는 두 단계 번호를 사용한다.

- `1.3.x`: 차수별 코드레벨 개발 요구서
- `1.3.x.1`: 해당 차수의 실제 구현 상세 항목

`1.3.x.1`에는 실제 함수 내부 구현 방식, 클래스 설계, argparse option, YAML schema, checkpoint/report schema, 검증 명령을 기록한다. 구현자는 각 차수 개발 전에 반드시 해당 문서의 `1.3.x.1 코드 구현 상세` 섹션을 먼저 확인한다.

## 1. 개발 원칙

v1.3의 목적은 YOLOv7을 무작정 크게 만드는 것이 아니라 CCTV 도메인에서 발생하는 실패 원인을 분리해 개선하는 것이다. 개발은 플래그 기반으로 통합하되 학습 서버에서는 한 번에 하나의 기능만 켜서 검증한다.

공통 원칙:
- Baseline, evaluation, Python ONNX export 기준선을 먼저 고정한다.
- 모델 구조, loss, assignment, augmentation, sampler, phase 변경을 한 번에 섞지 않는다.
- YOLOv7-L은 속도형 모델로 유지하고, YOLOv7-W6는 정확도형 모델로 분리한다.
- Backbone 구조는 변경하지 않는다. feature 추출 위치 변경은 별도 후순위 실험으로만 다룬다.
- C++ 후처리, TensorRT runtime, 추론 서버, 운영 배포 코드는 본 v1.3 개발 범위에서 제외한다.
- 각 단계는 `stage_result.yaml`, `profile.json`, `export_check.json` 중 필요한 산출물을 남긴 뒤 다음 단계로 넘어간다.

## 2. 문서 인덱스

| 차수 | 문서 | 역할 |
| --- | --- | --- |
| 1.3.1 | `doc/PLAN/development_requirements_1.3.1_baseline_export.md` | baseline, dataset 안정화, checkpoint, Python ONNX export |
| 1.3.2 | `doc/PLAN/development_requirements_1.3.2_train_loop_phase_logging.md` | phase 학습 루프, DataLoader rebuild, logging |
| 1.3.3 | `doc/PLAN/development_requirements_1.3.3_core_model_loss.md` | Decoupled Head, WIoU, TAL, VFL |
| 1.3.4 | `doc/PLAN/development_requirements_1.3.4_cctv_augmentation_sampler.md` | CCTV augmentation, hard negative, sampler |
| 1.3.5 | `doc/PLAN/development_requirements_1.3.5_w6_structure_expansion.md` | W6 P2 Anchor, SCDown |
| 1.3.6 | `doc/PLAN/development_requirements_1.3.6_optional_experiments.md` | PSA, FCOS, GELAN, L AUX option |
| 1.3.7 | `doc/PLAN/development_requirements_1.3.7_finetuning_continual_learning.md` | replay, pseudo label, LwF fine-tuning |
| 1.3.8 | `doc/PLAN/development_requirements_1.3.8_training_sequence_reporting.md` | training sequence, COCO128 quick run, final report automation |
| 주의사항 | `doc/PLAN/development_precautions_1.3.1-1.3.8.md` | 1.3.1~1.3.8 개발 사유, 예상 효과, 주의점 |

## 3. 전체 실행 순서

### 1.3.1 Baseline / Python Export

목표: 모델 개선 전 기존 YOLOv7-L/W6의 학습, 평가, export 기준선을 고정한다.

개발 순서:
1. `1.3.1-P1`: CLI alias, `images -> labels` 매핑, label cache invalidation, `persistent_workers` 정책 수정
2. `1.3.1-P2`: `best.pt`, `test.test()` 4-return, `opt.yaml`, `results.txt` parse 안정화
3. `1.3.1-P3`: raw ONNX export, `profile_model.py`, `verify_export.py`
4. `1.3.1-P4`: dataset manifest, metric summary, baseline report

통과 기준: L/W6 smoke 학습, 평가, raw ONNX export, PyTorch/ONNX Runtime 비교, GFLOPs baseline 기록이 모두 완료된다.

### 1.3.2 Train Loop / Phase / Logging

목표: 모델 구조를 바꾸기 전에 통합 학습 루프와 phase 전환 기반을 만든다.

개발 순서:
1. `1.3.2-P1`: `utils/phase.py`, `tools/check_phase_schedule.py`
2. `1.3.2-P2`: `utils/train_logger.py`, `results.csv`, `loss_detail.csv`
3. `1.3.2-P3`: DataLoader build helper, Phase 2/3 rebuild, Close Mosaic worker 정책
4. `1.3.2-P4`: `utils/train_common.py` 공통 helper 추출
5. `1.3.2-P5`: `train_aux.py` wrapper화 또는 얇은 entry 축소

통과 기준: epoch boundary, `workers=0/>0`, Phase 2/3 rebuild, Close Mosaic smoke가 통과한다.

### 1.3.3 Core Model / Loss

목표: 성능 개선의 핵심 요소를 하나씩 분리 적용한다.

검증 순서:
1. Decoupled Head 단독
2. WIoU v3 단독
3. TAL + VFL 단독
4. Decoupled Head + WIoU + TAL/VFL 누적

통과 기준: NaN/Inf 없음, positive count와 loss scale 기록, GFLOPs 증가 10% 미만, raw ONNX export 통과, baseline 대비 primary mAP 하락 없음.

### 1.3.4 CCTV Augmentation / Sampler

목표: 데이터 개선 효과를 모델 구조 효과와 분리해서 검증한다.

개발 순서:
1. augmentation profile dry-run
2. label-preserving pixel augmentation
3. Patch-Paste 안전장치
4. Hard Negative mining/paste
5. weighted sampler
6. scenario metric 연결

통과 기준: 시각 검증 이미지, bbox/class id 검사, sampler 통계, smoke 학습이 통과한다. label-changing augmentation은 visual audit 전 full training에 사용하지 않는다.

### 1.3.5 W6 Structure Expansion

목표: W6 정확도형 모델에만 소형 객체 개선 구조를 적용한다.

개발 순서:
1. SCDown only
2. P2 Anchor only
3. P2 Anchor + SCDown
4. output box 수, memory, Python NMS 비용 추정

통과 기준: W6 small AP/recall 개선, GFLOPs 증가 10% 미만, raw ONNX export 통과, output 증가가 허용 범위 안에 있어야 한다.

### 1.3.6 Optional Experiments

목표: 필수 구성이 안정화된 뒤 부족한 지표가 있을 때만 선택 실험을 진행한다.

진입 조건:
- 1.3.1~1.3.5 산출물이 모두 존재한다.
- 목표 metric이 아직 부족하다.
- GFLOPs/latency 여유가 남아 있다.
- `doc/REPORT/optional_decision_*.md`에 진입 사유를 먼저 기록한다.

실험 순서: L AUX on, PSA P5, FCOS P2 Python raw/decode, GELAN 일부 교체. 동시에 둘 이상 켜지 않는다.

### 1.3.7 Fine-tuning / Continual Learning

목표: scratch 학습 기준선 확정 후 catastrophic forgetting을 줄인다.

개발 순서:
1. class mapping 검사
2. pseudo label 생성/병합
3. replay buffer
4. Replay only fine-tuning
5. cls distillation
6. cls + reg distillation

통과 기준: 대상 클래스 mAP, 기존 클래스 mAP, forgetting 지표, export 검증이 분리 저장된다. Replay only가 먼저 통과해야 LwF를 켠다.

### 1.3.8 Training Sequence / Report Automation

목표: 1.3.1~1.3.7 개발 완료 후 COCO128 quick run과 대상 dataset full run을 stage별로 연속 실행하고, 최종 리포트를 자동 생성한다.

개발 순서:
1. `1.3.8-P1`: stage config/result schema 작성
2. `1.3.8-P2`: sequence runner dry-run
3. `1.3.8-P3`: COCO128 Stage 00~02 quick run
4. `1.3.8-P4`: stage metric 수집과 delta 계산
5. `1.3.8-P5`: stage/sequence/final report 생성
6. `1.3.8-P6`: COCO128 전체 sequence report 생성

통과 기준: COCO128 quick run은 orchestration과 report 판정 검증에만 사용하고, 최종 유지/제거 판단은 대상 dataset full run 결과로 확정한다.

## 4. 공통 중단 조건

아래 조건 중 하나라도 발생하면 다음 차수로 넘어가지 않는다.

- primary mAP가 baseline 대비 2 percentage points 이상 하락
- GFLOPs 증가율이 10% 이상
- ONNX export 또는 PyTorch/ONNX Runtime output 비교 실패
- NaN/Inf loss 발생
- DataLoader rebuild 또는 Close Mosaic smoke 실패
- label-changing augmentation visual audit 실패
- W6 P2 적용 후 output/NMS 비용 증가가 허용 범위를 초과
- optional 실험에서 route/channel/export 오류 발생

## 5. 현재 착수 위치

현재 구현 시작점은 `1.3.1-P1`이다.

우선 처리할 항목:
- `train.py`, `train_aux.py`, `test.py`, `export.py` CLI alias 정리
- `utils/datasets.py` 일반 YOLO layout label mapping 복구
- label cache hash/version invalidation 복구
- `persistent_workers`를 `workers > 0`일 때만 활성화하고 Close Mosaic과 충돌하지 않게 정리

`1.3.1-P1`이 끝나기 전에는 loss, head, augmentation, W6 구조 변경을 시작하지 않는다.
