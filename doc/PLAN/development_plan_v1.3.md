# YOLOv7 Custom Development Plan v1.3

- 기준 요구서: `doc/BASIC/YOLOv7_Custom_Design_Spec_v1.3.md`
- 작성일: 2026-05-21
- 원칙: 코드는 플래그 기반으로 통합 구현하고, 학습 서버에서는 차수별로 하나씩 기능을 켜서 검증한다.

## 1.3.1 Baseline / Python Export 기준선

목표: 원본 YOLOv7-L/W6의 기준값을 고정하고 Python-side ONNX export 검증 경로를 먼저 확보한다. C++/TensorRT runtime/추론 서버 구현은 본 차수에서 제외한다.

개발 범위:
- 원본 L/W6 학습, 평가, ONNX export 재현
- raw ONNX export
- PyTorch/ONNX Runtime output 비교용 `tools/verify_export.py` 작성
- params/GFLOPs 측정용 `tools/profile_model.py` 작성

산출물:
- baseline `best.pt`
- `results.txt`
- `profile.json`
- `export_check.json`
- validation set checksum

통과 기준:
- PyTorch/ONNX Runtime output 비교 통과
- baseline mAP, params/GFLOPs 기록 완료

## 1.3.2 Train Loop / Phase / Logging 기반

목표: 통합 학습 루프와 계측 기반을 만든다. 모델 구조 변경은 아직 하지 않는다.

개발 범위:
- `train.py`와 `train_aux.py` 통합
- Phase 1/2/3 자동 전환
- Rect finetune
- DataLoader rebuild
- Close Mosaic worker 정책
- `results.csv` canonical log
- `tools/profile_model.py`

산출물:
- `phase_transition.log`
- `stage_result.yaml`
- `results.csv`
- `profile.json`

통과 기준:
- epoch boundary test 통과
- `workers=0`과 `workers>0` smoke test 통과
- GFLOPs delta 자동 계산 가능

## 1.3.3 Core Model / Loss 분리 적용

목표: 모델 핵심 개선을 한 번에 넣지 않고 원인 분리 가능하게 순차 적용한다.

개발 범위:
- `--head decoupled`
- `--loss-box wiou_v3`
- `--assign tal`
- `--loss-cls vfl`
- CIoU/BCE/SimOTA fallback 유지
- WIoU state checkpoint/resume 저장

실행 순서:
1. Head만 변경
2. WIoU만 변경
3. TAL + VFL 변경

통과 기준:
- NaN/Inf loss 없음
- GFLOPs 증가 10% 미만
- primary mAP 하락 없음
- 각 단계 export 비교 통과

## 1.3.4 CCTV Augmentation / Sampler

목표: 데이터 증강 효과를 모델 구조 효과와 분리해서 검증한다.

개발 범위:
- pixel-level CCTV aug
- SpiderWeb / IR 반사 / 역광 / 저조도 / blur 계열
- Patch-Paste 안전장치
- Hard Negative Paste
- Weighted sampler
- `tools/check_aug_visual.py`

실행 순서:
1. label을 바꾸지 않는 pixel aug
2. label-changing aug
3. sampler

통과 기준:
- aug 시각 검증 통과
- bbox/class label 오염 없음
- smoke training 정상
- hard negative FP/image 개선 확인

## 1.3.5 W6 구조 확장

목표: W6 정확도형 모델의 소형 객체 성능을 구조적으로 개선한다.

개발 범위:
- W6 P2 Anchor Head
- SCDown
- output 증가에 따른 box 수와 Python NMS 비용 추정
- memory 사용량 측정

통과 기준:
- small AP/recall 개선
- GFLOPs 증가 10% 미만
- Python/ONNX export 검증 통과
- output box 수 증가가 허용 범위 내

## 1.3.6 Optional / 후순위 실험

목표: 필수 구성이 안정화된 뒤 선택 실험을 진행한다.

개발 범위:
- L AUX on 성능형 옵션
- PSA P5
- FCOS P2
- GELAN

진입 조건:
- 앞 단계 목표 미달
- latency/GFLOPs 여유 있음
- export 기준선 안정

중단 기준:
- 효과 미미
- 학습 불안정
- ONNX export 실패
- output shape 또는 후처리 복잡도 과다

## 1.3.7 Fine-tuning / Continual Learning

목표: scratch 학습 기준선 확정 후 catastrophic forgetting 억제 파이프라인을 구현한다.

개발 범위:
- Replay Buffer
- Pseudo Label 생성/병합
- YOLO LwF A/B
- `finetune.py`
- `utils/continual_loss.py`
- `utils/replay_buffer.py`
- `utils/pseudo_label.py`

통과 기준:
- 파인튜닝 대상 클래스 mAP 유지
- 기존/미포함 클래스 mAP 하락 억제
- teacher는 학습 시에만 사용하고 최종 모델 구조에는 영향 없음

## 차수 진행 규칙

각 차수는 이전 차수의 산출물이 있어야 시작한다. 실패 시 다음 차수로 넘어가지 않고 해당 차수의 플래그를 끈 뒤 직전 성공 차수의 config와 weight를 기준으로 재시작한다.

공통 중단 조건:
- primary mAP가 baseline 대비 2 percentage points 이상 하락
- GFLOPs 증가율 10% 이상
- ONNX export 또는 PyTorch/ONNX output 비교 실패
- NaN/Inf loss 발생
- DataLoader rebuild 또는 Close Mosaic 테스트 실패
- label-changing aug 시각 검증 실패
