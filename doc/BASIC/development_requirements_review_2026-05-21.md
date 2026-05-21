# YOLOv7 커스텀 학습 시스템 개발 요구서 검토

- 원본 문서: `doc/BASIC/YOLOv7_Custom_Design_Spec_v1.2_final.md`
- 검토일: 2026-05-21
- 검토 관점: 요구사항 내부 일관성, 구현 착수 가능성, 수용 기준, 운영 리스크

## 반영된 정리

다음 항목은 요구서 본문에 반영해 검토 이슈에서 제외했다.

- ONNX/TensorRT 입력 해상도는 YOLO stride 및 TensorRT 호환을 위한 32 배수 입력으로 정리했다. `640x360`, `1280x720`은 소스 해상도이고, 실제 TensorRT 입력은 `640x384`, `1280x736`으로 본다.
- 속도 기준은 기존 모델 대비 GFLOPs 증가 10% 미만으로 정리했다. TensorRT FP16 latency는 별도 실측 지표로 병행한다.
- Backbone은 weight freeze가 아니라 구조 원본 유지로 정리했다. 향후 feature 추출 위치 변경은 후순위 실험으로만 둔다.
- Warmup은 별도 Phase가 아니라 Phase 1 내부 상태로 정리했고, 문서 epoch는 1-based, 구현 루프는 0-based end-exclusive 기준으로 명시했다.
- 파인튜닝의 15개/5개/10개 클래스 수량은 예시로 명시했다. 실제 클래스 수량과 mapping은 `data/*.yaml` 기준이다.
- DataLoader rebuild, cache invalidation, persistent worker, sampler/DDP/seed 정책을 요구서에 추가했다.

## 남은 검토 의견

### [P1] GFLOPs 10% 예산을 자동 검증해야 함

- 관련 섹션: `1.2`, `10.1`
- 속도 기준이 "기존 모델 대비 GFLOPs 증가 10% 미만"으로 정리되었으므로, L/W6 각각 baseline GFLOPs를 고정하고 변경 모델의 GFLOPs를 같은 입력 크기에서 산출해야 한다.
- 수동 기록만으로는 실험별 예산 초과를 놓치기 쉽다.

권장:
- `tools/profile_model.py` 또는 동등한 스크립트로 params/GFLOPs를 출력한다.
- L은 `640x384`, W6는 `1280x736` 입력 기준으로 baseline과 변경 모델을 비교한다.
- 실험 결과표에 `baseline_gflops`, `current_gflops`, `delta_percent`를 저장한다.

### [P1] Phase 전환 구현 기준을 테스트로 고정해야 함

- 관련 섹션: `4.2`, `5.4`, `6.1`, `6.2`, `6.3`
- 요구서에서 Warmup/Phase/0-based 구현 기준은 정리되었지만, 실제 구현 시 off-by-one 오류가 발생하기 쉽다.
- Phase 2/3 전환 시 hyp, rect, mosaic, dataloader rebuild가 정확히 한 번만 실행되는지 확인해야 한다.

권장:
- epoch 0, 29, 30, 289, 290, 359, 360 기준의 Phase 상태 unit test를 추가한다.
- Phase 전환 시 `phase_transition.log`에 변경 전/후 설정을 기록한다.
- Close Mosaic 진입 시 worker가 새 Dataset을 쓰는지 테스트한다.

### [P1] DataLoader rebuild 정책은 구현 난도가 높음

- 관련 섹션: `6.3.1`
- 요구서에 정책은 추가되었지만, 80만장 규모에서는 cache rebuild 비용, DDP sampler, persistent worker 재시작이 학습 안정성에 직접 영향을 준다.
- 특히 Close Mosaic 시 parent dataset만 바꾸는 구현은 worker 내부 dataset에 반영되지 않는다.

권장:
- Phase 전환 시 train_loader 객체 id, dataset.mosaic, worker 재시작 여부를 로그로 남긴다.
- `workers=0`과 `workers>0` 양쪽 smoke test를 둔다.
- label cache hash/version 불일치 시 자동 재생성 테스트를 추가한다.

### [P2] A/B 실험 플래그와 실험 번호가 맞지 않음

- 관련 섹션: `9.2`, `10.2`
- 문서에는 `A/B 실험 플래그 7개`라고 되어 있지만 실제 표에는 8개 플래그가 있다.
- 실험 표는 A~F까지만 정의되어 있는데, export 검증 주석은 `E/F/G/H/I` 구조 변경을 언급한다.

권장:
- 플래그 개수와 실험 ID를 실제 표와 맞춘다.
- 실험별 목적, 성공 기준, 중단 기준을 한 표로 정리한다.

### [P2] mAP 향상 목표의 측정 기준이 부족함

- 관련 섹션: `1.2`, `10.1`, `13.5`
- `+6~12%`가 absolute percentage point인지 relative improvement인지 명확하지 않다.
- mAP@0.5 기준인지, mAP@0.5:0.95 기준인지, 전체 클래스 평균인지, 희귀/소형 클래스 가중 평균인지도 고정되어 있지 않다.

권장:
- primary metric: 예) `mAP@0.5:0.95`
- secondary metric: `mAP@0.5`, 소형 객체 AP, 희귀 클래스 recall
- baseline checkpoint와 validation set checksum 기록
- 목표는 `+N percentage points`처럼 단위를 명시한다.

### [P2] 로그 요구사항이 대규모 데이터셋에서 과할 수 있음

- 관련 섹션: `7.1`, `7.2`, `7.3`
- 매 epoch `Train P/R/mAP`, per-class metric, speed를 모두 출력하는 것은 80만장 데이터셋에서 비용이 크다.
- 기존 YOLO 계열의 `results.txt`와 새 `results.csv` 중 어떤 파일을 canonical log로 쓸지도 정해야 한다.

권장:
- train metric은 loss 중심으로 제한하고, mAP는 validation 기준으로 계산한다.
- per-class metric 저장 주기 지정: 예) 매 epoch 또는 best 갱신 시
- `results.csv`로 전환한다면 기존 plot/read 경로도 함께 변경한다.

### [P2] Custom Augmentation의 라벨 안전장치가 더 필요함

- 관련 섹션: `5.2`, `5.3`, `5.4`
- Patch-Paste, Hard Negative Paste, 안전모 paste는 성능 향상 가능성이 있지만 라벨 오염 위험이 크다.
- 현재 요구서는 확률과 목적은 있으나 bbox clipping/filtering, paste 실패 처리, 시각 검증 기준이 부족하다.

권장:
- paste 후 bbox 최소 면적, aspect ratio, visibility 기준 명시
- GT와 paste 영역 IoU 제한 및 중복 제거 규칙 명시
- Hard Negative crop mining 방식 명시
- `tools/check_aug_visual.py`에서 저장할 샘플 수와 승인 기준 명시

### [P2] WIoU v3, TAL, VFL의 상태 관리와 fallback 기준이 부족함

- 관련 섹션: `4.1`, `4.2`, `4.3`
- WIoU v3는 `.detach()`뿐 아니라 running mean/state 관리가 중요하다.
- TAL/VFL은 positive 할당, objectness, classification target의 결합 방식이 구현마다 달라질 수 있다.
- fallback인 CIoU/BCE/SimOTA로 언제 되돌릴지 기준이 없다.

권장:
- WIoU state를 checkpoint/resume에 저장할지 명시
- TAL top-k, alpha/beta, loss normalization 기준 명시
- VFL target score 정의와 obj BCE 병행 여부 명시
- fallback 전환 조건: NaN, loss divergence, mAP 하락 기준 등

### [P2] TensorRT 환경 매트릭스가 부족함

- 관련 섹션: `1.3`, `8.3`, `8.5`
- TensorRT 8.6 / 10.x를 모두 지원한다고 되어 있지만, CUDA, cuDNN, PyTorch, ONNX Runtime, Visual Studio 버전 조합이 고정되어 있지 않다.
- `build_trt.py`는 버전 자동 감지를 한다고 되어 있으나 예시 명령에는 `--trt-version`을 직접 넘긴다.

권장:
- 개발/검증 환경 matrix 작성
- `--trt-version` 옵션은 override인지 필수값인지 명시
- TensorRT 8.6과 10.x 각각의 최소 smoke test 정의

## 착수 전 확인 항목

1. L/W6 baseline GFLOPs와 10% 예산 계산 방식
2. Phase 전환 테스트 기준
3. DataLoader rebuild/cache/sampler/worker 구현 테스트
4. A/B 실험표와 fallback/중단 기준
5. mAP primary metric과 validation set checksum
6. 로그 파일 canonical format과 metric 계산 주기

## 결론

요구서의 큰 방향은 개발 착수 가능한 수준으로 정리되었다. 남은 리스크는 구현 중 흔히 어긋나는 수치 기준과 자동화 정책이다. 특히 GFLOPs 예산, Phase 전환, DataLoader rebuild는 코드 작업 초기에 테스트 기준까지 같이 고정하는 편이 좋다.
