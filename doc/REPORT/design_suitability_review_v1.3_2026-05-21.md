# v1.3 개발 요구서 적합성 재검토

- 대상 문서: `doc/BASIC/YOLOv7_Custom_Design_Spec_v1.3.md`
- 검토일: 2026-05-21
- 검토 관점: 개발 착수 적합성, 개선 방향 타당성, 단계별 원인 분리 가능성

## 총평

v1.3의 방향은 전반적으로 맞다. 특히 baseline 고정, GFLOPs 10% 예산, Phase/DataLoader 검증, `results.csv` 중심 로그, 단계별 착수 순서를 넣은 점은 개발 착수 문서로 적합하다.

다만 아직 "한 번에 여러 기법을 넣지 않는다"는 원칙과 충돌하는 항목이 남아 있다. 또한 CCTV 운영 목표인 오감지/미감지는 mAP만으로 판정하기 어렵다. 아래 P1 항목은 구현 전에 문서에 반영하는 것이 좋다.

## 주요 검토 의견

### [P1] 실험 A 정의가 단계별 원인 분리 원칙과 충돌함

- 관련 위치: `9.2 A/B 실험 플래그`, `9.3 단계별 개발 착수 순서`
- `9.2`의 실험 A는 `Decoupled Head + WIoU v3 + TAL + VFL`을 한 번에 묶는다.
- 반면 `9.3`은 `5A Head`, `5B WIoU`, `5C TAL+VFL`로 분리한다.
- 현재 상태로는 실험표와 착수 순서가 서로 다른 기준을 말한다.

권장:
- 실험 A를 `A1 Decoupled Head`, `A2 WIoU`, `A3 TAL+VFL`로 분리한다.
- `10.1 기대 성능 향상`의 실험 구분도 A1/A2/A3로 맞춘다.

### [P1] GFLOPs 10%만으로 속도 목표를 판정하기 부족함

- 관련 위치: `1.4`, `9.3`, `10.1`, `13.5`
- GFLOPs는 좋은 1차 게이트지만 TensorRT latency, memory bandwidth, output box 수, C++ NMS 비용을 대변하지 못한다.
- 특히 W6 P2 Anchor는 GFLOPs보다 `total_boxes`와 NMS latency 증가가 더 큰 문제가 될 수 있다.

권장:
- L/W6 각각 hard latency gate를 추가한다.
- 최소 기록 항목: `preprocess_ms`, `trt_infer_ms`, `decode_ms`, `nms_ms`, `total_ms`, `peak_memory`.
- P2 Anchor 단계는 GFLOPs와 별도로 `total_boxes` 및 NMS latency 상한을 둔다.

### [P1] CCTV 운영 목표가 metric으로 충분히 연결되지 않음

- 관련 위치: `1.2`, `1.4`, `5`, `10.1`
- 오감지 억제 대상은 나뭇가지/거미줄/그림자이고, 미감지 억제 대상은 소형 객체/역광/안전모/부분 가림이다.
- 현재 primary metric은 `mAP@0.5:0.95`라서 전체 성능 판단에는 좋지만, 운영 리스크를 직접 측정하지 못한다.

권장:
- validation set을 scenario subset으로 나눈다: `hard_negative`, `small_object`, `backlight`, `helmet`, `occlusion`, `IR/night`.
- 추가 지표를 명시한다: hard negative FP/image, scenario recall, small AP, rare class recall.
- 단계별 결과표에 scenario metric columns를 추가한다.

### [P1] Augmentation 단계가 모델/손실 변경보다 앞서 있어 원인 분리가 흐려질 수 있음

- 관련 위치: `9.3 단계별 개발 착수 순서`
- 현재 Stage 4에서 CCTV Aug, Patch-Paste, Hard Negative, Weighted Sampler가 먼저 들어가고, 이후 Stage 5A~5C에서 Head/Loss/Assignment가 들어간다.
- Label-changing augmentation과 sampler가 먼저 들어가면 뒤의 모델 개선 효과를 분리하기 어렵다.

권장:
- Stage 4를 두 단계로 나눈다.
- `4A`: label을 바꾸지 않는 pixel-level aug만 적용한다.
- `4B`: Patch-Paste, Hard Negative, Weighted Sampler는 Head/Loss 기준선이 안정화된 뒤 별도 실험으로 넣는다.

### [P2] Phase 2의 `rect=True + mosaic=True`는 별도 검증 조건이 필요함

- 관련 위치: `6.1`, `6.2`
- Phase 2의 목적은 TensorRT 입력 해상도 적응인데, Mosaic ON이 유지되면 실제 추론 분포 적응 효과가 약해질 수 있다.
- 문서에는 유지 이유가 있지만, 성능/분포 관점의 비교 조건은 없다.

권장:
- Phase 2에서 `rect=True, mosaic=True`와 `rect=True, mosaic=False`를 짧은 A/B로 비교한다.
- 기준은 primary mAP, small AP, final Phase 3 안정성으로 둔다.

### [P2] Baseline 고정 단계가 너무 무거울 수 있음

- 관련 위치: `9.3 단계별 개발 착수 순서`
- Stage 0이 원본 L/W6 전체 학습 재현으로 되어 있어, 80만장 기준 개발 착수 초기에 시간이 많이 든다.

권장:
- `0A smoke baseline`: 소형 subset, 1~3 epoch, train/eval/export 통과 확인
- `0B full baseline`: 전체 validation 기준 수치 고정

### [P2] AUX freeze 의미가 더 명확해야 함

- 관련 위치: `3.3`, `4.2`, `6.2`
- `λ_aux=0.0 (AUX freeze)`는 loss 비활성인지, branch parameter freeze인지 애매하다.

권장:
- Phase 2/3에서 AUX branch를 `loss off`, `requires_grad=False`, `export off` 중 어떤 상태로 둘지 표로 명시한다.

### [P2] 클래스 불균형의 `cls_pw`가 per-class인지 scalar인지 불명확함

- 관련 위치: `4.4`
- YOLO 계열의 `cls_pw`는 보통 BCE positive weight scalar로 쓰인다.
- 희귀 클래스별 2.0~3.0을 의도한다면 per-class weight 구현이 필요하다.

권장:
- scalar `cls_pw`와 per-class class weight를 구분한다.
- per-class weight를 쓸 경우 loss tensor shape, config key, logging 항목을 명시한다.

## 결론

v1.3은 개발 착수 문서로 방향이 맞다. 다만 구현 전 마지막 보정으로 실험 A 분리, latency gate 추가, CCTV scenario metric 추가, augmentation 단계 분리만 반영하면 원인 추적이 훨씬 쉬워진다.
