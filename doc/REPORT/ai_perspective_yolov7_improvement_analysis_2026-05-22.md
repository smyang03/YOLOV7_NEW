# AI 관점 YOLOv7 개선 방식 분석 리포트

## 문서 정보

- 작성일: 2026-05-22
- 기준 문서: `doc/BASIC/YOLOv7_Custom_Design_Spec_v1.3.md`, `doc/PLAN/development_plan_v1.3.md`
- 분석 관점: AI 모델 개선 전략, 실험 분리성, 운영 제약, 실패 원인 추적성

## 1. 이 레포지토리를 개선하는 이유

이 레포지토리의 개선 목적은 단순히 YOLOv7을 최신 구조로 바꾸는 것이 아니다. 핵심은 CCTV 환경에서 발생하는 실제 실패 유형을 줄이는 것이다.

- 소형 객체, 안전모, 부분 가림, 역광, IR/흑백, 압축 노이즈에 대한 미감지 감소
- CCTV 배치 환경에서 오감지와 희귀 클래스 recall 문제 개선
- 기존 YOLOv7-L/W6의 추론 속도와 배포 안정성을 크게 해치지 않는 범위에서 mAP 향상
- 파인튜닝 시 기존 클래스 성능이 무너지는 catastrophic forgetting 억제
- 학습, 검증, export 결과를 재현 가능한 산출물로 남기는 운영형 학습 파이프라인 확보

즉, 이 프로젝트는 “모델을 더 크게 만드는 작업”이 아니라 “현장 도메인에서 실패하는 지점을 좁혀서 고치는 작업”에 가깝다.

## 2. AI가 AI를 개선한다는 의미

AI가 AI를 개선한다는 것은 모델 구조를 감각적으로 바꾸는 것이 아니다. 현재 문서의 방향은 다음 루프에 가깝다.

1. 실패 환경을 관찰한다.
2. 실패 원인을 데이터, 학습 정책, loss, head, neck, export 중 하나로 분해한다.
3. 하나의 가설만 켠다.
4. 동일 validation set과 동일 측정 기준으로 비교한다.
5. 성능, GFLOPs, export, 안정성 조건을 통과하지 못하면 되돌린다.

이 방식은 AI 모델 개발에서 가장 중요한 원인 추적성을 확보한다. 여러 개선을 한 번에 넣으면 mAP가 올라도 무엇이 효과였는지 알 수 없고, mAP가 떨어져도 어디서 망가졌는지 찾기 어렵다. 현재 v1.3은 이 문제를 플래그 기반 구현과 차수별 학습 실행으로 줄이려는 설계다.

## 3. 현재 개선 방식의 적합성

현재 방향은 전반적으로 적합하다.

YOLOv7-L은 640×384 입력의 속도형 모델로 유지하고, YOLOv7-W6는 1280×736 입력의 정확도형 모델로 분리한 판단이 맞다. 두 모델의 역할이 다른데 동일한 구조 개선을 강제로 적용하면 L은 latency를 잃고, W6는 소형 객체 개선 기회를 놓칠 수 있다.

Backbone을 유지하는 것도 합리적이다. 현재 목표는 feature extractor 자체의 세대 교체가 아니라 CCTV 도메인 적응이다. Backbone을 바꾸면 pretrained weight, feature route, export 안정성, GFLOPs 기준이 동시에 흔들린다. 따라서 우선 head, loss, assignment, augmentation, phase schedule을 개선하는 순서가 더 안전하다.

W6에 P2 Anchor와 SCDown을 우선 적용하고, PSA/FCOS/GELAN을 후순위로 둔 것도 적합하다. P2 Anchor는 기존 YOLO output 체계와 맞기 때문에 FCOS보다 후처리와 export 리스크가 낮다. SCDown은 convolution 계열이라 TensorRT 계열 변환 리스크가 attention 계열보다 낮다.

## 4. 강점

- Baseline과 export 기준선을 먼저 고정한다.
- 학습 루프, 로그, phase 전환을 구조 변경보다 먼저 안정화한다.
- L/W6의 역할을 분리해 속도형과 정확도형 목표가 섞이지 않는다.
- WIoU, TAL, VFL을 fallback 가능한 옵션으로 둔다.
- CCTV augmentation을 구조 개선과 분리해서 검증한다.
- P2, SCDown, PSA, FCOS, GELAN을 우선순위로 나눠 원인 추적성을 확보한다.
- raw ONNX output을 기본으로 두어 모델 export 검증과 C++/TensorRT 후처리 문제를 분리한다.
- 파인튜닝은 Replay Buffer를 우선하고 LwF를 A/B로 두어 망각 억제를 단계화한다.

## 5. 주요 위험

첫 번째 위험은 개선 항목 간 상호작용이다. WIoU, TAL, VFL, Decoupled Head는 각각은 타당하지만 동시에 켜면 loss scale과 positive assignment가 크게 바뀐다. 반드시 한 항목씩 켜고 `loss_detail.csv`, positive 수, mAP 변화를 같이 봐야 한다.

두 번째 위험은 augmentation label pollution이다. Patch-Paste, Hard Negative, SpiderWeb 계열은 CCTV 도메인에 맞지만, bbox 잘림이나 가려짐 처리 기준이 부정확하면 모델이 잘못된 라벨을 학습한다. 증강은 시각 검증과 label check 도구가 먼저 있어야 한다.

세 번째 위험은 W6 P2 output 증가다. P2는 소형 객체 recall에 유리하지만 output box 수와 NMS 비용을 늘린다. 현재 C++/TensorRT runtime은 제외되어 있으므로 Python 기준 NMS 비용과 ONNX raw output shape를 먼저 관리해야 한다.

네 번째 위험은 phase 전환 구현이다. Close Mosaic은 단순히 `dataset.mosaic=False`만 바꾸면 worker 내부 dataset 상태가 남을 수 있다. DataLoader rebuild와 persistent worker 정책이 요구서대로 구현되어야 한다.

다섯 번째 위험은 문서 저장 규칙과 기존 문서 위치의 차이다. `AGENTS.md`는 신규 개발 문서를 `doc/PLAN/`에 작성하도록 정리되었지만, 현재 작성된 1.3.1~1.3.7 코드레벨 요구서는 `doc/dev/`에 남아 있다. 구현 착수 전 문서 위치 정책을 실제 파일 구조에도 맞출지 결정하는 것이 좋다.

## 6. AI 관점의 개선 제안

현 방식은 “한 번에 강한 모델을 만드는 방식”보다 “실패 원인을 분해해 누적 개선하는 방식”이므로 더 안전하다. 다만 구현 순서는 더 엄격해야 한다.

1. `1.3.1-P1`에서 데이터 경로, cache, worker, CLI alias를 먼저 고친다.
2. `1.3.1-P2`에서 checkpoint, resume, 결과 파일 parse 안정성을 고친다.
3. `1.3.1-P3`에서 ONNX raw export와 PyTorch/ONNX Runtime 비교를 고정한다.
4. `1.3.2`에서 phase/logging/rebuild 기반을 만든다.
5. `1.3.3`부터는 head, loss, assign을 반드시 개별 플래그로 하나씩 학습한다.
6. `1.3.4` augmentation은 visual audit 없이는 본 학습에 넣지 않는다.
7. `1.3.5` W6 구조 확장은 SCDown only, P2 only, P2+SCDown 순서로 검증한다.

## 7. 최종 판단

현재 개선 방식은 AI 모델을 개선하는 방식으로 적합하다. 특히 baseline, 단계별 플래그, 동일 validation 비교, GFLOPs 10% 제한, raw ONNX 검증, fallback 조건을 둔 점이 좋다.

이 프로젝트의 핵심은 “YOLOv7을 최신 논문 요소로 덮어쓰기”가 아니다. 핵심은 CCTV 도메인에서 실패하는 조건을 데이터와 학습 체계로 먼저 잡고, 그다음 손실 함수와 head 구조를 제한적으로 바꾸며, W6에만 소형 객체 구조 개선을 적용하는 것이다.

따라서 지금은 추가 아이디어를 더 넣는 단계가 아니라, `1.3.1-P1`부터 실제 코드 구현을 시작해 기준선을 흔들지 않는 것이 맞다.
