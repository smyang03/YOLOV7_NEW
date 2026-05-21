# YOLOv7 커스텀 학습 시스템 설계 요구서

## 문서 정보

| 버전 | v1.3 |
| --- | --- |
| 작성일 | 2026. 05. 21 |
| 대상 모델 | YOLOv7-L / YOLOv7-W6 |
| 학습 환경 | 최종 개발 착수안 확정 (요구서 검토 반영) |
| 추론 환경 | TensorRT 8.6 / 10.x (Windows) |
| 상태 | v1.3 개발 요구서 확정 |

## 1. 프로젝트 개요

### 1.1 목적

본 설계 요구서는 YOLOv7-L 및 YOLOv7-W6 모델을 기반으로, CCTV 영상 환경에 특화된 고성능 객체 탐지 시스템을 구축하기 위한 커스텀 학습 파이프라인 전체를 정의한다.

속도를 유지하면서 탐지 성능(mAP)을 향상시키는 것을 핵심 목표로 하며, 오감지 및 미감지 억제를 위한 CCTV 특화 Augmentation, 최신 학습 기법 통합, 단일 실행 자동화 학습 흐름을 포함한다.

### 1.2 핵심 목표

| 구분 | 목표 | 비고 |
| --- | --- | --- |
| 성능 | 탐지 mAP 향상 (+6~12 percentage points 예상) | Primary metric 기준 |
| 속도 | 기존 모델 대비 GFLOPs 증가 10% 미만 | L/W6 각각 기존 모델 GFLOPs 기준, TRT FP16 실측 병행 |
| 오감지 | 나뭇가지, 거미줄, 그림자 오감지 억제 | Hard Negative 학습 |
| 미감지 | 소형 객체, 역광, 안전모, 부분 가림 | W6 P2 Anchor 확정, FCOS P2 후순위 |
| 자동화 | 단일 실행으로 전체 학습 완결 | 3단계 자동 전환 |
| 호환성 | TRT 8.6 / 10.x 양쪽 export | 버전 자동 감지 |

### 1.3 환경 확정

| 항목 | 내용 |
| --- | --- |
| 베이스 모델 | YOLOv7 원본 repo (L / W6) |
| 학습 GPU | RTX A6000 48GB |
| 추론 환경 | Windows + TensorRT 8.6 / 10.x |
| 개발 환경 | Visual Studio 2019/2022, C++/Python |
| 학습 해상도 | 640×640 (1차) → 640×384 / 1280×736 (3차, TensorRT 32배수 입력 기준) |
| 추론 소스 해상도 | YOLOv7-L: 640×360 / YOLOv7-W6: 1280×720 |
| TensorRT 입력 해상도 | YOLOv7-L: 640×384 / YOLOv7-W6: 1280×736 (stride 32 배수 letterbox 입력) |
| 데이터셋 | 80만장 이상, RGB + IR 혼재 |
| 클래스 수 | 예: 5~15개 (불균형 있음, 실제 수량은 data yaml 기준) |
| 카메라 유형 | RGB 카메라 + IR 카메라 (혼재, 분리 불가) |

### 1.4 성능 및 수용 기준

| 항목 | 기준 | 기록 위치 |
| --- | --- | --- |
| Primary metric | `mAP@0.5:0.95` | `results.csv` |
| Secondary metric | `mAP@0.5`, 소형 객체 AP, 희귀 클래스 recall | `results.csv`, `results_per_class.csv` |
| 성능 향상 단위 | percentage points 기준 | baseline 대비 delta 기록 |
| 속도/연산량 | 기존 모델 대비 GFLOPs 증가 10% 미만 | `tools/profile_model.py` 결과 |
| 검증 데이터 | 동일 validation set 고정 | dataset path, image count, label count, checksum 기록 |
| Export 검증 | PyTorch/ONNX Runtime output 비교 | `tools/verify_export.py` 결과 |

## 2. 학습 코드 통합 설계

### 2.1 기존 구조 문제

YOLOv7 원본은 학습 코드가 두 개로 분리되어 있다.

train.py: YOLOv7-L 등 기본 모델 학습

train_aux.py: YOLOv7-W6 등 AUX Head 포함 모델 학습

두 파일 분리로 인한 문제:

L 모델에 AUX Head 추가 시 train_aux.py 사용 불가 (W6 전용 구조 하드코딩)

3단계 자동 전환(Phase 1→2→3) 구현 시 두 파일 동기화 필요

유지보수 이중화

### 2.2 통합 train.py 설계

단일 train.py로 완전 통합한다. AUX Head 포함 여부는 cfg 파일 기반으로 자동 감지한다.

| 기능 | 설계 방식 |
| --- | --- |
| 모델 타입 감지 | cfg 파일명 또는 레이어 구조로 L / W6 자동 판별 |
| AUX Head 감지 | model.yaml 내 aux 키 존재 여부로 자동 활성/비활성 |
| Phase 자동 전환 | epoch 기반 Phase 1→2→3 자동 분기 (단일 루프) |
| DataLoader rebuild | Phase 전환 시 해상도/rect 설정 변경 후 자동 재생성 |
| hyp 자동 전환 | Phase 진입 시 hyp_phase1.yaml / hyp_phase3.yaml 자동 로드 |
| Loss 분기 | AUX 감지 시 AUX Loss 자동 추가, Phase 3에서 자동 비활성 |

### 2.3 실행 명령어

최종 개발 착수 기본값은 L 경량형 / W6 공격형으로 분리한다. L은 속도 유지 목적상 AUX/P2/Neck 수정 없이 가볍게 두고, W6는 WIoU + TAL/VFL + P2 Anchor + SCDown을 기본 적용한다.

# YOLOv7-L 기본 실행 (경량형 확정: AUX off / P2 off / Neck 원본)

python train.py \

--cfg cfg/yolov7-l-custom.yaml \

--data data/custom.yaml \

--hyp data/hyp_phase1.yaml \

--epochs 400 --img 640 --batch 32 --workers 8 --device 0 \

--name exp_l_baseline \

--phase1-epochs 290 --phase2-epochs 70 --phase3-epochs 40 \

--rect-size-l 640 384 \

--head decoupled --aux off --p2-head none --neck-mod none \

--loss-box wiou_v3 --loss-cls vfl --assign tal \

--nms-mode none

# YOLOv7-W6 기본 실행 (공격형 확정: AUX on / P2 Anchor / SCDown)

python train.py \

--cfg cfg/yolov7-w6-custom.yaml \

--data data/custom.yaml --hyp data/hyp_phase1.yaml --epochs 400 --img 640 \

--batch 8 --grad-accumulate 4 --rect-size-w6 1280 736 \

--head decoupled --aux on --p2-head anchor --neck-mod scdown \

--loss-box wiou_v3 --loss-cls vfl --assign tal \

--nms-mode none

## 3. 아키텍처 설계

### 3.1 전체 구조 개요

| 영역 | 구성 요소 | 변경 여부 | 비고 |
| --- | --- | --- | --- |
| Backbone | YOLOv7-L / W6 원본 | 구조 고정 | weight freeze 의미 아님. 향후 feature 추출 위치 변경 실험 전까지 Backbone 구조는 변경하지 않음 |
| Neck | L: 원본 ELAN Neck / W6: ELAN + SCDown | 분리 적용 | PSA P5는 2차 후보, GELAN은 최후순위 |
| P2 경로 | W6 P2 Anchor Head | 신규 확정(W6) | L 제외, FCOS P2는 후순위 |
| AUX Head | P4 중간 피처 분기 | 모델별 분리 | W6 on 확정 / L off 기본, 성능형 옵션 |
| 앵커 헤드 | Decoupled Head (P3/P4/P5) | 수정 | cls/reg/obj 분리 |
| 앵커프리 헤드 | FCOS Head (P2) | 후순위 | W6 P2 Anchor 효과 부족 시만 검토 |

### 3.2 Neck 설계

#### 3.2.1 GELAN 부분 교체 (최후순위)

최후순위 실험 항목으로 유지한다. W6에서도 채널/concat/route 안정성 검증 후 단독 적용한다.

초기 개발 착수안에서는 L/W6 모두 GELAN을 기본 적용하지 않는다.

GELAN 적용 시 기존 Neck 원본 대비 mAP/latency/export 결과를 별도 기록한다.

GELAN은 FCOS P2 이후에도 추가 성능이 필요할 때만 검토한다.

#### 3.2.2 PSA (Partial Self-Attention, W6 2차 후보)

W6 2차 확정 후보로만 유지한다. 기본 착수안에는 포함하지 않는다.

적용 순서는 P5 단독 → P4 추가 → P3 추가이며, P3/P4/P5 동시 적용은 금지한다.

TRT FP16 latency와 profile별 편차를 반드시 측정한다.

L 모델에는 PSA를 적용하지 않는다.

#### 3.2.3 SCDown (Stride Convolution Downsampling, W6 확정)

W6 공격형 기본 구성으로 적용한다. L 모델에는 적용하지 않는다.

Conv 계열 연산으로 TensorRT 리스크가 낮고, W6 고해상도 추론에서 효율 개선을 기대한다.

적용 후 raw output ONNX/TRT 변환 및 latency를 중간 검증한다.

#### 3.2.4 P2 경로 (W6 P2 Anchor 확정)

W6는 P2 Anchor Head를 기본 적용한다. L은 P2를 제외한다.

W6 1280×736 기준 stride=4 P2 feature를 소형 객체 검출에 사용한다.

FCOS P2는 후순위로 두고, 먼저 P2 Anchor Head로 성능/속도 균형을 확인한다.

P2 추가 후 total_boxes 증가량과 Python 기준 NMS 비용 추정치를 함께 측정한다. C++ NMS는 별도 요청 전까지 구현 범위에서 제외한다.

### 3.3 AUX Head 설계 (L / W6 공통)

AUX Head는 W6에서 기본 ON으로 유지한다. L 모델은 속도형/경량형 역할을 유지하기 위해 기본 OFF로 두며, 성능형 옵션으로만 별도 실험한다.

| 항목 | 내용 |
| --- | --- |
| 연결 위치 | Neck PAN 중간 P4 스케일 ELAN 출력 |
| L 모델 | AUX off 기본 / 성능형 옵션으로 --aux on 실험 가능 |
| W6 모델 | AUX on 확정 / 기존 train_aux 계열 설계 유지 |
| Coarse-to-Fine | Lead Head 예측 기반 soft label → AUX: coarse / Main: fine |
| 희귀 클래스 효과 | L 성능형 옵션에서만 검증. W6는 희귀/소형 클래스 recall 보조 목적으로 유지 |
| Loss 가중치 | λ_aux = 0.25 (Phase 1~2), 0.0 (Phase 3) |
| 추론 시 | 브랜치 완전 제거 → 속도 영향 없음 |
| ONNX export | model.eval() + AUX 비활성 후 trace |

### 3.4 Head 설계

#### 3.4.1 Decoupled Head (앵커 헤드, P3/P4/P5)

cls branch: Conv3×3 → 클래스 분류

reg branch: Conv3×3 → 박스 회귀

obj branch: Conv1×1 → 객체 유무

cls/reg 분리로 학습 충돌 제거 → 특히 소형/희귀 클래스 효과

추론 속도 영향: ±2ms (TRT FP16 기준)

#### 3.4.2 P2 Head 설계 (W6 P2 Anchor 확정 / FCOS 후순위)

W6 기본: P2 Anchor Head를 적용한다. 기존 YOLO 출력 체계와 맞춰 obj/cls/reg 분기를 유지한다.

L 기본: P2 Head를 적용하지 않는다. L은 640×360 속도형 모델로 유지한다.

FCOS P2는 후순위다. P2 Anchor 적용 후에도 소형 객체 recall이 부족할 때만 검토한다.

W6 P2 Anchor 적용 시 output 증가에 따른 NMS 속도와 memory 사용량을 반드시 측정한다.

FCOS P2 적용 시 centerness/obj score 결합, raw output decode, C++ postprocess를 별도 구현한다.

FCOS P2는 기본 export 경로에 포함하지 않는다.

## 4. Loss 및 Label Assignment 설계

### 4.1 Loss 구성

최종 착수 기본값은 WIoU v3 + TAL + VFL이다. CIoU + BCE + SimOTA는 fallback 기준선으로 유지한다. VFL은 반드시 TAL과 세트로만 적용한다.

| 헤드 | Loss 유형 | 기본값 | 세부 설정 | 목적 |
| --- | --- | --- | --- | --- |
| 앵커 헤드 Box | CIoU / WIoU v3 | WIoU v3 | WIoU: .detach() 필수 / CIoU fallback 유지 | 박스 회귀 정밀도 |
| 앵커 헤드 Cls | BCE / VFL | VFL | TAL 적용 시 사용, 단독 사용 금지 | 클래스 분류 |
| 앵커 헤드 Obj | BCE | BCE | focal gamma=1.5 | 객체 유무 |
| 앵커프리 Box | GIoU | GIoU | FCOS 표준 | P2 박스 회귀 |
| 앵커프리 Cls | Focal Loss | Focal | γ=1.5, α=0.5 | 소형 클래스 불균형 |
| 앵커프리 Ctr | BCE | BCE | - | Centerness 학습 |
| AUX Head | 앵커 헤드와 동일 | - | λ=0.25 고정 | 보조 gradient |

### 4.2 Loss 가중치 스케줄

| Phase | 구간 | λ_main | λ_free (앵커프리) | λ_aux |
| --- | --- | --- | --- | --- |
| 워밍업 | ep 1~30 | 1.0 | 0.1 (앵커 안정화 우선) | 0.25 |
| Phase 1 | ep 31~290 | 1.0 | 0.5 | 0.25 |
| Phase 2 | ep 291~360 | 1.0 | 0.5 | 0.0 (AUX freeze) |
| Phase 3 | ep 361~400 | 1.0 | 0.5 | 0.0 |

> ※ WIoU v3 구현 주의: 동적 가중치 계산 시 .detach() 누락 시 gradient explosion 발생

> ※ VFL과 TAL 순서: TAL이 positive 결정 → VFL이 해당 positive에 loss 계산 (순서 고정)

### 4.3 Label Assignment

| 헤드 | 방식 | 파라미터 | 특징 |
| --- | --- | --- | --- |
| 앵커 헤드 | TAL (Task-Aligned Learning) | cls^0.5 × IoU^6.0, TopK | 최종 기본값. SimOTA는 fallback 기준선 |
| 앵커프리 헤드 | Center Sampling | radius = 1.5 × stride(4) | GT 중심 반경 내 positive |

### 4.4 클래스 불균형 대응

WeightedRandomSampler: 희귀 클래스 포함 이미지 샘플링 가중치 ↑

image_weights=False (Sampler와 중복 방지, Sampler 단일화)

cls_pw (positive weight): 희귀 클래스 2.0~3.0 설정

소형 객체 IoU 가중치 보정: iou_weight = 1.0 + (1.0 - normalize_area)

### 4.5 Loss 상태 관리 및 fallback 기준

| 항목 | 기준 |
| --- | --- |
| WIoU v3 state | running mean/state를 checkpoint와 resume에 포함한다. |
| TAL 파라미터 | `topk`, `alpha`, `beta`, loss normalization 기준을 hyp에 저장한다. |
| VFL target | TAL positive의 IoU-aware score를 target으로 사용하고, TAL 없는 VFL 단독 적용은 금지한다. |
| Obj BCE 병행 | 앵커 헤드는 obj BCE를 유지하고, VFL은 cls branch에만 적용한다. |
| fallback 조건 | NaN/Inf loss 발생, 3 epoch 연속 loss divergence, baseline 대비 mAP@0.5:0.95 2 percentage points 이상 하락 시 CIoU+BCE+SimOTA로 되돌린다. |
| fallback 기록 | 전환 epoch, 원인, 직전 loss/mAP를 `phase_transition.log`와 `train_log.txt`에 남긴다. |

## 5. Augmentation 설계

### 5.1 설계 원칙

RGB + IR 혼재, 분리 불가 → 도메인 무관하게 동작하는 aug 구성

박스 기반 (마스크 없음) → Patch-Paste 방식 적용

OneOf 구조 활용으로 과도한 중첩 방지

SpiderWeb Aug는 Mosaic 이전 단일 이미지 단계에서 적용 (단일 카메라 렌즈 현실 반영)

### 5.2 전체 파이프라인

| 단계 | 기법 | 확률 | 목적 |
| --- | --- | --- | --- |
| Mosaic 이전 | SpiderWeb Aug (커스텀) | p=0.05 | 거미줄 오감지 억제 |
| Mosaic 이전 | IR 고반사 시뮬레이션 (커스텀) | p=0.20 | 안전모 역광/과노출 학습 |
| Mosaic 이전 | ToGray→3채널 복제 | p=0.15 | IR 도메인 증강 |
| Stage 1 | Mosaic4 | p=0.85 | 기본 공간 다양성 |
| Stage 1 | Mosaic9 (OR, --mosaic9 플래그) | p=0.0 (기본 off) | 소형 객체 극단 부족 시 옵션 |
| Stage 1 | RandomAffine (scale+shear) | - | 기하 변환 |
| Stage 1 | Flip LR | p=0.50 | 좌우 대칭 |
| Stage 1 | Flip UD | p=0.00~0.05 | 천장 카메라 대응 |
| Stage 2 | Patch-Paste (희귀/소형 우선) | p=0.10~0.15 | 클래스 불균형 완화 (보수적 시작) |
| Stage 2 | Hard Negative Paste | p=0.20 | 나뭇가지 오감지 억제 |
| Stage 2 | 안전모→사람 박스 상단 배치 (조건 강화) | 기본 off / 옵션 p=0.05 | 라벨 오염 방지를 위해 시각 검증 후 사용 |
| Stage 3 | GridMask | p=0.25 | 부분 가림, 펜스 대응 |
| Stage 4 | RandomShift (translate 전담) | p=0.20 | 바람 흔들림 폴대 |
| Stage 4 | DirectionalMotionBlur | p=0.15 | 방향성 카메라 진동 |
| Stage 4 | 진동 블러 중첩 (OneOf 흡수) | - | OneOf[MotionBlur/Defocus/DirBlur]로 통합 |
| Stage 5 | OneOf[JPEG, Downscale, Pixelate] | p=0.40 | 비트레이트 열화 |
| Stage 5 | OneOf[GaussNoise, ISONoise, S&P] | p=0.30 | 센서 노이즈 |
| Stage 5 | OneOf[MotionBlur, Defocus, DirBlur] | p=0.25 | 모션/초점 열화 |
| Stage 5 | Unsharp Mask | p=0.30 | CCTV 샤프닝 펌웨어 |
| Stage 5 | Sharpen | p=0.20 | 과샤프닝 시뮬레이션 |
| Stage 5 | RandomFog | p=0.05~0.10 | 렌즈 오염 |
| Stage 5 | GlassBlur | ❌ 제외 | Defocus와 중복 |
| Stage 5 | LensFlare | p=0.05 | 역광 플레어 (과도한 적용 방지) |
| Stage 5 | RandomSunFlare (상단 집중) | p=0.05~0.10 | 안전모 역광 (확률 조정) |
| Stage 5 | 과노출 박스 집중 (커스텀) | p=0.20 | 흰 안전모 과노출 |
| Stage 5 | 역광 배경 시뮬레이션 (커스텀) | p=0.15 | 실루엣 극단 케이스 |
| Stage 5 | Gamma 조정 | p=0.20 | 하이라이트 압축 |
| Stage 5 | CLAHE (clip=1~4 랜덤) | p=0.20~0.40 | IR/야간 대비 강화 |
| Stage 5 | RandomBrightnessContrast | p=0.30 | 저대비 미감지 억제 |
| Stage 5 | RandomShadow | p=0.20 | 그림자 오감지 억제 |
| Stage 5 | HSV (h=0.015, s=0.7, v=0.5) | - | 색상 다양성 |
| Stage 5 | ISONoise (color_shift=0) | p=0.20 | IR 센서 노이즈 |
| Stage 5 | Posterize | ❌ 제외 | 실효성 낮음 |
| Stage 5 | OcclusionAug (커스텀) | p=0.15 | GT 위 패치 부분 가림 |
| Stage 6 | MixUp (alpha=0.3) | p=0.05 | 도메인 일반화 (최소화) |

### 5.3 커스텀 Aug 상세 설계

#### 5.3.1 SpiderWeb Aug

중심점: 이미지 코너 또는 엣지 랜덤 선택

방사형 선: 8~16개 방향, 두께 1~2px, 색상 회색~흰색

동심원 연결: 2~4레이어 곡선

반투명 합성: alpha=0.15~0.40

GT 변경 없음 (거미줄 = 배경 처리)

#### 5.3.2 Patch-Paste (박스 기반 Copy-Paste)

희귀/소형 클래스 인스턴스 우선 선택 (빈도 역수 가중치)

GT IoU < 0.3 위치 제약 (과도한 중첩 방지)

소프트 blending: alpha=0.1~0.2 (사각형 패턴 학습 방지)

소형 인스턴스 붙여넣기 시 추가 Downscale(0.5~0.8) 적용

paste 후 bbox는 image boundary로 clipping하고, clipping 후 면적이 원본 대비 30% 미만이거나 aspect ratio가 1:8 또는 8:1을 넘으면 폐기한다.

paste 결과의 visible area가 50% 미만이면 GT를 추가하지 않는다.

paste 실패 시 원본 이미지를 그대로 사용하고, 실패 횟수는 augmentation debug log에 누적한다.

#### 5.3.3 Hard Negative Paste

데이터셋 내 GT 없는 이미지 자동 분류 (별도 풀 불필요)

나뭇가지/전신주/펜스 영역 crop → 학습 이미지에 paste

paste 영역 GT 없음 → 배경으로 강제 학습

Hard Negative crop은 teacher/baseline 모델의 false positive 영역과 GT 없는 이미지의 edge/texture 영역에서 우선 채굴한다.

GT와 hard negative paste 영역의 IoU가 0.1 이상이면 해당 위치는 사용하지 않는다.

`tools/check_aug_visual.py`는 각 custom aug별 최소 200장 샘플을 저장하고, bbox 오염/클래스 오염/비현실적 합성이 발견되면 해당 aug를 기본 off로 되돌린다.

#### 5.3.4 카메라 흔들림 Aug

RandomShift: ±5~20px 전체 이동, 빈 영역 replicate 패딩, GT 동일 방향 이동

DirectionalMotionBlur: 좌우/상하 고정 방향, kernel=5~15, angle=0°/90° ±10°

진동 블러: 짧은 kernel(3~5) 다방향 중첩

Rolling Shutter: 구현 복잡도로 제외, 선택 옵션 문서화

### 5.4 Phase별 Aug 적용 범위

| Aug 그룹 | Phase 1<br>(ep1~290) | Phase 2<br>(ep291~360) | Phase 3<br>(ep361~400) |
| --- | --- | --- | --- |
| Mosaic4 / Mosaic9 | ON | ON | OFF |
| Patch-Paste / Hard Neg | ON | ON | OFF |
| GridMask | ON | ON | OFF |
| MixUp | ON | ON | OFF |
| 카메라 흔들림 | ON | ON | ON |
| 픽셀레벨 전체 | ON | ON | ON |
| SpiderWeb (단일이미지) | ON | ON | ON |

## 6. 학습 흐름 설계 (단일 실행 자동화)

### 6.1 3단계 자동화 개요

python train.py 단일 실행으로 아래 3단계가 자동으로 순서대로 실행된다. 각 Phase 전환은 epoch 기반으로 자동 감지하며, DataLoader / 해상도 / hyp / 학습 정책이 자동 전환된다. Backbone은 구조 변경 대상이 아니며, feature 추출 위치 변경은 후순위 실험으로만 별도 검토한다.

| Phase | 구간 | 해상도 | Mosaic | 목적 | 비고 |
| --- | --- | --- | --- | --- | --- |
| Phase 1 | ep 1~290 | 640×640 | ON | 최대 다양성 풀학습 | Warmup ep 1~30 포함, Multi-scale 0.5~1.5× |
| Phase 2 | ep 291~360 | 640×384 / 1280×736 | ON | TensorRT 입력 해상도 적응 | rect=True, lr=0.001, Backbone 구조 유지 |
| Phase 3 | ep 361~400 | 640×384 / 1280×736 | OFF | 분포 최종 안정화 | Close Mosaic, Early stopping p=20 |

> ※ Phase 2에서 Mosaic ON 유지 이유: 해상도 전환 시 다양한 스케일 커버로 적응 안정화. Close Mosaic(OFF)는 올바른 해상도에서 수행해야 의미있음.

> ※ 문서의 epoch 표기는 1-based 기준이다. 구현 루프는 0-based를 사용하되 `phase1_end`, `phase2_end`는 end-exclusive로 처리한다. Warmup은 별도 Phase가 아니라 Phase 1 내부 loss/LR 상태로 관리한다.

### 6.2 Phase 전환 자동화 로직

for epoch in range(total_epochs):

# Phase 1: 풀학습

if epoch < phase1_end:

set_augmentation(mosaic=True, full=True)

set_loss_weights(lam_free=0.1 if epoch<30 else 0.5, lam_aux=0.25)

# Phase 2: Rect Finetune (최초 1회 전환)

elif epoch < phase2_end:

if epoch == phase1_end:  # 최초 1회만

set_backbone_structure(original=True)

set_phase2_trainable_policy(backbone_trainable=True)

unfreeze_neck_p2_path()

unfreeze_head()

rebuild_dataloader(rect=True, mosaic=True)

rebuild_val_loader(rect=True)

reset_lr(lr=0.001)

reset_scheduler(cosine)

load_hyp("hyp_phase2.yaml")

set_loss_weights(lam_aux=0.0)  # AUX freeze

# Phase 3: Close Mosaic (최초 1회 전환)

else:

if epoch == phase2_end:  # 최초 1회만

rebuild_dataloader(rect=True, mosaic=False)

load_hyp("hyp_phase3.yaml")

if early_stopping(patience=20): break

### 6.3 DataLoader 자동 재생성

Phase 2 진입: del train_loader → 해상도/rect 변경 → 새 DataLoader 생성

Phase 3 진입: mosaic=False 설정 → 새 DataLoader 생성

val_loader도 Phase 2 진입 시 rect=True로 함께 재생성 (mAP 기준 일치)

모델 타입 자동 감지: L → 640×384 / W6 → 1280×736

#### 6.3.1 DataLoader rebuild 정책

Phase 전환 시 부모 dataset 객체의 속성만 변경하지 않고 Dataset/DataLoader/Worker를 모두 새로 생성한다.

`persistent_workers`는 Phase rebuild와 Close Mosaic 구간에서 기본 비활성화한다. `workers=0`이면 항상 `persistent_workers=False`로 강제한다.

Phase 3 진입 시 새 Dataset을 `mosaic=False`로 생성하고, 기존 worker가 mosaic=True 상태를 유지하지 않도록 train_loader를 완전히 폐기 후 재생성한다.

Label cache는 dataset version, image path list, image/label file hash, class count가 달라지면 무효화한다. 기존 `.cache`가 있어도 hash/version 불일치 시 자동 재생성한다.

Weighted sampler는 `image_weights=False`와 중복 사용하지 않는다. DDP에서는 rank별 shard와 class-balanced sampling이 충돌하지 않도록 distributed-aware sampler를 사용하고, 지원 전까지는 단일 GPU/단일 프로세스에서만 활성화한다.

재현성을 위해 `seed`, `rank`, `worker_id`, `epoch`를 조합해 `worker_init_fn`을 설정하고, DDP sampler는 매 epoch `set_epoch(epoch)`를 호출한다.

#### 6.3.2 Phase 전환 검증 기준

Phase 상태는 epoch 0, 29, 30, 289, 290, 359, 360에서 unit test로 고정한다.

Phase 2/3 전환은 각각 최초 진입 epoch에서 한 번만 실행되어야 하며, hyp reload, rect 변경, mosaic 변경, train/val loader rebuild 여부를 `phase_transition.log`에 기록한다.

Close Mosaic 검증은 `workers=0`과 `workers>0` 양쪽 smoke test를 수행한다. `workers>0`에서는 새 worker가 `mosaic=False` Dataset을 들고 있는지 로그로 확인한다.

Label cache 검증은 image 추가/삭제, label 수정, class count 변경 케이스를 포함하며 hash/version 불일치 시 자동 재생성되어야 한다.

### 6.4 학습 설정 상세

| 항목 | YOLOv7-L | YOLOv7-W6 |
| --- | --- | --- |
| Phase 1 Batch | 32 | 8 (grad_accum=4, effective=32) |
| Phase 1 해상도 | 640×640 | 640×640 |
| Phase 2/3 해상도 | 640×384 | 1280×736 |
| lr0 (Phase 1) | 0.01 | 0.01 |
| lr (Phase 2~3) | 0.001 | 0.001 |
| LR Schedule | Cosine Annealing | Cosine Annealing |
| Warmup | 3 epoch | 3 epoch |
| EMA | decay=0.9999 전구간 유지 | decay=0.9999 전구간 유지 |
| Workers | 8 | 8 |
| Cache | disk (80만장 RAM 초과) | disk |
| Early Stopping | Phase 3만, patience=20 | Phase 3만, patience=20 |

## 7. 학습 로그 및 결과 출력 설계

### 7.1 Verbose 출력 설계

학습 중 콘솔에 아래 정보를 매 epoch 출력한다. 기본값으로 활성화되며 --no-verbose 플래그로 비활성 가능하다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Phase 1] Epoch 045/400  |  lr: 0.00821  |  GPU: 38.2GB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Loss     │  box: 0.0312  cls: 0.0187  obj: 0.0094

│  aux: 0.0078  free: 0.0143  total: 0.0814

Train    │  P: 0.782  R: 0.751  mAP@0.5: 0.801

Val      │  P: 0.764  R: 0.739  mAP@0.5: 0.789  mAP@0.5:0.95: 0.512

Per-cls  │  person: 0.831  vehicle: 0.812  helmet: 0.623  ...

Speed    │  pre: 1.2ms  infer: 8.3ms  nms: 0.8ms  total: 10.3ms

Aug      │  mosaic: ON  copy_paste: ON  clahe: ON

Sampler  │  weighted: ON  rare_boost: 2.3×

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Best mAP@0.5:0.95: 0.512 (ep045)  │  No improve: 0/20

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 7.2 로그 파일 구조

| 파일명 | 내용 | 형식 | 갱신 주기 |
| --- | --- | --- | --- |
| results.csv | 전체 epoch 학습 loss 및 validation 지표 | CSV | 매 epoch |
| results_per_class.csv | 클래스별 P/R/AP | CSV | best 갱신 시 또는 `--per-class-log-interval` 주기 |
| loss_detail.csv | box/cls/obj/aux/free loss 분리 | CSV | 매 epoch |
| train_log.txt | 콘솔 출력 전체 저장 | TXT | 실시간 |
| phase_transition.log | Phase 전환 시점/설정 기록 | TXT | 전환 시 |
| hyp_used.yaml | 실제 사용된 hyp 스냅샷 | YAML | Phase 전환 시 |
| opt.yaml | 실행 옵션 전체 기록 | YAML | 학습 시작 시 |
| best.pt | 최고 primary metric weight | PT | best 갱신 시 |
| last.pt | 마지막 epoch weight | PT | 매 epoch |
| confusion_matrix.png | 혼동 행렬 시각화 | PNG | 학습 종료 시 |
| PR_curve.png | PR 곡선 전체 클래스 | PNG | 학습 종료 시 |
| F1_curve.png | F1 곡선 | PNG | 학습 종료 시 |
| results_plot.png | loss/mAP 학습 곡선 | PNG | 학습 종료 시 |

`results.csv`를 canonical metric log로 사용한다. 기존 `results.txt`를 유지하는 경우에도 plot/read 경로는 `results.csv`를 우선 사용한다.

Train metric은 loss와 learning rate 중심으로 기록하고, P/R/mAP는 validation 기준으로만 계산한다. 대규모 dataset에서는 per-class metric 기본 주기를 10 epoch로 두고, best 갱신 시에는 항상 저장한다.

### 7.3 results.csv 컬럼 정의

epoch, phase,

train/box_loss, train/cls_loss, train/obj_loss,

train/aux_loss, train/free_loss, train/total_loss,

metrics/precision, metrics/recall,

metrics/mAP_0.5, metrics/mAP_0.5:0.95,

val/box_loss, val/cls_loss, val/obj_loss,

x/lr0, x/lr1, x/lr2,

gpu_mem_gb, epoch_time_sec

### 7.4 Phase 전환 알림

╔══════════════════════════════════════════════════════╗

║  PHASE TRANSITION: 1 → 2 (Rect Finetune)           ║

║  Epoch: 291                                         ║

║  Resolution: 640×640 → 640×384                     ║

║  Mosaic: ON (유지)                                  ║

║  Backbone: ORIGINAL / TRAINABLE                     ║

║  LR reset: 0.00821 → 0.001                         ║

║  DataLoader: REBUILT                                ║

╚══════════════════════════════════════════════════════╝

### 7.5 Early Stopping 알림

⚠ Early Stopping: no improvement for 20 epochs

Best mAP@0.5:0.95: 0.547 @ epoch 378

Saving best.pt → runs/train/exp_l_custom/weights/best.pt

Training complete.

## 8. TensorRT Export 설계

### 8.1 Export 전처리

model.eval() + AUX Head 브랜치 비활성 후 torch.onnx.export

앵커프리 출력 정규화 노드 포함 (ltrb→xywh, centerness→obj_score)

표준 ONNX 연산만 사용 (Reshape, Mul, Add, Sigmoid)

두 헤드 출력 concat → flatten → [batch, total_boxes, 5+C]

ONNX opset=16 (TRT 8.6 / 10.x 양쪽 지원)

ONNX/TensorRT 입력 해상도는 TensorRT 및 YOLO stride 호환을 위해 32 배수로 고정한다. 실제 16:9 소스 해상도는 letterbox로 640×384 또는 1280×736 입력에 맞춘다.

### 8.2 NMS 모드 설계 — raw output 기본

현재 개발 범위에서는 raw ONNX output을 기본값으로 한다. C++ 후처리, TensorRT runtime, EfficientNMS 내장은 별도 요청 전까지 구현하지 않는다.

| NMS 모드 | 방식 | 상태 | 비고 |
| --- | --- | --- | --- |
| none (raw) | ONNX raw output | ✅ 기본 | Python 검증 도구에서 output shape/value 비교 |
| efficient_nms | TRT Plugin 내장 | 제외 | TensorRT runtime 차수에서 별도 검토 |

### 8.3 ONNX Runtime 검증 환경

| 항목 | 기준 |
| --- | --- |
| OS | Windows |
| GPU | RTX A6000 48GB |
| PyTorch/ONNX Runtime | `requirements.txt`와 export 환경 기록 |
| Smoke test | PyTorch forward → ONNX Runtime forward → output diff 비교 |

### 8.4 Dynamic Shape Profile

| 모델 | min | opt | max |
| --- | --- | --- | --- |
| YOLOv7-L | 1×3×320×192 | 4×3×640×384 | 8×3×640×384 |
| YOLOv7-W6 | 1×3×640×384 | 4×3×1280×736 | 8×3×1280×736 |

### 8.5 Export 명령어

# ONNX export

python export.py --weights best.pt --img 640 384 --opset 16 --nms-mode none

TensorRT engine build, C++ 후처리, 추론 서버 실행 명령은 현재 개발 범위에서 제외한다.

## 9. 구현 파일 맵 및 작업 순서

### 9.1 수정/신규 파일 목록

| 파일 | 작업 | 난이도 | 의존성 |
| --- | --- | --- | --- |
| utils/loss.py | WIoU v3 + VFL + AUX Loss + A/B 플래그 | 중 | 없음 |
| utils/augmentations.py | CCTV 특화 Aug 파이프라인 (조정 버전) | 중 | 없음 |
| utils/datasets.py | Mosaic9, Patch-Paste(조건강화), CLAHE | 중 | augmentations.py |
| utils/tal.py | TAL assigner 신규 구현 | 중 | loss.py |
| utils/sampler.py | WeightedRandomSampler 구현 | 낮 | 없음 |
| utils/loggers.py | Verbose 출력 + CSV/PNG 로그 전체 | 낮 | 없음 |
| models/common.py | SCDown 확정(W6) / PSA P5 후보 / GELAN 후순위 | 중 | 없음 |
| models/yolo.py | Decoupled Head + W6 P2 Anchor + AUX 통합 / FCOS 후순위 | 높 | common.py |
| cfg/yolov7-l-custom.yaml | L 모델 커스텀 아키텍처 정의 | 중 | models/ |
| cfg/yolov7-w6-custom.yaml | W6 모델 커스텀 아키텍처 정의 | 중 | models/ |
| data/hyp_phase1.yaml | Phase 1 하이퍼파라미터 | 낮 | 없음 |
| data/hyp_phase2.yaml | Phase 2 하이퍼파라미터 | 낮 | 없음 |
| data/hyp_phase3.yaml | Phase 3 하이퍼파라미터 | 낮 | 없음 |
| train.py | train+train_aux 통합, 3단계 자동화, A/B 플래그 | 높 | 전체 |
| export.py | AUX 비활성 + 두 헤드 정규화 export | 높 | models/ |
| tools/profile_model.py | params/GFLOPs 측정 및 10% 예산 검증 | 낮 | models/, cfg/ |
| tools/verify_export.py | PyTorch/ONNX Runtime raw output 수치 비교 | 중 | export.py |
| tools/check_aug_visual.py | Augmentation 결과/라벨 오염 시각 검증 | 낮 | augmentations.py |

### 9.2 A/B 실험 플래그 (train.py)

원인 분리를 위해 모든 핵심 기법을 플래그로 제어한다. 기본값은 안정적인 설정으로 고정한다.

| 플래그 | 옵션 | 기본값 | 설명 |
| --- | --- | --- | --- |
| --loss-box | ciou / wiou_v3 | wiou_v3 | 박스 Loss 선택 |
| --loss-cls | bce / vfl | vfl | 분류 Loss 선택 (TAL 적용 시 vfl 권장) |
| --assign | simota / tal | tal | Label Assignment 방식 |
| --head | coupled / decoupled | decoupled | Detection Head 구조 |
| --aux | on / off | L: off / W6: on | AUX Head 활성 여부 |
| --p2-head | none / anchor / fcos | L: none / W6: anchor | P2 소형객체 헤드 (W6 우선 실험) |
| --neck-mod | none / scdown / psa / gelan | L: none / W6: scdown | Neck 모듈 교체 (단독 실험) |
| --nms-mode | none / efficient_nms | none | NMS 모드 (raw output 기본) |

L 모델은 실험 A와 C까지 진행한다. W6는 A/B/D/E/F를 진행하되, D/E/F는 후순위 조건을 만족할 때만 수행한다. Export 검증은 각 구조 변경 직후 중간 체크포인트로 반드시 수행한다.

| 실험 | 적용 모델 | 변경 항목 | 기본값 대비 |
| --- | --- | --- | --- |
| A | L / W6 | Decoupled Head + WIoU v3 + TAL + VFL | 최종 공통 기본값 |
| B | W6 | 실험A + P2 Anchor Head + SCDown | W6 공격형 확정 구성 |
| C | L | 실험A + AUX on | L 성능형 옵션 검증 |
| D | W6 | 실험B + PSA P5 단독 | 2차 확정 후보, P4/P3 순차 확장 |
| E | W6 | 실험B + FCOS P2 Head | 후순위, P2 Anchor 부족 시 검토 |
| F | W6 | 실험B + GELAN | 최후순위, 채널/route 검증 후 적용 |

> ※ Export 중간 검증 체크포인트: 실험 A 완료 후 즉시 raw output ONNX → TRT 빌드 검증. 이후 구조 변경(B/D/E/F)마다 재검증.

> ※ PSA는 P3/P4/P5 동시 적용 금지. P5 단독 → P4 추가 → P3 추가 순서로 효과 분리.

> ※ VFL 단독 적용 금지. TAL 없는 VFL은 설계에서 제외.

#### 9.2.1 실험 성공 및 중단 기준

| 실험 | 성공 기준 | 중단 기준 |
| --- | --- | --- |
| A | baseline 대비 primary mAP 상승, GFLOPs +10% 미만 | NaN/Inf 또는 mAP 2 points 이상 하락 |
| B | W6 소형 객체 AP/recall 상승, GFLOPs +10% 미만 | output box 수 또는 Python NMS 비용 과다 |
| C | L 희귀 클래스 recall 상승 | L GFLOPs +10% 초과 또는 효과 미미 |
| D | PSA P5 단독에서 mAP 상승 | TRT FP16 profile 편차 증가 |
| E | P2 Anchor 대비 소형 객체 recall 추가 상승 | postprocess 복잡도 또는 latency 과다 |
| F | 구조 안정성/export 통과 후 mAP 상승 | route/concat/export 오류 발생 |

### 9.3 단계별 개발 착수 순서

한 번에 여러 기법을 넣지 않는다. 각 단계는 한 축의 변경만 포함하고, baseline 대비 지표를 남긴 뒤 다음 단계로 진행한다. 실패 시 해당 단계만 되돌리거나 fallback 설정으로 전환한다.

| 단계 | 우선순위 | 변경 범위 | 검증/산출물 | 다음 단계 진입 조건 |
| --- | --- | --- | --- | --- |
| 0. Baseline 고정 | 최우선 | 원본 YOLOv7-L/W6 학습, 평가, export 재현 | baseline `best.pt`, `results.csv`, GFLOPs, TRT FP16 latency | L/W6 baseline 수치와 validation set checksum 확보 |
| 1. Export 기준선 | 최우선 | raw ONNX export, PyTorch/ONNX Runtime 비교 | `tools/verify_export.py`, PyTorch/ONNX output 비교 | export 오차 허용 범위 통과 |
| 2. 학습 루프 통합 | 높음 | `train.py` 통합, Phase 1/2/3, rect finetune, DataLoader rebuild | `phase_transition.log`, epoch boundary test, `workers=0/>0` smoke test | Phase 전환과 Close Mosaic 정상 동작 |
| 3. 계측/로그 기반 | 높음 | `results.csv` canonical log, `tools/profile_model.py`, per-class 주기 저장 | mAP/GFLOPs/latency 비교표 | GFLOPs delta 자동 계산 가능 |
| 4. 데이터/Aug 기반 | 중간 | CCTV pixel aug, Patch-Paste 안전장치, Hard Negative, Weighted Sampler | `tools/check_aug_visual.py` 샘플, 라벨 오염 점검 | 시각 검증 통과, 학습 smoke run 정상 |
| 5A. Head 구조 | 중간 | Decoupled Head만 적용 | 실험 A-1 결과, export 비교 | GFLOPs +10% 미만, mAP 하락 없음 |
| 5B. Box Loss | 중간 | WIoU v3만 적용, CIoU fallback 유지 | loss 안정성, resume state 확인 | NaN/Inf 없음, mAP 하락 없음 |
| 5C. Assignment/Cls Loss | 중간 | TAL + VFL 적용, SimOTA/BCE fallback 유지 | positive 수, loss scale, mAP 비교 | baseline 대비 primary metric 개선 |
| 6. W6 구조 확장 | 중간 | W6 P2 Anchor + SCDown | W6 소형 객체 AP/recall, output box 수, GFLOPs | GFLOPs +10% 미만, output 증가 허용 |
| 7. L 옵션 검증 | 낮음 | L AUX on 성능형 옵션만 별도 실험 | L recall/mAP, export 영향 확인 | 효과 없거나 불안정하면 off 유지 |
| 8. 후순위 구조 | 낮음 | PSA P5 → FCOS P2 → GELAN 순차 단독 실험 | 각 실험별 mAP/GFLOPs/export 결과 | 앞 단계 목표 미달이고 latency 여유가 있을 때만 진행 |
| 9. 파인튜닝 | 별도 | Replay Buffer, Pseudo Label, YOLO LwF A/B | 기존/대상 클래스 mAP, forgetting 지표 | scratch 학습 기준선 확정 후 진행 |

단계별 결과는 동일 validation set에서 비교한다. 결과표에는 `stage`, `config`, `weights`, `primary_mAP`, `mAP@0.5`, `small_AP`, `rare_recall`, `GFLOPs_delta`, `TRT_latency`, `export_status`를 기록한다.

### 9.4 플래그 기반 통합 구현 및 단계별 학습 실행 계획

개발 방식은 "전체 코드를 플래그 기반으로 먼저 준비하고, 학습 서버에서는 stage별로 하나씩 기능을 켜서 검증"하는 구조로 한다. 기능을 한 번에 모두 활성화하지 않는다.

#### 9.4.1 구현 원칙

모든 신규 기능은 독립 플래그로 제어한다. 기본값은 baseline과 최대한 동일해야 하며, 플래그를 켜지 않으면 기존 YOLOv7 학습/평가/export 동작이 유지되어야 한다.

핵심 플래그는 `--head`, `--loss-box`, `--loss-cls`, `--assign`, `--aux`, `--p2-head`, `--neck-mod`, `--aug-profile`, `--sampler-mode`, `--nms-mode`로 관리한다.

플래그 조합은 stage config로 저장한다. 학습 서버는 stage config를 순서대로 읽고, 이전 stage가 성공한 경우에만 다음 stage를 실행한다.

#### 9.4.2 학습 서버 실행 순서

| Stage | 목적 | 활성 플래그 예시 | 성공 조건 |
| --- | --- | --- | --- |
| 0 | baseline 고정 | baseline 기본값 | 학습/평가/export 기준값 확보 |
| 1 | export 기준선 | `--nms-mode none` | PyTorch/ONNX 비교 통과 |
| 2 | 통합 학습 루프 | `--phase-train on` | Phase/DataLoader 전환 테스트 통과 |
| 3 | 계측/로그 | `--profile on --log-format csv` | GFLOPs/latency/results.csv 생성 |
| 4A | Head만 변경 | `--head decoupled` | GFLOPs +10% 미만, mAP 하락 없음 |
| 4B | Box loss만 변경 | `--loss-box wiou_v3` | NaN/Inf 없음, resume 정상 |
| 4C | Assignment/cls 변경 | `--assign tal --loss-cls vfl` | primary mAP 개선 |
| 5A | pixel aug | `--aug-profile cctv_pixel` | 시각 검증 및 smoke run 통과 |
| 5B | label-changing aug | `--aug-profile cctv_paste --sampler-mode weighted` | 라벨 오염 없음 |
| 6 | W6 구조 확장 | `--p2-head anchor --neck-mod scdown` | small AP/recall 개선, NMS latency 허용 |
| 7 | L AUX 옵션 | `--aux on` | 효과 없으면 기본 off 유지 |
| 8 | 후순위 구조 | `--neck-mod psa/fcos/gelan` | 앞 단계 목표 미달이고 latency 여유 있을 때만 |

#### 9.4.3 자동 중단 및 산출물

각 stage 종료 후 `results.csv`, `stage_result.yaml`, `profile.json`, `export_check.json`, `phase_transition.log`를 저장한다.

다음 조건 중 하나라도 발생하면 다음 stage로 진행하지 않는다.

- primary mAP가 baseline 대비 2 percentage points 이상 하락
- GFLOPs 증가율이 10% 이상
- ONNX export 또는 PyTorch/ONNX output 비교 실패
- NaN/Inf loss 발생
- Close Mosaic/DataLoader rebuild 테스트 실패
- label-changing aug 시각 검증 실패

실패한 stage는 플래그를 끄고 직전 성공 stage의 config와 weight를 기준으로 재시작한다.

## 10. 기대 효과 및 확정 체크리스트

### 10.1 기대 성능 향상

| 적용 항목 | 예상 mAP 향상 | 속도 영향 | 실험 구분 |
| --- | --- | --- | --- |
| CCTV 특화 Aug (조정 버전) | +2~4% | 없음 | 기본 적용 |
| Hard Negative Paste | +1~2% | 없음 | 기본 적용 |
| 역광/과노출 Aug (미감지 억제) | +1~2% | 없음 | 기본 적용 |
| Decoupled Head | +1~2% | ±2ms | 실험 A |
| WIoU v3 | +0.5~1% | 없음 | 공통 확정 |
| TAL + VFL | +1~2% | 없음 | 공통 확정 |
| SCDown (W6) / PSA P5 후보 | +0.5~1% | 없음~-2ms | SCDown 확정, PSA 후보 |
| P2 Anchor Head (W6) | +1~2% | +2~3ms | W6 확정 |
| FCOS P2 Head | +1~3% | +2~4ms | 후순위 |
| AUX Head L 추가 | +0.5~1% | 없음 (추론 제거) | L 옵션 / W6 확정 |
| Rect Finetune | +0.5~1% | 없음 | 기본 적용 |
| GELAN | +0.5~1% | 없음 | 실험 F (최후순위) |
| 합계 (중복 제외) | +6~12% 예상 | 기존 모델 대비 GFLOPs 증가 10% 미만 | TensorRT FP16 latency 실측 병행 |

GFLOPs는 `tools/profile_model.py`로 측정한다. L은 `640×384`, W6는 `1280×736` 입력 기준으로 baseline/current/delta_percent를 저장하고, delta가 10% 이상이면 해당 구조 실험은 중단한다.

### 10.2 최종 확정 체크리스트

| 영역 | 항목 | 상태 | 비고 |
| --- | --- | --- | --- |
| Architecture | Backbone 구조 원본 유지 | ✅ 확정 | weight freeze 의미 아님, feature 변경은 후순위 실험 |
| Architecture | PSA (P5 단독 시작 → P4→P3 순차) | ⚠ 2차 후보 | W6 P5 단독부터, 전체 적용 금지 |
| Architecture | SCDown (Downsampling) | ✅ W6 확정 | L 제외 |
| Architecture | P2 upsample + Conv×2 | ✅ W6 확정 | P2 Anchor 우선, L 제외 |
| Architecture | AUX Head L/W6 분리 (W6 on / L off 기본) | ✅ 모델별 확정 | W6 on / L off 기본, L on은 옵션 |
| Architecture | Decoupled Head (P3/P4/P5) | ✅ 확정 | 실험 A |
| Architecture | FCOS 앵커프리 헤드 (P2) | ⚠ 후순위 | P2 Anchor 부족 시만 검토 |
| Architecture | 출력 포맷 정규화 노드 | ✅ 확정 |  |
| Architecture | GELAN | ⚠ 후순위 | 채널 검증 후 단독 실험 |
| Loss | WIoU v3 (.detach 명시) | ✅ 공통 확정 | CIoU fallback 유지 |
| Loss | VFL (TAL 할당 후 순서 고정) | ✅ 공통 확정 | TAL과 세트, 단독 금지 |
| Loss | TAL + Center Sampling | ✅ 공통 확정 | SimOTA fallback 유지 |
| Loss | λ 워밍업 ep30 기준 | ✅ 확정 |  |
| Loss | AUX Phase 3 비활성 | ✅ 확정 |  |
| Augmentation | CCTV 특화 파이프라인 (조정) | ✅ 확정 | GlassBlur/Posterize 제외, Mosaic9 기본 off |
| Augmentation | IR/RGB 혼재 대응 | ✅ 확정 |  |
| Augmentation | 카메라 흔들림 | ✅ 확정 | Rolling Shutter 제외 |
| Augmentation | 오감지 억제 (나뭇가지/거미줄/그림자) | ✅ 확정 |  |
| Augmentation | 미감지 억제 (소형/역광/저대비/부분가림) | ✅ 확정 |  |
| Augmentation | OneOf 구조 (중복 방지) | ✅ 확정 |  |
| Augmentation | 안전모 Paste 조건 강화 | ⚠ 옵션 | 기본 off, p=0.05 시각 검증 후 사용 |
| Augmentation | image_weights 제거 → Sampler 단일화 | ✅ 확정 |  |
| Augmentation | GlassBlur / Posterize | ❌ 제외 | Defocus 중복 / 실효성 낮음 |
| Augmentation | LensFlare | ⚠ 조정 | p=0.1 → 0.05 |
| Training | 단일 실행 3단계 자동 전환 | ✅ 확정 |  |
| Training | train.py + train_aux.py 통합 | ✅ 확정 |  |
| Training | A/B 실험 플래그 8개 | ✅ 확정 | 신규 추가 |
| Training | DataLoader / val_loader 자동 rebuild | ✅ 확정 |  |
| Training | EMA 전구간 유지 | ✅ 확정 |  |
| Training | Early Stopping Phase 3만 | ✅ 확정 |  |
| Training | W6 batch=8 + grad_accum=4 | ✅ 확정 |  |
| Logging | Verbose 매 epoch 상세 출력 (기본 활성) | ✅ 확정 |  |
| Logging | results.csv / per_class.csv / loss_detail.csv | ✅ 확정 |  |
| Logging | Phase 전환 알림 / 로그 | ✅ 확정 |  |
| Logging | 학습 곡선 / PR / F1 / 혼동행렬 PNG | ✅ 확정 |  |
| Export | raw ONNX output 기본 | ✅ 확정 | 구조 변경마다 PyTorch/ONNX 중간 검증 필수 |
| Export | EfficientNMS 옵션 (--nms-mode efficient_nms) | 제외 | TensorRT runtime 차수에서 별도 검토 |
| Export | ONNX opset=16 | ✅ 확정 |  |
| Export | TRT 버전 자동 감지 / 분기 빌드 | 제외 | 별도 요청 전까지 제외 |
| Export | L / W6 Dynamic Profile 분리 | ⚠ 후순위 | ONNX 입력 shape 기준만 유지 |
| Export | FP16 기본 | 제외 | TensorRT runtime 차수에서 검토 |
| 제외 | PGI (성능 검증 실패) | ❌ 제외 |  |
| 제외 | SAHI (속도 조건 위반) | ❌ 제외 |  |
| 제외 | FlashAttention (GPU 행렬 한계) | ❌ 제외 |  |
| 제외 | R-ELAN (채널 불일치 리스크) | ❌ 제외 |  |
| 제외 | Area Attention (TRT 호환성) | ❌ 제외 |  |
| 선택 | Rolling Shutter Aug | ⚠ 선택옵션 | 구현 복잡도 |
| 선택 | GELAN | ⚠ 후순위 | 채널 검증 후 단독 실험 |

## 11. 파인튜닝 설계 — Catastrophic Forgetting 억제

### 11.1 문제 정의

아래 15개/5개/10개 클래스 수량은 설명을 위한 예시다. 실제 파인튜닝 클래스 수량과 class index mapping은 대상 프로젝트의 `data/*.yaml`을 기준으로 확정한다.

예를 들어 15개 클래스 스크래치 학습 완료 후, 일부 클래스 데이터로만 파인튜닝 시 학습 데이터에 포함되지 않은 기존 클래스 성능이 급락하는 현상이 발생할 수 있다. 이를 Catastrophic Forgetting이라 한다.

| 원인 | 설명 |
| --- | --- |
| Gradient 편중 | 일부 클래스 데이터만 존재 → 미포함 기존 클래스 gradient = 0 |
| Weight Decay | Gradient 없는 기존 클래스 관련 weight가 decay로 소실 |
| BN 통계 변화 | 배치 내 클래스 분포 변화 → BatchNorm 통계 틀어짐 |
| cls branch 충돌 | Head cls branch 전체 클래스 공유 → 일부 클래스만 학습 시 미포함 클래스 node 소실 |
| 기존 해결책 한계 | 스크래치 전체 데이터 혼합 → 데이터 관리 복잡, 학습 시간 증가 |

### 11.2 적용 방법 — YOLO LwF + Replay Buffer

2025년 3월 발표된 "Teach YOLO to Remember" (arXiv:2503.04688) 연구에서 YOLO 전용 Self-Distillation 방법인 YOLO LwF를 제안했다. 기존 LwF가 one-stage YOLO에서 noisy regression 출력으로 인해 비효과적인 문제를 해결하고, Replay Buffer와 함께 사용 시 SOTA 성능을 달성했다.

기존 LwF의 YOLO 적용 문제점:

Teacher 모델의 regression 출력(박스 좌표)은 노이즈가 많음

노이즈 있는 regression을 distillation target으로 쓰면 손상된 지식 전달

cls 출력은 상대적으로 안정적 → cls만 선택적 distillation 필요

YOLO LwF 해결 방식:

cls distillation: 전체 예측 위치에 적용 (안정적)

reg distillation: confidence > threshold 박스만 선택적 적용 (노이즈 필터링)

Replay Buffer 병행으로 distillation 효과 극대화

### 11.3 파인튜닝 전체 흐름

┌──────────────────────────────────────────────────────┐

│               준비 단계                              │

│                                                      │

│  1. Teacher 모델 준비                                │

│     스크래치 완료 best.pt → freeze (변경 없음)      │

│                                                      │

│  2. Pseudo Label 생성                                │

│     Teacher로 파인튜닝 이미지 추론                   │

│     conf > 0.5 박스 → 기존 클래스 Pseudo GT        │

│     파인튜닝 데이터 레이블에 자동 병합               │

│                                                      │

│  3. Replay Buffer 구성                               │

│     스크래치 데이터에서 클래스당 200장 선택          │

│     Hard case 위주 (conf 낮은 샘플 우선)            │

│     야간/IR/역광/소형/부분가림 다양성 포함           │

└──────────────────────────────────────────────────────┘

│

▼

┌──────────────────────────────────────────────────────┐

│               학습 단계                              │

│                                                      │

│  Freeze:  옵션(기본 off), 필요 시 Neck 하위 중심    │

│  Unfreeze: Neck 상위 + Head 전체                    │

│                                                      │

│  배치 구성:                                          │

│    새 데이터 + Pseudo GT        70%                 │

│    Replay Buffer                 30%                │

│                                                      │

│  Loss:                                               │

│    L_total = L_yolo                                  │

│            + α × L_distill_cls                      │

│            + β × L_distill_reg (conf>0.5만)        │

│                                                      │

│  LR: 스크래치의 1/20, cosine, warmup 3ep           │

└──────────────────────────────────────────────────────┘

### 11.4 Loss 설계

| Loss 항목 | 수식 | 파라미터 | 목적 |
| --- | --- | --- | --- |
| L_yolo | YOLOv7 기본 Loss | - | 파인튜닝 대상 클래스 + Pseudo GT 학습 |
| L_distill_cls | KLDiv(Student cls, Teacher cls) | α=0.2→0.5 스케줄, 망각 심하면 0.8 | 미포함 기존 클래스 cls 기억 유지 |
| L_distill_reg | SmoothL1(Student box, Teacher box) | β=0.1→0.3 스케줄, conf>0.5만 | 신뢰도 높은 박스만 reg 유지 |
| L_total | L_yolo + α×L_cls + β×L_reg | α/β A/B 조정 | 망각 억제 + 새 클래스 학습 균형 |

> ※ reg distillation은 conf > 0.5 박스만 적용. 노이즈 있는 박스를 target으로 쓰면 오히려 성능 저하.

> ※ α/β는 클래스 불균형 정도에 따라 조정. 기존 클래스 망각이 심하면 α 높임 (0.5→0.8).

### 11.5 Replay Buffer 구성 기준

| 항목 | 내용 |
| --- | --- |
| 선택 기준 | Hard case 우선: 스크래치 모델 추론 시 conf 낮은 샘플 |
| 클래스별 수량 | 200장 기본 / 희귀 클래스는 300장까지 확대 / replay-ratio 기본 0.3 |
| 다양성 조건 | 야간/주간/IR/역광 균등 포함, 소형 객체 포함, 부분 가림 포함 |
| 총 용량 | 15클래스 × 200장 = 3,000장 (전체 데이터의 0.4% 이하) |
| 갱신 주기 | 파인튜닝마다 재선택 (새 Hard case 반영) |
| 저장 형식 | 이미지 + 레이블 쌍으로 저장 (기존 데이터 포맷 동일) |

### 11.6 Pseudo Label 생성 파이프라인

# 1. Teacher 모델로 파인튜닝 이미지 추론

python generate_pseudo_labels.py \

--weights scratch_best.pt \

--source finetune_data/images/ \

--conf-thres 0.5 \

--iou-thres 0.45 \

--output finetune_data/pseudo_labels/

# 2. 기존 레이블 + Pseudo Label 병합

# 파인튜닝 대상 클래스 GT + 기존 클래스 Pseudo GT

python merge_labels.py \

--gt-labels finetune_data/labels/ \

--pseudo-labels finetune_data/pseudo_labels/ \

--output finetune_data/merged_labels/

# 3. Pseudo Label 품질 검증

# conf < 0.5 박스 자동 제거 (이미 필터됨)

# GT와 IoU > 0.8 중복 제거

### 11.7 파인튜닝 실행 명령어

# 파인튜닝 실행 (YOLO LwF + Replay Buffer)

python finetune.py \

--weights scratch_best.pt \

--teacher-weights scratch_best.pt \

--data data/finetune.yaml \

--replay-buffer data/replay_buffer/ \

--replay-ratio 0.3 \

--hyp data/hyp_finetune.yaml \

--epochs 100 \

--img 640 \

--batch 32 \

--freeze neck_lower \

--distill-alpha 0.5 \

--distill-beta 0.3 \

--distill-conf-thres 0.5 \

--name exp_finetune_v1

### 11.8 hyp_finetune.yaml 설정

| 파라미터 | 파인튜닝 값 | 스크래치 값 | 이유 |
| --- | --- | --- | --- |
| lr0 | 0.0005 | 0.01 | 스크래치의 1/20, 기존 weight 보호 |
| lrf | 0.01 | 0.1 | 최종 LR 낮게 유지 |
| warmup_epochs | 3 | 3 | 학습 초반 안정화 |
| mosaic | 0.0 | 0.85 | Close Mosaic 유지 (파인튜닝 전구간) |
| mixup | 0.0 | 0.05 | 도메인 혼재 방지 |
| copy_paste | 0.0 | 0.15 | 파인튜닝 단계 불필요 |
| weight_decay | 0.0005 | 0.0005 | 유지 |
| hsv_v | 0.3 | 0.5 | aug 강도 낮춤 |
| scale | 0.3 | 0.5 | aug 강도 낮춤 |

### 11.9 파인튜닝 단계별 검증 기준

| 검증 항목 | 기준 | 조치 |
| --- | --- | --- |
| 파인튜닝 대상 클래스 mAP | 스크래치 대비 90% 이상 | 미달 시 α 낮춤 (대상 클래스 학습 강화) |
| 기존/미포함 클래스 mAP | 스크래치 대비 95% 이상 | 미달 시 α 높임 + Replay 비율 증가 |
| 전체 mAP | 스크래치 전체 대비 93% 이상 | 미달 시 Replay Buffer 크기 증가 |
| 추론 속도 | 스크래치 모델 대비 동일 | Teacher는 학습 시에만 사용, 추론 무관 |

### 11.10 구현 파일 추가

| 파일 | 역할 | 난이도 |
| --- | --- | --- |
| finetune.py | YOLO LwF + Replay 통합 파인튜닝 스크립트 | 높 |
| utils/continual_loss.py | L_distill_cls + L_distill_reg 구현 | 중 |
| utils/replay_buffer.py | Hard case 선택 + Buffer 관리 | 중 |
| utils/pseudo_label.py | Teacher 추론 → Pseudo GT 생성 + 병합 | 중 |
| data/hyp_finetune.yaml | 파인튜닝 전용 하이퍼파라미터 | 낮 |
| data/finetune.yaml | 파인튜닝 데이터 경로 정의 | 낮 |

## 12. 최종 개발 착수안 반영 요약

### 12.1 확정 방향

공통 확정: Decoupled Head, WIoU v3, TAL + VFL, CCTV Aug 조정, Hard Negative, Rect Finetune, raw ONNX output export.

YOLOv7-L 확정: 경량형 모델로 운용한다. AUX off, P2 off, Neck 원본 유지, Decoupled Head + WIoU + TAL/VFL만 적용한다.

YOLOv7-W6 확정: 공격형 모델로 운용한다. AUX on, P2 Anchor Head, SCDown, Decoupled Head, WIoU + TAL/VFL을 기본 적용한다.

### 12.2 후순위 유지

PSA는 W6 P5 단독부터 시작하는 2차 후보로 유지한다. P3/P4/P5 동시 적용은 금지한다.

FCOS P2는 W6 P2 Anchor Head로도 소형 객체 성능이 부족할 때만 검토한다.

GELAN은 최후순위 실험으로 유지한다. 채널/route/concat 안정성 검증 전 기본 적용하지 않는다.

### 12.3 구현 기준

구조 변경 후에는 즉시 raw output ONNX export와 PyTorch/ONNX Runtime 출력 비교를 수행한다.

L은 속도 유지가 핵심이고 W6는 소형 객체 성능 개선이 핵심이다. 모델별 역할을 혼동하지 않는다.

## 13. 논문 비교 기반 최종 검토 결과

본 절은 v1.3 최종 개발 착수안을 YOLOv7, WIoU, TOOD/TAL, VarifocalNet, FCOS, YOLOv9, YOLOv10, YOLO LwF 관련 논문 방향과 비교하여 최종 확정/후순위 판단을 정리한다.

### 13.1 최종 판단 요약

| 항목 | 최종 상태 | 판단 근거 | 개발 적용 기준 |
| --- | --- | --- | --- |
| L 경량형 유지 | 확정 | L은 640×360 소스 / 640×384 TRT 입력 속도형이므로 구조 추가보다 latency 보존 우선 | Backbone/Neck 원본, AUX off, P2 off |
| W6 공격형 적용 | 확정 | W6는 1280×720 소스 / 1280×736 TRT 입력 정확도형이므로 소형 객체/박스 품질 개선 여지 큼 | AUX on, P2 Anchor, SCDown, WIoU, TAL+VFL |
| Decoupled Head | 확정 | cls/reg 분리로 task 충돌 완화 가능 | L/W6 공통 적용 |
| WIoU v3 | 공통 확정 | 추론 구조 영향 없이 box regression 품질 개선 가능 | CIoU fallback 유지 |
| TAL + VFL | 공통 확정 | classification/localization alignment 및 IoU-aware score 학습 목적과 부합 | VFL 단독 금지, TAL과 세트 |
| P2 Anchor Head | W6 확정 | FCOS보다 기존 YOLO head와 output/postprocess 일관성 높음 | W6 적용, L 제외 |
| SCDown | W6 확정 | Conv 계열로 TRT 리스크 낮고 고해상도 W6 효율 개선 여지 있음 | W6 적용, L 제외 |
| PSA | 2차 후보 | attention 계열로 latency/profile 편차 가능 | W6 P5 단독부터 검증 |
| FCOS P2 | 후순위 | anchor head와 score/loss/postprocess 결합 복잡도 높음 | P2 Anchor 부족 시만 검토 |
| GELAN | 최후순위 | YOLOv9 PGI/GELAN 전체 맥락과 분리 적용 시 효과 보장 어려움 | 채널/route 검증 후 단독 실험 |
| raw ONNX output | 확정 | 후처리 구현과 모델 export 검증을 분리하기 위함 | --nms-mode none 기본 |
| Replay Buffer + YOLO LwF | Replay 확정 / LwF A-B | continual learning에서 기존 클래스 망각 억제 목적과 부합 | replay-ratio 0.3 기본, LwF는 성능 비교 |

### 13.2 논문별 반영 기준

| 논문/기술 | 핵심 내용 | 요구서 반영 | 주의점 |
| --- | --- | --- | --- |
| YOLOv7 | E-ELAN, trainable bag-of-freebies, lead/auxiliary head | W6 AUX on 유지, L은 AUX off 기본 | L AUX는 성능형 옵션으로만 유지 |
| WIoU v3 | dynamic focusing mechanism 기반 box loss | L/W6 공통 box loss 확정 | .detach 및 running mean 관리 필요 |
| TOOD / TAL | classification-localization task alignment | TAL 적용 확정 | VFL과 세트로 적용 |
| VarifocalNet / VFL | IoU-aware classification score 학습 | TAL 적용 시 cls loss로 사용 | TAL 없는 VFL 단독 금지 |
| FCOS | anchor-free + centerness 기반 per-pixel detector | FCOS P2는 후순위 | YOLO anchor output과 score/NMS 결합 복잡 |
| YOLOv10 | NMS-free 방향, SCDown/PSA 등 효율 구조 | SCDown은 W6 확정, PSA는 P5 후보 | YOLOv7 단독 이식 효과는 실측 필요 |
| YOLOv9 | PGI + GELAN 구조 | GELAN 최후순위 | PGI 없이 GELAN만 이식 시 효과 보장 어려움 |
| YOLO LwF | YOLO continual learning용 self-distillation + replay | Replay Buffer 확정, LwF A/B | teacher forward 비용 및 distill loss 튜닝 필요 |

### 13.3 최종 모델별 개발 착수 기준

YOLOv7-L 최종 착수 기준

• 역할: 속도형 모델, 640×360 추론 중심

• 구조: Backbone 원본, Neck 원본, Decoupled Head 적용

• 제외: AUX 기본 off, P2 Head off, SCDown/PSA/GELAN 제외

• 학습: WIoU v3 + TAL + VFL 확정, CIoU/BCE/SimOTA fallback 유지

• 운영: raw ONNX output export. C++/TensorRT 추론은 별도 차수에서 검토

YOLOv7-W6 최종 착수 기준

• 역할: 정확도형 모델, 1280×720 추론 중심

• 구조: Backbone 원본, Decoupled Head, AUX on, P2 Anchor Head, SCDown 적용

• 학습: WIoU v3 + TAL + VFL 확정

• 후순위: PSA P5, FCOS P2, GELAN

• 운영: raw ONNX output export. C++/TensorRT 추론은 별도 차수에서 검토

### 13.4 후순위 항목 적용 조건

| 후순위 항목 | 적용 조건 | 적용 순서 | 중단 기준 |
| --- | --- | --- | --- |
| PSA | W6 기본안에서 mAP/recall이 부족하고 latency 여유가 있을 때 | P5 단독 → P4 추가 → P3 추가 | TRT FP16 latency/profile 편차가 커지면 중단 |
| FCOS P2 | W6 P2 Anchor로도 소형 객체 recall이 부족할 때 | W6 단독 실험 | postprocess 복잡도 또는 속도 손실이 크면 중단 |
| GELAN | P2/SCDown/PSA 이후에도 구조 성능 상한 개선이 필요할 때 | W6 Neck 일부 단독 교체 | 채널/route/export 안정성 문제 발생 시 중단 |
| L AUX on | L 경량형에서 희귀 클래스 recall이 부족할 때 | L 성능형 옵션으로 별도 실험 | 학습 안정성/효과 미미하면 off 유지 |

### 13.5 개발 게이트

1. Baseline L/W6 학습 결과와 v1.3 최종안 결과를 반드시 동일 validation set에서 비교한다.
2. `tools/profile_model.py`로 L/W6 GFLOPs delta가 10% 미만인지 확인한다.
3. Phase 전환 unit test와 DataLoader rebuild smoke test를 통과해야 한다.
4. 구조 변경 직후 `tools/verify_export.py`로 PyTorch/ONNX Runtime raw output을 비교한다.
5. W6 P2 Anchor/SCDown 적용 후 latency 또는 GFLOPs가 목표 범위를 벗어나면 PSA/FCOS/GELAN 실험을 보류한다.
6. L 모델은 구조 경량 유지 원칙을 깨지 않는다. L의 성능 개선은 WIoU/TAL/VFL, Aug, Hard Negative 중심으로 처리한다.
7. 파인튜닝은 Replay Buffer를 먼저 적용하고, LwF는 forgetting이 남을 때 A/B로 활성화한다.

기밀 — 사내 배포용 / v1.3 최종 개발 요구서
