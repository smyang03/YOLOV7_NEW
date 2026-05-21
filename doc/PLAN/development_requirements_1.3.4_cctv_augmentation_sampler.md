# 1.3.4 Code-Level Development Requirements

## 1.3.4.1 코드 구현 상세

이 세부 항목은 현재 augmentation이 `utils/datasets.py` 안에 집중되어 있는 구조를 고려해 policy, pixel augmentation, label-changing augmentation, sampler를 분리하는 구현 기준을 고정한다.

### 대상 파일과 함수

| 파일 | 클래스/함수 | 구현 방식 |
| --- | --- | --- |
| `utils/datasets.py` | `LoadImagesAndLabels.__getitem__` | 기존 mosaic, mixup, random_perspective 흐름은 유지하고 `AugmentPolicy` 호출 지점을 명확히 삽입한다. |
| `utils/datasets.py` | `augment_hsv`, `copy_paste`, `load_mosaic`, `load_mosaic9` | 기존 함수는 regression 방지를 위해 그대로 두고, CCTV 전용 aug는 신규 모듈에서 호출한다. |
| `utils/cctv_augmentations.py` | pixel aug 함수들 | `spider_web`, `to_gray3`, `clahe`, `motion_blur`, `compression_noise`를 label-preserving 함수로 구현한다. |
| `utils/augment_policy.py` | `AugmentPolicy` | `off`, `cctv_pixel`, `cctv_paste` profile과 phase별 확률을 관리한다. |
| `utils/sampler.py` | `build_weighted_sampler()` | class/image weight를 계산하고 DDP 미지원 시 명확히 실패 또는 warning 처리한다. |
| `tools/check_aug_visual.py` | CLI | aug sample을 저장하고 bbox overlay를 그린다. |
| `tools/check_labels.py` | CLI | bbox range, class id, empty label, tiny box 비율을 검사한다. |

### argparse 구현 규칙

```python
parser.add_argument('--aug-profile', choices=['off', 'cctv_pixel', 'cctv_paste'], default='off')
parser.add_argument('--sampler-mode', choices=['off', 'weighted'], default='off')
parser.add_argument('--aug-debug-samples', type=int, default=0)
parser.add_argument('--hard-negative-manifest', type=str, default='')
```

### AugmentPolicy schema

```python
@dataclass
class AugmentPolicy:
    profile: str
    phase: str
    spider_web_p: float = 0.0
    gray_p: float = 0.0
    clahe_p: float = 0.0
    blur_p: float = 0.0
    patch_paste_p: float = 0.0
    hard_negative_p: float = 0.0
```

`profile='off'`일 때는 기존 YOLOv7 augmentation만 동작해야 한다. CCTV aug 함수는 image dtype, shape, label count를 바꾸지 않아야 한다. label-changing aug는 반드시 label validator를 통과해야 한다.

### sampler 구현 규칙

기존 `utils/general.py::labels_to_image_weights()`를 재사용하되, 새 sampler는 아래를 기록한다.

- class별 image count
- class별 sampling weight
- image별 final weight
- epoch별 sample histogram

DDP에서는 distributed-aware sampler가 준비되기 전까지 `--sampler-mode weighted`를 단일 GPU 전용으로 제한한다.

### 검증 명령

```bash
python tools/check_aug_visual.py --data data/coco128.yaml --aug-profile cctv_pixel --samples 200 --output runs/aug_check/cctv_pixel
python tools/check_labels.py --data data/coco128.yaml
python train.py --data data/coco128.yaml --epochs 1 --aug-profile cctv_pixel --sampler-mode off --name smoke_1341_aug
```

필수 확인:
- visual sample 200장 저장
- bbox 좌표가 `[0, 1]` 범위 유지
- class id가 `0 <= cls < nc`
- label-changing aug는 full training 전 반드시 별도 audit 완료

## 리포트 기반 정비 기준

- 문서 위치 기준: 본 코드레벨 개발 요구서는 `doc/PLAN/`에 둔다.
- 기준 리포트: `doc/REPORT/ai_perspective_yolov7_improvement_analysis_2026-05-22.md`
- augmentation 효과는 모델 구조 효과와 분리해 검증한다.
- label-changing augmentation은 visual audit과 bbox/class id 검사를 통과하기 전 full training에 사용하지 않는다.
- sampler는 augmentation 검증 후 별도 PR/run으로 적용한다.

- 기준 계획: `doc/PLAN/development_plan_v1.3.md`
- 대상 차수: `1.3.4 CCTV Augmentation / Sampler`
- 선행 조건: `1.3.3` core model/loss 단계 통과
- 목적: CCTV 환경 증강과 sampler 효과를 모델 구조 효과와 분리해 검증한다.

## 1. 범위

포함:
- label-preserving pixel augmentation
- label-changing Patch-Paste
- Hard Negative Paste
- Weighted sampler
- augmentation 시각 검증 도구
- sampler 통계와 label 오염 검사
- phase별 augmentation 적용 범위 고정
- CCTV scenario metric 기록

제외:
- Head/Loss/Assignment 추가 변경
- W6 P2 Anchor, SCDown
- FCOS, PSA, GELAN
- C++ 후처리, TensorRT runtime, 추론 서버

## 2. 하위 단계

| 단계 | 목적 | 실행 플래그 |
| --- | --- | --- |
| `1.3.4-B1` | pixel aug만 적용 | `--aug-profile cctv_pixel --sampler-mode none` |
| `1.3.4-B2` | label-changing aug 적용 | `--aug-profile cctv_paste --sampler-mode none` |
| `1.3.4-B3` | sampler 적용 | `--aug-profile cctv_paste --sampler-mode weighted` |

B2/B3는 B1 시각 검증이 통과한 뒤 진행한다.

## 3. 파일 단위 개발 요구사항

| 파일 | 구분 | 요구사항 |
| --- | --- | --- |
| `train.py` | 수정 | `--aug-profile`, `--sampler-mode`, `--aug-debug` 플래그를 저장하고 dataloader에 전달한다. |
| `utils/datasets.py` | 수정 | 기존 mosaic/random_perspective 흐름에 CCTV aug hook을 추가한다. bbox 좌표 변환 순서를 명확히 유지한다. |
| `utils/cctv_augmentations.py` | 신규 | pixel aug, Patch-Paste, Hard Negative Paste를 함수 단위로 분리한다. |
| `utils/augment_policy.py` | 신규 | `cctv_pixel`, `cctv_paste`, `off` profile과 phase별 aug enable/disable 정책을 제공한다. |
| `utils/sampler.py` | 신규 | class-balanced image sampler를 구현한다. DDP 미지원 시 단일 GPU에서만 활성화한다. |
| `tools/check_aug_visual.py` | 신규 | aug별 샘플 이미지를 저장하고 bbox/class 오염 여부를 요약한다. |
| `tools/check_labels.py` | 신규/확인 | bbox clipping, normalize range, class id 범위를 검사한다. |
| `tools/mine_hard_negatives.py` | 신규 | baseline model의 false positive 영역과 GT 없는 이미지에서 hard negative crop 후보를 생성한다. |
| `data/hyp_phase*.yaml` | 수정 | aug 확률과 sampler weight key를 추가한다. 기본값은 보수적으로 둔다. |

## 4. CLI 요구사항

Pixel aug smoke:

```bash
python train.py --cfg cfg/training/yolov7.yaml --data data/custom_example.yaml --hyp data/hyp_phase1.yaml --epochs 1 --img 640 --batch 16 --aug-profile cctv_pixel --sampler-mode none --name aug_pixel_smoke
```

시각 검증:

```bash
python tools/check_aug_visual.py --data data/custom_example.yaml --hyp data/hyp_phase1.yaml --aug-profile cctv_paste --samples 200 --output runs/aug_check/cctv_paste
```

Sampler smoke:

```bash
python train.py --cfg cfg/training/yolov7.yaml --data data/custom_example.yaml --hyp data/hyp_phase1.yaml --epochs 1 --img 640 --batch 16 --aug-profile cctv_paste --sampler-mode weighted --name sampler_smoke
```

## 5. Augmentation 계약

### 5.1 Aug Profile Matrix

| Aug | 기본 profile | 확률 | label 변경 | 비고 |
| --- | --- | --- | --- | --- |
| SpiderWeb | `cctv_pixel` | 0.05 | 없음 | Mosaic 이전 단일 이미지 단계 |
| IR reflection / overexposure | `cctv_pixel` | 0.20 | 없음 | 안전모/흰 물체 과노출 |
| ToGray 3-channel | `cctv_pixel` | 0.15 | 없음 | RGB/IR 혼재 대응 |
| Mosaic9 | 옵션 | 0.0 | 있음 | 기본 off, 소형 객체 극단 부족 시만 |
| RandomAffine scale/shear | 기본 | hyp | 있음 | 기존 YOLO 경로 유지 |
| Flip LR | 기본 | 0.50 | 있음 | 기존 YOLO 경로 유지 |
| Flip UD | 옵션 | 0.00~0.05 | 있음 | 천장 카메라 대응 |
| Patch-Paste | `cctv_paste` | 0.10~0.15 | 있음 | 희귀/소형 우선 |
| Hard Negative Paste | `cctv_paste` | 0.20 | 없음 | GT 추가 금지 |
| Helmet-on-person paste | 옵션 | 0.05 | 있음 | 기본 off, 시각 검증 후만 |
| GridMask | `cctv_pixel` | 0.25 | 없음 | 부분 가림 |
| RandomShift | `cctv_pixel` | 0.20 | 있음 | GT 동일 이동 |
| DirectionalMotionBlur | `cctv_pixel` | 0.15 | 없음 | 카메라 진동 |
| OneOf[JPEG, Downscale, Pixelate] | `cctv_pixel` | 0.40 | 없음 | 비트레이트 열화 |
| OneOf[GaussNoise, ISONoise, S&P] | `cctv_pixel` | 0.30 | 없음 | 센서 노이즈 |
| OneOf[MotionBlur, Defocus, DirBlur] | `cctv_pixel` | 0.25 | 없음 | 모션/초점 열화 |
| RandomFog | `cctv_pixel` | 0.05~0.10 | 없음 | 렌즈 오염 |
| LensFlare | `cctv_pixel` | 0.05 | 없음 | 역광 플레어 |
| RandomSunFlare | `cctv_pixel` | 0.05~0.10 | 없음 | 상단 집중 |
| CLAHE | `cctv_pixel` | 0.20~0.40 | 없음 | IR/야간 대비 |
| RandomShadow | `cctv_pixel` | 0.20 | 없음 | 그림자 오감지 억제 |
| OcclusionAug | `cctv_pixel` | 0.15 | 없음 | GT 위 부분 가림 |
| MixUp | 옵션 | 0.05 | 있음 | 최소화 |
| GlassBlur | 제외 | 0.0 | 없음 | Defocus와 중복 |
| Posterize | 제외 | 0.0 | 없음 | 실효성 낮음 |
| Rolling Shutter | 선택 | 0.0 | 있음 | 구현 복잡도, 기본 제외 |

### 5.2 Phase별 적용 범위

| Aug 그룹 | Phase 1 | Phase 2 | Phase 3 |
| --- | --- | --- | --- |
| Mosaic4 / Mosaic9 | ON | ON | OFF |
| Pixel CCTV aug | ON | ON | 제한 ON |
| Patch-Paste / Hard Negative | ON | ON | OFF |
| SpiderWeb 단일 이미지 | ON | ON | ON |
| Sampler | ON 가능 | ON 가능 | OFF 권장 |

Phase 3는 최종 입력 분포 안정화 구간이므로 label-changing aug를 끈다.

### 5.3 Pixel Aug

대상:
- low-light, brightness/contrast, gamma, CLAHE
- JPEG/downscale/pixelate
- noise, IR-like noise
- motion/defocus/directional blur
- shadow, lens flare, sun flare
- spider web, occlusion overlay

계약:
- GT bbox와 class id를 변경하지 않는다.
- image dtype, range, channel order를 기존 dataloader와 동일하게 유지한다.
- 확률은 hyp에서 읽고, `--aug-profile off`면 완전히 비활성화한다.
- 원본 image/label cache에는 증강 결과를 저장하지 않는다. 증강은 `__getitem__` runtime에서만 적용한다.

### 5.4 Patch-Paste

계약:
- hook 지점은 pixel xyxy label 상태로 고정한다. 최종 return 직전에는 기존 dataloader와 동일하게 normalized xywh로 변환한다.
- paste 후보는 희귀/소형 클래스 우선으로 선택한다.
- 기존 GT와 IoU/IoA가 `0.3` 이상이면 해당 위치를 버린다.
- clipping 후 bbox 면적이 원본 대비 30% 미만이면 GT를 추가하지 않는다.
- visible area가 50% 미만이면 GT를 추가하지 않는다.
- aspect ratio가 `1:8` 또는 `8:1`을 넘으면 폐기한다.
- 실패 시 원본 이미지를 그대로 반환하고 실패 횟수를 로그에 남긴다.

### 5.5 Hard Negative Paste

계약:
- GT 없는 이미지 또는 baseline false positive 영역에서 crop을 만든다.
- paste 영역에는 GT를 추가하지 않는다.
- 기존 GT와 IoU가 `0.1` 이상이면 해당 위치를 버린다.
- false positive 억제 목적이므로 class label을 만들지 않는다.
- hard negative 후보는 `tools/mine_hard_negatives.py`로 생성할 수 있다.
- 후보 source는 baseline false positive crop, GT 없는 이미지의 edge/texture 영역이다.

### 5.6 Dataloader 통합 지점

- `utils/datasets.py`의 기존 `load_mosaic`, `load_mosaic9`, `random_perspective`, `augment_hsv`, `pastein` 순서를 깨지 않는다.
- label-preserving pixel aug는 geometric transform 이후, tensor 변환 이전에 적용한다.
- label-changing aug는 bbox가 pixel xyxy인 구간에서만 실행한다.
- 각 aug 함수는 `(image, labels, debug_info)` 형태로 결과를 반환하고, 실패 시 원본 image/labels를 반환한다.

## 6. Sampler 계약

- `--sampler-mode none|weighted`
- `weighted` 사용 시 `--image-weights`와 중복 사용하지 않는다.
- argparse `store_true` 옵션인 `--image-weights`에는 `False` 값을 전달하지 않는다. sampler 사용 시 해당 옵션을 생략한다.
- class frequency는 label cache에서 계산한다.
- 빈 라벨 이미지는 hard negative 비율 기준으로 제한한다.
- DDP distributed-aware sampler가 구현되기 전에는 `world_size > 1`에서 에러를 낸다.
- `sampler_stats.csv`에 epoch별 class sampling count를 기록한다.

## 7. 산출물

필수 산출물:
- `aug_check.json`
- `aug_samples/*.jpg`
- `sampler_stats.csv`
- `scenario_metrics.csv`
- `stage_result.yaml`
- `results.csv`

`aug_check.json` 필수 필드:
- `aug_profile`
- `samples`
- `label_preserving`
- `bbox_range_errors`
- `class_id_errors`
- `paste_failures`
- `hard_negative_pastes`
- `manual_review_required`
- `status`

`scenario_metrics.csv` 컬럼:
- `epoch`
- `phase`
- `scenario`: `small`, `rare`, `backlight`, `low_light`, `ir`, `occlusion`, `hard_negative`
- `AP_0.5`
- `recall`
- `false_positive_per_image`
- `sample_count`

## 8. 통과 기준

1. B1 시각 샘플 200장 생성 후 bbox/class 오염이 없다.
2. B2 시각 샘플 200장 생성 후 paste bbox 오염이 없다.
3. B3 sampler smoke가 단일 GPU에서 완료된다.
4. `sampler_stats.csv`에서 희귀 클래스 sampling count 증가가 확인된다.
5. label-changing aug 실패 시 학습이 중단되지 않는다.
6. hard negative paste가 GT를 추가하지 않는 것이 확인된다.
7. 후속 ONNX export 검증 경로가 계속 통과한다.
8. `scenario_metrics.csv`에 오감지/미감지 관련 scenario metric이 기록된다.
9. Phase 3에서 Patch-Paste/Hard Negative/Mosaic이 꺼진 것이 로그로 확인된다.

## 9. 구현 순서

1. `--aug-profile`, `--sampler-mode` CLI 추가
2. `utils/cctv_augmentations.py`에 pixel aug 구현
3. `utils/augment_policy.py`에 profile/phase 정책 작성
4. `tools/check_aug_visual.py` 작성
5. Patch-Paste 안전장치 구현
6. `tools/mine_hard_negatives.py` 작성
7. Hard Negative Paste 구현
8. `utils/sampler.py` 작성
9. scenario metric 기록 연결
10. B1, B2, B3 순서로 smoke와 report 저장

## 10. 리스크 및 주의사항

- label-changing aug는 시각 검증 전 full training에 사용하지 않는다.
- 클래스 수량 기준은 예시일 뿐 고정값으로 하드코딩하지 않는다.
- 파인튜닝 클래스 수량도 본 차수 기준이 아니다.
- 데이터 변경 후 label cache 재생성 정책은 1.3.1/1.3.2 기준을 그대로 따른다.
- 증강 확률 변경만으로 label cache를 무효화하지 않는다. cache는 원본 데이터 manifest 기준으로만 관리한다.

## 11. 개발 착수 분리 기준

augmentation은 시각적으로 정상이어도 label 오염을 만들 수 있으므로 작은 단위로만 병합한다.

| PR | 범위 | 완료 기준 |
| --- | --- | --- |
| `1.3.4-P1` | `utils/augment_policy.py`, profile/phase on/off | `--aug-profile off/cctv_pixel/cctv_paste` dry-run 통과 |
| `1.3.4-P2` | label-preserving pixel aug 최소 세트 | SpiderWeb, ToGray, CLAHE, blur 계열 샘플 200장 저장 |
| `1.3.4-P3` | Patch-Paste 안전장치 | bbox range/class id 검사 통과 |
| `1.3.4-P4` | Hard Negative mining/paste | GT 추가 없음, false positive crop manifest 생성 |
| `1.3.4-P5` | weighted sampler | 단일 GPU smoke와 `sampler_stats.csv` 생성 |
| `1.3.4-P6` | scenario metric 연결 | `scenario_metrics.csv` 생성 |

처음 구현하는 pixel aug 최소 세트는 SpiderWeb, ToGray, CLAHE, blur 계열로 제한한다. LensFlare, RandomSunFlare, Helmet paste, MixUp, Rolling Shutter는 기본 세트 통과 후 옵션으로만 추가한다.
