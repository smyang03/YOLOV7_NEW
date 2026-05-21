# 1.3.1 Code-Level Development Requirements

- 기준 계획: `doc/PLAN/development_plan_v1.3.md`
- 대상 차수: `1.3.1 Baseline / Python Export 기준선`
- 목적: 모델 개선 전 원본 YOLOv7-L/W6의 학습, 평가, Python-side ONNX export 기준선을 고정한다.

## 1. 범위

1.3.1에서는 모델 구조, loss, augmentation, sampler를 변경하지 않는다. 기존 YOLOv7 동작을 보존하면서 baseline 학습/평가/export 산출물과 Python 검증 도구만 만든다.

포함:
- baseline 학습을 막는 기존 경로 회귀 보정
- L/W6 baseline smoke 학습 명령 정리
- `best.pt`, `last.pt`, `results.txt` 생성 보장
- `test.py` 평가 반환값과 학습 후 평가 호출 정합성 확보
- ONNX raw output export 기준화
- raw output layout/shape 계약 고정
- PyTorch/ONNX Runtime output 비교 스크립트
- params/GFLOPs profile 스크립트
- train/val dataset manifest 및 validation checksum 생성
- baseline primary/secondary metric snapshot 저장
- baseline 산출물 manifest/report 저장

제외:
- C++ 후처리 및 C++ NMS
- TensorRT runtime, TensorRT engine build, TensorRT output 비교
- 별도 추론 서버, 운영 배포 코드
- Decoupled Head, WIoU, TAL/VFL
- P2 Anchor, SCDown, AUX 신규 실험
- CCTV augmentation, sampler
- Phase 자동 전환 구현

## 2. 파일 단위 개발 요구사항

| 파일 | 구분 | 요구사항 |
| --- | --- | --- |
| `train.py` | 수정 | baseline 학습 시 `best.pt`, `last.pt`, `results.txt` 생성 보장. `--img`, `--batch` alias를 추가하거나 문서 명령과 parser를 일치시킨다. |
| `train_aux.py` | 수정 | W6 baseline도 `best.pt`, `last.pt`, `results.txt` 생성 보장. `test.test()` 반환값 unpack 정합성 수정. |
| `test.py` | 확인/최소 수정 | standalone 평가와 training 호출 양쪽에서 `(results, maps, times, per_class_results)` 반환 계약을 유지한다. |
| `export.py` | 수정 | `--opset` 기본값 16, `--nms-mode none` 옵션 추가. AUX는 eval/export 시 비활성화하고 raw ONNX output을 `[batch, total_boxes, 5 + nc]`로 고정한다. |
| `utils/datasets.py` | 수정 | 일반 YOLO layout의 `images` -> `labels` 매핑과 label cache hash/version 검증을 보장한다. `persistent_workers`는 `workers > 0`에서만 켜지게 한다. |
| `utils/general.py` | 확인/최소 수정 | `opt.yaml`에 저장되는 `save_dir`이 `yaml.SafeLoader`로 다시 읽을 수 있는 문자열이어야 한다. |
| `utils/plots.py` | 확인 | `results.txt`는 `plot_results()`가 읽을 수 있는 numeric column 포맷을 유지한다. per-class 텍스트는 별도 파일에 저장한다. |
| `tools/verify_export.py` | 신규 | PyTorch와 ONNX Runtime output을 같은 입력으로 비교하고 `export_check.json`, `output_contract.json`을 저장. |
| `tools/profile_model.py` | 신규 | params, GFLOPs, input shape, weight path를 `profile.json`으로 저장. |
| `tools/dataset_manifest.py` | 신규 | data yaml 기준 train/val image, label count, file hash/checksum을 `dataset_manifest.json`으로 저장. |
| `tools/summarize_metrics.py` | 신규 | `results.txt` 또는 test 결과에서 baseline primary/secondary metric을 추출해 `baseline_metrics.json`으로 저장. |
| `requirements.txt` | 확인/최소 수정 | 1.3.1 실행에 필요한 `onnx`, `onnxruntime`, `thop` 의존성 상태를 명확히 한다. |
| `doc/REPORT/baseline_1.3.1_*.md` | 신규 | 실행 명령, 환경, 산출물 경로, 측정 결과를 기록. |

## 3. CLI 요구사항

기본 방침은 사용자 편의를 위해 `--img`/`--batch` alias를 지원하는 것이다. 내부 변수명은 기존 `img_size`/`batch_size`를 유지해도 된다.

학습 스크립트와 export/검증 스크립트의 `--img` 의미를 혼동하지 않는다.
- `train.py`, `train_aux.py`: 기존 호환을 위해 `--img 640`은 square train/test size, `--img 640 640`은 기존 YOLOv7 방식의 train/test scalar size로 해석한다.
- `export.py`, `tools/verify_export.py`, `tools/profile_model.py`: `--img H W`를 input tensor shape으로 해석한다.
- rectangular finetune의 실제 H/W 입력은 1.3.2의 `--phase2-img H W`, `--phase3-img H W`에서 처리한다.

### 3.1 Baseline Smoke 학습

YOLOv7-L:

```bash
python train.py --cfg cfg/training/yolov7.yaml --data data/custom_example.yaml --hyp data/hyp.scratch.p5.yaml --epochs 1 --img 640 --batch 32 --name baseline_l_smoke
```

YOLOv7-W6:

```bash
python train_aux.py --cfg cfg/training/yolov7-w6.yaml --data data/custom_example.yaml --hyp data/hyp.scratch.p6.yaml --epochs 1 --img 1280 --batch 8 --name baseline_w6_smoke
```

실제 full baseline은 학습 서버에서 동일 명령 구조로 epoch와 dataset yaml만 운영 기준에 맞춘다.

### 3.1.1 학습 경로 안전성

1.3.1은 모델 성능 개선이 아니라 기준선 고정 차수다. 따라서 아래 항목은 신규 기능이 아니어도 baseline blocker로 처리한다.

- 일반 YOLO dataset layout은 `.../images/...` 이미지와 `.../labels/...` 라벨을 기본으로 매핑한다.
- label cache는 파일 목록, label 파일, cache version/hash가 바뀌면 자동으로 재생성한다.
- `workers 0`에서도 DataLoader가 생성되어야 한다.
- `--close-mosaic`가 켜진 경우 persistent worker가 이전 dataset 상태를 계속 들고 있지 않아야 한다. 필요하면 해당 옵션에서 persistent worker를 끄거나 DataLoader 재생성 정책을 명시한다.
- `results.txt`에는 숫자 컬럼만 기록한다. validation set 이름, per-class 결과, 부가 텍스트는 별도 report/json/md에 저장한다.
- `opt.yaml`은 `yaml.SafeLoader`로 다시 읽을 수 있어야 한다. `Path` 객체가 Python 전용 YAML tag로 저장되면 실패로 본다.

### 3.2 평가

```bash
python test.py --data data/custom_example.yaml --weights runs/train/baseline_l_smoke/weights/best.pt --img 640
```

`test.py`도 `--img` alias를 지원한다. 기존 `--img-size`와 동작은 동일해야 한다.

### 3.3 ONNX Export

L:

```bash
python export.py --weights runs/train/baseline_l_smoke/weights/best.pt --img 640 384 --opset 16 --nms-mode none
```

W6:

```bash
python export.py --weights runs/train/baseline_w6_smoke/weights/best.pt --img 1280 736 --opset 16 --nms-mode none
```

### 3.4 Export 검증

```bash
python tools/verify_export.py --weights runs/train/baseline_l_smoke/weights/best.pt --onnx runs/train/baseline_l_smoke/weights/best.onnx --img 640 384 --output runs/train/baseline_l_smoke/export_check.json
```

1.3.1의 비교 대상은 PyTorch와 ONNX Runtime이다. TensorRT engine 비교는 본 차수에서 제외한다.

검증 스크립트는 같은 위치에 `output_contract.json`도 저장한다.

### 3.5 Profile

```bash
python tools/profile_model.py --weights runs/train/baseline_l_smoke/weights/best.pt --cfg cfg/training/yolov7.yaml --img 640 384 --output runs/train/baseline_l_smoke/profile.json
```

### 3.6 Dataset Manifest

```bash
python tools/dataset_manifest.py --data data/custom_example.yaml --output runs/train/baseline_l_smoke/dataset_manifest.json
```

### 3.7 Baseline Metric Snapshot

```bash
python tools/summarize_metrics.py --results runs/train/baseline_l_smoke/results.txt --output runs/train/baseline_l_smoke/baseline_metrics.json
```

## 4. 산출물 스키마

### 4.1 `export_check.json`

필수 필드:
- `weights`
- `onnx`
- `input_shape`
- `providers`
- `max_abs_diff`
- `mean_abs_diff`
- `passed`
- `error`

통과 기준:
- PyTorch/ONNX Runtime `max_abs_diff <= 1e-3` 권장
- 실제 허용치는 baseline smoke 실행 후 report에 확정 기록

### 4.1.1 `output_contract.json`

필수 필드:
- `weights`
- `onnx`
- `nms_mode`: `none`
- `opset`
- `input_shape`
- `nc`
- `names`
- `output_layout`: `xywh_obj_cls`
- `output_shape`
- `total_boxes`
- `aux_exported`: `false`
- `dynamic_axes`
- `postprocess_in_graph`: `false`

계약:
- raw output은 후처리 전 tensor만 포함한다.
- NMS, C++ postprocess, TensorRT plugin node는 포함하지 않는다.
- AUX branch가 있는 모델도 export output은 main head 기준으로 고정한다.

### 4.2 `profile.json`

필수 필드:
- `model`
- `weights`
- `input_shape`
- `params`
- `gflops`
- `device`
- `torch_version`
- `cuda_available`

### 4.2.1 `baseline_metrics.json`

필수 필드:
- `primary_metric`: `mAP@0.5:0.95`
- `metrics/mAP_0.5`
- `metrics/mAP_0.5:0.95`
- `metrics/precision`
- `metrics/recall`
- `small_AP`
- `rare_recall`
- `source_results`
- `validation_checksum`

`small_AP`, `rare_recall`을 아직 계산할 수 없으면 `null`로 저장하되 report에 미계산 사유를 적는다.

### 4.3 Baseline report

`doc/REPORT/baseline_1.3.1_YYYY-MM-DD.md`에 아래 항목을 기록한다.

- 실행 환경: GPU, CUDA, PyTorch, ONNX Runtime 버전
- dataset yaml, image count, label count, validation checksum
- L/W6 학습 명령
- L/W6 평가 결과
- L/W6 params/GFLOPs
- ONNX export 및 PyTorch/ONNX 비교 결과
- dataset manifest 경로와 validation checksum
- output contract 경로
- baseline metrics snapshot 경로
- 산출물 경로

### 4.4 `dataset_manifest.json`

필수 필드:
- `data_yaml`
- `nc`
- `names`
- `train_image_count`
- `val_image_count`
- `train_label_count`
- `val_label_count`
- `train_hash`
- `val_hash`
- `missing_labels`
- `empty_labels`
- `created_at`

### 4.5 `manifest_1.3.1.json`

학습 서버 실행 결과는 run directory마다 manifest를 남긴다.

필수 필드:
- `stage`: `1.3.1`
- `run_name`
- `command`
- `git_commit`
- `data_yaml`
- `cfg`
- `hyp`
- `dataset_manifest`
- `validation_checksum`
- `weights_dir`
- `best_pt`
- `last_pt`
- `results_txt`
- `profile_json`
- `export_check_json`
- `output_contract_json`
- `baseline_metrics_json`
- `status`
- `created_at`

## 5. 통과 기준

1. L/W6 baseline smoke 학습이 완료된다.
2. `best.pt`, `last.pt`, `results.txt`가 생성된다.
3. `test.py` 평가가 정상 완료된다.
4. ONNX export가 `opset 16`, `nms-mode none`으로 완료된다.
5. `tools/verify_export.py`가 `export_check.json`을 생성한다.
6. `tools/profile_model.py`가 `profile.json`을 생성한다.
7. baseline report가 `doc/REPORT`에 저장된다.
8. `workers 0`과 `workers > 0` smoke 경로가 모두 DataLoader 생성 단계에서 실패하지 않는다.
9. 기존 `--img-size`, `--batch-size` 명령도 계속 동작한다.
10. `opt.yaml`을 `yaml.SafeLoader`로 다시 읽을 수 있다.
11. `dataset_manifest.json`이 생성되고 validation checksum이 baseline report에 기록된다.
12. `output_contract.json`의 `nms_mode=none`, `postprocess_in_graph=false`, `aux_exported=false`가 확인된다.
13. `baseline_metrics.json`에 primary metric과 secondary metric snapshot이 저장된다.

## 6. 구현 순서

1. CLI alias 정렬: `--img`, `--batch`
2. dataset label mapping/cache/persistent worker 안전성 보정
3. `best.pt` 저장 보장: `train.py`, `train_aux.py`
4. `test.test()` 반환값 unpack 정합성 수정
5. `results.txt`, `opt.yaml` 포맷 안전성 확인
6. `export.py`에 `--opset`, `--nms-mode none` 추가
7. `tools/profile_model.py` 최소 구현
8. `tools/verify_export.py` 최소 구현
9. `tools/dataset_manifest.py` 최소 구현
10. `tools/summarize_metrics.py` 최소 구현
11. L smoke 학습/평가/export 검증
12. W6 smoke 학습/평가/export 검증
13. 학습 서버 full baseline 실행 후 manifest/report 저장

## 7. 리스크 및 주의사항

- `onnx`가 없으면 export가 불가능하므로 설치 안내를 명확히 출력한다.
- `onnxruntime`이 없으면 `verify_export.py`는 PyTorch output만 저장하고 비교는 skip 처리한다.
- `train.py`의 기존 `results.txt` 포맷은 1.3.1에서 변경하지 않는다.
- baseline 산출물은 후속 차수 비교 기준이므로 덮어쓰지 않는다.
- C++/TensorRT runtime/추론 서버 구현은 본 요구서 범위가 아니다.
- smoke 단계에서 실패한 항목이 있으면 full baseline으로 넘어가지 않는다.
- L baseline과 W6 baseline은 서로 다른 run directory에 저장한다.

## 8. 개발 착수 분리 기준

1.3.1은 이후 모든 차수의 기반이므로 한 번에 구현하지 않는다. 아래 PR 단위로 분리한다.

| PR | 범위 | 완료 기준 |
| --- | --- | --- |
| `1.3.1-P1` | CLI alias, label mapping, cache invalidation, persistent worker | `--img/--batch`와 기존 옵션 모두 동작, `workers=0/>0` dataloader 생성 |
| `1.3.1-P2` | `best.pt`, `test.test()` 4-return, `opt.yaml`, `results.txt` 포맷 | smoke 학습 후 `best.pt/last.pt/results.txt` 존재, resume yaml load 통과 |
| `1.3.1-P3` | `export.py`, `tools/profile_model.py`, `tools/verify_export.py` | raw ONNX export, `profile.json`, `export_check.json`, `output_contract.json` 생성 |
| `1.3.1-P4` | `tools/dataset_manifest.py`, `tools/summarize_metrics.py`, baseline report | `dataset_manifest.json`, `baseline_metrics.json`, `manifest_1.3.1.json` 생성 |

개발자는 `P1`과 `P2`를 먼저 끝낸 뒤 export/profile 도구를 작성한다. 데이터 경로와 checkpoint가 안정화되기 전에는 `verify_export.py` 구현 검증이 흔들릴 수 있다.
