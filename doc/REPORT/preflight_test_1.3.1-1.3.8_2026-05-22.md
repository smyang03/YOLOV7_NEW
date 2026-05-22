# YOLOv7 v1.3.1-1.3.8 Preflight Test Report

- 작성일: 2026-05-22
- 목적: 학습 서버 full run 전에 코드, 데이터, 모델, loss, phase, sequence runner가 최소 smoke 기준을 통과하는지 확인한다.
- 환경: Windows, Python/Anaconda, PyTorch 1.12.0, CUDA 사용 가능, NVIDIA GeForce RTX 3050 Laptop GPU 4GB
- 데이터: `data/coco128.yaml`, 실제 경로 `../coco128`

## 1. 종합 결과

판정: `조건부 통과`

COCO128 quick 기준으로 L baseline, Phase/DataLoader rebuild, W6/AUX smoke, loss smoke, cfg/model build, stage sequence dry-run은 통과했다. 테스트 중 발견된 산출물/경로 문제는 즉시 수정했다.

Full 학습 전 남은 주의점:
- `epochs=0`은 scheduler에서 division by zero를 만들므로 smoke에도 사용하지 않는다. 최소 `epochs=1`을 사용한다.
- COCO128은 성능 판단용이 아니며, 이번 결과의 mAP는 사용하지 않는다.
- ONNX export는 이번 preflight에서 강제하지 않았다. 필요 시 sequence runner에 `--require-export`를 켠 별도 실행으로 본다.

## 2. 테스트 중 수정한 항목

| 파일 | 수정 내용 | 이유 |
| --- | --- | --- |
| `data/coco128.yaml` | `./coco128/...` -> `../coco128/...` | YOLOv7 `check_dataset()` 다운로드 위치와 맞춤 |
| `train.py` | `log-format csv/both`이면 baseline에서도 `TrainLogger` 생성 | Stage 00 baseline도 `results.csv`, `loss_detail.csv`, `stage_result.yaml` 필요 |
| `train_aux.py` | `train.py`와 동일하게 structured logging 적용 | W6/AUX baseline 산출물 parity 필요 |
| `tools/run_training_sequence.py` | COCO128 quick Stage 01에 `phase1/2/3=1` override 및 `epochs>=3` 적용 | sequence runner에서도 Phase 2/3 rebuild를 실제로 밟게 함 |

## 3. 실행한 테스트

### 3.1 문법/CLI

통과:
- `python -m py_compile train.py train_aux.py test.py export.py ...`
- `python train.py --help`
- `python train_aux.py --help`
- `python test.py --help`
- `python export.py --help`
- `python tools/check_phase_schedule.py --phase1-epochs 1 --phase2-epochs 1 --phase3-epochs 1`

비고:
- `--img 64 --batch 1` alias는 parser에서 `img_size=[64,64]`, `batch_size=1`로 잡히는 것을 확인했다.
- 단, `epochs=0` 실행은 scheduler에서 실패하므로 preflight 명령으로 쓰지 않는다.

### 3.2 YAML/schema/data

통과:
- `data/coco128.yaml`: `nc=80`, `names=80`
- `data/hyp.scratch.p5.yaml`, `data/hyp.scratch.p6.yaml`, `data/hyp_finetune.yaml` load
- `utils.stage_schema.StageConfig`, `StageResult` validation
- COCO128 다운로드: `E:\code\coco128`
- `python tools/check_labels.py --data data/coco128.yaml`
- `python tools/dataset_manifest.py --data data/coco128.yaml --output runs/tmp_preflight/coco128_dataset_manifest.json`

Label check 결과:
- label rows: `1858`
- format errors: `0`
- class id errors: `0`
- bbox range errors: `0`
- tiny boxes: `278`
- status: `pass`

### 3.3 모델 build/profile/output contract

통과:
- `cfg/training/yolov7.yaml`: `IDetect`, stride `[8,16,32]`
- `cfg/training/yolov7-w6.yaml`: `IAuxDetect`, stride `[8,16,32,64]`
- `cfg/training/yolov7-w6-scdown.yaml`: build 통과
- `cfg/training/yolov7-w6-p2.yaml`: `IAuxDetect`, stride `[4,8,16,32,64]`
- `cfg/training/yolov7-w6-p2-scdown.yaml`: `IAuxDetect`, stride `[4,8,16,32,64]`
- `cfg/experiments/yolov7-w6-psa-p5.yaml`: build 통과
- `cfg/experiments/yolov7-w6-gelan-neck.yaml`: build 통과
- `tools/check_output_contract.py --cfg cfg/training/yolov7-w6-p2-scdown.yaml --img 128 128`: `status=pass`
- `tools/decode_fcos_outputs.py --allow-synthetic`: `status=pass`

Profile smoke:
- L 64x64: `1.0647 GFLOPs`
- W6 P2+SCDown 128x128: `4.4168 GFLOPs`

## 4. Loss Smoke

통과:
- L coupled + CIoU + SimOTA + BCE
- L coupled + WIoU v3 + SimOTA + BCE
- L coupled + CIoU + TAL + VFL
- L decoupled + WIoU v3 + TAL + VFL + empty targets
- W6 decoupled AUX + WIoU v3 + TAL + VFL + empty targets
- W6 CUDA AUX + CIoU + SimOTA + BCE + non-empty targets

W6 CUDA AUX non-empty 결과:
- `loss_name`: `ComputeLossAuxOTA`
- `positive_count`: `11`
- `status`: `pass`

## 5. 실제 COCO128 학습 Smoke

### L baseline 1 epoch

명령 요약:

```bash
python train.py --weights= --cfg cfg/training/yolov7.yaml --data data/coco128.yaml --hyp data/hyp.scratch.p5.yaml --epochs 1 --img-size 64 64 --batch-size 2 --workers 0 --device 0 --project runs/tmp_preflight_train --name smoke_l_csv --exist-ok --noautoanchor
```

통과 산출물:
- `weights/best.pt`
- `weights/last.pt`
- `results.txt`
- `results.csv`
- `loss_detail.csv`
- `stage_result.yaml`
- `opt.yaml` SafeLoader load 성공

`stage_result.yaml`:
- `stage: 1.3.1`
- `status: completed`

### Phase/DataLoader rebuild 3 epoch

명령 요약:

```bash
python train.py --weights= --cfg cfg/training/yolov7.yaml --data data/coco128.yaml --hyp data/hyp.scratch.p5.yaml --epochs 3 --img-size 64 64 --batch-size 2 --workers 0 --device 0 --project runs/tmp_preflight_train --name smoke_phase --exist-ok --noautoanchor --phase-train on --phase1-epochs 1 --phase2-epochs 1 --phase3-epochs 1 --phase2-img 64 64 --phase3-img 64 64
```

통과 산출물:
- `phase_transition.log`
- `results.csv`
- `loss_detail.csv`
- `stage_result.yaml`
- `hyp_used.yaml`
- `weights/best.pt`
- `weights/last.pt`

검증:
- `to_phase=phase2`: 확인
- `to_phase=phase3`: 확인
- `stage: 1.3.2`
- `status: completed`

### W6/AUX 1 epoch

명령 요약:

```bash
python train_aux.py --weights= --cfg cfg/training/yolov7-w6.yaml --data data/coco128.yaml --hyp data/hyp.scratch.p6.yaml --epochs 1 --img-size 128 128 --batch-size 1 --workers 0 --device 0 --project runs/tmp_preflight_train --name smoke_w6_aux --exist-ok --noautoanchor
```

통과 산출물:
- `weights/best.pt`
- `weights/last.pt`
- `results.txt`
- `results.csv`
- `loss_detail.csv`
- `stage_result.yaml`
- `opt.yaml`

`stage_result.yaml`:
- `stage: 1.3.1`
- `status: completed`

## 6. Sequence Runner

통과:

```bash
python tools/run_training_sequence.py --plan doc/PLAN/training_execution_plan_v1.8.md --data data/coco128.yaml --dataset-profile coco128_quick --model-family l,w6 --output runs/tmp_preflight_seq_coco128_phase_override --start-stage 00 --end-stage 02 --epochs 1 --batch 1 --img 64 --cfg cfg/training/yolov7.yaml --w6-cfg cfg/training/yolov7-w6.yaml --dry-run --stop-on-hard-fail
```

검증:
- 6개 stage record 생성
- Stage 01 L/W6 모두 `epochs=3`
- Stage 01 L/W6 enabled flags:
  - `phase-train: on`
  - `phase1-epochs: 1`
  - `phase2-epochs: 1`
  - `phase3-epochs: 1`
  - `phase2-img: [64,64]`
  - `phase3-img: [64,64]`

## 7. 최종 판단

학습 서버 full run 전 최소 preflight는 통과했다. 지금 바로 target full training으로 넘어가기 전에 권장되는 다음 단계는 아래 순서다.

1. COCO128 full sequence dry-run이 아니라 실제 Stage 00~02 quick sequence를 한 번 실행한다.
2. Stage 00~02 산출물의 `decision_table.csv`, `metrics_delta_all.csv`, `stage_summary.md`를 확인한다.
3. 문제가 없으면 target dataset의 manifest/checksum을 먼저 만들고 Stage 00 baseline full run을 시작한다.
4. ONNX 검증까지 포함하려면 `--require-export`를 켠 별도 run으로 분리한다.
