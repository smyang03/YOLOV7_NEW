# 1.3.2 Code-Level Development Requirements

- 기준 계획: `doc/PLAN/development_plan_v1.3.md`
- 대상 차수: `1.3.2 Train Loop / Phase / Logging 기반`
- 선행 조건: `1.3.1` baseline 학습, 평가, ONNX export, `profile.json`, `export_check.json` 완료
- 목적: 모델 구조를 바꾸지 않고 통합 학습 루프, Phase 전환, DataLoader rebuild, canonical logging 기반을 만든다.

## 1. 범위

포함:
- `train.py`를 canonical training entry로 정리
- `train_aux.py`는 호환 wrapper 또는 AUX 전용 얇은 entry로 축소
- Phase 1/2/3 epoch boundary 계산
- Phase 전환 시 train/val DataLoader rebuild
- Rect finetune, Close Mosaic 정책
- `results.csv`, `results_per_class.csv`, `loss_detail.csv`, `train_log.txt`, `phase_transition.log`, `hyp_used.yaml`, `stage_result.yaml`
- confusion/PR/F1/results plot 산출 경로 정리
- EMA 전구간 유지, Early Stopping Phase 3 전용 정책
- W6 `batch=8`, `grad_accumulate=4` 기준
- `workers=0`, `workers>0` smoke 검증

제외:
- Decoupled Head, WIoU, TAL, VFL
- CCTV augmentation, sampler
- P2 Head, SCDown
- C++ 후처리, TensorRT runtime, 추론 서버

## 2. 파일 단위 개발 요구사항

| 파일 | 구분 | 요구사항 |
| --- | --- | --- |
| `train.py` | 수정 | 단일 학습 루프의 기준 entry. `--phase-train`, Phase epoch, logging 옵션을 처리한다. |
| `train_aux.py` | 수정 | 중복 루프를 줄이고 `train.py`의 공통 함수를 호출한다. 기존 명령 호환은 유지한다. |
| `utils/datasets.py` | 수정 | Phase 전환 시 새 Dataset/DataLoader를 생성할 수 있게 `mosaic`, `rect`, `imgsz`, `persistent_workers` 정책을 인자로 제어한다. |
| `utils/phase.py` | 신규 | `PhaseConfig`, `PhaseState`, `resolve_phase(epoch)`를 제공한다. epoch 기준은 0-based, end-exclusive다. |
| `utils/train_logger.py` | 신규 | `results.csv`, `loss_detail.csv`, `stage_result.yaml`, `phase_transition.log` 기록을 담당한다. |
| `utils/train_common.py` | 신규 | `train.py`와 `train_aux.py`가 공유할 checkpoint 저장, dataloader build, loss 선택, 평가 호출 helper를 제공한다. |
| `utils/early_stopping.py` | 신규/확인 | Phase 3에서만 patience 기반 early stopping을 적용한다. Phase 1/2에서는 best 갱신만 수행한다. |
| `data/hyp_phase1.yaml` | 신규 | Phase 1 기본 hyp. 기존 `hyp.scratch.p5/p6.yaml`에서 복사 후 phase key만 추가한다. |
| `data/hyp_phase2.yaml` | 신규 | Rect finetune용 hyp. `lr0=0.001`, mosaic 정책 명시. |
| `data/hyp_phase3.yaml` | 신규 | Close Mosaic용 hyp. `mosaic=0.0`, early stopping 기준 명시. |
| `tools/check_phase_schedule.py` | 신규 | boundary epoch에서 Phase 상태와 rebuild 이벤트를 dry-run으로 검증한다. |
| `tools/profile_model.py` | 수정/확인 | 1.3.1 baseline과 current run의 GFLOPs delta를 계산할 수 있어야 한다. |

## 3. CLI 요구사항

신규 옵션:
- `--phase-train {off,on}`: 기본 `off`, 1.3.2 검증 시 `on`
- `--phase1-epochs`: 기본 `290`
- `--phase2-epochs`: 기본 `70`
- `--phase3-epochs`: 기본 `40`
- `--phase2-img`: L `640 384`, W6 `1280 736`
- `--phase3-img`: L `640 384`, W6 `1280 736`
- `--rect-size-l`: `--phase2-img/--phase3-img`의 L alias. 기본 `640 384`
- `--rect-size-w6`: `--phase2-img/--phase3-img`의 W6 alias. 기본 `1280 736`
- `--phase2-rect`: 기본 `True`
- `--phase2-mosaic {on,off}`: 기본 `on`, 짧은 A/B 검증 지원
- `--phase3-mosaic {off}`: 1.3.2에서는 `off`만 허용
- `--aux {auto,on,off}`: 기본 `auto`. 1.3.2에서는 AUX 구조를 새로 만들지 않고 기존 cfg의 `IAuxDetect` 여부로 loss 경로만 선택한다.
- `--grad-accumulate`: 기본 L `1`, W6 `4`
- `--early-stop-phase {phase3,off}`: 기본 `phase3`
- `--patience`: 기본 `20`
- `--profile {off,on}`: 기본 `off`, stage 종료 시 `profile.json` 생성
- `--per-class-log-interval`: 기본 `10`
- `--log-format {txt,csv,both}`: 기본 `both`
- `--no-verbose`: 콘솔 상세 출력 비활성

`--img` 호환 정책:
- `train.py`, `train_aux.py`의 기존 `--img-size` 의미를 바꾸지 않는다.
- `--img 640 384`를 학습 입력 H/W로 재해석하지 않는다.
- Phase 2/3 rectangular 입력은 `--phase2-img H W`, `--phase3-img H W`로만 지정한다.

Smoke 예시:

```bash
python train.py --cfg cfg/training/yolov7.yaml --data data/custom_example.yaml --hyp data/hyp_phase1.yaml --epochs 3 --phase-train on --phase1-epochs 1 --phase2-epochs 1 --phase3-epochs 1 --img 640 --batch 16 --workers 0 --name phase_l_smoke
```

Worker smoke는 같은 명령을 `--workers 2`로 한 번 더 실행한다.

## 4. Phase 정책

Phase 계산:
- `phase1_end = phase1_epochs`
- `phase2_end = phase1_epochs + phase2_epochs`
- `epoch < phase1_end`: Phase 1
- `phase1_end <= epoch < phase2_end`: Phase 2
- `epoch >= phase2_end`: Phase 3

전환 이벤트:
- Phase 2 최초 진입 시 train/val loader를 모두 재생성한다.
- Phase 3 최초 진입 시 train loader를 `mosaic=False`로 재생성한다.
- 기존 worker가 이전 Dataset 상태를 들고 있으면 실패로 본다.
- AUX freeze는 1.3.2에서 loss weight `lambda_aux=0.0` 의미만 기록하고, 파라미터 freeze는 1.3.3 이후에 다룬다.
- EMA는 Phase 전환과 무관하게 전구간 유지한다.
- Early stopping은 Phase 3에서만 활성화한다.
- W6는 `batch=8`, `grad_accumulate=4`, effective batch 32를 기본 smoke 기준으로 둔다.

## 5. 산출물 스키마

### 5.1 `results.csv`

필수 컬럼:
- `epoch`
- `phase`
- `train/box_loss`
- `train/cls_loss`
- `train/obj_loss`
- `train/aux_loss`
- `train/total_loss`
- `metrics/precision`
- `metrics/recall`
- `metrics/mAP_0.5`
- `metrics/mAP_0.5:0.95`
- `val/box_loss`
- `val/cls_loss`
- `val/obj_loss`
- `x/lr0`
- `x/lr1`
- `x/lr2`
- `gpu_mem_gb`
- `epoch_time_sec`

### 5.1.1 `results_per_class.csv`

필수 컬럼:
- `epoch`
- `phase`
- `class_id`
- `class_name`
- `precision`
- `recall`
- `AP_0.5`
- `AP_0.5:0.95`
- `is_rare`

기록 주기:
- best 갱신 시 항상 기록
- 그 외에는 `--per-class-log-interval` epoch마다 기록

### 5.1.2 `loss_detail.csv`

필수 컬럼:
- `epoch`
- `phase`
- `box_loss`
- `cls_loss`
- `obj_loss`
- `aux_loss`
- `free_loss`
- `total_loss`
- `lambda_aux`
- `lambda_free`
- `positive_count`

### 5.1.3 기타 로그/시각화 산출물

필수 산출물:
- `train_log.txt`
- `hyp_used.yaml`
- `confusion_matrix.png`
- `PR_curve.png`
- `F1_curve.png`
- `results_plot.png`

`hyp_used.yaml`은 학습 시작과 Phase 전환 시점마다 실제 적용 hyp snapshot을 저장한다.

### 5.2 `phase_transition.log`

전환마다 아래 항목을 남긴다.
- `epoch`
- `from_phase`
- `to_phase`
- `imgsz`
- `rect`
- `mosaic`
- `hyp_path`
- `train_loader_rebuilt`
- `val_loader_rebuilt`
- `persistent_workers`
- `reason`

### 5.3 `stage_result.yaml`

필수 필드:
- `stage: 1.3.2`
- `baseline_run`
- `current_run`
- `phase_train`
- `phase_boundaries`
- `best_epoch`
- `best_map_50_95`
- `profile_json`
- `baseline_gflops`
- `current_gflops`
- `gflops_delta_percent`
- `primary_mAP`
- `mAP_0.5`
- `small_AP`
- `rare_recall`
- `trt_latency`: `null` 허용, TensorRT runtime 차수 전에는 측정하지 않음
- `export_check_json`
- `output_contract_json`
- `status`

## 6. 통과 기준

1. `tools/check_phase_schedule.py`가 epoch `0, 29, 30, 289, 290, 359, 360`의 Phase를 올바르게 판정한다.
2. `workers=0` smoke 학습이 완료된다.
3. `workers>0` smoke 학습이 완료된다.
4. Phase 2/3 전환은 각각 최초 진입 시 1회만 발생한다.
5. Phase 3 진입 후 새 Dataset의 `mosaic=False`가 로그로 확인된다.
6. `results.csv`, `results_per_class.csv`, `loss_detail.csv`, `train_log.txt`, `phase_transition.log`, `hyp_used.yaml`, `stage_result.yaml`이 생성된다.
7. `results.txt`는 기존 plot 경로를 깨지 않는다.
8. 1.3.1 export 검증 경로가 계속 통과한다.
9. `train_aux.py`와 `train.py`의 Phase/DataLoader/checkpoint 로직이 서로 다른 방식으로 중복 구현되지 않는다.
10. `--aux auto`에서 W6 cfg의 AUX 경로가 자동 선택되고 L cfg는 기본 main loss 경로를 사용한다.
11. Early stopping은 Phase 1/2에서 동작하지 않고 Phase 3에서만 동작한다.
12. W6 smoke에서 `grad_accumulate=4`가 적용되고 effective batch가 로그에 기록된다.

## 7. 구현 순서

1. `utils/phase.py` 작성 및 boundary dry-run 검증
2. `utils/train_logger.py` 작성
3. `train.py`에 Phase CLI와 PhaseState 연결
4. DataLoader build 함수를 분리하고 Phase rebuild 적용
5. `train_aux.py` 중복 루프 제거 또는 wrapper화
6. Phase별 hyp 파일 작성
7. per-class/loss/detail/plot 산출물 연결
8. Early stopping Phase 3 전용 정책 연결
9. `workers=0`, `workers>0` smoke 실행
10. `results.csv`, `stage_result.yaml` 산출물 검증

## 8. 리스크 및 주의사항

- Phase 도입 중 모델/loss/augmentation 변경을 섞지 않는다.
- `results.csv`를 canonical log로 사용하되, 기존 `results.txt` 호환은 유지한다.
- Phase 2의 `rect=True, mosaic=True`는 짧은 A/B가 가능해야 한다.
- smoke 실패 시 full training으로 넘어가지 않는다.

## 9. 개발 착수 분리 기준

`train.py`와 `train_aux.py` 통합은 가장 큰 변경점이므로 loop를 한 번에 갈아엎지 않는다.

| PR | 범위 | 완료 기준 |
| --- | --- | --- |
| `1.3.2-P1` | `utils/phase.py`, `tools/check_phase_schedule.py` | boundary epoch dry-run 통과 |
| `1.3.2-P2` | `utils/train_logger.py`, `results.csv`, `loss_detail.csv` | 기존 학습 loop 변경 최소화 상태에서 CSV 산출 |
| `1.3.2-P3` | DataLoader build helper와 Phase rebuild | `workers=0/>0`, Phase 2/3 rebuild 로그 통과 |
| `1.3.2-P4` | `utils/train_common.py`로 checkpoint/eval/helper 추출 | `train.py`와 `train_aux.py` 결과 산출물 parity 확인 |
| `1.3.2-P5` | `train_aux.py` wrapper화 또는 얇은 entry 축소 | 기존 W6 명령 호환 유지 |

`train_aux.py`는 즉시 삭제하거나 대체하지 않는다. 먼저 공통 helper를 도입하고, W6 AUX smoke가 통과한 뒤 wrapper화한다.
