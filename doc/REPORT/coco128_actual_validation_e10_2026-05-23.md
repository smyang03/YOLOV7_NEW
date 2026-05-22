# COCO128 Actual Validation e10

- 작성일: 2026-05-23
- 목적: v1.3.1~v1.3.8 및 로그/리포트 인프라를 COCO128 quick 실제 실행으로 검증한다.
- 실행 profile: `coco128_quick`, `l_only`, `epochs=10`, `img=320`, `batch=2`, `workers=0`, `device=0`, `debug-log=error`
- 결과 경로: `runs/train_seq/coco128_actual_e10_stage00_02`

## 1. COCO128 확인

`data/coco128.yaml`은 존재했고 `../coco128` 데이터도 이미 준비되어 있었다.

확인 결과:

```text
../coco128/images/train2017: 128 images
../coco128/labels/train2017: 128 labels
GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4GB
```

## 2. 실행 명령

Stage 00~02 1차 실제 실행:

```powershell
python tools\run_training_sequence.py --plan doc\PLAN\training_execution_plan_v1.8.md --data data\coco128.yaml --dataset-profile coco128_quick --model-family l_only --output runs\train_seq\coco128_actual_e10_stage00_02 --start-stage 00 --end-stage 02 --cfg cfg\training\yolov7.yaml --epochs 10 --batch-size 2 --img 320 320 --workers 0 --device 0 --debug-log error --stop-on-hard-fail
```

Stage 03~08 이어서 실행:

```powershell
python tools\run_training_sequence.py --plan doc\PLAN\training_execution_plan_v1.8.md --data data\coco128.yaml --dataset-profile coco128_quick --model-family l_only --output runs\train_seq\coco128_actual_e10_stage00_02 --resume-sequence --end-stage 08 --cfg cfg\training\yolov7.yaml --epochs 10 --batch-size 2 --img 320 320 --workers 0 --device 0 --debug-log error --stop-on-hard-fail
```

Stage 12~13 defer 포함 최종 report 갱신:

```powershell
python tools\run_training_sequence.py --plan doc\PLAN\training_execution_plan_v1.8.md --data data\coco128.yaml --dataset-profile coco128_quick --model-family l_only --output runs\train_seq\coco128_actual_e10_stage00_02 --resume-sequence --end-stage 13 --cfg cfg\training\yolov7.yaml --epochs 10 --batch-size 2 --img 320 320 --workers 0 --device 0 --debug-log error --stop-on-hard-fail
```

## 3. Stage 결과

| Stage | Train Type | Decision | GFLOPs | 결과 |
| --- | --- | --- | ---: | --- |
| 00 baseline | baseline_export | keep | 26.1284 | 통과 |
| 01 phase_logging | phase_training | keep | 26.1284 | 통과 |
| 02 head_decoupled | core_loss_model | keep | 26.1284 | 통과 |
| 03 wiou_v3 | core_loss_model | keep | 26.1284 | 통과 |
| 04 tal_vfl | core_loss_model | keep | 26.1284 | 통과 |
| 05 core_cumulative | core_loss_model | keep | 26.1284 | 통과 |
| 06 cctv_pixel_aug | augmentation_data | keep | 26.1284 | 통과 |
| 07 patch_paste_hard_negative | augmentation_data | keep | 26.1284 | 통과 |
| 08 weighted_sampler | augmentation_data | keep | 26.1284 | 통과 |
| 12 optional_gate | optional_gate | defer |  | defer 정상 |
| 13 finetune_continual | finetune_distill | defer |  | defer 정상 |

COCO128 quick은 성능 결론용이 아니므로 mAP 증감은 유지/제거 판단에 사용하지 않는다. 이번 판단 기준은 crash, checkpoint, log, summary, final report 생성 여부다.

## 4. 산출물 검증

각 실제 학습 stage 00~08에서 아래 파일이 모두 생성되었다.

```text
stage_config.yaml
stage_result.yaml
stage_summary.md
metrics_delta.csv
results.csv
loss_detail.csv
run_summary.md
weights/best.pt
error_trace.log
```

`error_trace.log`는 모든 실제 학습 stage에서 0 byte였다. 즉, 구조화 error event는 발생하지 않았다.

최종 report 산출물:

```text
final_report/sequence_summary.md
final_report/decision_table.csv
final_report/metrics_delta_all.csv
final_report/baseline_export_summary.md
final_report/phase_training_summary.md
final_report/core_loss_model_summary.md
final_report/augmentation_data_summary.md
final_report/optional_gate_summary.md
final_report/finetune_distill_summary.md
doc/REPORT/final_training_report_v1.8_2026-05-23.md
```

## 5. Phase 로그 확인

Stage 01에서 phase 전환 로그가 정상 생성되었다.

```text
epoch=1 from_phase=phase1 to_phase=phase2 imgsz=320 rect=True mosaic=True train_loader_rebuilt=True val_loader_rebuilt=True persistent_workers=False reason=phase boundary
epoch=2 from_phase=phase2 to_phase=phase3 imgsz=320 rect=True mosaic=False train_loader_rebuilt=True val_loader_rebuilt=True persistent_workers=False reason=phase boundary
```

이는 close-mosaic phase에서 `mosaic=False`, dataloader rebuild, `persistent_workers=False`가 기록된 것을 의미한다.

## 6. 판단

COCO128 actual validation 기준으로는 통과다.

통과 근거:

- COCO128 데이터 경로와 label 수량 확인 완료
- Stage 00~08 실제 학습 완료
- Stage 12~13 defer 처리 완료
- 모든 실제 학습 stage에서 `best.pt`, `results.csv`, `loss_detail.csv`, `stage_result.yaml`, `run_summary.md` 생성
- `error_trace.log`에 error event 없음
- `phase_transition.log`에서 phase2/phase3 전환과 close-mosaic 상태 확인
- `final_report`와 `doc/REPORT/final_training_report_v1.8_2026-05-23.md` 생성

## 7. 남은 검증

이번 검증은 `l_only` COCO128 quick이다. 아래는 아직 최종 성능 판정으로 보지 않는다.

- W6 전용 Stage 09~11 실제 학습
- target full dataset 성능 검증
- long run 안정성
- 실제 성능 개선 여부
- ONNX export 강제 검증

다음 단계는 W6 quick 검증 또는 target full sequence 중 하나를 선택해 진행한다.
