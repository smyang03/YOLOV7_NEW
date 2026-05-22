# COCO128 e10 Test Observation

- 작성일: 2026-05-23
- 실행 목적: 개발 완료 후 COCO128 quick에서 실제 학습, 로그, 리포트 산출물이 정상 생성되는지 확인한다.
- 실행 성격: 성능 최종 판단이 아니라 10 epoch 테스트 관찰 리포트다.
- 실행 경로: `runs/train_seq/coco128_actual_e10_stage00_02`
- 조건: `l_only`, `epochs=10`, `img=320`, `batch=2`, `workers=0`, `device=0`, `debug-log=error`

## 1. 전체 결론

전체 실행은 통과다.

- 실제 학습 Stage 00~08 모두 `keep`
- optional/finetune Stage 12~13은 계획대로 `defer`
- hard fail 없음
- `error_trace.log`는 모든 stage에서 0 byte
- `Traceback`, `RuntimeError`, `AssertionError`, `OutOfMemory`, `NaN/Inf` 에러 없음
- 각 실제 학습 stage에서 `best.pt`, `results.csv`, `loss_detail.csv`, `stage_result.yaml`, `stage_summary.md`, `run_summary.md`, `metrics_delta.csv` 생성
- 최종 `sequence_summary.md`, `decision_table.csv`, train type별 summary 생성

COCO128 10 epoch라 mAP 수치는 매우 작고 흔들림이 크다. 이 결과는 기능 유지/제거 판단이 아니라, 실행 안정성과 로그 체계 검증으로 해석해야 한다.

## 2. Stage별 결과 요약

| Stage | 기능 | Decision | Best epoch | Best mAP50:95 | Best mAP50 | Final loss | Positive count | GFLOPs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00 | baseline | keep | 3 | 0.00023477 | 0.00110269 | 0.150314 | 4897 | 26.1284 |
| 01 | phase_logging | keep | 3 | 0.00032586 | 0.00162660 | 0.138727 | 1246 | 26.1284 |
| 02 | head_decoupled | keep | 0 | 0.00008427 | 0.00066328 | 0.152167 | 4912 | 26.1284 |
| 03 | wiou_v3 | keep | 6 | 0.00006872 | 0.00026014 | 0.146857 | 4961 | 26.1284 |
| 04 | tal_vfl | keep | 0 | 0.00006142 | 0.00018110 | 0.137404 | 13981 | 26.1284 |
| 05 | core_cumulative | keep | 4 | 0.00024926 | 0.00157752 | 0.132407 | 14069 | 26.1284 |
| 06 | cctv_pixel_aug | keep | 1 | 0.00014658 | 0.00041293 | 0.145704 | 4204 | 26.1284 |
| 07 | patch_paste_hard_negative | keep | 0 | 0.00015264 | 0.00034956 | 0.150075 | 4559 | 26.1284 |
| 08 | weighted_sampler | keep | 0 | 0.00018340 | 0.00040432 | 0.152064 | 5309 | 26.1284 |
| 12 | optional_gate | defer |  |  |  |  |  |  |
| 13 | finetune_continual | defer |  |  |  |  |  |  |

## 3. Metric 관찰

Best mAP50:95 기준으로는 Stage 01이 가장 높았다.

```text
Stage 01 phase_logging: 0.00032586
Stage 05 core_cumulative: 0.00024926
Stage 00 baseline: 0.00023477
```

Final epoch mAP50:95 기준으로는 Stage 05가 가장 높았다.

```text
Stage 05 core_cumulative: 0.00009002
Stage 01 phase_logging: 0.00004383
Stage 00 baseline: 0.00004222
```

단, 10 epoch와 COCO128 조건에서는 이 차이를 성능 개선으로 해석하지 않는다. 짧은 epoch에서는 best epoch가 0~6 사이로 흔들리고 final metric도 안정되지 않는다.

## 4. Loss/Assignment 관찰

Final loss만 보면 Stage 05가 가장 낮았다.

```text
Stage 05 core_cumulative: 0.132407
Stage 04 tal_vfl: 0.137404
Stage 01 phase_logging: 0.138727
```

TAL/VFL 계열은 positive count가 크게 증가했다.

```text
Stage 04 tal_vfl: 13981
Stage 05 core_cumulative: 14069
Baseline: 4897
```

이는 예상된 방향이다. TAL matching은 positive assignment가 늘어날 수 있으므로, full run에서는 `loss_detail.csv`의 positive count와 loss scale을 계속 감시해야 한다.

## 5. Phase 로그

Stage 01에서 phase 전환은 정상 기록됐다.

```text
epoch=1 phase1 -> phase2 imgsz=320 rect=True mosaic=True train_loader_rebuilt=True val_loader_rebuilt=True persistent_workers=False
epoch=2 phase2 -> phase3 imgsz=320 rect=True mosaic=False train_loader_rebuilt=True val_loader_rebuilt=True persistent_workers=False
```

의미:

- phase2 진입 시 dataloader rebuild 정상
- phase3 진입 시 close-mosaic 적용 정상
- `persistent_workers=False`로 worker dataset 상태 고착 문제 없음

## 6. Error/Warning 확인

구조화 에러:

```text
00~08 error_trace.log: 0 byte
12~13 error_trace.log: 0 byte
```

에러 검색 결과:

```text
Traceback: 없음
Exception: 없음
RuntimeError: 없음
AssertionError: 없음
OutOfMemory: 없음
NaN/Inf: 없음
```

공통 warning:

```text
torch.meshgrid UserWarning: future version requires indexing argument
```

이 warning은 모든 실제 학습 stage에서 1회씩 발생했다. 현재 실행에는 영향이 없고, PyTorch future warning이다.

stderr 파일 크기가 큰 이유는 에러 때문이 아니라 YOLOv7의 logger/tqdm 출력이 stderr 쪽에 많이 기록되기 때문이다.

## 7. 산출물 확인

각 실제 학습 stage 00~08에서 아래 산출물이 생성됐다.

```text
weights/best.pt
stage_config.yaml
stage_result.yaml
stage_summary.md
metrics_delta.csv
results.csv
loss_detail.csv
run_summary.md
error_trace.log
```

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

## 8. 해석

이 테스트에서 확인된 것은 성능 개선이 아니라 실행 안정성이다.

확인된 점:

- baseline부터 augmentation/sampler까지 L 계열 stage가 실제 학습에서 crash 없이 통과했다.
- checkpoint, CSV, YAML, Markdown report가 모두 생성됐다.
- phase 전환과 close-mosaic 로그가 정상이다.
- TAL/VFL 및 cumulative loss path가 최소한 10 epoch에서는 NaN/Inf 없이 동작했다.
- 구조화 error logging은 정상적으로 파일을 만들고, 에러가 없을 때 빈 파일로 남는다.

아직 판단할 수 없는 점:

- 실제 성능 개선 여부
- 장시간 학습 안정성
- W6 Stage 09~11 안정성
- target full dataset 성능
- ONNX export 강제 검증

## 9. 다음 권장 테스트

COCO128에서 성능 경향을 조금 더 보려면 아래 조건이 적합하다.

```text
epochs=50
img=320
batch=2
stage=00~08
```

더 의미 있는 비교는 `epochs=100`부터 가능하지만, 최종 판단은 target full dataset에서 해야 한다.
