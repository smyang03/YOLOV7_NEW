# YOLOv7 Training Report Format v1.8

## 문서 정보

- 작성일: 2026-05-22
- 목적: `training_execution_plan_v1.8.md` 실행 후 생성할 stage별/최종 리포트 형식을 정의한다.
- 저장 기준: 리포트 규격은 `doc/PLAN/`, 실제 실행 결과 리포트는 `doc/REPORT/`와 `runs/train_seq/v1.8/final_report/`에 저장한다.

## 1. 리포트 레벨

리포트는 세 단계로 만든다.

| 레벨 | 파일 | 목적 |
| --- | --- | --- |
| Stage 요약 | `runs/train_seq/v1.8/<stage>/stage_summary.md` | 해당 stage의 성공/실패와 즉시 판단 |
| Sequence 요약 | `runs/train_seq/v1.8/final_report/sequence_summary.md` | 전체 stage 흐름과 delta 비교 |
| 최종 리포트 | `doc/REPORT/final_training_report_v1.8_YYYY-MM-DD.md` | 유지/제거할 기능과 다음 개선점 결정 |

## 2. 입력 데이터

최종 리포트는 아래 파일을 읽어 작성한다.

- `stage_config.yaml`: stage 이름, 활성 flag, 시작 weight, epoch, seed
- `stage_result.yaml`: decision, best epoch, 주요 metric, 실패 사유
- `results.csv`: epoch별 train/val metric
- `results_per_class.csv`: class별 AP/precision/recall
- `loss_detail.csv`: box/cls/obj/aux/free loss와 positive count
- `profile.json`: params, GFLOPs, memory, inference ms
- `export_check.json`: ONNX export 상태, output diff
- `dataset_manifest.json`: image/label checksum
- 선택 파일: `aug_check.json`, `sampler_stats.csv`, `nms_cost.json`, `pseudo_label_manifest.json`

`stage_config.yaml`에는 아래 dataset profile을 반드시 기록한다.

| Field | 값 | 의미 |
| --- | --- | --- |
| `dataset_profile` | `coco128_quick` | 최초 빠른 검증. orchestration, crash, 산출물, 리포트 판정만 확인 |
| `dataset_profile` | `target_full` | 실제 대상 dataset full run. 최종 성능 판단 기준 |

COCO128 quick report는 최종 성능 결론에 사용하지 않는다. quick report의 목적은 stage 실행과 리포트 생성 체계가 동작하는지 확인하는 것이다.

## 3. Delta 계산 기준

각 stage는 세 가지 기준으로 비교한다.

| 비교 기준 | 계산 | 목적 |
| --- | --- | --- |
| Baseline delta | `current - Stage 00` | 전체 개선량 판단 |
| Previous delta | `current - previous_success_stage` | 방금 켠 기능의 효과 판단 |
| Best delta | `current - best_previous_stage` | 최종 후보 가치 판단 |

주요 계산식:

```text
primary_mAP_delta = current.primary_mAP - baseline.primary_mAP
small_AP_delta = current.small_AP - baseline.small_AP
rare_recall_delta = current.rare_recall - baseline.rare_recall
GFLOPs_delta_percent = (current.GFLOPs - baseline.GFLOPs) / baseline.GFLOPs * 100
NMS_delta_ms = current.python_nms_ms - baseline.python_nms_ms
FP_delta = current.FP_per_image - baseline.FP_per_image
FN_delta = current.FN_per_image - baseline.FN_per_image
```

## 4. Stage 판정 값

각 stage의 `decision`은 아래 중 하나만 사용한다.

| Decision | 의미 | 다음 동작 |
| --- | --- | --- |
| `keep` | 성능/비용/안정성 모두 통과 | 다음 stage에서 누적 사용 |
| `keep_candidate` | 개선은 있으나 full run 확인 필요 | 다음 stage 진행, 최종 후보 표시 |
| `drop` | 비용 증가 또는 성능 하락이 명확함 | 해당 flag 끄고 직전 성공 weight로 복귀 |
| `retry_tune` | 방향은 맞지만 설정 조정 필요 | 동일 stage에서 1회만 조정 재실행 |
| `blocker` | crash/export/label 등 hard fail | sequence 중단 |
| `defer` | optional 또는 후순위로 이동 | 현재 sequence에서는 제외 |

## 5. Stage 요약 템플릿

각 stage가 끝나면 아래 형식으로 `stage_summary.md`를 저장한다.

```markdown
# Stage NN Summary - <stage_name>

## Decision

- decision:
- reason:
- start_weight:
- best_weight:
- failed_flag:
- fallback_weight:

## Metric Delta

| Metric | Baseline | Previous | Current | Delta vs Baseline | Delta vs Previous |
| --- | ---: | ---: | ---: | ---: | ---: |
| primary_mAP | | | | | |
| mAP50 | | | | | |
| small_AP | | | | | |
| rare_recall | | | | | |
| FP/image | | | | | |
| FN/image | | | | | |
| GFLOPs | | | | | |
| python_nms_ms | | | | | |

## What Changed

- increased:
- decreased:
- cost_increased:
- unchanged:

## Risk Check

- loss_stability:
- export_status:
- label_status:
- per_class_regression:
- runtime_cost:

## Next Action

- next_stage:
- carry_flags:
- disabled_flags:
- notes:
```

## 6. 최종 리포트 목차

최종 리포트는 아래 순서로 작성한다.

1. 실행 요약
2. 실행 환경, dataset profile, dataset checksum
3. Stage별 결정표
4. L 모델 최종 후보
5. W6 모델 최종 후보
6. Metric delta 표
7. 증가한 항목
8. 감소한 항목
9. 비용 증가 항목
10. 실패/탈락 stage 원인
11. 유지할 flag
12. 제거할 flag
13. 재실험할 flag
14. 최종 full run 필요 항목
15. 다음 개발 개선점

## 7. 최종 결정표

최종 리포트에는 아래 표를 반드시 포함한다.

```text
stage, model, enabled_flags, decision, reason,
primary_mAP_delta, small_AP_delta, rare_recall_delta,
FP_delta, FN_delta, GFLOPs_delta_percent, NMS_delta_ms,
export_status, risk_level, next_action
```

`risk_level` 기준:

| Risk | 조건 |
| --- | --- |
| `low` | mAP 상승, 비용 허용, export 통과 |
| `medium` | 일부 class regression 또는 비용 증가가 있으나 개선이 큼 |
| `high` | 성능/비용 trade-off가 불리하거나 재현성 확인 필요 |
| `blocker` | hard fail 조건 발생 |

## 8. 그래프와 시각화

가능하면 아래 이미지를 `runs/train_seq/v1.8/final_report/plots/`에 저장한다.

- `primary_map_by_stage.png`
- `small_ap_by_stage.png`
- `rare_recall_by_stage.png`
- `gfops_delta_by_stage.png`
- `nms_ms_by_stage.png`
- `fp_fn_by_stage.png`
- `loss_scale_by_stage.png`
- `per_class_delta_heatmap.png`
- `decision_waterfall.png`

그래프가 없어도 최종 리포트는 작성할 수 있어야 한다. 그래프 생성 실패는 hard fail로 처리하지 않는다.

## 9. 실패 원인 분류

실패 stage는 아래 category 중 하나로 분류한다.

| Category | 예시 | 기본 대응 |
| --- | --- | --- |
| `data` | label missing, class mapping 불일치 | dataset manifest/cache 재생성 |
| `loader` | Close Mosaic 미반영, worker 상태 불일치 | DataLoader rebuild |
| `loss` | NaN/Inf, loss scale 폭주 | loss gain/fallback |
| `assignment` | positive 과소/과다 | TAL topk/alpha/beta 조정 |
| `architecture` | route/channel mismatch | cfg diff 축소 |
| `export` | ONNX 실패, output diff 초과 | raw output contract 확인 |
| `runtime_cost` | GFLOPs/NMS 초과 | channel/top-k/filter 축소 |
| `augmentation` | label pollution | 해당 aug off, visual audit 재실행 |
| `finetune` | forgetting 증가 | replay/distill 비율 조정 |

## 10. 내가 최종 리포트에서 해줄 판단

최종 결과가 주어지면 다음 순서로 판단한다.

1. `dataset_profile`이 `coco128_quick`인지 `target_full`인지 먼저 분리한다.
2. COCO128 quick 결과에서는 hard fail, 산출물 누락, 리포트 판정 오류만 본다.
3. target full 결과에서는 baseline 대비 전체 개선량을 본다.
4. previous stage 대비 방금 켠 기능의 순효과를 본다.
5. L/W6 역할에 맞는지 분리 판단한다.
6. 성능 증가보다 비용 증가가 큰 기능을 제거 후보로 둔다.
7. class별 regression이 큰 기능을 재실험 후보로 둔다.
8. 최종 유지 flag와 제거 flag를 표로 확정한다.
9. 최종 full run 후보를 1~2개로 줄인다.

최종 답변은 단순 요약이 아니라 아래 형태로 제공한다.

- `유지`: 근거가 명확한 기능
- `제거`: 비용 또는 성능 문제가 큰 기능
- `재실험`: 조정하면 가능성이 있는 기능
- `원인`: 문제가 생긴 stage와 가장 가능성 높은 원인
- `다음 액션`: 바로 실행할 명령 또는 수정할 코드 영역

## 11. 학습 1회 종료 종합 리포트

학습 1회가 끝나면 성공/실패와 관계없이 run 단위 종합 리포트를 Markdown으로 저장한다.

| 실행 형태 | 저장 위치 | 목적 |
| --- | --- | --- |
| 단일 학습 | `runs/train/<exp>/run_summary.md` | 해당 학습 1회의 결과, 문제, 다음 액션 정리 |
| sequence stage | `runs/train_seq/v1.8/<stage>/stage_summary.md` | stage별 keep/drop/retry/blocker 즉시 판정 |
| 학습 종류별 종합 | `runs/train_seq/v1.8/final_report/<train_type>_summary.md` | 같은 종류의 학습 결과를 L/W6, dataset profile별로 비교 |
| 최종 통합 | `doc/REPORT/final_training_report_v1.8_YYYY-MM-DD.md` | 전체 개발 항목의 유지/제거/재실험 결론 |

`train_type`은 아래 값 중 하나로 기록한다.

| train_type | 대상 |
| --- | --- |
| `baseline_export` | v1.3.1 baseline, label/cache/checkpoint/export 안전성 |
| `phase_training` | v1.3.2 phase schedule, close-mosaic, dataloader 재구성 |
| `core_loss_model` | v1.3.3 loss, assignment, head, AUX 안정성 |
| `augmentation_data` | v1.3.4 mosaic/rect/augmentation/data policy |
| `model_family_export` | v1.3.5 L/W6/W6-P2/SCDown/PSA/GELAN, ONNX export |
| `optional_gate` | v1.3.6 optional branch와 fallback |
| `finetune_distill` | v1.3.7 fine-tuning, class mapping, distillation |
| `sequence_report` | v1.3.8 stage orchestration과 final report |

## 12. 종합 판정 기준

run 단위 리포트는 아래 순서로 판단한다.

1. 실행 식별: stage id, train_type, model family, dataset profile, seed, 시작 weight, 활성 flag
2. 산출물 완성도: `best.pt`, `last.pt`, `results.csv`, `loss_detail.csv`, `stage_result.yaml`, export 산출물 존재 여부
3. 성능 변화: primary mAP, mAP50, precision, recall, class별 AP, small/rare metric
4. 안정성: NaN/Inf, loss 폭주, positive count 이상, label drop, cache rebuild, warning count
5. 비용 변화: GFLOPs, parameter count, epoch time, GPU memory, inference latency, NMS time
6. export 가능성: ONNX 생성 여부, 32배수 입력 검증, output diff, simplify 결과
7. 문제 원인: `data`, `loader`, `loss`, `assignment`, `architecture`, `export`, `runtime_cost`, `augmentation`, `finetune`
8. 최종 판정: `keep`, `keep_candidate`, `drop`, `retry_tune`, `blocker`, `defer`
9. 다음 액션: 다음 stage 진행, flag off, threshold 조정, dataset rebuild, 코드 수정 위치

COCO128 quick run은 성능 개선 결론에 사용하지 않는다. 이 경우 리포트는 crash, 산출물 누락, flag 연결, stage 전환, 리포트 생성이 정상인지에만 초점을 둔다.

target full run은 baseline 대비 성능과 비용을 함께 판단한다. GFLOPs 증가는 기존 모델 대비 최대 10% 미만을 원칙으로 보고, 이를 넘는 기능은 성능 개선이 있더라도 `retry_tune` 또는 `drop` 후보로 둔다.

## 13. run_summary.md 템플릿

```markdown
# Run Summary - <train_type>

## Decision

- decision:
- reason:
- dataset_profile:
- model_family:
- start_weight:
- best_weight:
- enabled_flags:
- disabled_flags:

## Artifact Check

| Artifact | Status | Path | Note |
| --- | --- | --- | --- |
| best.pt | | | |
| last.pt | | | |
| results.csv | | | |
| loss_detail.csv | | | |
| stage_result.yaml | | | |
| export | | | |

## Metric Summary

| Metric | Baseline | Previous | Current | Delta vs Baseline | Delta vs Previous |
| --- | ---: | ---: | ---: | ---: | ---: |
| primary_mAP | | | | | |
| mAP50 | | | | | |
| precision | | | | | |
| recall | | | | | |
| GFLOPs | | | | | |
| epoch_time | | | | | |
| inference_ms | | | | | |

## Stability And Risk

- loss_status:
- label_status:
- cache_status:
- export_status:
- warning_count:
- failed_category:

## What Changed

- increased:
- decreased:
- unchanged:
- risk:

## Next Action

- next_stage:
- carry_flags:
- rollback_flags:
- code_area:
```
