# YOLOv7 v1.3.1-1.3.8 Integrated AI Review

- 작성일: 2026-05-22
- 평가 대상:
  - `doc/PLAN/development_plan_v1.3.md`
  - `doc/PLAN/development_requirements_1.3.1_baseline_export.md`
  - `doc/PLAN/development_requirements_1.3.2_train_loop_phase_logging.md`
  - `doc/PLAN/development_requirements_1.3.3_core_model_loss.md`
  - `doc/PLAN/development_requirements_1.3.4_cctv_augmentation_sampler.md`
  - `doc/PLAN/development_requirements_1.3.5_w6_structure_expansion.md`
  - `doc/PLAN/development_requirements_1.3.6_optional_experiments.md`
  - `doc/PLAN/development_requirements_1.3.7_finetuning_continual_learning.md`
  - `doc/PLAN/development_requirements_1.3.8_training_sequence_reporting.md`
  - `doc/PLAN/development_precautions_1.3.1-1.3.8.md`
  - `doc/PLAN/training_execution_plan_v1.8.md`
  - `doc/PLAN/training_report_format_v1.8.md`
- 평가 관점: AI가 작성한 AI 모델 개선 개발 문서를 실제 개발자가 구현, 학습, 평가, 회귀 방어에 사용할 수 있는지 점검한다.

## 1. 종합 판정

현재 문서 세트는 `구현 착수 가능` 수준이다. 각 차수는 개선 사유, 대상 파일, CLI flag, 산출물, 검증 명령, 중단 조건을 대부분 갖추고 있다. 특히 원 코드 유지, flag 기반 비활성 기본값, COCO128 quick run과 target full run 분리, L/W6 역할 분리, optional gate, stage별 report 자동화 방향은 적합하다.

다만 통합 기준에서는 `조건부 통과`다. 이유는 아래 세 가지다.

1. 일부 문서에는 아직 ONNX export가 공통 hard fail처럼 남아 있지만, 최신 1.3.8 정책은 `--require-export`를 켠 경우에만 hard fail이다.
2. 1.3.8 실행 문서는 `retry_tune`, plot 산출물, full sequence report까지 요구하지만 현재 실행기 구현은 dry-run/기본 stage report 중심이다.
3. `development_plan_v1.3.md`의 현재 착수 위치와 공통 중단 조건 일부가 이전 상태를 가리켜, 현재 1.3.8까지 진행된 상태와 어긋난다.

종합 점수: `8.0 / 10`

## 2. 평가 기준

| 기준 | 의미 | 평가 |
| --- | --- | --- |
| 추적성 | 왜 바꾸는지, 어디를 바꾸는지, 무엇을 검증하는지 연결되는가 | 좋음 |
| 원 코드 보호 | 기존 YOLOv7 명령/출력/동작을 기본값에서 유지하는가 | 좋음 |
| 단계 분리 | 한 번에 여러 기능을 섞지 않고 원인 추적 가능한가 | 좋음 |
| 산출물 계약 | metric, profile, report, manifest가 기계적으로 비교 가능한가 | 보통 이상 |
| 학습 실행성 | COCO128 quick과 target full run으로 이어지는가 | 보통 이상 |
| 문서 최신성 | 구현 완료 상태와 문서가 일치하는가 | 일부 보정 필요 |

## 3. 차수별 평가

| 차수 | 판정 | 평가 |
| --- | --- | --- |
| 1.3.1 Baseline / Python Export | 조건부 통과 | baseline 안정화 항목은 적합하다. label mapping, cache, persistent worker, best.pt, SafeLoader, results.txt 회귀 방어가 명확하다. 단, ONNX export를 필수 통과 기준처럼 쓴 부분은 1.3.8의 optional 정책과 맞춰야 한다. |
| 1.3.2 Train Loop / Phase / Logging | 통과 | phase, dataloader rebuild, close mosaic, canonical CSV logging 방향이 적합하다. 다만 1.3.8 sequence runner가 COCO128 quick에서 Phase 1/2/3를 실제로 짧게 통과시키려면 Stage 01에 `--phase1-epochs 1 --phase2-epochs 1 --phase3-epochs 1` 같은 stage 전용 override가 필요하다. |
| 1.3.3 Core Model / Loss | 통과 | Decoupled Head, WIoU, TAL/VFL을 단독 검증 후 누적하는 방식이 타당하다. loss_detail, positive count, WIoU state, AUX device 회귀 방어도 적합하다. ONNX hard fail 문구만 optional 정책과 정렬해야 한다. |
| 1.3.4 CCTV Augmentation / Sampler | 통과 | label-preserving과 label-changing augmentation을 분리하고 visual audit를 요구한 점이 좋다. 데이터 리빌드와 cache 정책도 반영되어 있다. 남은 gap은 `false_positive_per_image`가 아직 빈 값이라는 점이며, 문서가 이를 후속 evaluator 확장으로 명시하고 있어 허용 가능하다. |
| 1.3.5 W6 Structure Expansion | 통과 | W6 전용 P2/SCDown 분리, L 모델 보호, 5-level IAuxDetect, output box/NMS 비용 측정 기준이 적합하다. 실제 profile 수치도 문서에 반영되어 있다. 추후 target full run에서 Python NMS ms와 memory를 반드시 같이 봐야 한다. |
| 1.3.6 Optional Experiments | 조건부 통과 | optional gate와 `optional_decision_*.md` 요구는 적합하다. PSA/FCOS/GELAN 동시 적용 차단도 맞다. 현재 1.3.8에서는 Stage 12가 defer이므로, optional을 최종 sequence에 넣을지 별도 sequence로 뺄지 운영 기준을 더 명시해야 한다. |
| 1.3.7 Fine-tuning / Continual Learning | 통과 | replay, pseudo label, LwF, class mapping, teacher exclusion 방향이 타당하다. ONNX/TensorRT 제외도 사용자 요구와 맞다. target dataset full run 뒤에만 진입해야 하는 조건이 중요하다. |
| 1.3.8 Training Sequence / Reporting | 조건부 통과 | schema, runner, collect, compare, report 구조는 적합하고 COCO128 dry-run 기준 동작한다. 다만 `--max-retry-per-stage`, `--skip-plots`, plot 산출물, `retry_tune` 실제 재실행 정책은 문서 요구 대비 구현 gap이 있다. |

## 4. 통합 강점

1. 원본 YOLOv7 보존 원칙이 모든 차수에 반복되어 있다.
2. 새 기능은 기본값 off, flag/helper/wrapper 중심으로 추가하도록 설계되어 회귀 위험을 낮춘다.
3. L은 속도형, W6는 정확도형으로 역할을 분리한 점이 좋다.
4. Backbone 변경을 후순위로 밀어 과도한 구조 변경을 피한다.
5. COCO128 quick run을 성능 판단이 아니라 orchestration 검증으로 제한한 점이 적합하다.
6. target full run에서만 최종 유지/제거 판단을 하도록 설계되어 있다.
7. Stage별 delta를 baseline, previous success, best previous로 나누는 방식은 원인 추적에 유리하다.
8. optional 실험을 기본 경로에 섞지 않고 gate 문서로 통제한다.

## 5. 통합 리스크

### R1. ONNX 정책 불일치

`development_plan_v1.3.md`와 `development_precautions_1.3.1-1.3.8.md`에는 ONNX export 실패와 PyTorch/ONNX Runtime 비교 실패가 공통 중단 조건처럼 남아 있다. 반면 1.3.5, 1.3.7, 1.3.8은 C++/TensorRT runtime을 제외하고 ONNX 검증도 기본 필수가 아니며 `--require-export`일 때만 hard fail로 본다.

판정:
- 문서 정책을 1.3.8 기준으로 통일해야 한다.
- 권장 문구: `ONNX/Python export 검증은 기본 diagnostic이며, --require-export 실행에서만 hard fail로 본다. TensorRT/C++ runtime은 본 범위에서 제외한다.`

### R2. 1.3.8 runner와 문서 요구의 차이

문서에는 retry, plot, full sequence report가 요구되어 있다. 현재 실행기는 기본 stage registry, dry-run, result/report 생성은 갖췄지만 아래 항목은 구현 또는 명시가 더 필요하다.

- `--max-retry-per-stage`는 CLI에 있으나 실제 retry loop와 tune config가 아직 약하다.
- `--skip-plots`는 CLI에 있으나 plot 생성 도구는 선택 항목으로 남아 있다.
- `decision_waterfall.png`, `per_class_delta_heatmap.png`는 report format에 있으나 필수 구현은 아니다.
- Stage 12 optional, Stage 13 finetune은 defer 처리이므로 full sequence의 최종 판단 범위에서 분리해야 한다.

판정:
- 현재 1.3.8은 `COCO128 quick orchestration/report MVP`로는 통과다.
- `target full 자동판정 runner`로 승격하려면 retry/plot/Stage12/13 운영 정책을 보강해야 한다.

### R3. Stage 01 quick 검증 부족 가능성

Stage 01은 `--phase-train on`만 켜면 기본 phase epoch가 길어서 COCO128 quick run에서 Phase 2/3 rebuild와 close mosaic을 실제로 밟지 않을 수 있다. 문서의 목적은 Stage 01에서 phase boundary와 rebuild를 확인하는 것이므로 quick profile에서는 phase epoch를 짧게 override해야 한다.

권장:
- COCO128 quick Stage 01 enabled flags에 아래를 추가한다.

```text
--phase1-epochs 1
--phase2-epochs 1
--phase3-epochs 1
```

### R4. 현재 상태 문구가 오래됨

`development_plan_v1.3.md`의 `현재 착수 위치`는 아직 `1.3.1-P1`로 되어 있다. 현재 문서와 코드 상태는 1.3.8까지 진행되어 있으므로 historical note로 유지하거나 `현재 구현 진행 상태` 섹션으로 바꾸는 것이 맞다.

### R5. 목표 metric threshold 부족

공통 기준으로 primary mAP 하락 2pp, GFLOPs 10%는 있다. 그러나 CCTV 도메인 핵심인 small AP, rare recall, FP/image, FN/image의 target threshold는 아직 명확하지 않다.

권장:
- target full run 전 dataset별 목표를 별도 표로 둔다.
- 예: `small_AP +x`, `rare_recall +y`, `FP/image -z`, `NMS ms <= n`.

## 6. 코드/파일 존재 대조 결과

문서가 요구하는 핵심 구현 파일은 현재 작업트리에 대부분 존재한다.

확인된 주요 파일:
- 1.3.2: `utils/phase.py`, `utils/train_logger.py`, `utils/train_common.py`, `utils/early_stopping.py`, `tools/check_phase_schedule.py`
- 1.3.3: `utils/wiou.py`, `utils/tal.py`, `utils/loss_components.py`, `tools/check_loss_smoke.py`
- 1.3.4: `utils/cctv_augmentations.py`, `utils/augment_policy.py`, `utils/sampler.py`, `tools/check_aug_visual.py`, `tools/check_labels.py`, `tools/mine_hard_negatives.py`
- 1.3.5: `tools/estimate_nms_cost.py`, `tools/check_output_contract.py`, W6 P2/SCDown cfg
- 1.3.6: `utils/model_options.py`, `utils/fcos.py`, `tools/decode_fcos_outputs.py`, `cfg/experiments/*`
- 1.3.7: `finetune.py`, `utils/class_mapping.py`, `utils/replay_buffer.py`, `utils/pseudo_label.py`, `utils/continual_loss.py`
- 1.3.8: `utils/stage_schema.py`, `tools/run_training_sequence.py`, `tools/collect_stage_results.py`, `tools/compare_stage_metrics.py`, `tools/generate_training_report.py`

이 결과는 문서가 단순 아이디어가 아니라 실제 구현 파일과 연결된다는 점에서 긍정적이다. 단, 본 리포트는 전체 학습 성능 검증이 아니라 문서-구현 요구의 통합 적합성 평가다.

## 7. 권장 정비 순서

1. `development_plan_v1.3.md`와 `development_precautions_1.3.1-1.3.8.md`의 ONNX 공통 hard fail 문구를 1.3.8 정책에 맞춘다.
2. `development_plan_v1.3.md`의 현재 착수 위치를 현행 상태로 갱신한다.
3. 1.3.8 runner의 Stage 01 quick flags에 phase epoch override를 넣는다.
4. 1.3.8 문서에서 plot 산출물은 optional로 명확히 분리한다.
5. `retry_tune`을 실제 구현할지, 아니면 v1.8.1로 넘길지 결정한다.
6. target full run 전에 small AP, rare recall, FP/image, FN/image의 목표 threshold를 확정한다.
7. Stage 12 optional과 Stage 13 finetune은 기본 full sequence에서 defer인지, 별도 sequence인지 문서에 고정한다.

## 8. 최종 결론

AI 관점에서 이 문서 세트의 개선 방향은 맞다. 무작정 YOLOv7을 크게 만드는 문서가 아니라, baseline 안정화, 학습 루프 통합, core loss/head ablation, CCTV 데이터 대응, W6 소형 객체 구조, optional gate, finetune 보존, sequence report까지 단계적으로 이어진다.

가장 중요한 보완점은 성능 아이디어가 아니라 운영 판정 정책이다. ONNX optional 정책, retry/plot 구현 범위, COCO128 quick에서 실제 phase 검증 여부, target full run threshold를 정리하면 문서 품질은 `구현 가능`에서 `학습 서버 실행 가능` 수준으로 올라간다.
