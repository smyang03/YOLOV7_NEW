# Logging Items Recheck for v1.3.1-1.3.8

- 작성일: 2026-05-22
- 범위: `train.py`, `train_aux.py`, `utils/train_logger.py`, `tools/run_training_sequence.py`, `tools/generate_training_report.py`, `utils/stage_schema.py`, 관련 `doc/PLAN`
- 목적: 학습 중 오류 로그, 학습 1회 종료 요약, sequence 종합 리포트가 서로 끊기지 않는지 재검토한다.

## 1. 결론

현재 기본 metric logging과 sequence report 뼈대는 들어가 있다. `TrainLogger`는 `results.csv`, `loss_detail.csv`, `results_per_class.csv`, `scenario_metrics.csv`, `phase_transition.log`, `stage_result.yaml`을 만들고, sequence runner는 stage별 `stdout.log`, `stderr.log`, `stage_summary.md`, `final_report` 산출물을 만든다.

하지만 문서에서 정의한 `debug_trace.log`, `error_trace.log`, 단일 학습 종료 `run_summary.md`, 학습 종류별 `<train_type>_summary.md`는 아직 코드 구현이 부족하다. 학습 서버에서 연속 실험을 돌리기 전에는 아래 P0/P1 항목을 먼저 맞춰야 한다.

## 2. 현재 구현 확인

| 항목 | 현재 상태 | 확인 파일 |
| --- | --- | --- |
| epoch metric CSV | 구현됨 | `utils/train_logger.py` |
| loss detail CSV | 구현됨 | `utils/train_logger.py` |
| per-class CSV | 구현됨 | `utils/train_logger.py` |
| scenario metric CSV | 구현됨 | `utils/train_logger.py` |
| phase transition log | 구현됨 | `utils/train_logger.py`, `train.py`, `train_aux.py` |
| stage stdout/stderr capture | 구현됨 | `tools/run_training_sequence.py` |
| stage summary markdown | 구현됨 | `tools/generate_training_report.py` |
| sequence summary/final report | 구현됨 | `tools/generate_training_report.py` |
| debug JSONL | 미구현 | `utils/debug_logging.py` 없음 |
| error JSONL | 미구현 | `utils/debug_logging.py` 없음 |
| 단일 run summary | 미구현 | `run_summary.md` writer 없음 |
| train_type별 summary | 미구현 | `<train_type>_summary.md` writer 없음 |

## 3. P0 수정 필요 항목

### 3.1 `stage_result.yaml` schema 정렬

현재 `TrainLogger.write_stage_result()`는 `stage`, `status`, 개별 metric key 중심으로 저장한다. 반면 `utils/stage_schema.py`와 sequence report는 `stage_id`, `decision`, `reason`, `metrics`, `hard_fail`, `failed_category` 중심으로 읽는다.

개선 방향:
- 단일 학습도 `StageResult` schema와 호환되게 저장한다.
- 기존 필드는 호환용으로 유지하되, 표준 필드를 추가한다.
- 최소 필드:

```yaml
schema_version: "1.3"
stage_id: "1.3.2"
stage_name: "phase_training"
train_type: "phase_training"
decision: "keep"
reason: "training completed"
status: "completed"
hard_fail: false
soft_fail: false
failed_category: null
metrics:
  primary_mAP: 0.0
  mAP50: 0.0
  GFLOPs: null
artifacts:
  results_csv: results.csv
  loss_detail_csv: loss_detail.csv
  run_summary: run_summary.md
log_paths:
  train_log: train_log.txt
  phase_transition: phase_transition.log
  debug_trace: debug_trace.log
  error_trace: error_trace.log
```

### 3.2 `run_summary.md` 자동 생성

학습 1회가 끝나면 성공/실패와 관계없이 `runs/train/<exp>/run_summary.md`를 저장해야 한다. 현재는 sequence stage에만 `stage_summary.md`가 생성된다.

개선 방향:
- `utils/train_logger.py`에 `write_run_summary()`를 추가한다.
- `train.py`, `train_aux.py` 종료 블록에서 `stage_result.yaml` 저장 직후 호출한다.
- summary에는 decision, artifact check, metric summary, stability/risk, next action을 포함한다.

### 3.3 `debug_trace.log` / `error_trace.log` 구현

문서에는 JSONL debug/error 로그가 정의되어 있지만 구현 모듈이 없다.

개선 방향:
- `utils/debug_logging.py` 추가
- `--debug-log`, `--debug-log-file`, `--debug-log-modules` argparse 추가
- 기본값은 `off`
- rank 0만 기록
- tensor/image/label 원본 dump 금지
- 예외는 `error_trace.log`, 정상 경계 이벤트는 `debug_trace.log`

### 3.4 sequence runner 실패 로그 보강

`tools/run_training_sequence.py`는 `stdout.log`, `stderr.log`를 캡처하지만, command 실패 시 `error_trace.log`와 missing artifact list를 별도로 남기지 않는다.

개선 방향:
- stage 시작/종료 이벤트를 JSONL로 기록한다.
- 실패 시 `exit_code`, `stdout_path`, `stderr_path`, `missing_artifacts`, `failed_category`를 `error_trace.log`와 `stage_result.yaml`에 모두 남긴다.
- stage command에 debug 옵션을 전달할 수 있게 runner CLI를 확장한다.

## 4. P1 수정 필요 항목

### 4.1 학습 종류별 summary 생성

`training_report_format_v1.8.md`에 정의한 `<train_type>_summary.md`가 아직 생성되지 않는다.

개선 방향:
- `StageSpec`에 `train_type` 필드를 추가한다.
- report writer에 `write_train_type_summaries(results)`를 추가한다.
- 저장 위치: `runs/train_seq/v1.8/final_report/<train_type>_summary.md`

### 4.2 stage별 `metrics_delta.csv`

문서에는 `<stage>/metrics_delta.csv`가 정의되어 있지만 현재 구현은 `final_report/metrics_delta_all.csv` 중심이다.

개선 방향:
- 각 stage summary 생성 시 해당 stage의 delta row만 `<stage>/metrics_delta.csv`로 저장한다.
- 최종 통합은 기존 `metrics_delta_all.csv` 유지.

### 4.3 `log_format=txt`일 때 stage 산출물 보장

현재 `train.py`, `train_aux.py`는 `log_format`이 `csv` 또는 `both`일 때만 `TrainLogger`를 만든다. 사용자가 `--log-format txt`를 선택하면 `stage_result.yaml`과 `train_log.txt`도 생성되지 않을 수 있다.

개선 방향:
- metric CSV writer와 stage/run summary writer를 분리한다.
- 최소 `train_log.txt`, `stage_result.yaml`, `run_summary.md`는 log_format과 무관하게 rank 0에서 생성한다.

## 5. P2 수정 필요 항목

| 영역 | 필요 로그 |
| --- | --- |
| dataset/cache | label path sample, cache hash/version, rebuild 여부 |
| dataloader/phase | workers, persistent_workers, rect/mosaic, rebuild 결과 |
| loss/assignment | target count, positive count, assigner, device, NaN/Inf |
| augmentation | label count before/after, bbox drop/clipping |
| W6/export | cfg, stride, anchor shape, output contract, GFLOPs delta |
| finetune | class mapping, pseudo/replay count, distillation alpha/beta |

## 6. 최종 로그 구조 기준

단일 학습:

```text
runs/train/<exp>/
  train_log.txt
  debug_trace.log
  error_trace.log
  results.csv
  loss_detail.csv
  results_per_class.csv
  scenario_metrics.csv
  phase_transition.log
  stage_result.yaml
  run_summary.md
```

sequence stage:

```text
runs/train_seq/<sequence>/<stage>/
  stage_config.yaml
  stdout.log
  stderr.log
  debug_trace.log
  error_trace.log
  results.csv
  loss_detail.csv
  stage_result.yaml
  stage_summary.md
  metrics_delta.csv
```

sequence final:

```text
runs/train_seq/<sequence>/final_report/
  sequence_manifest.yaml
  sequence_summary.md
  decision_table.csv
  metrics_delta_all.csv
  baseline_export_summary.md
  phase_training_summary.md
  core_loss_model_summary.md
  augmentation_data_summary.md
  model_family_export_summary.md
  optional_gate_summary.md
  finetune_distill_summary.md
  sequence_report_summary.md
```

## 7. 실행 전 통과 기준

학습 서버 실행 전 아래 항목을 통과해야 한다.

1. `python train.py ... --epochs 1` 후 `run_summary.md`, `stage_result.yaml`, `results.csv`, `loss_detail.csv` 생성
2. `--debug-log error` 사용 시 `error_trace.log` 파일 생성
3. 강제 실패 command에서 `stage_result.yaml`에 `decision=blocker`, `exit_code`, `stdout_path`, `stderr_path`, `failed_category` 기록
4. sequence dry-run 후 stage별 `stage_summary.md`, stage별 `metrics_delta.csv`, final `decision_table.csv` 생성
5. `train_type`별 summary가 final_report 아래 생성
6. COCO128 quick report가 성능 결론 대신 산출물/실패/전환 상태만 판정

## 8. 최종 판단

로그 방향은 맞다. 다만 현재 상태로는 metric CSV와 sequence summary는 가능하지만, 장애 원인 분석과 학습 1회 종료 종합 문서가 완전히 닫히지 않는다.

개발 우선순위는 `stage_result.yaml` schema 정렬, `run_summary.md` 생성, JSONL debug/error logger, sequence runner 실패 로그 보강 순서가 적합하다. 이 네 가지가 들어가면 학습 서버에서 1번 학습 종료 후 바로 다음 학습으로 넘어가도 결과를 비교하고 문제 위치를 분리할 수 있다.
