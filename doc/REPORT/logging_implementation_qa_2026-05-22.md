# Logging Implementation QA

- 작성일: 2026-05-22
- 범위: debug/error JSONL, 단일 run summary, sequence stage/final report

## 1. 구현 요약

| 영역 | 결과 |
| --- | --- |
| `utils/debug_logging.py` | `debug_trace.log`, `error_trace.log` JSONL writer 추가 |
| `utils/train_logger.py` | 표준 `stage_result.yaml`, `run_summary.md` 생성 추가 |
| `train.py` | `--debug-log*` CLI 추가, `--log-format txt`에서도 최소 로그 생성 |
| `train_aux.py` | `train.py`와 동일한 logging CLI/summary 연결 |
| `utils/stage_schema.py` | `train_type`, `missing_artifacts`, `log_paths` 필드 추가 |
| `tools/run_training_sequence.py` | stage command debug flag 전달, 실패 로그/누락 산출물 기록 |
| `tools/generate_training_report.py` | stage별 `metrics_delta.csv`, train_type별 summary 생성 |

## 2. QA 명령

문법 검사:

```powershell
python -m py_compile utils\debug_logging.py utils\train_logger.py utils\stage_schema.py tools\collect_stage_results.py tools\generate_training_report.py tools\run_training_sequence.py train.py train_aux.py
```

단일 run logger 생성 검사:

```powershell
@'
from pathlib import Path
from utils.debug_logging import get_debug_logger
from utils.train_logger import TrainLogger

save_dir = Path('runs/qa_logging_unit')
save_dir.mkdir(parents=True, exist_ok=True)
logger = get_debug_logger(save_dir, 'debug', 'train,runner', debug_file='debug_trace.log')
logger.log_event('debug', 'train', 'qa', 'debug_event', 'debug ok', summary={'shape': [1, 2, 3]})
try:
    raise RuntimeError('qa failure sample')
except Exception as exc:
    logger.log_exception('runner', 'qa', exc, summary={'stage_id': 'qa'})

train_logger = TrainLogger(save_dir, log_format='txt')
result = train_logger.write_stage_result(
    stage='1.3.2', current_run=str(save_dir), phase_train=True,
    primary_mAP=0.123, mAP_0_5=0.456, status='completed')
train_logger.write_run_summary(result)
'@ | python -
```

sequence dry-run:

```powershell
python tools\run_training_sequence.py --plan doc\PLAN\training_execution_plan_v1.8.md --data data\coco128.yaml --dataset-profile coco128_quick --model-family l_only --output runs\train_seq\qa_logging_dry_run --dry-run --start-stage 00 --end-stage 01 --debug-log error
```

의도 실패 검사:

```powershell
python tools\run_training_sequence.py --plan doc\PLAN\training_execution_plan_v1.8.md --data data\coco128.yaml --dataset-profile coco128_quick --model-family l_only --output runs\train_seq\qa_logging_fail --start-stage 00 --end-stage 00 --cfg missing_cfg_for_logging_qa.yaml --epochs 1 --batch-size 1 --img 64 64 --workers 0 --debug-log error --stop-on-hard-fail
```

CLI 노출 검사:

```powershell
python train.py --help
python train_aux.py --help
python tools\run_training_sequence.py --help
```

diff 검사:

```powershell
git diff --check -- utils\debug_logging.py utils\train_logger.py utils\stage_schema.py tools\collect_stage_results.py tools\generate_training_report.py tools\run_training_sequence.py train.py train_aux.py
```

## 3. QA 결과

| 항목 | 결과 |
| --- | --- |
| `py_compile` | 통과 |
| 단일 run logger 생성 | `debug_trace.log`, `error_trace.log`, `stage_result.yaml`, `run_summary.md`, `train_log.txt` 생성 확인 |
| sequence dry-run | Stage 00~01 `stage_config.yaml`, `stage_result.yaml`, `stage_summary.md`, `metrics_delta.csv` 생성 확인 |
| train_type summary | `baseline_export_summary.md`, `phase_training_summary.md` 생성 확인 |
| 의도 실패 | `decision=blocker`, `exit_code=1`, `failed_category=train`, `missing_artifacts` 기록 확인 |
| 에러 원문 | `stderr.log`에 `AssertionError: File Not Found: missing_cfg_for_logging_qa.yaml` 확인 |
| 구조화 에러 | `error_trace.log`에 `command_end`, `stage_failed` JSONL event 확인 |
| CLI | `train.py`, `train_aux.py`, `run_training_sequence.py`에서 `--debug-log*` 노출 확인 |
| `git diff --check` | 통과. CRLF 변환 경고만 존재 |

## 4. 확인된 산출물

단일 QA:

```text
runs/qa_logging_unit/
  debug_trace.log
  error_trace.log
  stage_result.yaml
  run_summary.md
  train_log.txt
```

sequence dry-run QA:

```text
runs/train_seq/qa_logging_dry_run/
  00_baseline_l/metrics_delta.csv
  00_baseline_l/stage_summary.md
  01_phase_logging_l/metrics_delta.csv
  01_phase_logging_l/stage_summary.md
  final_report/baseline_export_summary.md
  final_report/phase_training_summary.md
  final_report/decision_table.csv
  final_report/sequence_summary.md
```

실패 QA:

```text
runs/train_seq/qa_logging_fail/00_baseline_l/
  stderr.log
  error_trace.log
  stage_result.yaml
  stage_summary.md
```

## 5. 남은 주의점

- 실제 학습 full run은 아직 수행하지 않았다. 이번 QA는 logging 기능의 생성/연결/실패 기록 검증이다.
- `trace` level batch logging은 아직 기본 구현 범위에 넣지 않았다.
- dataset/cache, loss/assignment, augmentation 내부 세부 debug event는 다음 단계에서 위험 경계 함수별로 추가한다.
