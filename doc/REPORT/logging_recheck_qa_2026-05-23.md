# Logging Recheck QA

- 작성일: 2026-05-23
- 범위: 로그 구현 재검토, 직접 학습 실패 경로, sequence dry-run/실패 경로

## 1. 재검토 결론

로그 구현은 전반적으로 동작한다. 재검토 중 직접 학습 실패 경로에서 `stage_result.yaml`과 `run_summary.md`가 남지 않을 수 있는 누락을 발견했고 수정했다.

수정 후 아래 경로가 모두 통과했다.

- 단일 run 성공/실패 summary writer
- `debug_trace.log` / `error_trace.log` JSONL writer
- sequence dry-run stage/final report
- sequence command 실패 시 `stage_result.yaml`, `stderr.log`, `error_trace.log` 연결
- CLI 옵션 노출

## 2. 발견 및 수정 사항

| 항목 | 상태 | 내용 |
| --- | --- | --- |
| 직접 학습 실패 summary 누락 | 수정 | `train.py`, `train_aux.py`에서 예외 발생 시 `write_failed_run_artifacts()`로 `stage_result.yaml`, `run_summary.md` 생성 |
| `TrainLogger` save_dir 가정 | 수정 | `TrainLogger.__init__()`에서 `save_dir.mkdir(parents=True, exist_ok=True)` 보장 |
| error module filter | 수정 | `--debug-log error`는 module filter와 무관하게 error event를 기록 |

## 3. 재검토 QA 명령

문법 검사:

```powershell
python -m py_compile utils\debug_logging.py utils\train_logger.py utils\stage_schema.py tools\collect_stage_results.py tools\generate_training_report.py tools\run_training_sequence.py train.py train_aux.py
```

직접 학습 실패 artifact helper 검사:

```powershell
@'
from argparse import Namespace
from pathlib import Path
from train import write_failed_run_artifacts

save_dir = Path('runs/qa_direct_failure_summary')
opt = Namespace(
    save_dir=save_dir,
    log_format='txt',
    per_class_log_interval=10,
    aug_profile='off',
    sampler_mode='off',
    phase_train='off',
    head='coupled',
    loss_box='ciou',
    assign='simota',
    loss_cls='bce')
write_failed_run_artifacts(opt, RuntimeError('direct train qa failure'))
'@ | python -
```

sequence dry-run:

```powershell
python tools\run_training_sequence.py --plan doc\PLAN\training_execution_plan_v1.8.md --data data\coco128.yaml --dataset-profile coco128_quick --model-family l_only --output runs\train_seq\qa_logging_recheck_dry --dry-run --start-stage 00 --end-stage 01 --debug-log error
```

sequence 실패 경로:

```powershell
python tools\run_training_sequence.py --plan doc\PLAN\training_execution_plan_v1.8.md --data data\coco128.yaml --dataset-profile coco128_quick --model-family l_only --output runs\train_seq\qa_logging_recheck_fail --start-stage 00 --end-stage 00 --cfg missing_cfg_for_logging_recheck.yaml --epochs 1 --batch-size 1 --img 64 64 --workers 0 --debug-log error --stop-on-hard-fail
```

CLI 확인:

```powershell
python train.py --help
python train_aux.py --help
python tools\run_training_sequence.py --help
```

diff 검사:

```powershell
git diff --check -- train.py train_aux.py utils\debug_logging.py utils\train_logger.py utils\stage_schema.py tools\collect_stage_results.py tools\generate_training_report.py tools\run_training_sequence.py
```

## 4. 결과

| 검사 | 결과 |
| --- | --- |
| `py_compile` | 통과 |
| 직접 실패 artifact helper | `stage_result.yaml`, `run_summary.md`, `train_log.txt` 생성 확인 |
| sequence dry-run | `stage_summary.md`, `metrics_delta.csv`, train type summary 생성 확인 |
| sequence 실패 | `decision=blocker`, `exit_code=1`, `failed_category=train`, `missing_artifacts` 기록 확인 |
| `error_trace.log` | `command_end`, `stage_failed` JSONL event 기록 확인 |
| CLI | `--debug-log`, `--debug-log-file`, `--debug-log-modules` 노출 확인 |
| `git diff --check` | 통과. CRLF 변환 경고만 존재 |

## 5. 남은 범위

이번 재검토는 로그 인프라와 실패/summary 산출물 검증이다. 실제 학습 full run, batch-level `trace`, dataset/cache/loss/augmentation 내부 위험 경계 함수별 상세 debug event는 다음 구현/검증 범위로 남긴다.
