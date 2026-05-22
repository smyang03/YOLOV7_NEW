# Debug Logging Structure v1.3

## 1. 목적

v1.3.1부터 v1.3.8까지의 개발 항목은 학습, 검증, export, 단계 실행 결과를 연속으로 비교해야 한다. 따라서 로그는 단순 콘솔 출력이 아니라, 실패 원인과 성능 변화 원인을 단계별로 추적할 수 있는 구조로 남긴다.

이 문서는 Python 기반 학습/검증/export/시퀀스 실행 로그만 다룬다. C++ 추론 런타임과 TensorRT 런타임 내부 로그는 본 범위에서 제외한다.

## 2. 기본 원칙

- 기존 원본 동작을 유지하고, 신규 개선 코드에서 발생한 판단과 예외를 별도 로그로 남긴다.
- 사람이 읽는 로그와 자동 리포트가 읽는 로그를 분리한다.
- 로그는 재현 가능해야 하므로 stage id, 모델 계열, 데이터셋 profile, epoch, phase 정보를 항상 포함한다.
- Tensor, 이미지, 전체 label 내용은 그대로 저장하지 않는다. shape, count, dtype, device, min/max, 경로 basename 중심으로 남긴다.
- YAML에는 Python 객체가 들어가지 않도록 문자열, 숫자, bool, list, dict만 기록한다.

## 3. 디렉터리 구조

단일 학습 run은 기존 `runs/train/<exp>/` 아래에 로그를 남긴다.

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
  hyp_used.yaml
  stage_result.yaml
  weights/
```

연속 학습/평가 run은 `runs/train_seq/<sequence_name>/` 아래에 stage별로 남긴다.

```text
runs/train_seq/v1.8_coco128_quick/
  00_baseline_l/
    stage_config.yaml
    stdout.log
    stderr.log
    debug_trace.log
    error_trace.log
    stage_result.yaml
    stage_summary.md
  01_phase_l/
    stage_config.yaml
    stdout.log
    stderr.log
    debug_trace.log
    error_trace.log
    stage_result.yaml
    stage_summary.md
  final_report/
    sequence_summary.md
    decision_table.csv
    metrics_delta_all.csv
```

## 4. 로그 파일 역할

`train_log.txt`는 사람이 바로 확인하는 요약 로그다. epoch 진행, 주요 옵션, checkpoint 저장 여부, 최종 결과를 간결하게 남긴다.

`stdout.log`와 `stderr.log`는 시퀀스 러너가 stage별 하위 프로세스 출력을 원본 그대로 캡처한다. 장애가 발생했을 때 재현 명령과 콘솔 흐름을 확인하는 1차 자료다.

`debug_trace.log`는 구조화된 JSONL 로그다. 함수 진입/종료, phase 변경, dataloader 구성, label cache 판단, loss 입력 요약, model head 구성 같은 디버깅 정보를 남긴다.

`error_trace.log`는 구조화된 JSONL 오류 로그다. 예외 타입, 메시지, traceback, stage 상태, 관련 artifact 경로를 남긴다.

`results.csv`, `loss_detail.csv`, `results_per_class.csv`, `scenario_metrics.csv`는 수치 비교용 정식 산출물이다. `plot_results()`가 읽는 기존 numeric 결과 포맷을 깨지 않는다.

`phase_transition.log`는 phase 전환과 close-mosaic 상태 변화를 별도 기록한다.

`stage_config.yaml`과 `stage_result.yaml`은 각 stage의 입력 설정과 종료 결과를 저장한다. 최종 리포트는 이 파일들을 기준으로 생성한다.

## 5. JSONL Schema

`debug_trace.log`와 `error_trace.log`는 한 줄에 하나의 JSON object를 저장한다.

```json
{
  "time": "2026-05-22T13:10:22.183+09:00",
  "level": "debug",
  "stage_id": "1.3.2",
  "stage_name": "phase_training",
  "model_family": "yolov7-l",
  "dataset_profile": "coco128_quick",
  "rank": 0,
  "epoch": 2,
  "phase": "phase2",
  "module": "train",
  "function": "resolve_phase",
  "event": "phase_transition",
  "message": "phase changed",
  "summary": {
    "from": "phase1",
    "to": "phase2",
    "img_size": 512,
    "mosaic": true,
    "rect": false
  },
  "artifact_paths": {
    "save_dir": "runs/train/exp"
  },
  "duration_ms": 3.1
}
```

오류 로그는 아래 필드를 추가한다.

```json
{
  "level": "error",
  "event": "exception",
  "exception_type": "RuntimeError",
  "message": "Expected all tensors to be on the same device",
  "traceback": "...",
  "hard_fail": true
}
```

## 6. CLI 정책

초기 구현은 아래 옵션을 기준으로 한다.

```text
--debug-log off|error|debug|trace
--debug-log-file debug_trace.log
--debug-log-modules dataset,phase,model,loss,aug,sampler,finetune,runner
```

기본값은 `off`다. `tools/run_training_sequence.py`로 실행하는 v1.8 연속 실험은 stage 실패 분석이 목적이므로 `error` 이상을 권장한다. `trace`는 batch 단위 정보가 커질 수 있으므로 COCO128 사전 점검이나 특정 stage 재현에만 사용한다.

## 7. Stage별 기록 항목

### 7.1 v1.3.1 Baseline/Export

- label path 변환 결과와 missing label count
- cache 파일 경로, hash, version, rebuild 여부
- persistent workers 사용 여부와 `num_workers`
- `best.pt`, `last.pt`, `best_###.pt` 저장 결과
- `opt.yaml` 저장 전 YAML-safe 변환 결과
- ONNX export 입력 크기, stride, dynamic/static 여부

### 7.2 v1.3.2 Phase Training

- phase 결정 결과: epoch, phase name, image size, batch size
- `rect`, `mosaic`, close-mosaic 상태
- phase별 dataloader 재생성 여부
- persistent worker 재사용/해제 판단
- `phase_transition.log`와 동일한 핵심 전환 이벤트

### 7.3 v1.3.3 Core Loss/Model

- model family, head type, anchor/stride 요약
- loss branch 선택 결과: 기존 loss, WIoU, TAL, AUX OTA
- target count, positive count, ignored count
- CPU/GPU device summary
- NaN/Inf loss 감지 결과

### 7.4 v1.3.4 Augmentation/Data Policy

- augmentation policy 이름과 활성 옵션
- mosaic/mixup/copy-paste 적용 전후 label count
- bbox clipping/drop count
- sampler class histogram 요약
- rect와 mosaic 조합 사용 시 적용 mode

### 7.5 v1.3.5 Model Family/Export

- cfg path, resolved model family, parameter count, GFLOPs
- stride, detection layer count, aux head 존재 여부
- export 입력 shape, opset, simplify 여부
- ONNX 파일 생성 여부와 파일 크기
- TensorRT 호환을 위한 32배수 입력 검증 결과

### 7.6 v1.3.6 Optional Feature Gate

- optional feature enable/disable 상태
- gate에서 reject된 이유
- fallback branch 사용 여부
- FCOS/extra branch decode 요약
- 기존 branch와 신규 branch의 출력 shape 비교

### 7.7 v1.3.7 Fine-tuning/Distillation

- class mapping 파일, source/target class count
- freeze layer 목록과 BN 정책
- pseudo label 생성/필터링 count
- replay sample 선택 count
- distillation alpha/beta, teacher checkpoint 경로

### 7.8 v1.3.8 Sequence/Report

- stage별 실제 실행 command
- exit code, start/end time, duration
- stdout/stderr/debug/error log path
- stage_result 생성 여부
- baseline 대비 metric delta
- 다음 stage 진행/중단 decision reason

## 8. 예외 처리 방식

함수 내부에서 복구 가능한 fallback을 선택한 경우 `debug_trace.log`에 `event=fallback`으로 남긴다. fallback이 성능 비교에 영향을 줄 수 있으면 `stage_result.yaml`의 `warnings`에도 기록한다.

복구 불가능한 예외는 `error_trace.log`에 남기고 기존 예외를 다시 발생시킨다. 시퀀스 러너는 프로세스 종료 코드를 `stage_result.yaml`에 저장하고 다음 stage 진행 여부를 정책에 따라 결정한다.

`stage_result.yaml`에는 최소 아래 필드를 남긴다.

```yaml
stage_id: "1.3.3"
status: failed
exit_code: 1
hard_fail: true
failed_category: loss_device_mismatch
stdout_path: stdout.log
stderr_path: stderr.log
debug_log_path: debug_trace.log
error_log_path: error_trace.log
warnings: []
```

## 9. 구현 위치

공통 JSONL writer는 `utils/debug_logging.py`에 둔다. 학습 코드에서는 기존 `utils/train_logger.py`의 metric logging을 유지하고, 디버깅 이벤트만 신규 writer로 위임한다.

`train.py`와 `train_aux.py`는 argparse 옵션을 추가하고, dataloader/model/loss/phase 결정 지점에 필요한 summary만 전달한다.

`tools/run_training_sequence.py`는 stage별 subprocess 실행 전후에 command, log path, exit code를 기록한다. stage가 실패해도 `stage_result.yaml`은 반드시 생성한다.

## 10. 리포트 반영

최종 리포트는 `results.csv`, `loss_detail.csv`, `scenario_metrics.csv`, `stage_result.yaml`을 우선 사용한다. `debug_trace.log`와 `error_trace.log`는 원인 분석 부록으로 연결한다.

학습 1회가 끝나면 `runs/train/<exp>/run_summary.md`를 생성한다. sequence 실행에서는 동일한 내용을 stage 단위로 `runs/train_seq/<sequence>/<stage>/stage_summary.md`에 저장하고, 같은 학습 종류끼리는 `runs/train_seq/<sequence>/final_report/<train_type>_summary.md`로 다시 묶는다.

리포트에는 stage별로 아래 내용을 요약한다.

- 증가한 항목: mAP, recall, precision, FPS, 안정성 지표
- 감소한 항목: GFLOPs, latency, loss 변동성, 실패 횟수
- 문제 발생 위치: dataset, phase, model, loss, augmentation, export, runner
- 개선 방안: rollback, gate off, threshold 조정, phase 조정, dataset rebuild

종합 리포트의 기준은 아래 순서로 적용한다.

1. 산출물 완성도: `best.pt`, `last.pt`, `results.csv`, `loss_detail.csv`, `stage_result.yaml`
2. 성능 변화: primary mAP, mAP50, precision, recall, per-class AP
3. 안정성: NaN/Inf, loss 폭주, positive count, label/cache warning
4. 비용 변화: GFLOPs, parameter count, epoch time, GPU memory, inference latency
5. export 가능성: ONNX 생성, 32배수 입력, output diff, simplify 결과
6. 실패 원인: data, loader, loss, assignment, architecture, export, runtime_cost, augmentation, finetune
7. 최종 판정: keep, keep_candidate, drop, retry_tune, blocker, defer

COCO128 quick 학습은 성능 개선 결론에 사용하지 않고, crash, 산출물 누락, stage 전환, 로그/리포트 생성 여부만 확인한다. target full 학습은 baseline 대비 성능과 비용을 함께 판단하며, GFLOPs 증가는 기존 모델 대비 10% 미만을 기준으로 둔다.

## 11. 구현 우선순위

1. `utils/debug_logging.py` JSONL writer 추가
2. `tools/run_training_sequence.py` stage 실패 로그 보강
3. `train.py`, `train_aux.py` argparse와 logger 연결
4. v1.3.1~v1.3.4 핵심 함수 summary logging
5. v1.3.5~v1.3.8 export, finetune, report logging 확장

초기 개발은 `error`와 `debug` 수준까지만 구현한다. `trace` 수준의 batch-level logging은 COCO128 사전 점검에서 필요성이 확인되면 별도 단계로 확장한다.

## 12. 재검토 후 확정 체크리스트

로그 항목 재검토 결과는 `doc/REPORT/logging_items_recheck_1.3.1-1.3.8_2026-05-22.md`에 저장한다. 구현 시 아래 순서로 처리한다.

### 12.1 P0

- `stage_result.yaml` schema를 단일 학습과 sequence runner에서 동일하게 맞춘다.
- 단일 학습 종료 시 `runs/train/<exp>/run_summary.md`를 생성한다.
- `utils/debug_logging.py`를 추가하고 `debug_trace.log`, `error_trace.log` JSONL writer를 제공한다.
- `train.py`, `train_aux.py`, `tools/run_training_sequence.py`에 `--debug-log`, `--debug-log-file`, `--debug-log-modules`를 연결한다.
- command 실패 시 `error_trace.log`와 `stage_result.yaml`에 `exit_code`, `stdout_path`, `stderr_path`, `missing_artifacts`, `failed_category`를 같이 기록한다.

### 12.2 P1

- `StageSpec`에 `train_type`을 추가하고 `final_report/<train_type>_summary.md`를 생성한다.
- 각 stage directory에 `metrics_delta.csv`를 저장한다.
- `--log-format txt`에서도 `train_log.txt`, `stage_result.yaml`, `run_summary.md`는 생성되게 한다.
- COCO128 quick report는 성능 결론 대신 crash, 산출물, stage 전환, 리포트 생성 여부만 판정한다.

### 12.3 P2

- dataset/cache, dataloader/phase, loss/assignment, augmentation, W6/export, finetune 경계 함수에 debug summary를 추가한다.
- `trace` level batch logging은 기본 비활성으로 두고, 필요 시 sampling 방식으로만 확장한다.
