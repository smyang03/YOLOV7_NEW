# Debug Logging Requirements Review for v1.3.1-1.3.8

- 작성일: 2026-05-22
- 목적: 1.3.1~1.3.8 개발 범위에서 학습 전/중 오류를 빠르게 찾기 위한 디버그/에러 로그 필요 항목을 정리한다.
- 결론: 전 함수 무차별 logging은 적용하지 않는다. stage별 위험 경계 함수에만 opt-in 방식으로 적용한다.

## 1. 기본 정책

디버그 로그는 기본값 `off`로 둔다. 학습 속도와 디스크 사용량을 보호하기 위해 사용자가 명시적으로 켠 경우에만 기록한다.

권장 CLI:

```bash
--debug-log off|error|debug|trace
--debug-log-file debug_trace.log
--debug-log-modules dataset,phase,model,loss,aug,sampler,finetune,runner
```

권장 파일:

```text
runs/train/<exp>/
  train_log.txt
  debug_trace.log
  error_trace.log
  phase_transition.log
  stage_result.yaml
```

기록 원칙:
- `error`: exception, stack trace, stage/function/input summary만 기록
- `debug`: 주요 함수 진입/종료와 shape/count/device 요약 기록
- `trace`: batch별 세부 요약까지 허용하되 tensor 원본/label 전체는 금지
- DDP에서는 기본적으로 rank 0만 기록
- 개인정보/원본 이미지 경로 전체/label 전체 dump는 금지

## 2. 공통 구현 항목

신규 모듈 후보:

| 파일 | 역할 |
| --- | --- |
| `utils/debug_logging.py` | debug logger 생성, safe summary, exception wrapper, rank filter |
| `utils/train_logger.py` | 기존 `TrainLogger`에 debug/error 파일 경로 연결 |
| `train.py`, `train_aux.py` | CLI flag 추가, logger 초기화 |
| `tools/run_training_sequence.py` | stage command에 debug flag 전달, stdout/stderr/stage_result 연결 |

핵심 함수:

```python
get_debug_logger(save_dir, level, modules, rank=0)
safe_summary(obj)
log_event(module, function, message, **summary)
log_exception(module, function, exc, **summary)
debug_scope(module, function, **summary)
```

## 3. 차수별 필요 항목

### 1.3.1 Baseline / Export

필수 로그 지점:
- `utils/datasets.py::img2label_paths()`
- `utils/datasets.py::LoadImagesAndLabels.__init__()`
- cache load/rebuild block
- `utils/datasets.py::create_dataloader()`
- checkpoint save branch
- `opt.yaml` 저장 직전

기록 내용:
- image path count, label path count
- `images -> labels` 변환 sample 3개
- cache path, cache hit/miss, hash/version match
- workers, persistent_workers, close_mosaic
- best/last checkpoint path, save success
- SafeLoader 가능 여부

필요 이유:
- label missing, stale cache, best.pt 미생성, resume 실패를 가장 빨리 잡아야 한다.

### 1.3.2 Phase / Logging / DataLoader

필수 로그 지점:
- `utils/phase.py::resolve_phase()`
- `utils/train_common.py::build_train_dataloader()`
- Phase 2/3 rebuild 직전/직후
- `TrainLogger.log_epoch()`
- `TrainLogger.log_phase_transition()`

기록 내용:
- epoch, from_phase, to_phase
- imgsz, rect, mosaic, allow_rect_mosaic
- dataloader rebuilt 여부
- workers, persistent_workers
- dataset.mosaic 실제 값
- results.csv/loss_detail.csv write success

필요 이유:
- Close Mosaic이 worker dataset에 반영되지 않는 문제가 재발하기 쉽다.

### 1.3.3 Core Model / Loss

필수 로그 지점:
- `models/yolo.py::Model.__init__()`
- head replacement 지점
- `utils/loss.py::ComputeLossOTA.__call__()`
- `utils/loss_aux.py::ComputeLossAuxOTA.__call__()`
- TAL matching block
- WIoU state update/resume

기록 내용:
- cfg, head type, Detect/IAuxDetect class
- prediction output shape, dtype, device
- target count, positive count
- assigner, loss_box, loss_cls
- WIoU running mean
- matching tensor device
- NaN/Inf 발생 여부

필요 이유:
- TAL/VFL/WIoU는 crash보다 loss scale 폭주나 positive count 급변이 더 위험하다.

### 1.3.4 CCTV Augmentation / Sampler

필수 로그 지점:
- `utils/augment_policy.py::AugmentPolicy`
- `utils/cctv_augmentations.py`의 label-preserving/paste 함수
- `utils/datasets.py::__getitem__()` augmentation hook
- `utils/sampler.py::build_weighted_sampler()`
- `log_sampler_stats()`

기록 내용:
- aug profile, phase, 확률
- image shape/dtype 변경 여부
- labels count before/after
- bbox range/class id validator 결과
- hard negative manifest path/count
- sampler class count, weight min/max, sampled histogram

필요 이유:
- label pollution은 학습 후 mAP만 보면 원인 추적이 늦다.

### 1.3.5 W6 Structure Expansion

필수 로그 지점:
- `models/yolo.py::load_model_yaml()`
- `models/yolo.py::parse_model()`
- `IAuxDetect.__init__()`
- stride build
- `tools/check_output_contract.py`
- `tools/estimate_nms_cost.py`

기록 내용:
- resolved cfg path
- base cfg path
- p2_head, neck_mod
- detect levels, anchors shape
- main/aux feature count
- stride list
- total boxes
- NMS input count/ms
- GFLOPs delta

필요 이유:
- W6 P2/SCDown은 route/channel/feature count mismatch가 핵심 리스크다.

### 1.3.6 Optional Experiments

필수 로그 지점:
- `utils/model_options.py` validation
- PSA/GELAN cfg parse
- FCOS decode helper
- optional decision gate

기록 내용:
- optional_decision path 존재 여부
- requested experiment
- rejected reason
- p2_head/neck_mod/psa_level 조합
- FCOS raw shape, decoded box count, score combine rule

필요 이유:
- optional 실험은 기본 경로에 섞이면 안 된다. 실패 이유를 명확히 남겨야 한다.

### 1.3.7 Fine-tuning / Continual Learning

필수 로그 지점:
- `utils/class_mapping.py::validate_class_mapping()`
- `utils/pseudo_label.py::PseudoLabelGenerator`
- `utils/replay_buffer.py::ReplayBufferBuilder`
- `utils/continual_loss.py::DistillationLoss`
- `finetune.py` dry-run/main flow

기록 내용:
- old/new nc, names length
- class index mismatch
- pseudo label generated/filtered/merged count
- replay selected image count, class distribution
- teacher/student output shape
- distill alpha/beta current value
- BN/freeze policy, trainable parameter count

필요 이유:
- forgetting, pseudo label pollution, class index mismatch는 학습 후에 발견하면 비용이 크다.

### 1.3.8 Training Sequence / Reporting

필수 로그 지점:
- `tools/run_training_sequence.py::build_train_command()`
- `run_command()`
- `collect_stage_result()`
- `decide_result()`
- `generate_training_report.py`

기록 내용:
- stage_id, stage_name, model_family, dataset_profile
- command list
- start_weight, fallback_weight, best_weight
- stdout/stderr path
- exit code
- missing artifact list
- decision reason
- soft/hard fail category
- carry/disabled flags

필요 이유:
- 학습 서버에서 여러 stage를 연속 실행할 때 실패 원인을 stage별로 즉시 분리해야 한다.

## 4. 구현 우선순위

### P1 - 즉시 필요

1. `utils/debug_logging.py` 추가
2. `train.py`, `train_aux.py`에 `--debug-log`, `--debug-log-file`, `--debug-log-modules` 추가
3. dataset/cache/dataloader/checkpoint 예외 logging
4. phase rebuild logging 보강
5. loss NaN/Inf, positive count 급변, tensor device mismatch logging
6. sequence runner command/exit/artifact missing logging 보강

### P2 - 학습 서버 전 권장

1. W6 cfg/stride/anchor/output contract debug logging
2. augmentation label before/after summary
3. sampler histogram 요약
4. optional gate reject reason 기록
5. finetune class mapping/replay/pseudo summary

### P3 - 후순위

1. decorator 기반 함수별 trace
2. batch 단위 trace sampling
3. debug log rotation
4. per-rank log 분리

## 5. 제외 기준

아래는 하지 않는다.

- 모든 함수에 decorator를 일괄 적용
- tensor 전체 값 dump
- label 전체 dump
- image binary/base64 dump
- 매 batch마다 전체 prediction dump
- 기본값에서 trace logging 활성화
- C++/TensorRT runtime logging

## 6. 권장 schema

`debug_trace.log`는 JSONL을 권장한다.

```json
{"time":"2026-05-22T12:00:00","level":"debug","stage":"1.3.3","module":"loss","function":"ComputeLossOTA.__call__","message":"loss computed","summary":{"pred_shapes":["1x3x8x8x85"],"target_count":12,"positive_count":24,"device":"cuda:0"}}
```

`error_trace.log`는 exception 중심으로 쓴다.

```json
{"time":"2026-05-22T12:00:00","level":"error","stage":"1.3.3","module":"loss_aux","function":"matching","message":"device mismatch","summary":{"cost_device":"cuda:0","matching_matrix_device":"cpu"},"traceback":"..."}
```

## 7. 최종 판단

1.3.1~1.3.8 전체에서 디버그/에러 로그 체계는 필요하다. 특히 학습 서버에서 stage를 연속 실행할 계획이면 `error_trace.log`, `debug_trace.log`, `stage_result.yaml`, `stdout.log`, `stderr.log`가 서로 연결되어야 한다.

다만 구현은 `전 함수 logging`이 아니라 `위험 경계 함수 logging`으로 시작해야 한다. 가장 먼저 넣을 곳은 dataset/cache/dataloader, phase rebuild, loss matching, W6 output contract, sequence runner command/decision이다.
