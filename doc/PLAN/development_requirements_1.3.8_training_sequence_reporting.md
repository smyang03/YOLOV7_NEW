# 1.3.8 Code-Level Development Requirements

## 공통 예외 사항 - 원 코드 유지 개발

- 본 차수의 새 기능은 기본값으로 비활성화한다. 플래그를 켜지 않으면 기존 YOLOv7 학습, 평가, export 동작이 유지되어야 한다.
- 기존 함수/클래스는 버그 수정, 호환성 보강, 공통 helper 호출 연결에 한해서만 직접 수정한다.
- 신규 기능은 가능한 `utils/*`, `models/*`의 새 helper/class/wrapper로 분리하고, 기존 entrypoint는 기존 CLI와 출력 경로를 유지한다.
- `train.py`, `train_aux.py`, `test.py`, `export.py`는 기존 옵션명을 삭제하지 않는다. alias를 추가할 때도 기존 `dest`와 결과 파일명을 바꾸지 않는다.
- `train_aux.py`는 즉시 삭제하거나 대체하지 않는다. 공통 helper를 먼저 만들고 AUX/W6 smoke 검증 후 얇은 wrapper로 축소한다.

## 1.3.8.1 코드 구현 상세

이 세부 항목은 stage sequence runner, YAML schema validator, report generator의 내부 함수/클래스 구조를 고정한다.

### 신규 모듈 구조

| 파일 | 클래스/함수 | 역할 |
| --- | --- | --- |
| `utils/stage_schema.py` | `StageConfig` | stage 실행 입력 schema. YAML load/save/validate를 포함한다. |
| `utils/stage_schema.py` | `StageResult` | stage 결과 schema. decision, metric, failure category를 포함한다. |
| `utils/stage_schema.py` | `Decision` | `keep`, `keep_candidate`, `drop`, `retry_tune`, `blocker`, `defer` enum. |
| `tools/run_training_sequence.py` | `TrainingSequenceRunner` | stage 순서, carry flags, fallback weight, retry 정책을 관리한다. |
| `tools/collect_stage_results.py` | `collect_stage_result()` | stage output directory에서 metric/report 파일을 읽는다. |
| `tools/compare_stage_metrics.py` | `compute_stage_delta()` | baseline, previous success, best previous 기준 delta를 계산한다. |
| `tools/generate_training_report.py` | `TrainingReportWriter` | stage summary, sequence summary, final report를 생성한다. |

### StageConfig dataclass

```python
@dataclass
class StageConfig:
    stage_id: str
    stage_name: str
    dataset_profile: str
    model_family: str
    data: str
    start_weight: str
    enabled_flags: dict
    disabled_flags: dict
    seed: int
    epochs: int
    output_dir: str

    def validate(self) -> None:
        ...
```

validation 규칙:
- `dataset_profile`은 `coco128_quick` 또는 `target_full`
- `model_family`는 `l`, `w6`, `l_only`, `w6_only`
- `stage_id`, `stage_name`, `output_dir`는 비어 있으면 안 됨
- `start_weight`는 Stage 00이 아니면 존재해야 함
- quick run과 target full run output dir를 섞으면 실패

### StageResult dataclass

```python
@dataclass
class StageResult:
    stage_id: str
    decision: str
    reason: str
    best_weight: str
    fallback_weight: str
    metrics: dict
    export_status: str
    hard_fail: bool
    soft_fail: bool
    failed_category: str | None

    def validate(self) -> None:
        ...
```

validation 규칙:
- decision은 enum 값만 허용
- `hard_fail=True`이면 decision은 `blocker`
- `drop`이면 `fallback_weight`가 있어야 함
- `target_full`에서 `primary_mAP`가 null이면 fail
- `coco128_quick`에서 small AP/rare recall null은 허용

### TrainingSequenceRunner 실행 흐름

```text
load plan
build stage list
for stage in stages:
    create StageConfig
    validate StageConfig
    write stage_config.yaml
    build command
    if dry_run: continue
    run command
    collect artifacts
    create StageResult
    compute decision
    write stage_result.yaml
    write stage_summary.md
    update carry_flags/fallback_weight
    stop if blocker and stop_on_hard_fail
write sequence_summary.md
write final report
```

### command 생성 규칙

command는 문자열 조합 대신 list 형태로 구성한다.

```python
cmd = [
    sys.executable, 'train.py',
    '--data', config.data,
    '--weights', config.start_weight,
    '--project', str(stage_project),
    '--name', config.stage_name,
]
```

flag value가 `None`이면 command에 넣지 않는다. boolean flag는 `True`일 때만 넣는다. 실패 시 exit code, stdout path, stderr path를 `stage_result.yaml`에 기록한다.

### report writer 규칙

`TrainingReportWriter`는 아래 출력을 모두 지원한다.

- `write_stage_summary(stage_config, stage_result, delta)`
- `write_sequence_summary(all_results)`
- `write_final_report(all_results, output_path)`
- `write_decision_table_csv(all_results)`
- `write_metrics_delta_csv(all_results)`

Markdown report와 csv/yaml은 같은 source dict에서 생성한다. 사람이 읽는 report와 기계가 읽는 data가 서로 다르면 안 된다.

### COCO128 quick run 제한

`dataset_profile='coco128_quick'`이면 아래 decision만 성능 기반으로 내리지 않는다.

- mAP 낮음으로 `drop` 금지
- small AP null로 fail 금지
- rare recall null로 fail 금지

단, 아래는 hard fail이다.

- command crash
- label/class mapping 오류
- `best.pt` 누락
- stage yaml 누락
- `--require-export`를 켠 경우의 export 실패
- metric csv 생성 실패

## 리포트 기반 정비 기준

- 문서 위치 기준: 본 코드레벨 개발 요구서는 `doc/PLAN/`에 둔다.
- 기준 실행 계획: `doc/PLAN/training_execution_plan_v1.8.md`
- 기준 리포트 규격: `doc/PLAN/training_report_format_v1.8.md`
- 목적: 1.3.1~1.3.7 개발 완료 후 COCO128 quick run과 대상 dataset full run을 stage별로 연속 실행하고, 최종 리포트를 자동 생성한다.

## 1. 범위

포함:
- stage sequence 실행 도구
- stage별 config/result 수집
- COCO128 quick run profile 구분
- target full run profile 구분
- metric delta 계산
- stage별 decision 판정
- 최종 report markdown 생성

제외:
- 모델 구조 추가 변경
- loss/augmentation/sampler 신규 개발
- C++ 후처리
- TensorRT runtime
- 추론 서버

## 2. 파일 단위 개발 요구사항

| 파일 | 구분 | 요구사항 |
| --- | --- | --- |
| `tools/run_training_sequence.py` | 신규 | stage 목록을 순서대로 실행하고 각 stage의 시작 weight, enabled flags, output dir을 관리한다. |
| `tools/collect_stage_results.py` | 신규 | `results.csv`, `profile.json`, `export_check.json`, `stage_result.yaml`을 읽어 stage metric을 표준 dict로 변환한다. |
| `tools/generate_training_report.py` | 신규 | stage 결과를 모아 `stage_summary.md`, `sequence_summary.md`, 최종 report markdown을 생성한다. |
| `tools/compare_stage_metrics.py` | 신규 | baseline, previous success, best previous 기준 delta를 계산한다. |
| `tools/plot_training_sequence.py` | 선택 | stage별 mAP, small AP, GFLOPs, NMS ms, FP/FN 그래프를 생성한다. |
| `utils/stage_schema.py` | 신규 | `stage_config.yaml`, `stage_result.yaml`, decision enum schema를 정의한다. |
| `doc/REPORT/final_training_report_v1.8_*.md` | 산출 | 최종 유지/제거/재실험/원인/다음 액션을 기록한다. |

## 3. CLI 요구사항

기본 실행:

```bash
python tools/run_training_sequence.py \
  --plan doc/PLAN/training_execution_plan_v1.8.md \
  --data data/coco128.yaml \
  --dataset-profile coco128_quick \
  --model-family l,w6 \
  --output runs/train_seq/v1.8_coco128_quick \
  --stop-on-hard-fail
```

대상 dataset full run:

```bash
python tools/run_training_sequence.py \
  --plan doc/PLAN/training_execution_plan_v1.8.md \
  --data data/custom.yaml \
  --dataset-profile target_full \
  --model-family l,w6 \
  --output runs/train_seq/v1.8 \
  --stop-on-hard-fail
```

필수 옵션:
- `--plan`
- `--data`
- `--dataset-profile {coco128_quick,target_full}`
- `--model-family l,w6,l_only,w6_only`
- `--output`
- `--stop-on-hard-fail`

선택 옵션:
- `--start-stage`
- `--end-stage`
- `--resume-sequence`
- `--dry-run`
- `--max-retry-per-stage`, 기본 `1`
- `--skip-plots`

## 4. Stage Schema

`stage_config.yaml` 필수 필드:

```yaml
stage_id: "02"
stage_name: "head_decoupled"
dataset_profile: "coco128_quick"
model_family: "l"
data: "data/coco128.yaml"
start_weight: "runs/train_seq/v1.8_coco128_quick/01_phase/weights/best.pt"
enabled_flags:
  head: "decoupled"
disabled_flags:
  p2_head: "none"
seed: 0
epochs: 3
output_dir: "runs/train_seq/v1.8_coco128_quick/02_head_decoupled_l"
```

`stage_result.yaml` 필수 필드:

```yaml
stage_id: "02"
decision: "keep"
reason: "COCO128 quick run passed. Artifacts generated."
best_weight: "runs/.../weights/best.pt"
fallback_weight: "runs/.../01_phase/weights/best.pt"
primary_mAP: 0.0
mAP50: 0.0
small_AP: null
rare_recall: null
GFLOPs: 0.0
python_nms_ms: null
export_status: "pass"
hard_fail: false
soft_fail: false
failed_category: null
```

허용 decision:
- `keep`
- `keep_candidate`
- `drop`
- `retry_tune`
- `blocker`
- `defer`

## 5. COCO128 Quick Run 정책

COCO128은 성능 판단용이 아니다. 아래 항목만 확인한다.

- stage command 생성
- stage output directory 생성
- `stage_config.yaml` 생성
- `stage_result.yaml` 생성
- `stage_summary.md` 생성
- `metrics_delta.csv` 생성
- `export_check.json` 생성 가능 여부. 기본값은 `status: skip`이며 `--require-export`를 켠 경우에만 hard fail로 본다.
- hard fail 분류 가능 여부

COCO128에서 mAP가 낮거나 stage 간 mAP 변화가 이상해도 최종 성능 결론으로 사용하지 않는다. 단, crash, 산출물 누락, label 오류는 hard fail로 처리한다. export 실패는 `--require-export`를 켠 경우에만 hard fail로 처리한다.

## 6. Delta 계산 계약

각 stage는 아래 delta를 모두 저장한다.

```text
delta_vs_baseline
delta_vs_previous_success
delta_vs_best_previous
```

필수 metric:
- `primary_mAP`
- `mAP50`
- `small_AP`
- `rare_recall`
- `FP_per_image`
- `FN_per_image`
- `GFLOPs`
- `GFLOPs_delta_percent`
- `python_infer_ms`
- `python_nms_ms`
- `onnx_max_abs_diff`

metric이 없는 경우:
- COCO128에서 계산 불가능한 small AP/rare recall은 `null` 허용
- target full run에서는 primary metric이 `null`이면 fail

## 7. Report 생성 계약

각 stage 종료 후:
- `<stage>/stage_summary.md`
- `<stage>/stage_result.yaml`
- `<stage>/metrics_delta.csv`
- `<stage>/debug_trace.log`
- `<stage>/error_trace.log`

sequence 종료 후:
- `final_report/sequence_summary.md`
- `final_report/metrics_delta_all.csv`
- `final_report/decision_table.csv`
- `final_report/<train_type>_summary.md`
- `final_report/plots/*.png`, plot 실패는 soft warning

최종 report:
- `doc/REPORT/final_training_report_v1.8_YYYY-MM-DD.md`

최종 report에는 반드시 아래 섹션을 포함한다.
- 유지
- 제거
- 재실험
- 원인
- 다음 액션

## 8. 통과 기준

1. COCO128 quick run에서 Stage 00~02가 crash 없이 실행된다.
2. COCO128 quick run에서 stage별 summary/result/config가 생성된다.
3. `dataset_profile`이 모든 stage config에 기록된다.
4. hard fail과 soft fail이 구분된다.
5. `drop` stage 이후 carry flags에서 해당 flag가 제거된다.
6. `retry_tune`은 stage당 1회만 실행된다.
7. target full run 결과로 최종 report가 생성된다.
8. 최종 report가 유지/제거/재실험/원인/다음 액션을 포함한다.
9. command 실패 시 `stage_result.yaml`에 `exit_code`, `stdout_path`, `stderr_path`, `failed_category`, `missing_artifacts`가 기록된다.
10. `--debug-log error` 사용 시 stage별 `error_trace.log`가 생성된다.
11. `final_report/<train_type>_summary.md`가 학습 종류별로 생성된다.

## 9. 개발 착수 분리 기준

| PR | 범위 | 완료 기준 |
| --- | --- | --- |
| `1.3.8-P1` | `utils/stage_schema.py`, schema validation | config/result yaml validation 통과 |
| `1.3.8-P2` | `tools/run_training_sequence.py` dry-run | stage command와 output dir 계획 출력 |
| `1.3.8-P3` | COCO128 Stage 00~02 실행 | stage 산출물 생성 |
| `1.3.8-P4` | `collect_stage_results.py`, `compare_stage_metrics.py` | delta csv 생성 |
| `1.3.8-P5` | `generate_training_report.py` | stage/sequence/final report 생성 |
| `1.3.8-P6` | train_type summary | 학습 종류별 summary 생성 |
| `1.3.8-P7` | full sequence 연결 | COCO128 전체 sequence report 생성 |

## 10. 리스크 및 주의사항

- sequence tool이 학습 실패를 덮어쓰면 안 된다.
- command 실패는 exit code와 stderr path를 stage result에 남긴다.
- 이전 stage의 `best.pt`가 없으면 다음 stage를 시작하지 않는다.
- quick run과 target full run의 report를 섞지 않는다.
- final report는 사람이 읽는 markdown과 기계가 읽는 csv/yaml을 모두 남긴다.
- 그래프는 보조 자료다. 그래프 실패가 metric 누락을 숨기면 안 된다.

## 11. 구현 반영 메모

- `utils/stage_schema.py`는 `StageConfig`, `StageResult`, `Decision`을 정의하고 YAML load/save/validate를 담당한다.
- `tools/run_training_sequence.py`는 Stage 00~13 registry를 사용한다. Stage 12/13은 optional/defer로 두고, Stage 00~02 dry-run부터 산출물 경로와 command를 검증한다.
- `tools/collect_stage_results.py`는 `results.csv`, `results.txt`, `profile.json`, `export_check.json`, `stage_result.yaml`을 표준 dict로 수집한다.
- `tools/compare_stage_metrics.py`는 baseline, previous success, best previous 세 기준 delta를 CSV로 저장한다.
- `tools/generate_training_report.py`는 `stage_summary.md`, `sequence_summary.md`, `metrics_delta_all.csv`, `decision_table.csv`, `final_report/<train_type>_summary.md`, `doc/REPORT/final_training_report_v1.8_YYYY-MM-DD.md`를 같은 source dict에서 생성한다.
- `tools/run_training_sequence.py`는 stage command에 debug flag를 전달할 수 있어야 하며, 실패 시 `error_trace.log`와 `stage_result.yaml`을 동시에 남긴다.
- target full run에서는 기본적으로 직전 성공 stage 대비 `primary_mAP < -0.02` 또는 family baseline 대비 `GFLOPs > +10%`이면 `drop` soft fail로 분류한다. COCO128 quick run에서는 성능 기반 drop을 금지한다.
- ONNX/TensorRT 검증은 기본 필수 조건에서 제외한다. `export_check.json`은 기본 `skip`을 허용하고, Python export 검증을 강제하려면 sequence runner에 `--require-export`를 명시한다.
