# Finetune Command Guide 2026-06-08

## 목적

기존 모델 성능을 최대한 유지하면서 파인튜닝 데이터 성능이 오르는 조건을 찾기 위한 실행 명령 모음이다.

핵심 출력은 아래 두 파일이다.

```text
runs/fine/<prefix>_sweep/retention_summary.csv
runs/fine/<prefix>_sweep/retention_summary.json
```

## Data YAML

`--best-val-set combined`를 쓰려면 validation set을 두 개 이상 넣는다.

```yaml
train: /workspace/datasets/Congress-1/file/list/train.txt
val:
  - name: congress_valid
    path: /workspace/datasets/Congress-1/file/list/valid.txt
  - name: base_valid
    path: /workspace/datasets/Congress-1/file/list/valil_falldown.txt
nc: <same_nc>
names: <same_names>
```

`--finetune-scenario`와 `--base-scenario`는 위 `name`과 맞춘다.

## 전체 실험 1epoch Smoke

`core_pseudo`는 기존 core 실험과 pseudo old-label 실험을 모두 실행한다.

```bash
python tools/run_finetune_sweep.py \
  --weights /workspace/datasets/svms/file/SIAV2_Detector_YOLOV7_SafeEnv_V7.0.0_FP32_260223.pt \
  --teacher-weights /workspace/datasets/svms/file/SIAV2_Detector_YOLOV7_SafeEnv_V7.0.0_FP32_260223.pt \
  --base-data /workspace/datasets/svms/file/data/data.yaml \
  --data /workspace/datasets/svms/file/data/data.yaml \
  --hyp /workspace/datasets/svms/file/data/hyp.scratch.custom.yaml \
  --epochs 1 \
  --img 640 \
  --batch 64 \
  --workers 8 \
  --project runs/fine \
  --prefix congress_all_smoke \
  --preset core_pseudo \
  --launcher distributed \
  --nproc-per-node 4 \
  --master-port 9527 \
  --best-val-set combined \
  --replay-ratio-source finetune \
  --pseudo-conf 0.6 \
  --save-best-only \
  --execute \
  --summarize \
  --continue-on-error \
  --baseline-run runs/fine/base_eval_or_original_run \
  --finetune-scenario congress_valid \
  --base-scenario base_valid \
  --summary-select-scenario combined
```

## 전체 실험 실사용

1epoch smoke가 통과하면 `--epochs`만 늘린다. 현재 결과 기준으로는 30epoch가 과적합이었으므로 먼저 `6~8` epoch를 권장한다.

```bash
python tools/run_finetune_sweep.py \
  --weights /workspace/datasets/svms/file/SIAV2_Detector_YOLOV7_SafeEnv_V7.0.0_FP32_260223.pt \
  --teacher-weights /workspace/datasets/svms/file/SIAV2_Detector_YOLOV7_SafeEnv_V7.0.0_FP32_260223.pt \
  --base-data /workspace/datasets/svms/file/data/data.yaml \
  --data /workspace/datasets/svms/file/data/data.yaml \
  --hyp /workspace/datasets/svms/file/data/hyp.scratch.custom.yaml \
  --epochs 8 \
  --img 640 \
  --batch 64 \
  --workers 8 \
  --project runs/fine \
  --prefix congress_all_e8 \
  --preset core_pseudo \
  --launcher distributed \
  --nproc-per-node 4 \
  --master-port 9527 \
  --best-val-set combined \
  --replay-ratio-source finetune \
  --pseudo-conf 0.6 \
  --save-best-only \
  --execute \
  --summarize \
  --continue-on-error \
  --baseline-run runs/fine/base_eval_or_original_run \
  --finetune-scenario congress_valid \
  --base-scenario base_valid \
  --summary-select-scenario combined
```

## Pseudo 실험만 실행

teacher pseudo old-label 병합 효과만 먼저 보고 싶을 때 사용한다.

```bash
python tools/run_finetune_sweep.py \
  --weights /workspace/datasets/svms/file/SIAV2_Detector_YOLOV7_SafeEnv_V7.0.0_FP32_260223.pt \
  --teacher-weights /workspace/datasets/svms/file/SIAV2_Detector_YOLOV7_SafeEnv_V7.0.0_FP32_260223.pt \
  --base-data /workspace/datasets/svms/file/data/data.yaml \
  --data /workspace/datasets/svms/file/data/data.yaml \
  --hyp /workspace/datasets/svms/file/data/hyp.scratch.custom.yaml \
  --epochs 1 \
  --img 640 \
  --batch 64 \
  --workers 8 \
  --project runs/fine \
  --prefix congress_pseudo_smoke \
  --preset pseudo \
  --launcher distributed \
  --nproc-per-node 4 \
  --master-port 9527 \
  --best-val-set combined \
  --replay-ratio-source finetune \
  --pseudo-conf 0.6 \
  --save-best-only \
  --execute \
  --summarize \
  --continue-on-error \
  --baseline-run runs/fine/base_eval_or_original_run \
  --finetune-scenario congress_valid \
  --base-scenario base_valid \
  --summary-select-scenario combined
```

## 결과 해석 기준

먼저 `retention_summary.csv`를 본다.

우선순위:

```text
1. status == keep_candidate
2. base_retention_percent >= 95
3. finetune_delta > 0
4. base_worst_class_delta가 과도하게 낮지 않음
5. best_weight 경로의 best.pt 사용
```

`last.pt`가 아니라 항상 `best.pt`를 비교한다.

## 주요 산출물

```text
runs/fine/<prefix>_sweep/sweep_plan.yaml
runs/fine/<prefix>_sweep/commands.sh
runs/fine/<prefix>_sweep/sweep_run_status.json
runs/fine/<prefix>_sweep/retention_summary.csv
runs/fine/<prefix>_sweep/retention_summary.json
```

각 실험 폴더:

```text
runs/fine/<prefix>_<experiment_id>/weights/best.pt
runs/fine/<prefix>_<experiment_id>/results_detail.txt
runs/fine/<prefix>_<experiment_id>/scenario_metrics.csv
runs/fine/<prefix>_<experiment_id>/stage_result.yaml
runs/fine/<prefix>_<experiment_id>/replay_manifest.json
```
