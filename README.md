# YOLOV7_NEW

YOLOv7 기반 학습, 평가, export, 파인튜닝 실험 저장소입니다.

## 파인튜닝 유지율 Sweep

목적은 단순 파인튜닝이 아니라, 튜닝 데이터 성능을 올리면서 기존/base validation 성능 하락을 줄이는 것입니다. `data.yaml`에 튜닝 validation과 기존 validation을 함께 넣고 `--best-val-set combined`를 사용합니다.

권장 validation yaml 형식:

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

먼저 sweep plan과 각 실험의 dry-run 산출물만 생성합니다.

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
  --prefix congress_retention_core \
  --preset core \
  --launcher distributed \
  --nproc-per-node 4 \
  --master-port 9527 \
  --best-val-set combined \
  --save-best-only
```

실제 학습까지 순차 실행하려면 같은 명령에 `--execute`를 추가합니다.

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
  --prefix congress_retention_core \
  --preset core \
  --launcher distributed \
  --nproc-per-node 4 \
  --master-port 9527 \
  --best-val-set combined \
  --save-best-only \
  --execute
```

최근 논문 검토를 반영한 pseudo old-label completion 실험만 돌리려면 `--preset pseudo`를 사용합니다. 이 preset은 teacher로 파인튜닝 train 이미지의 누락된 기존 클래스 후보를 pseudo label로 만들고, GT와 병합한 shadow dataset을 생성한 뒤 학습합니다.

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
  --best-val-set combined \
  --replay-ratio-source finetune \
  --pseudo-conf 0.6 \
  --save-best-only \
  --execute
```

주요 산출물:

- `runs/fine/congress_retention_core_sweep/sweep_plan.yaml`
- `runs/fine/congress_retention_core_sweep/commands.sh`
- 각 실험 폴더의 `weights/best.pt`
- 각 실험 폴더의 `results_detail.txt`, `scenario_metrics.csv`, `stage_result.yaml`, `replay_manifest.json`

판정은 `last.pt`가 아니라 `best.pt` 기준으로 합니다. 기존/base validation 유지율을 먼저 보고, 유지율이 통과한 후보 중 튜닝 validation `mAP50-95`가 가장 높은 설정을 선택합니다.

결과 요약:

```bash
python tools/summarize_finetune_retention.py \
  --baseline runs/fine/base_eval_or_original_run \
  --runs runs/fine/congress_pseudo_smoke_* \
  --finetune-scenario congress_valid \
  --base-scenario base_valid \
  --select-scenario combined \
  --output runs/fine/congress_pseudo_retention_summary.csv
```

전체 core + pseudo 실험을 한 번에 실행하고 최종 summary까지 생성하려면 아래 명령을 사용합니다.

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

최종 결과는 `runs/fine/congress_all_smoke_sweep/retention_summary.csv`와 `retention_summary.json`에 저장됩니다. 각 실험의 성공/실패 상태는 `sweep_run_status.json`에 저장됩니다.

상세 실험 매트릭스와 keep/drop 기준은 `doc/PLAN/finetune_retention_experiment_matrix_2026-06-07.md`를 참고합니다.
