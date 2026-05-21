# v1.3 기반 doc/dev 코드레벨 요구서 재검토

- 점검일: 2026-05-22
- 기준 문서: `doc/BASIC/YOLOv7_Custom_Design_Spec_v1.3.md`
- 대상 문서: `doc/dev/development_requirements_1.3.1_*.md` ~ `doc/dev/development_requirements_1.3.7_*.md`
- 목적: v1.3 설계 요구가 코드레벨 개발 요구서에 빠짐없이 내려왔는지 재점검
- 제외 유지: C++ 후처리, TensorRT runtime, TensorRT engine build, 별도 추론 서버

## 결론

`doc/dev` 문서들은 전체 차수 순서와 개발 방향은 맞다. 다만 v1.3 원문에 있던 세부 산출물, raw output 계약, phase별 logging, augmentation 세부 정책, W6 구조 실험 분리, LwF 세부 스케줄이 충분히 상세하지 않았다. 이번 재검토에서 해당 항목을 각 dev 문서에 직접 반영했다.

현재 상태는 1.3.1부터 실제 코드 개발 착수 기준으로 사용할 수 있다. 단, 문서는 요구서이며 현재 코드가 이미 충족한다는 의미는 아니다.

## 전체 공통 기준

확인한 공통 기준:
- 개발은 플래그 기반으로 통합하되 학습 서버에서는 차수별로 하나씩 켠다.
- 각 차수는 직전 차수 산출물이 있어야 시작한다.
- `primary mAP` 2 percentage points 이상 하락, GFLOPs 10% 이상 증가, ONNX 비교 실패, NaN/Inf loss는 중단 조건이다.
- C++/TensorRT runtime/추론 서버는 별도 요청 전까지 제외한다.
- raw ONNX output과 PyTorch/ONNX Runtime 비교는 구조 변경마다 유지한다.

## 차수별 재검토 결과

### 1.3.1 Baseline / Python Export

보강 전 누락:
- v1.3의 validation checksum이 구체적 산출 도구로 내려오지 않았다.
- raw output의 layout/shape 계약이 명확하지 않았다.
- baseline metric snapshot이 `results.txt`에만 묻힐 수 있었다.

반영:
- `tools/dataset_manifest.py`와 `dataset_manifest.json` 요구사항 추가
- `output_contract.json` 스키마 추가
- `baseline_metrics.json` 스키마 추가
- export output을 `[batch, total_boxes, 5 + nc]`, `nms_mode=none`, `postprocess_in_graph=false`, `aux_exported=false`로 고정

### 1.3.2 Train Loop / Phase / Logging

보강 전 누락:
- v1.3의 `results_per_class.csv`, `train_log.txt`, `hyp_used.yaml`, PR/F1/confusion/results plot 산출물이 빠져 있었다.
- Early Stopping Phase 3 전용 조건과 EMA 전구간 유지가 약했다.
- W6 `batch=8`, `grad_accumulate=4` 기준이 명시되지 않았다.
- L/W6 rect 입력 alias가 부족했다.

반영:
- `results_per_class.csv`, `loss_detail.csv`, `train_log.txt`, `hyp_used.yaml`, plot 산출물 추가
- `--grad-accumulate`, `--early-stop-phase`, `--patience`, `--profile` 추가
- `--rect-size-l`, `--rect-size-w6`, `--phase3-img` 추가
- `stage_result.yaml`에 `primary_mAP`, `small_AP`, `rare_recall`, `trt_latency: null`, `output_contract_json` 추가
- Early stopping은 Phase 3에서만 동작하도록 통과 기준 보강

### 1.3.3 Core Model / Loss

보강 전 누락:
- v1.3의 warmup/Phase별 `lambda_aux`, `lambda_free` schedule이 코드레벨로 부족했다.
- `cls_pw`, small object IoU weight 기준이 약했다.
- WIoU/TAL/VFL 적용 후 loss scale 기록 기준이 부족했다.

반영:
- `Loss Weight Schedule` 섹션 추가
- warmup 30 epoch, Phase 1/2/3 lambda 기준 추가
- `cls_pw`, `cls_pw_per_class`, `small_iou_weight` 정책 추가
- `stage_result.yaml`에 `lambda_aux`, `lambda_free`, `cls_pw_mode`, `small_iou_weight_enabled` 추가
- epoch boundary에서 loss weight schedule 확인 기준 추가

### 1.3.4 CCTV Augmentation / Sampler

보강 전 누락:
- v1.3의 aug 목록이 dev 문서에서는 너무 요약되어 있었다.
- phase별 aug 적용 범위가 명확하지 않았다.
- 오감지/미감지 scenario metric이 산출물로 잡히지 않았다.
- hard negative 후보 채굴 도구가 빠져 있었다.

반영:
- `Aug Profile Matrix` 추가
- SpiderWeb, IR reflection, ToGray, Mosaic9, Patch-Paste, Hard Negative, Helmet paste, GridMask, RandomShift, blur/noise/flare/CLAHE 등 v1.3 항목 반영
- GlassBlur/Posterize 제외, Rolling Shutter 선택 옵션 명시
- Phase 1/2/3별 aug 적용 표 추가
- `tools/mine_hard_negatives.py`, `utils/augment_policy.py` 추가
- `scenario_metrics.csv` 추가
- Phase 3에서 label-changing aug와 Mosaic이 꺼져야 한다는 통과 기준 추가

### 1.3.5 W6 Structure Expansion

보강 전 누락/오류:
- `utils/autoanchor.py` 요구사항에 4개 detection layer 표현이 남아 있었다.
- P2와 SCDown 효과를 분리할 하위 단계가 없었다.
- P2 path의 기본 구조와 output contract 산출물이 약했다.

반영:
- autoanchor 기준을 5 detection layer로 수정
- `1.3.5-D1` SCDown 단독, `1.3.5-D2` P2 Anchor 단독, `1.3.5-D3` 누적 적용으로 분리
- P2 path는 upsample + Conv 2회 기본으로 명시
- full run에서 `--rect-size-w6 1280 736 --grad-accumulate 4` 사용 명시
- `output_contract.json`, `sub_stage`, `memory_peak_mb`, `decision` 산출 필드 추가

### 1.3.6 Optional Experiments

보강 전 누락:
- PSA가 P5 단독부터 시작한다는 v1.3 조건이 약했다.
- FCOS P2의 Python decode 검증 산출물이 없었다.
- optional 기본값 승격/보류 결정 기록은 있었지만 실험별 이유 필드가 부족했다.

반영:
- `--psa-level p5|p4p5|p3p4p5` 추가
- PSA는 P5 단독 없이 P3/P4/P5 동시 적용 금지
- `tools/decode_fcos_outputs.py`, `fcos_decode_check.json` 추가
- `optional_ablation.csv`에 `reason` 추가

### 1.3.7 Fine-tuning / Continual Learning

보강 전 누락:
- Replay only와 LwF A/B가 하위 단계로 분리되어 있지 않았다.
- alpha/beta schedule과 reg distillation conf threshold 기준이 약했다.
- BN/freeze 정책과 class mapping check가 부족했다.

반영:
- `1.3.7-E1` Replay only, `E2` Replay + cls distill, `E3` Replay + cls/reg distill 추가
- `distill-alpha 0.2:0.5`, 필요 시 0.8까지 상향 가능
- `distill-beta 0.1:0.3`, reg distillation은 conf threshold 이상만 적용
- `tools/check_class_mapping.py`, `class_mapping_check.json` 추가
- `--bn-policy train|eval`, freeze/trainable parameter count 기록 기준 추가

## 남은 구현 전 주의점

- 현재 코드에는 아직 신규 플래그와 신규 도구 대부분이 없다.
- 1.3.1의 blocker인 label mapping, cache invalidation, persistent worker, best checkpoint, test return unpack, ONNX 옵션을 먼저 처리해야 한다.
- W6 P2 적용 시 기존 `IAuxDetect`의 4-level 전제가 남아 있으면 반드시 `m.nl` 기반으로 고쳐야 한다.
- `--img`는 학습 스크립트와 export/profile 도구에서 의미가 다르므로 문서의 구분을 코드에도 반영해야 한다.
- TensorRT 관련 문구는 v1.3 원문에 남아 있지만, 현재 개발 차수에서는 Python ONNX 검증까지만 수행한다.

## 최종 판단

이번 보강 후 `doc/dev` 문서는 v1.3 설계 요구를 코드레벨 요구사항으로 충분히 반영한다. 다음 단계는 문서 추가 작성이 아니라 `1.3.1` 요구서 기준으로 코드 구현을 시작하고, 완료 시 `doc/REPORT/stage_result_1.3.1_YYYY-MM-DD.md`를 저장하는 것이다.
