# YOLOv7 Development Precautions v1.3.1-1.3.8

## 문서 정보

- 작성일: 2026-05-22
- 목적: 1.3.1부터 1.3.8까지 개발 시 항목별 개선 사유, 개선 예측, 주의점, 검증 기준을 정리한다.
- 기준 계획: `doc/PLAN/development_plan_v1.3.md`
- 학습 실행 기준: `doc/PLAN/training_execution_plan_v1.8.md`
- 리포트 기준: `doc/PLAN/training_report_format_v1.8.md`

## 세부 번호 체계

각 코드레벨 요구서에는 `1.3.x.1 코드 구현 상세` 섹션을 둔다.

| 차수 | 세부 구현 섹션 | 기준 문서 |
| --- | --- | --- |
| 1.3.1 | `1.3.1.1` | `development_requirements_1.3.1_baseline_export.md` |
| 1.3.2 | `1.3.2.1` | `development_requirements_1.3.2_train_loop_phase_logging.md` |
| 1.3.3 | `1.3.3.1` | `development_requirements_1.3.3_core_model_loss.md` |
| 1.3.4 | `1.3.4.1` | `development_requirements_1.3.4_cctv_augmentation_sampler.md` |
| 1.3.5 | `1.3.5.1` | `development_requirements_1.3.5_w6_structure_expansion.md` |
| 1.3.6 | `1.3.6.1` | `development_requirements_1.3.6_optional_experiments.md` |
| 1.3.7 | `1.3.7.1` | `development_requirements_1.3.7_finetuning_continual_learning.md` |
| 1.3.8 | `1.3.8.1` | `development_requirements_1.3.8_training_sequence_reporting.md` |

`1.3.x.1`은 요구사항이 아니라 구현 직전 기준이다. 실제 코드의 함수, 클래스, argparse, YAML/schema, checkpoint/report 저장 방식을 이 섹션 기준으로 맞춘다.

## 0. 공통 개발 원칙

개발자는 기능을 한 번에 모두 넣지 않는다. 코드는 플래그 기반으로 통합하되, 검증은 항상 한 stage 또는 한 PR 단위로 끝낸다.

공통 주의점:
- 기존 baseline 동작을 깨면 성능 개선 개발을 시작하지 않는다.
- `best.pt`, `results.csv`, `profile.json`, `export_check.json` 산출물이 없으면 다음 차수로 넘어가지 않는다.
- L 모델은 속도형, W6 모델은 정확도형 역할을 유지한다.
- Backbone 구조는 변경하지 않는다.
- C++ 후처리, TensorRT runtime, 추론 서버는 현재 개발 범위에서 제외한다.
- COCO128은 quick validation 전용이다. 최종 성능 판단은 대상 dataset full run으로 한다.

공통 중단 조건:
- NaN/Inf loss
- label 또는 class mapping 오류
- ONNX export 실패
- PyTorch/ONNX Runtime output 비교 실패
- GFLOPs 증가율 10% 이상
- primary mAP가 baseline 대비 2 percentage points 이상 하락
- 특정 핵심 클래스 recall 급락

## 1.3.1 Baseline / Python Export

### 개선 사유

모델 개선 전에 현재 학습/평가/export 경로가 정상인지 고정해야 한다. label path, cache, checkpoint, resume, result parsing이 불안정하면 이후 mAP 변화가 모델 개선 때문인지 코드 버그 때문인지 구분할 수 없다.

### 개선 예측

- mAP 자체 상승은 목표가 아니다.
- label missing, stale cache, `best.pt` 미생성, resume 실패, COCO 평가 crash 같은 baseline blocker가 줄어든다.
- ONNX raw export와 PyTorch/ONNX Runtime 비교 기준이 생긴다.
- 이후 모든 stage의 비교 기준이 안정화된다.

### 개발 주의점

- `img2label_paths()`는 일반 YOLO layout인 `images -> labels`를 기본 지원해야 한다.
- `JPEGImages` 특수 케이스를 유지하더라도 일반 `images` layout을 깨면 안 된다.
- label cache hash/version 검증을 반드시 복구한다.
- `persistent_workers=True`는 `workers > 0`일 때만 허용한다.
- `--close-mosaic`과 persistent worker가 충돌하지 않도록 이후 1.3.2 rebuild 정책과 맞춘다.
- `results.txt`는 numeric column을 유지하고, per-class text는 별도 파일로 뺀다.
- `opt.yaml`에 `Path` 객체가 Python-specific YAML tag로 저장되지 않게 문자열로 저장한다.
- `test.test()` return value 변경은 `train.py`, `train_aux.py` 호출부와 같이 맞춘다.
- 기본 checkpoint branch에서도 `weights/best.pt`가 항상 갱신되어야 한다.

### 봐야 할 지표/산출물

- `best.pt`, `last.pt`
- `results.csv` 또는 numeric `results.txt`
- `dataset_manifest.json`
- `baseline_metrics.json`
- `profile.json`
- `export_check.json`
- `output_contract.json`

### 예상 문제와 대응

| 문제 | 원인 | 대응 |
| --- | --- | --- |
| label missing | `images -> labels` mapping 깨짐 | `img2label_paths()` 표준 layout 복구 |
| cache stale | hash/version check 누락 | cache invalidation 복구 |
| resume 실패 | `opt.yaml`에 `Path` tag 저장 | `save_dir` 문자열 저장 |
| final COCO eval crash | `test.test()` unpack 불일치 | 4-return 호출부 정리 |
| plot 실패 | `results.txt`에 text token 삽입 | numeric log와 report 분리 |

## 1.3.2 Train Loop / Phase / Logging

### 개선 사유

학습 루프가 `train.py`와 `train_aux.py`로 갈라져 있으면 phase 전환, logging, checkpoint, eval 정책이 쉽게 어긋난다. 구조 개선 전에 학습 루프와 산출물 기준을 통합해야 이후 실험 결과를 비교할 수 있다.

### 개선 예측

- Phase 1/2/3 전환이 자동화된다.
- Close Mosaic이 실제 worker dataset에 반영된다.
- `results.csv`, `loss_detail.csv`, `phase_transition.log`로 문제 위치 추적이 쉬워진다.
- W6 grad accumulation과 AUX 경로 관리가 일관된다.

### 개발 주의점

- Phase 전환은 0-based epoch와 end-exclusive 기준을 명확히 한다.
- Phase 2/3 진입 시 부모 dataset 속성만 바꾸지 말고 Dataset/DataLoader를 재생성한다.
- `workers=0`과 `workers>0`을 모두 smoke test한다.
- `train_aux.py`를 바로 삭제하지 않는다. 먼저 공통 helper를 도입하고 wrapper화한다.
- Early stopping은 Phase 3에서만 작동한다.
- `results.csv`를 canonical source로 두고, console 출력과 plot 입력을 분리한다.
- `phase_transition.log`에는 epoch, img size, rect, mosaic, hyp, dataloader rebuild 여부를 남긴다.

### 봐야 할 지표/산출물

- `phase_transition.log`
- `results.csv`
- `loss_detail.csv`
- `stage_result.yaml`
- `hyp_used.yaml`
- dataloader build/rebuild 로그

### 예상 문제와 대응

| 문제 | 원인 | 대응 |
| --- | --- | --- |
| Phase boundary off-by-one | 0-based/1-based 혼동 | `tools/check_phase_schedule.py`로 boundary 고정 |
| Close Mosaic 미반영 | worker가 이전 dataset 복사본 유지 | DataLoader 완전 재생성 |
| W6 batch 불안정 | effective batch 관리 누락 | `grad_accumulate` 기록 및 검증 |
| 로그 파싱 실패 | CSV/text 혼합 | `results.csv` schema 고정 |

## 1.3.3 Core Model / Loss

### 개선 사유

Decoupled Head, WIoU, TAL, VFL은 모델 품질을 직접 올릴 수 있는 핵심 변경이다. 그러나 동시에 켜면 성능 변화의 원인을 알 수 없으므로 단독 검증 후 누적해야 한다.

### 개선 예측

- Decoupled Head: cls/reg task conflict 감소, 일부 class AP 개선 가능.
- WIoU v3: box regression 품질과 `mAP@0.5:0.95` 개선 가능.
- TAL + VFL: localization과 classification score alignment 개선, rare class recall 개선 가능.
- 누적 적용: primary mAP 상승 가능.

### 개발 주의점

- 최초 검증 순서는 `Decoupled Head 단독 -> WIoU 단독 -> TAL+VFL 단독 -> 누적`이다.
- VFL은 TAL positive 없이 단독 실행하지 않는다.
- TAL matching tensor는 prediction tensor와 같은 device에 둔다.
- WIoU dynamic weight 계산에서 `.detach()` 누락 시 loss 폭주 가능성이 있다.
- WIoU running state는 checkpoint/resume에 포함한다.
- Decoupled Head 적용 시 raw output shape는 기존 Detect와 동일하게 유지한다.
- 기존 weight load는 strict 실패가 아니라 partial load로 처리하고 missing/unexpected key를 로그에 남긴다.
- L 모델은 GFLOPs/latency 초과 시 head 변경을 fallback할 수 있어야 한다.

### 봐야 할 지표/산출물

- `loss_detail.csv`
- positive count
- `train/box_loss`, `train/cls_loss`
- per-class AP/recall
- `profile.json`
- `export_check.json`

### 예상 문제와 대응

| 문제 | 원인 | 대응 |
| --- | --- | --- |
| NaN/Inf loss | WIoU weight 또는 TAL matching 불안정 | `.detach()`, loss gain, fallback 확인 |
| positive 과소/과다 | TAL topk/alpha/beta 부적합 | hyp 조정 후 1회 retry |
| cls loss 과대 | VFL scale 문제 | cls gain 조정 |
| export shape 변경 | head output contract 변경 | raw output layout 유지 |
| L 속도 초과 | head 비용 증가 | L은 loss/assign만 유지하고 head fallback |

## 1.3.4 CCTV Augmentation / Sampler

### 개선 사유

CCTV 환경의 실패는 구조만으로 해결하기 어렵다. 오염, IR/흑백, 역광, 흔들림, 부분 가림, hard negative는 데이터 단계에서 반영해야 한다.

### 개선 예측

- pixel aug: IR/흑백/오염/압축 환경 robustness 증가.
- Patch-Paste: 부분 가림과 희귀 클래스 recall 개선 가능.
- Hard Negative: 배경 false positive 감소 가능.
- Weighted sampler: long-tail class recall 개선 가능.

### 개발 주의점

- label-preserving augmentation과 label-changing augmentation을 분리한다.
- label-changing augmentation은 visual audit 전 full training에 넣지 않는다.
- bbox range, class id, min area, occlusion ratio를 검사한다.
- 원본 image/label cache에는 증강 결과를 저장하지 않는다.
- sampler는 `image_weights`와 중복 적용하지 않는다.
- DDP에서는 sampler 충돌 가능성이 있으므로 단일 GPU smoke 후 확장한다.
- augmentation 확률은 stage별로 낮게 시작하고 metric을 보며 올린다.

### 봐야 할 지표/산출물

- `aug_check.json`
- `aug_samples/`
- `sampler_stats.csv`
- `scenario_metrics.csv`
- FP/image, FN/image
- class별 recall/precision

### 예상 문제와 대응

| 문제 | 원인 | 대응 |
| --- | --- | --- |
| label pollution | paste bbox/class 처리 오류 | 해당 aug off, visual audit 재실행 |
| clean mAP 하락 | aug 확률 과다 | probability 축소 |
| rare recall만 상승하고 precision 하락 | paste 과다 | class별 cap, conf threshold 조정 |
| dataloader 병목 | augmentation CPU 비용 | workers/cache/prefetch 조정 |
| sampler overfit | rare class 반복 과다 | weight 상한과 repeat cap 적용 |

## 1.3.5 W6 Structure Expansion

### 개선 사유

W6는 고해상도 정확도형 모델이므로 소형 객체 개선 여지가 크다. P2 Anchor는 stride 4 feature를 활용해 원거리/소형 객체 recall을 올릴 수 있고, SCDown은 고해상도 처리 효율을 개선할 가능성이 있다.

### 개선 예측

- P2 Anchor: small AP, small recall 증가 가능.
- SCDown: W6 neck 효율 개선, 일부 latency 완화 가능.
- P2 + SCDown: 소형 객체 개선과 비용 완화의 균형 가능.

### 개발 주의점

- W6 전용이다. L 구조는 변경하지 않는다.
- D1 SCDown only, D2 P2 only, D3 P2+SCDown을 분리한다.
- P2 추가 시 Detect/IAuxDetect는 5-level을 처리해야 한다.
- stride는 `4, 8, 16, 32, 64` 기준으로 검증한다.
- output box 수 증가와 Python NMS 비용을 반드시 측정한다.
- W6 AUX 구조에서 main 5 + aux 5 입력 수를 맞춘다.
- cfg 하나로 모든 실험을 처리하지 말고 route diff를 작게 유지한다.
- GFLOPs 10% 초과 시 P2 channel 축소 또는 fallback한다.

### 봐야 할 지표/산출물

- small AP
- small recall
- total boxes
- Python NMS ms
- memory usage
- `nms_cost.json`
- `profile.json`
- `export_check.json`

### 예상 문제와 대응

| 문제 | 원인 | 대응 |
| --- | --- | --- |
| route/channel mismatch | cfg route 오류 | D1/D2/D3 cfg 분리 |
| NMS 비용 급증 | P2 output 증가 | score threshold, pre-NMS top-k 검토 |
| duplicate detection 증가 | anchor/NMS 기준 부적합 | anchor size, NMS IoU 조정 |
| export 실패 | unsupported op 또는 shape 불일치 | 표준 Conv/BN/activation 유지 |
| 전체 mAP 하락 | P2 loss 비중 과다 | P2 loss gain 조정 |

## 1.3.6 Optional Experiments

### 개선 사유

PSA, FCOS, GELAN, L AUX는 기본 개발 흐름이 아니라 목표 미달 시 검토하는 후순위 실험이다. 필수 구성으로도 성능/속도 목표를 만족하면 개발하지 않는다.

### 개선 예측

- L AUX on: L 희귀 클래스 recall 개선 가능.
- PSA P5: W6 고수준 feature attention 개선 가능.
- FCOS P2: P2 Anchor로 부족한 소형 객체 recall 추가 개선 가능.
- GELAN 일부 교체: neck 표현력 개선 가능.

### 개발 주의점

- 진입 전 `doc/REPORT/optional_decision_*.md`에 사유를 기록한다.
- optional 구조는 동시에 둘 이상 켜지 않는다.
- PSA는 P5 단독부터 시작하고 P4/P3 동시 적용을 금지한다.
- FCOS P2는 Python raw/decode까지만 다루며 C++/runtime은 별도 차수다.
- GELAN은 route/channel/export 리스크가 크므로 일부 block부터 시작한다.
- 효과가 작으면 기본 off로 유지한다.

### 봐야 할 지표/산출물

- optional decision report
- target metric delta
- GFLOPs/latency 여유
- export 상태
- route/channel smoke 결과

### 예상 문제와 대응

| 문제 | 원인 | 대응 |
| --- | --- | --- |
| 효과 미미 | 병목이 optional 구조가 아님 | optional off 유지 |
| latency 초과 | attention 또는 output 증가 | 실험 중단 |
| FCOS decode 복잡 | anchor/anchor-free score 결합 문제 | Python 검증까지만 유지 |
| GELAN route 오류 | concat/channel mismatch | 적용 범위 축소 |

## 1.3.7 Fine-tuning / Continual Learning

### 개선 사유

신규 데이터나 일부 클래스만으로 파인튜닝하면 기존 클래스 성능이 급락할 수 있다. Replay, Pseudo Label, LwF를 통해 catastrophic forgetting을 줄인다.

### 개선 예측

- Replay only: 기존 클래스 mAP 유지율 증가.
- Pseudo Label: 파인튜닝 이미지 내 기존 클래스 누락 완화.
- LwF cls/reg distill: 기존 teacher의 분포를 보존해 forgetting 감소 가능.

### 개발 주의점

- scratch 기준선과 class mapping이 확정된 뒤 시작한다.
- class count는 예시값에 의존하지 않고 data yaml 기준으로 읽는다.
- Replay only를 먼저 통과한 뒤 LwF를 켠다.
- Teacher model은 학습 보조에만 사용하고 최종 추론 구조에는 포함하지 않는다.
- pseudo label은 confidence threshold와 IoU 중복 제거 기준을 기록한다.
- distill alpha/beta는 신규 클래스 학습을 막지 않도록 schedule을 지원한다.
- BN policy와 freeze policy를 명시한다.

### 봐야 할 지표/산출물

- `class_mapping_check.json`
- `pseudo_label_manifest.json`
- `merge_report.json`
- `replay_manifest.json`
- 기존 클래스 mAP
- 대상 클래스 mAP
- forgetting delta

### 예상 문제와 대응

| 문제 | 원인 | 대응 |
| --- | --- | --- |
| 신규 클래스 학습 부족 | distill alpha 과다 | alpha 낮춤 |
| 기존 클래스 하락 | replay 부족 | replay ratio 증가 |
| pseudo label 오염 | threshold 낮음 | confidence 상향, IoU 중복 제거 강화 |
| teacher 비용 과다 | forward 반복 | pseudo label cache 사용 |

## 1.3.8 Training Sequence / Report Automation

### 개선 사유

개발이 끝난 뒤 모든 기능을 한 번에 full training하면 어떤 기능이 성능을 올렸고 어떤 기능이 문제를 만들었는지 알기 어렵다. 1.3.8은 v1.8 학습 실행 플랜을 코드/운영 도구로 연결해 stage별 실행과 리포트 생성을 자동화하는 차수다.

### 개선 예측

- COCO128 quick run으로 orchestration 문제를 빠르게 찾는다.
- Stage별 `keep/drop/retry_tune/blocker` 판정이 자동 기록된다.
- baseline, previous, best 기준 delta가 자동 계산된다.
- 최종 리포트에서 유지/제거/재실험 기능을 근거 있게 결정할 수 있다.
- 학습 서버 테스트 시간이 줄어든다.

### 개발 범위

- `tools/run_training_sequence.py`
- `tools/collect_stage_results.py`
- `tools/generate_training_report.py`
- `stage_config.yaml` schema
- `stage_result.yaml` schema
- `stage_summary.md` template
- `sequence_summary.md` template
- `final_training_report_v1.8_YYYY-MM-DD.md` 생성

### 개발 주의점

- COCO128 quick run은 최종 성능 판단에 사용하지 않는다.
- `dataset_profile`은 `coco128_quick` 또는 `target_full`로 반드시 기록한다.
- quick run에서는 crash, 산출물 누락, report 판정 오류만 본다.
- target full run에서만 최종 유지/제거 결정을 한다.
- stage 실패 시 다음 stage를 어떤 weight에서 시작할지 명확히 기록한다.
- `drop` 판정된 flag는 다음 stage carry flags에서 제거한다.
- `retry_tune`은 무한 반복하지 않고 stage당 1회만 허용한다.
- 그래프 생성 실패는 hard fail이 아니지만, metric CSV 누락은 fail로 처리한다.
- 리포트는 단순 mAP 표가 아니라 `유지`, `제거`, `재실험`, `원인`, `다음 액션`을 포함해야 한다.

### 봐야 할 지표/산출물

- `runs/train_seq/v1.8_coco128_quick/*/stage_summary.md`
- `runs/train_seq/v1.8_coco128_quick/final_report/sequence_summary.md`
- `doc/REPORT/final_training_report_v1.8_YYYY-MM-DD.md`
- `metrics_delta.csv`
- `decision_waterfall.png`
- `per_class_delta_heatmap.png`

### 예상 문제와 대응

| 문제 | 원인 | 대응 |
| --- | --- | --- |
| stage 순서가 꼬임 | carry flags 관리 오류 | `stage_config.yaml`에 enabled/disabled flags 기록 |
| COCO128 결과를 성능 결론으로 오해 | dataset profile 미기록 | `dataset_profile` 필수화 |
| 리포트 delta가 틀림 | baseline/previous/best 기준 혼동 | 세 비교 기준을 별도 컬럼으로 저장 |
| 실패 후 잘못된 weight로 계속 진행 | fallback weight 기록 누락 | `fallback_weight` 필수화 |
| retry 반복 | retry policy 없음 | stage당 1회 제한 |
| 최종 판단 불명확 | decision enum 없음 | `keep/drop/retry_tune/blocker/defer`만 허용 |

## 최종 개발 체크리스트

개발 완료 전 아래 질문에 모두 답할 수 있어야 한다.

1. 이 변경의 개선 사유가 명확한가?
2. 기대되는 증가/감소 항목이 문서화되어 있는가?
3. 실패 시 끌 수 있는 flag가 있는가?
4. baseline과 비교할 산출물이 있는가?
5. COCO128 quick run에서 crash 없이 지나가는가?
6. target full run에서만 최종 성능 판단하도록 분리되어 있는가?
7. final report에서 유지/제거/재실험 판단이 가능한가?
