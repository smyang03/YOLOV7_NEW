# YOLOv7 Training Execution Plan v1.8

## 문서 정보

- 작성일: 2026-05-22
- 목적: v1.3 개발 완료 후 학습 서버에서 단계별 학습을 연속 실행하고 최종 리포트로 개선/문제 원인을 분리한다.
- 기준 문서: `doc/PLAN/development_plan_v1.3.md`
- 기준 리포트: `doc/REPORT/ai_perspective_yolov7_improvement_analysis_2026-05-22.md`
- 리포트 규격: `doc/PLAN/training_report_format_v1.8.md`
- 적용 범위: Python 학습, 평가, stage/report 자동화. ONNX raw export 검증은 선택이며, C++ 후처리, TensorRT runtime, 추론 서버는 제외한다.

## 1. 실행 철학

모든 기능은 코드에 구현되어 있어도 학습에서는 한 번에 모두 켜지 않는다. 학습 서버에서는 stage를 순서대로 실행하고, 각 stage가 끝날 때마다 metric, GFLOPs, export, 로그 안정성을 저장한다.

기본 실행 방식은 `누적 연속 실행`이다. 즉, `Stage N`은 직전 성공 stage의 `best.pt`에서 시작한다. 이렇게 하면 전체 테스트 시간이 줄고, 어느 stage에서 문제가 생겼는지 빠르게 찾을 수 있다.

단, 최종 후보는 반드시 `확인용 full run`을 한 번 더 수행한다. 누적 연속 실행은 빠른 원인 추적용이고, 최종 성능 확정은 같은 dataset, seed, epoch 정책으로 다시 학습한 결과를 기준으로 한다.

## 2. 실행 전 고정 조건

### 2.1 최초 빠른 테스트 데이터셋

최초 테스트는 `COCO128`로 진행한다. 목적은 최종 성능 판단이 아니라 stage orchestration, flag 조합, output 생성, 리포트 판정 로직을 빠르게 검증하는 것이다.

권장 데이터:
- `data/coco128.yaml`

COCO128 quick run에서 확인할 항목:
- Stage 00~02가 짧은 시간 안에 연속 실행되는지
- `stage_summary.md`, `sequence_summary.md`, `stage_result.yaml`이 생성되는지
- `keep/drop/retry_tune/blocker` 판정 값이 정상 기록되는지
- `profile.json`, `export_check.json`, `metrics_delta.csv`가 생성되는지
- label cache, DataLoader rebuild, Close Mosaic이 crash 없이 동작하는지

COCO128 quick run에서 판단하지 않을 항목:
- 최종 mAP 우열
- CCTV domain robustness
- rare class recall 개선
- W6 P2/SCDown의 실제 운영 성능
- 파인튜닝 forgetting 억제 효과

COCO128 결과는 `quick_validation`으로만 기록한다. 최종 유지/제거 결정은 실제 대상 dataset full run 결과로 확정한다.

예시 실행:

```bash
python tools/run_training_sequence.py \
  --plan doc/PLAN/training_execution_plan_v1.8.md \
  --data data/coco128.yaml \
  --model-family l,w6 \
  --output runs/train_seq/v1.8_coco128_quick \
  --stop-on-hard-fail
```

### 2.2 Full Run 고정 조건

아래 항목은 전체 stage에서 바꾸지 않는다.

- dataset yaml, train/val split, class index mapping
- validation image/label checksum
- seed, device, batch/effective batch, workers
- baseline weight와 baseline cfg
- primary metric: `mAP@0.5:0.95`
- secondary metric: `mAP@0.5`, small AP, rare recall, FP/image, FN/image
- speed metric: params, GFLOPs, Python inference ms, Python NMS ms. ONNX Runtime diff는 `--require-export`를 켠 실행에서만 기록한다.

공통 산출물:
- `stage_config.yaml`
- `stage_result.yaml`
- `stage_summary.md`
- `results.csv`
- `results_per_class.csv`
- `loss_detail.csv`
- `profile.json`
- `export_check.json`
- `metrics_delta.csv`
- `sequence_summary.md`
- `final_training_report_v1.8.md`

저장 위치 예시:

```text
runs/train_seq/v1.8/
  00_baseline/
  01_phase/
  02_head_decoupled/
  03_wiou/
  ...
  final_report/
```

## 3. 실행 도구 요구

권장 실행기는 `tools/run_training_sequence.py`다. 개발 완료 후 이 도구가 준비되어 있으면 아래처럼 실행한다.

```bash
python tools/run_training_sequence.py \
  --plan doc/PLAN/training_execution_plan_v1.8.md \
  --data data/custom.yaml \
  --model-family l,w6 \
  --output runs/train_seq/v1.8 \
  --stop-on-hard-fail
```

도구가 아직 없으면 동일 stage 순서를 shell script로 실행해도 된다. 중요한 것은 모든 stage가 같은 규칙으로 결과를 저장하는 것이다. 최초 검증은 위 full run 명령이 아니라 `data/coco128.yaml`을 사용하는 quick run으로 먼저 수행한다.

## 4. Stage 실행 정책

Hard fail이면 즉시 중단한다.

- NaN/Inf loss
- 학습 프로세스 crash
- validation 불가
- `--require-export`를 켠 실행에서 ONNX export 실패
- `--require-export`를 켠 실행에서 PyTorch/ONNX Runtime output 비교 실패
- `best.pt` 미생성
- label 검증 실패
- class mapping 불일치

Soft fail이면 해당 기능을 끄고 직전 성공 stage에서 다음 stage를 진행한다.

- primary mAP 2 percentage points 이상 하락
- GFLOPs 증가율 10% 이상
- Python NMS 비용 급증
- 특정 클래스 recall 급락
- train loss는 감소하지만 val mAP가 계속 하락
- rare class만 개선되고 common class가 크게 하락

## 5. 전체 Stage 순서

### Stage 00. Baseline 고정

목적: 개발 완료 후에도 원본 기준선이 재현되는지 확인한다.

활성 기능:
- 원본 YOLOv7-L/W6
- 기본 loss
- 기본 augmentation
- raw ONNX export 검증

예상 변화:
- 증가: 없음
- 감소: 없음
- 기준값으로 고정할 항목: mAP, per-class AP, small AP, GFLOPs, Python inference ms, ONNX diff

봐야 할 부분:
- `best.pt`, `last.pt`, `results.csv` 생성 여부
- dataset manifest와 validation checksum
- 기존 repo 대비 mAP 차이

예상 문제와 개선:
- label missing 발생: `images -> labels` 매핑과 cache invalidation 확인
- resume 실패: `opt.yaml` safe load 확인
- best 미생성: checkpoint save branch 확인

### Stage 01. Phase / Logging / Rebuild

목적: 모델 성능 개선 없이 학습 루프만 통합한다.

활성 기능:
- `--phase-train on`
- Phase 1/2/3
- DataLoader rebuild
- Close Mosaic
- canonical logging

예상 변화:
- 증가: 로그 정합성, phase별 추적성, 최종 분포 적응 안정성
- 감소: 잘못된 mosaic 지속, 결과 파일 파싱 오류, 재현 불가능성
- 성능: 큰 mAP 상승을 기대하지 않는다. Phase 3에서 val 안정성이 좋아질 수 있다.

봐야 할 부분:
- `phase_transition.log`
- Phase boundary epoch
- Phase 2/3에서 train/val image size
- `workers=0/>0` 동작
- Phase 3에서 mosaic이 실제로 꺼졌는지

예상 문제와 개선:
- Close Mosaic이 worker에 반영되지 않음: DataLoader 완전 재생성
- Phase 2 rect 전환 후 mAP 급락: rect image size, letterbox, batch shape 확인
- 로그 컬럼 깨짐: `results.csv`를 canonical source로 사용하고 `results.txt`는 numeric 유지

### Stage 02. Decoupled Head 단독

목적: cls/reg 분리 효과만 확인한다.

활성 기능:
- `--head decoupled`
- 기존 CIoU/BCE/SimOTA 유지

예상 변화:
- 증가: cls/reg task 분리, 일부 클래스 AP, localization 안정성
- 감소: head 내부 task conflict
- 비용: params/GFLOPs 소폭 증가 가능, L 모델은 특히 latency 확인 필요

봐야 할 부분:
- `train/cls_loss`, `train/box_loss` 분리 추세
- per-class AP 변화
- GFLOPs 증가율
- raw output shape 유지
- 기존 weight partial load 로그

예상 문제와 개선:
- 기존 weight load mismatch: missing/unexpected key를 로그로 남기고 head만 재초기화
- L latency 초과: L은 decoupled head 폭을 줄이거나 fallback
- export shape 변경: Detect output contract를 baseline과 동일하게 유지

### Stage 03. WIoU v3 단독

목적: 추론 구조 변경 없이 box regression 품질을 개선한다.

활성 기능:
- `--loss-box wiou_v3`
- `--head coupled`
- `--assign simota`
- `--loss-cls bce`

예상 변화:
- 증가: box 품질, `mAP@0.5:0.95`, localization 안정성
- 감소: box regression outlier 영향
- 비용: GFLOPs와 inference latency 변화 없음

봐야 할 부분:
- box loss scale
- WIoU running mean/state
- NaN/Inf 발생 여부
- resume 후 WIoU state 유지
- AP50보다 AP50:95가 더 좋아지는지

예상 문제와 개선:
- loss 폭주: dynamic weight 계산에 `.detach()` 적용 확인
- resume 후 metric 흔들림: WIoU state checkpoint 저장 확인
- mAP 하락: CIoU fallback으로 되돌리고 outlier label 품질 점검

### Stage 04. TAL + VFL 단독

목적: classification-localization alignment와 IoU-aware class score 효과를 확인한다.

활성 기능:
- `--assign tal`
- `--loss-cls vfl`
- 기존 head/loss box 유지

예상 변화:
- 증가: precision, cls score 품질, rare class recall, hard positive 학습
- 감소: cls/localization mismatch, overconfident false positive
- 비용: 학습 중 assignment 비용 증가 가능, 추론 비용 변화 없음

봐야 할 부분:
- positive count
- `train/cls_loss` scale
- class별 recall과 precision
- false positive 유형
- TAL top-k 분포

예상 문제와 개선:
- positive가 과소/과다: `topk`, `alpha`, `beta` 조정
- VFL 단독 적용 오류: TAL positive 없으면 실행 실패 처리
- CUDA/CPU mismatch: matching tensor를 prediction device에 유지

### Stage 05. Core 누적

목적: Decoupled Head + WIoU + TAL/VFL 조합이 실제로 누적 이득을 내는지 확인한다.

활성 기능:
- `--head decoupled`
- `--loss-box wiou_v3`
- `--assign tal`
- `--loss-cls vfl`

예상 변화:
- 증가: primary mAP, AP50:95, rare recall
- 감소: box/cls task conflict, FP 일부
- 비용: head 변경분만 GFLOPs 증가 가능

봐야 할 부분:
- Stage 02/03/04 대비 누적 delta
- loss scale 균형
- mAP 상승이 특정 클래스에만 몰리는지
- export 통과 여부

예상 문제와 개선:
- 단독은 좋은데 누적이 나쁨: Stage 03 또는 04를 끄고 조합 재실행
- cls loss가 box loss를 압도: loss gain 조정
- L 모델 속도 초과: L은 WIoU/TAL/VFL만 유지하고 head fallback 검토

### Stage 06. CCTV Pixel Augmentation

목적: 라벨을 바꾸지 않는 CCTV 도메인 증강 효과를 확인한다.

활성 기능:
- `--aug-profile cctv_pixel`
- SpiderWeb, ToGray, CLAHE, blur/noise 계열

예상 변화:
- 증가: IR/흑백/오염/압축 환경 robustness, scenario mAP
- 감소: 특정 깨끗한 이미지에 대한 과적합
- 비용: dataloader CPU 사용량과 학습 시간 증가 가능

봐야 할 부분:
- `aug_check.json`
- scenario별 mAP
- clean val mAP 하락 여부
- dataloader time
- visual sample 품질

예상 문제와 개선:
- clean mAP 하락: aug probability 낮춤
- dataloader 병목: cache, workers, prefetch 조정
- 비현실적인 이미지 생성: 해당 aug off 또는 확률 축소

### Stage 07. Patch-Paste / Hard Negative

목적: 부분 가림, 희귀 클래스, false positive 억제를 검증한다.

활성 기능:
- `--aug-profile cctv_paste`
- Patch-Paste
- Hard Negative mining/paste

예상 변화:
- 증가: occlusion recall, rare class recall, hard-case mAP
- 감소: FP/image, 배경 오탐
- 비용: 라벨 처리 복잡도와 데이터 생성 시간 증가

봐야 할 부분:
- bbox range check
- class id check
- paste 후 작은 box/잘린 box 비율
- FP/image 변화
- hard negative crop manifest

예상 문제와 개선:
- label pollution: 해당 paste 단계 중단, visual audit 재실행
- rare recall은 오르지만 precision 하락: paste 확률과 conf threshold 조정
- 특정 클래스 과샘플링: class별 paste cap 적용

### Stage 08. Weighted Sampler

목적: 클래스 불균형 대응 효과를 검증한다.

활성 기능:
- `--sampler-mode weighted`

예상 변화:
- 증가: rare class recall, long-tail class AP
- 감소: common class 편향, 특정 클래스 미학습
- 비용: epoch별 sample distribution 변화, DDP 적용 복잡도 증가

봐야 할 부분:
- `sampler_stats.csv`
- class별 image sampling 비율
- rare/common class AP trade-off
- overfit 징후

예상 문제와 개선:
- common class AP 급락: weight 상한 적용
- rare class overfit: repeat cap 적용
- DDP sampler 충돌: 단일 GPU에서 먼저 통과 후 distributed-aware sampler 사용

### Stage 09. W6 SCDown Only

목적: W6에서 효율 구조만 먼저 확인한다.

적용 대상:
- YOLOv7-W6만

활성 기능:
- `--neck-mod scdown`
- `--p2-head none`

예상 변화:
- 증가: 고해상도 feature 처리 효율, 일부 latency 개선 가능
- 감소: 불필요한 downsampling 비용
- 비용: 구조 변경에 따른 route/export 리스크

봐야 할 부분:
- GFLOPs delta
- memory usage
- ONNX export
- W6 mAP 유지 여부

예상 문제와 개선:
- route/channel mismatch: cfg diff를 SCDown only로 최소화
- mAP 하락: SCDown 위치 축소 또는 원본 neck fallback
- export 실패: Conv/BN/activation 표준 op만 사용

### Stage 10. W6 P2 Anchor Only

목적: 소형 객체 성능 개선 효과만 확인한다.

적용 대상:
- YOLOv7-W6만

활성 기능:
- `--p2-head anchor`
- `--neck-mod none`

예상 변화:
- 증가: small AP, small object recall, 원거리 객체 검출
- 감소: 소형 객체 미감지
- 비용: output box 수, memory, Python NMS ms, GFLOPs 증가

봐야 할 부분:
- stride 4 P2 output shape
- total boxes
- Python NMS ms
- small AP/recall
- duplicate detection 증가 여부

예상 문제와 개선:
- NMS 비용 급증: score threshold, pre-NMS top-k, per-level filtering 검토
- duplicate 증가: anchor size와 NMS IoU threshold 조정
- GFLOPs 10% 초과: P2 channel 축소 또는 W6 P2 fallback

### Stage 11. W6 P2 Anchor + SCDown

목적: W6 최종 구조 후보를 검증한다.

활성 기능:
- `--p2-head anchor`
- `--neck-mod scdown`

예상 변화:
- 증가: small AP/recall, W6 정확도형 최종 mAP
- 감소: SCDown이 P2 비용 일부를 상쇄할 가능성
- 비용: output 증가와 구조 변경 상호작용

봐야 할 부분:
- Stage 09/10 대비 누적 delta
- GFLOPs 10% 제한
- Python NMS ms
- rare/small class recall
- export diff

예상 문제와 개선:
- SCDown only와 P2 only는 좋지만 누적이 나쁨: 둘 중 하나만 최종 후보로 유지
- small AP만 오르고 전체 mAP 하락: P2 loss gain 또는 anchor 재조정
- output 증가 과다: P2 confidence filtering 추가

### Stage 12. Optional Gate

목적: 필수 stage 이후에도 목표가 부족할 때만 후순위 실험을 수행한다.

진입 조건:
- Stage 00~11 결과가 모두 정리됨
- 목표 mAP 또는 small recall이 부족함
- GFLOPs/latency 여유가 남아 있음
- `doc/REPORT/optional_decision_*.md` 작성 완료

실험 후보:
- L AUX on
- W6 PSA P5
- W6 FCOS P2 Python raw/decode
- W6 GELAN 일부 교체

예상 변화:
- 증가: 특정 병목 metric
- 감소: 남은 latency/GFLOPs 여유
- 비용: export, route, 후처리 복잡도 증가

예상 문제와 개선:
- 효과가 작음: optional 기본 off 유지
- export 실패: optional 실험 폐기
- FCOS postprocess 복잡: Python raw/decode까지만 유지하고 C++/runtime은 별도 차수로 분리

### Stage 13. Fine-tuning / Continual Learning

목적: scratch 기준선 이후 신규 데이터 파인튜닝에서 기존 클래스 망각을 줄인다.

활성 기능:
- Replay only
- Pseudo Label
- LwF cls distill
- LwF cls + reg distill

예상 변화:
- 증가: 기존 클래스 유지율, 신규 데이터 적응성
- 감소: catastrophic forgetting
- 비용: teacher forward 시간, replay storage, distillation tuning 복잡도

봐야 할 부분:
- 대상 클래스 mAP
- 기존 클래스 mAP
- forgetting delta
- pseudo label precision
- replay class coverage

예상 문제와 개선:
- 신규 클래스 학습 부족: distill alpha 낮춤
- 기존 클래스 하락: replay ratio와 distill alpha 높임
- pseudo label 오염: confidence threshold 상향, IoU 중복 제거 강화

## 6. 최종 리포트 구조

최종 리포트는 `doc/REPORT/final_training_report_v1.8_YYYY-MM-DD.md`로 저장한다.
상세 리포트 규격은 `doc/PLAN/training_report_format_v1.8.md`를 따른다.

리포트는 세 단계로 생성한다.

| 레벨 | 파일 | 용도 |
| --- | --- | --- |
| Stage 요약 | `runs/train_seq/v1.8/<stage>/stage_summary.md` | 해당 stage의 keep/drop/retry/blocker 즉시 판정 |
| Sequence 요약 | `runs/train_seq/v1.8/final_report/sequence_summary.md` | Stage 00부터 최종 stage까지 delta 흐름 정리 |
| 최종 리포트 | `doc/REPORT/final_training_report_v1.8_YYYY-MM-DD.md` | 유지할 기능, 제거할 기능, 재실험할 기능 확정 |

각 stage는 `baseline 대비`, `직전 성공 stage 대비`, `이전 최고 stage 대비` 세 기준으로 비교한다. 이 세 비교를 분리하지 않으면 누적 학습에서 어느 기능이 실제로 효과를 냈는지 판단하기 어렵다.

필수 섹션:
- 실행 환경: GPU, CUDA, PyTorch, commit, dataset checksum
- Stage별 config와 weight path
- Stage별 metric delta
- L/W6 별 최종 후보
- 증가한 항목: mAP, small AP, recall, robustness
- 감소한 항목: FP/image, FN/image, loss instability, export error
- 비용 증가 항목: GFLOPs, memory, train time, NMS ms
- 실패 stage와 원인 추정
- 최종 유지할 flag
- 끌 flag와 이유
- 재실험 필요 항목

비교 테이블 기본 컬럼:

```text
stage, model, weights, primary_mAP, mAP50, small_AP, rare_recall,
FP_per_image, FN_per_image, GFLOPs, GFLOPs_delta,
python_infer_ms, python_nms_ms, train_hours,
export_status, onnx_max_abs_diff, decision, reason
```

판정 값은 아래 값만 사용한다.

| Decision | 의미 |
| --- | --- |
| `keep` | 다음 stage에 누적 적용 |
| `keep_candidate` | 최종 full run 후보로 유지 |
| `drop` | 해당 flag 제거 후 직전 성공 stage에서 계속 진행 |
| `retry_tune` | 같은 stage에서 설정 조정 후 1회 재실행 |
| `blocker` | hard fail로 sequence 중단 |
| `defer` | optional 또는 후순위로 이동 |

최종 리포트에서 내가 판단할 항목은 `유지`, `제거`, `재실험`, `원인`, `다음 액션` 다섯 가지다. 단순히 mAP만 요약하지 않고, 기능별로 성능 이득과 비용 증가를 같이 판단한다.

## 7. 최종 의사결정 기준

최종 모델로 승격하려면 아래 조건을 만족해야 한다.

- primary mAP가 baseline보다 상승
- small AP 또는 rare recall이 목표 방향으로 개선
- GFLOPs 증가율 10% 미만
- Python/ONNX raw output 비교 통과
- `best.pt`, `results.csv`, `profile.json`, `export_check.json` 존재
- 특정 클래스가 치명적으로 하락하지 않음
- L은 속도형 역할을 유지
- W6는 소형 객체 개선 역할을 달성

탈락 기준:
- mAP 상승 없이 비용만 증가
- small AP 상승보다 FP 증가가 큼
- label pollution 의심
- export 불안정
- loss scale 불안정
- 누적 stage에서 원인 추적 불가

## 8. 예상되는 전체 개선 방향

예상 증가:
- `mAP@0.5:0.95`
- small AP
- rare class recall
- CCTV 악조건 robustness
- 파인튜닝 후 기존 클래스 유지율

예상 감소:
- 미감지
- 배경 false positive
- close-mosaic 미반영 문제
- resume/export 불안정
- class imbalance 영향

예상 증가 비용:
- 일부 head/neck GFLOPs
- W6 P2 output box 수
- Python NMS ms
- augmentation dataloader time
- fine-tuning teacher forward time

비용이 증가해도 허용 가능한 경우:
- W6에서 small AP/recall 개선이 명확하고 GFLOPs 증가율이 10% 미만
- FP/image가 감소하거나 rare class recall이 의미 있게 상승
- export 검증이 안정적으로 통과

## 9. 실행 결론

v1.8은 개발 문서가 아니라 학습 실행 문서다. 모든 기능이 개발된 뒤에도 한 번에 최종 조합만 학습하지 않는다. stage를 연속 실행해 어느 기능이 성능을 올리고, 어느 기능이 비용이나 문제를 만드는지 기록한다.

이 방식의 장점은 세 가지다.

1. 테스트 시간을 줄인다.
2. 문제 발생 stage를 바로 찾는다.
3. 최종 리포트에서 유지할 기능과 제거할 기능을 근거 있게 결정한다.

첫 실제 실행은 COCO128 quick run으로 `Stage 00 -> Stage 01 -> Stage 02`까지만 먼저 짧게 돌린다. orchestration과 리포트 생성이 정상인지 확인한 뒤 COCO128 전체 sequence를 실행하고, 그 다음 실제 대상 dataset full run으로 넘어간다.
