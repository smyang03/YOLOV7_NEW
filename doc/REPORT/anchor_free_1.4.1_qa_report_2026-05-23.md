# 1.4.1 Anchor-Free QA Report

- 작성일: 2026-05-23
- 브랜치: `anchor-free`
- 기준 문서: `doc/PLAN/development_requirements_1.4.1_anchor_free.md`
- 실행 검증: `runs/tmp_141_sequence/coco128_stage00_08_e3_retry`

## 검증 목적

1.4.1 앵커프리 개발 요구서를 기준으로 코드 구현 여부를 점검하고, 기존 1.3.1~1.3.8 단계가 실제 COCO128에서 3 epoch씩 연속 학습되는지 확인했다. 이번 실행은 학습 루프와 스테이지 안정성 검증 목적이며, 성능 판단용 full-size 학습은 아니다.

## 구현 반영 요약

| 영역 | 반영 내용 | 상태 |
| --- | --- | --- |
| 모델 헤드 | `FCOSDetect`, `HybridDetect` 추가, `Model(..., det_head, anchor_free_levels)` 지원 | 완료 |
| FCOS 유틸 | target assign, centerness, decode helper 추가 | 완료 |
| 손실 함수 | `ComputeLossFCOS`, `ComputeLossHybrid` 추가 | 완료 |
| 학습 CLI | `--det-head`, `--anchor-free-levels`, `--lambda-free`, FCOS 옵션 추가 | 완료 |
| 검증/평가 | validation loss가 anchor/FCOS/hybrid 출력을 처리하도록 수정 | 완료 |
| cfg | L FCOS, W6 FCOS, W6 P2 hybrid cfg 추가 | 완료 |
| runner | stage 14~16 앵커프리 후보 stage 추가 | 완료 |
| export/profile | 앵커프리 메타데이터와 dict output 처리 추가 | 부분 완료 |

## 단계별 QA 결과

| 검증 | 명령/대상 | 결과 |
| --- | --- | --- |
| 문법 검사 | `py_compile` 주요 학습/모델/도구 파일 | 통과 |
| FCOS decode smoke | `tools/decode_fcos_outputs.py --allow-synthetic` | 통과 |
| 모델 build smoke | L FCOS, W6 FCOS, W6 P2 hybrid | 통과 |
| profile smoke | `tools/profile_model.py` L FCOS CPU | 통과 |
| FCOS loss | synthetic forward/backward | 통과 |
| Hybrid loss | synthetic forward/backward | 통과 |
| 실제 학습 smoke | COCO128 L FCOS 1 epoch | 통과 |
| stage 00~08 | COCO128 3 epoch 연속 실행 | 통과 |

## COCO128 Stage 00~08 실행 결과

실행 조건:

```bash
python tools/run_training_sequence.py \
  --plan doc/PLAN/training_execution_plan_v1.8.md \
  --data data/coco128.yaml \
  --dataset-profile coco128_quick \
  --model-family l_only \
  --output runs/tmp_141_sequence/coco128_stage00_08_e3_retry \
  --start-stage 00 \
  --end-stage 08 \
  --stop-on-hard-fail \
  --weights= \
  --cfg cfg/training/yolov7.yaml \
  --hyp data/hyp.scratch.p5.yaml \
  --epochs 3 \
  --batch-size 8 \
  --img 128 128 \
  --workers 0 \
  --device cpu \
  --skip-profile \
  --debug-log error \
  --console-log stderr \
  --progress-log-interval 20
```

| Stage | 이름 | exit_code | decision | hard_fail | soft_fail | mAP50 |
| --- | --- | ---: | --- | --- | --- | ---: |
| 00 | baseline_l | 0 | keep | false | false | 0.0 |
| 01 | phase_logging_l | 0 | keep | false | false | 0.0 |
| 02 | head_decoupled_l | 0 | keep | false | false | 0.0 |
| 03 | wiou_v3_l | 0 | keep | false | false | 0.0000315021 |
| 04 | tal_vfl_l | 0 | keep | false | false | 0.0 |
| 05 | core_cumulative_l | 0 | keep | false | false | 0.0016524754 |
| 06 | cctv_pixel_aug_l | 0 | keep | false | false | 0.0 |
| 07 | patch_paste_hard_negative_l | 0 | keep | false | false | 0.0 |
| 08 | weighted_sampler_l | 0 | keep | false | false | 0.0005400691 |

9개 stage result가 생성됐고, stage 00~08 모두 학습 명령이 정상 종료됐다.

## 실행 중 발견한 문제와 수정

| 문제 | 원인 | 수정 |
| --- | --- | --- |
| stage 00 validation 실패 | `test.py`에서 OTA validation loss 호출 시 `imgs`를 넘기지 않음 | dict/list output 모두 `compute_loss(..., targets, img)` 우선 호출 후 fallback 처리 |
| 순수 FCOS 학습 실패 | FCOS에도 autoanchor check가 실행됨 | `det_head=fcos`일 때 `check_anchors()` skip |
| 순수 FCOS 손실 초기화 실패 | FCOS head에 anchor OTA loss를 초기화함 | `det_head=fcos`는 `ComputeLossFCOS`만 사용 |
| W6 hybrid synthetic 실패 | W6 training cfg의 `IAuxDetect`와 train.py loss 경로가 충돌 | W6 FCOS/hybrid cfg는 deploy base Detect 경로로 고정 |

## 재검토 추가 수정

| 문제 | 영향 | 수정 |
| --- | --- | --- |
| `--anchor-free-levels p2`인데 `HybridDetect` FCOS branch가 모든 level을 사용 | W6 P2 hybrid 실험이 실제로는 P2 전용 보조 branch가 아니게 됨 | `FCOSDetect`/`HybridDetect`에 level 선택을 연결하고, hybrid P2는 FCOS branch가 stride 4 P2 1개 level만 사용하도록 수정 |
| `--fcos-score-mode`가 실제 head inference에 반영되지 않음 | 문서 기본값 `sqrt_cls_centerness`와 실제 NMS 입력 점수 계산이 달라질 수 있음 | `FCOSDetect` decode에서 `sqrt_cls_centerness`/`mul_cls_centerness`를 반영하고 `Model(..., fcos_score_mode=...)`를 train/train_aux에서 전달 |

재검토 후 확인값:

- L FCOS: `fcos_nl=3`, stride `[8, 16, 32]`
- W6 full FCOS: `fcos_nl=5`, stride `[4, 8, 16, 32, 64]`
- W6 P2 hybrid: `anchor_nl=5`, `fcos_nl=1`, FCOS stride `[4]`
- `sqrt_cls_centerness`와 `mul_cls_centerness` inference score mode 모두 forward 통과

## 요구서 대비 차이점

| 요구 항목 | 현재 상태 | 판단 |
| --- | --- | --- |
| 기존 anchor 경로 보존 | `--det-head anchor` 기본값 유지 | 충족 |
| FCOS trainable head | L/W6 cfg와 `FCOSDetect` 구현 | 충족 |
| Hybrid P2 경로 | `HybridDetect` 구현, W6 P2 cfg 추가 | 충족 |
| FCOS target/loss | point assign, ltrb, centerness, cls loss 구현 | 충족 |
| train/test 연동 | 학습 loss, validation loss, output 처리 수정 | 충족 |
| stage runner | 14~16 후보 stage 추가 | 충족 |
| stage 00~08 실제 학습 | COCO128 3 epoch 전체 통과 | 충족 |
| W6 P2 hybrid의 P2 전용 FCOS branch | 재검토 후 `fcos_nl=1`, stride 4로 제한 | 충족 |
| `fcos_score_mode` inference 반영 | 재검토 후 head decode에 연결 | 충족 |
| report schema에 FCOS 상세 지표 저장 | loss 객체에는 통계가 있으나 `stage_result.yaml` 영구 저장은 미완 | 추가 필요 |
| raw ONNX output name/diff 상세 검증 | export/profile 기초 처리는 있으나 요구서의 전체 raw diff schema는 미완 | 추가 필요 |
| test.py 별도 FCOS normalize helper | 현재는 `FCOSDetect`가 YOLO NMS 호환 tensor를 반환하는 방식 | 설계 차이 |
| stage 14~16 실제 COCO128 연속 실행 | 이번 요청 범위는 00~08이므로 미실행 | 별도 실행 필요 |

## 남은 작업

1. `stage_result.yaml`에 `fcos_positive_count`, `fcos_loss_box`, `fcos_loss_cls`, `fcos_loss_ctr`, `python_decode_ms`를 저장하도록 runner/report schema를 확장한다.
2. `tools/verify_export.py`와 `export.py`에서 FCOS raw output name, shape, PyTorch/ONNX diff를 요구서 schema대로 기록한다.
3. GPU 환경에서 stage 14~16을 3 epoch로 별도 실행해 L FCOS, W6 P2 hybrid, W6 full FCOS의 실제 학습 안정성을 확인한다.
4. 이후 CrowdHuman full 학습에서는 3 epoch smoke 결과가 아니라 100~300 epoch 누적 결과를 기준으로 baseline 대비 mAP, head AP, recall, FP/FN, GFLOPs, latency를 비교한다.

## 결론

1.4.1의 핵심 코드 경로는 구현됐고, 기존 stage 00~08은 COCO128 3 epoch 연속 실행에서 모두 통과했다. 다만 요구서의 리포트/ONNX 상세 schema는 아직 완전하지 않으므로, 앵커프리 stage 14~16을 성능 평가에 투입하기 전 해당 두 항목을 보강하는 것이 좋다.
