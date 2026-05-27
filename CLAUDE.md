# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 브랜치 정책

- `main`과 `anchor-free` 브랜치는 **사용자가 명시적으로 요청하기 전까지 절대 머지하지 않는다.**
- 브랜치 간 머지, rebase, cherry-pick은 사용자의 명시적 지시 없이 수행하지 않는다.

## 개발 원칙

기본 실행 경로는 기존 YOLOv7 동작을 유지한다. 새 기능은 플래그/helper/wrapper/신규 모듈로 추가하고, 플래그를 켜지 않으면 기존 학습·평가·export 동작과 CLI 호환이 유지되어야 한다. 기존 함수와 클래스는 버그 수정, 호환성 보강, 공통 helper 연결에 한해서만 직접 수정한다.

## 주요 명령어

```bash
# 의존성 설치
pip install -r requirements.txt

# 기본 학습 (YOLOv7-L)
python train.py --cfg cfg/training/yolov7.yaml --data data/coco.yaml --hyp data/hyp.scratch.p5.yaml --epochs 300 --img 640

# W6 계열 학습 (AUX head 포함)
python train_aux.py --cfg cfg/training/yolov7-w6.yaml --data data/coco.yaml --hyp data/hyp.scratch.p6.yaml --img 1280

# 평가
python test.py --data data/coco.yaml --weights runs/train/exp/weights/best.pt --img 640

# Export (ONNX 등)
python export.py --weights runs/train/exp/weights/best.pt --img 640

# 연속 학습 시퀀스 실행
python tools/run_training_sequence.py --data data/coco.yaml --sequence-dir runs/my_seq

# 연속 학습 시퀀스 (dry-run)
python tools/run_training_sequence.py --data data/coco.yaml --dry-run

# Smoke run (빠른 검증)
python train.py --cfg cfg/training/yolov7.yaml --data data/coco128.yaml --epochs 1 --img 640
```

## 아키텍처 개요

### 진입점 스크립트
- `train.py` — 메인 학습 진입점. Phase 학습, Decoupled Head, WIoU, TAL, VFL, augmentation profile 등 v1.3 기능 플래그를 모두 포함.
- `train_aux.py` — W6 계열용 AUX head 학습. 점진적으로 `train.py`의 공통 helper로 수렴 중.
- `test.py` / `detect.py` / `export.py` — 평가·추론·export.
- `finetune.py` — Replay/Pseudo-label 기반 continual fine-tuning 진입점.

### 핵심 커스텀 모듈 (utils/)

| 모듈 | 역할 |
|------|------|
| `phase.py` | Phase 기반 학습 스케줄 (`PhaseConfig`, `resolve_phase`) |
| `loss_components.py` | WIoU, TAL, VFL 등 loss 옵션 조합 |
| `wiou.py` | Wise IoU loss 구현 |
| `tal.py` | Task-Aligned Learning (TAL) assigner |
| `cctv_augmentations.py` | CCTV 도메인 특화 augmentation |
| `augment_policy.py` | Augmentation profile 선택 및 검증 |
| `sampler.py` | Hard negative weighted sampler |
| `train_common.py` | DataLoader 생성, phase 전환 공통 helper |
| `train_logger.py` | 단계별 로깅 (`TrainLogger`) |
| `debug_logging.py` | 구조화된 debug 로거 |
| `model_options.py` | `--neck-mod`, `--p2-head` 등 구조 옵션 검증 |
| `early_stopping.py` | Phase별 early stopping |
| `replay_buffer.py` / `pseudo_label.py` / `continual_loss.py` | Continual learning 지원 |
| `fcos.py` | FCOS head 실험용 (1.3.6 optional) |
| `stage_schema.py` | `StageConfig`, `StageResult`, `Decision` 데이터 구조 |

### 학습 시퀀스 자동화 (tools/)

`tools/run_training_sequence.py`가 `STAGES` 리스트에 정의된 14개 스테이지(00~13)를 순서대로 실행한다. 각 스테이지는 `stage_config.yaml` → subprocess 학습 → `stage_result.yaml` 수집 → delta CSV 비교 → 최종 리포트 생성 흐름을 따른다.

주요 tools:
- `collect_stage_results.py` — 스테이지 결과 수집
- `compare_stage_metrics.py` — 스테이지 간 mAP/GFLOPs delta 비교
- `generate_training_report.py` — 최종 훈련 리포트 자동 생성
- `profile_model.py` / `verify_export.py` — GFLOPs 측정 및 ONNX 검증
- `check_loss_smoke.py` / `check_phase_schedule.py` — 사전 검증 도구

### 모델 설정 (cfg/)

- `cfg/training/` — 학습용 YAML (yolov7-w6.yaml, yolov7-w6-p2.yaml, yolov7-w6-scdown.yaml 등)
- `cfg/deploy/` — 추론용 경량 YAML
- `cfg/experiments/` — PSA, FCOS, GELAN 실험용 YAML (1.3.6 optional)

### v1.3 개발 차수 구조

개발 문서는 모두 `doc/PLAN/`에 위치한다. 각 차수(`1.3.x`)는 독립 플래그로 활성화되며 동시에 두 개 이상 켜서 검증하지 않는다.

| 차수 | 주요 플래그 | 내용 |
|------|------------|------|
| 1.3.1 | (baseline) | Dataset 안정화, checkpoint, ONNX export |
| 1.3.2 | `--phase-train on` | Phase 학습 루프, logging |
| 1.3.3 | `--head decoupled`, `--loss-box wiou_v3`, `--assign tal` | Decoupled Head, WIoU, TAL, VFL |
| 1.3.4 | `--aug-profile cctv_pixel/cctv_paste`, `--sampler-mode weighted` | CCTV augmentation, hard negative |
| 1.3.5 | `--neck-mod scdown`, `--p2-head anchor` | W6 P2 Anchor, SCDown |
| 1.3.6 | (defer) | PSA, FCOS, GELAN 실험 |
| 1.3.7 | (defer) | Replay, pseudo label, LwF fine-tuning |
| 1.3.8 | (자동화) | Training sequence, COCO128 quick run, report |

## 문서 저장 규칙

- `doc/BUG/` — 버그·회귀 이슈
- `doc/PLAN/` — 개발 계획, 코드레벨 요구서
- `doc/REPORT/` — 검토 결과, 완료 리포트
- `doc/BASIC/` — 설계 명세서 (기준 문서)

파일명 형식: `snake_case` + 날짜. 예: `development_requirements_1.3.2_train_loop_phase_logging.md`

## 테스트 방법

별도 unit test suite 없음. 변경 범위에 맞는 최소 검증을 수행한다.

- Dataset/DataLoader 변경: `python test.py`
- 학습 루프 변경: `--epochs 1` smoke run
- Export 변경: `export.py` 후 PyTorch/ONNX Runtime 출력 비교 (`tools/verify_export.py`)
- Phase 스케줄 변경: `python tools/check_phase_schedule.py`
- Loss 변경: `python tools/check_loss_smoke.py`

TensorRT runtime, C++ 후처리, 추론 서버 검증은 명시적 요청이 있을 때만 다룬다.
