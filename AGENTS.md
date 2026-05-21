# Repository Guidelines

## 프로젝트 구조 및 모듈 구성

이 저장소는 YOLOv7 기반의 학습, 평가, 추론, export 프로젝트입니다.

- 루트 스크립트: `train.py`, `train_aux.py`, `test.py`, `detect.py`, `export.py`
- 모델 코드: `models/`에는 네트워크 정의와 공통 레이어가 있습니다.
- 학습 유틸리티: `utils/`에는 dataset, loss, metric, plotting, logging, torch helper가 있습니다.
- 설정 파일: `cfg/`에는 모델 YAML, `data/`에는 dataset 및 hyperparameter YAML이 있습니다.
- 배포 자료: `deploy/`와 `tools/`에는 TensorRT, ONNX, notebook, runtime 예제가 있습니다.
- 문서: `doc/`에는 설계 문서와 리뷰 기록이 있습니다.

## 문서 저장 규칙

Markdown 문서는 목적별 폴더에 분리해 저장합니다. 버그와 회귀 이슈는 `doc/BUG/`, 개발 문서와 작업 분해 문서는 `doc/PLAN/`, 검토 결과와 완료 리포트는 `doc/REPORT/`에 둡니다. 코드레벨 개발 요구서도 신규 작성 또는 수정 시 `doc/PLAN/`에 저장합니다. 파일명은 내용을 드러내는 `snake_case`와 날짜를 함께 사용합니다. 예: `doc/BUG/code_review_regressions_2026-05-21.md`, `doc/PLAN/phase_training_plan_2026-05-21.md`, `doc/PLAN/development_requirements_1.3.1_baseline_export.md`.

## 계획 문서 범위 기준

`doc/PLAN/`의 개발 문서는 학습, 평가, Python export, 검증 도구를 우선 대상으로 합니다. C++ 후처리, TensorRT runtime, 별도 추론 서버, 운영 배포 코드는 사용자가 명시적으로 요청하기 전까지 개발 범위에서 제외합니다. 추론/배포 관련 항목이 필요하면 `doc/PLAN/`에 별도 차수로 분리한 뒤 코드레벨 요구서도 `doc/PLAN/`에 작성합니다.

## 빌드, 테스트, 개발 명령

의존성 설치:

```bash
pip install -r requirements.txt
```

기본 학습 실행:

```bash
python train.py --cfg cfg/training/yolov7.yaml --data data/coco.yaml --hyp data/hyp.scratch.p5.yaml --epochs 300 --img 640
```

AUX head가 필요한 W6 계열 학습:

```bash
python train_aux.py --cfg cfg/training/yolov7-w6.yaml --data data/coco.yaml --hyp data/hyp.scratch.p6.yaml --img 1280
```

가중치 평가:

```bash
python test.py --data data/coco.yaml --weights runs/train/exp/weights/best.pt --img 640
```

가중치 export:

```bash
python export.py --weights runs/train/exp/weights/best.pt --img 640
```

## 코딩 스타일 및 이름 규칙

Python 3 기준으로 4칸 들여쓰기를 사용합니다. 기존 YOLOv7 스타일을 따르며, 공통 helper는 `utils/`, 모델 모듈은 `models/`, CLI flag는 각 스크립트의 argument parser 근처에 둡니다. 변수, 함수, YAML key, 출력 파일명은 명확한 `snake_case`를 사용합니다. YAML 이름은 `yolov7-w6-custom.yaml`, `hyp_phase1.yaml`처럼 용도가 드러나게 작성합니다.

## 테스트 지침

별도 unit test suite는 없습니다. 변경 범위에 맞는 가장 작은 검증을 수행합니다. Dataset 변경은 `test.py`, 학습 루프 변경은 짧은 `--epochs 1` smoke run, export 변경은 `export.py`로 확인합니다. ONNX 관련 변경은 완료 처리 전에 PyTorch 출력과 export 출력이 일치하는지 비교합니다. TensorRT runtime, C++ 후처리, 추론 서버 검증은 별도 요청이 있을 때만 다룹니다.

## 커밋 및 Pull Request 지침

최근 이력은 짧은 요약형 커밋 메시지를 사용합니다. 예: `데이터로더 캐시 정책 수정`, `Fix best checkpoint saving`. PR에는 목적, 변경된 파일 또는 모듈, 실행한 명령, 사용한 dataset/config, 측정한 mAP/GFLOPs/latency/export 결과를 포함합니다. 관련 issue나 설계 문서가 있으면 `doc/` 경로를 함께 연결합니다.

## Agent 전용 지침

명시 요청 없이 사용자 데이터, dataset, weight, experiment output을 덮어쓰지 않습니다. 변경 범위는 작게 유지하고, 가능한 기존 학습 동작을 보존합니다. 동작에 영향을 주는 변경은 `doc/`에 기록합니다.
