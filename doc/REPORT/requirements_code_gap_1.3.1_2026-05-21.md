# 1.3.1 요구서-코드 비교 점검

- 대상 요구서: `doc/dev/development_requirements_1.3.1_baseline_export.md`
- 대상 코드: `train.py`, `train_aux.py`, `test.py`, `export.py`, `tools/`
- 점검일: 2026-05-21
- 제외 범위: C++ 후처리, TensorRT runtime, TensorRT engine build, 별도 추론 서버

## 결론

1.3.1은 Python 학습/평가/ONNX export 기준선으로 범위를 줄이면 구현 가능성이 높다. 현재 코드는 baseline 학습/평가의 골격은 있으나, CLI 정합성, `best.pt` 저장, ONNX export 옵션, Python 검증 도구가 아직 요구서를 충족하지 않는다.

## 요약 매트릭스

| 요구사항 | 현재 코드 상태 | 판정 |
| --- | --- | --- |
| L/W6 baseline 학습 실행 | `train.py`, `train_aux.py` 존재 | 부분 충족 |
| 요구서 CLI (`--img`, `--batch`) | 실제 코드는 `--img-size`, `--batch-size` | 미충족 |
| `best.pt`, `last.pt`, `results.txt` 생성 | `last.pt`/`results.txt`는 가능, `best.pt` 기본 분기 누락 가능 | 미충족 |
| `test.py` 반환 계약 | `test.py`는 4개 반환, 일부 호출부는 3개 unpack | 미충족 |
| ONNX opset 16 | `export.py`는 `opset_version=12` 고정 | 미충족 |
| `--nms-mode none` | `export.py`에 옵션 없음 | 미충족 |
| `tools/verify_export.py` | 파일 없음 | 미충족 |
| `tools/profile_model.py` | 파일 없음 | 미충족 |
| `export_check.json`, `profile.json` | 생성 코드 없음 | 미충족 |
| baseline report | 템플릿/생성 절차 없음 | 미충족 |

## 주요 갭

### [P1] 요구서 CLI가 실제 코드 옵션과 맞지 않음

- 요구서: `--img`, `--batch`
- 실제 코드: `train.py`, `train_aux.py`, `export.py`는 `--img-size`, `--batch-size` 사용

조치:
- `--img`를 `--img-size` alias로 추가한다.
- `--batch`를 `--batch-size` alias로 추가한다.
- 기존 옵션명은 유지해 backward compatibility를 보존한다.

### [P1] 기본 학습 분기에서 `best.pt` 생성이 보장되지 않음

- 실제 코드: `train.py`, `train_aux.py`
- `--model-saveoptimizer`가 꺼진 기본 분기에서 best 갱신 시 `best_###.pt`만 저장될 수 있다.

조치:
- 기본 분기에서도 best 갱신 시 `weights/best.pt`를 항상 저장한다.
- 1.3.1 통과 기준은 `best.pt`, `last.pt`, `results.txt` 존재 여부다.

### [P1] `test.py` 반환값과 일부 호출부가 맞지 않음

- `test.test()`는 `(results, maps, times, per_class_results)` 4개 값을 반환한다.
- `train.py`, `train_aux.py`의 일부 학습 후 평가 호출은 3개 unpack을 사용한다.

조치:
- 호출부를 4개 unpack으로 수정한다.
- standalone `test.py` 실행은 기존처럼 유지한다.

### [P1] Export 요구사항이 `export.py`에 반영되어 있지 않음

- `--opset` 옵션 없음
- ONNX export가 `opset_version=12`로 고정
- `--nms-mode none` 옵션 없음

조치:
- `--opset` 기본값 16 추가
- `--nms-mode none` 추가
- 1.3.1에서는 `none`만 지원하고, 다른 값은 명확한 error 또는 unsupported message를 출력한다.

### [P1] Python 검증 도구가 없음

필요 파일:
- `tools/profile_model.py`
- `tools/verify_export.py`

조치:
- `profile_model.py`는 params/GFLOPs를 `profile.json`으로 저장한다.
- `verify_export.py`는 PyTorch와 ONNX Runtime output을 비교해 `export_check.json`으로 저장한다.

### [P2] 요구서의 dataset 예시는 현재 repo 기준으로 조정 필요

- `data/custom.yaml`은 현재 repo에 없다.
- 현재 예시로는 `data/custom_example.yaml`, `data/data.yaml`, `data/coco.yaml`이 있다.

조치:
- smoke 명령은 `data/custom_example.yaml`을 사용한다.
- 실제 학습 서버에서는 운영 dataset yaml 경로로 교체한다.

### [P2] 의존성 상태 확인

현재 확인:
- `onnx`: 미설치
- `onnxruntime`: 설치됨
- `thop`: 설치됨

조치:
- ONNX export를 위해 `onnx` 설치가 필요하다.
- `verify_export.py`는 `onnxruntime`이 없으면 비교를 skip하고 명확한 메시지를 남긴다.

## 1.3.1 구현 우선순위

1. CLI alias 정렬: `--img`, `--batch`
2. `best.pt` 저장 보장: `train.py`, `train_aux.py`
3. `test.test()` 반환값 unpack 수정
4. `export.py`에 `--opset`, `--nms-mode none` 추가
5. `tools/profile_model.py` 작성
6. `tools/verify_export.py` 작성
7. L/W6 smoke 학습/평가/export 실행
8. baseline report 작성

## 제외 항목

다음은 1.3.1 요구사항으로 보지 않는다.

- `build_trt.py`
- `deploy/cpp/postprocess.h`
- `deploy/cpp/postprocess.cpp`
- TensorRT engine build
- TensorRT output 비교
- C++ NMS
- 추론 서버 또는 운영 배포 코드

## 판정

1.3.1은 Python 기준선 구축 차수로 재정의되었다. 현재 코드는 아직 미충족 항목이 있지만, 위 P1 항목을 구현하면 C++/TensorRT 없이도 baseline 학습, 평가, ONNX export, PyTorch/ONNX 비교 기준선을 만들 수 있다.
