# 1.3.1 코드레벨 재검토

- 기준 요구서: `doc/dev/development_requirements_1.3.1_baseline_export.md`
- 점검일: 2026-05-21
- 범위: Python 학습, 평가, ONNX export, Python 검증 도구
- 제외: C++ 후처리, TensorRT runtime, TensorRT engine build, 추론 서버

## 결론

1.3.1 범위를 Python 기준선으로 줄인 것은 적합하다. 하지만 현재 코드는 아직 1.3.1 요구사항을 충족하지 않는다. 기존 YOLOv7 학습/평가/export 골격은 있으므로, 대규모 구조 변경 없이 P1 항목부터 순서대로 보정하면 된다.

## 요구사항별 판정

| 항목 | 요구사항 | 현재 코드 | 판정 |
| --- | --- | --- | --- |
| CLI alias | `--img`, `--batch` 지원 | `--img-size`, `--batch-size`만 있음 | 미충족 |
| L/W6 smoke 학습 | `train.py`, `train_aux.py` 실행 | 진입점 존재 | 부분 충족 |
| best checkpoint | `best.pt`, `last.pt`, `results.txt` 생성 보장 | 기본 분기에서 `best.pt` 누락 가능 | 미충족 |
| 평가 반환 계약 | 4개 반환값 일관 처리 | `test.py`는 4개 반환, 일부 호출부는 3개 unpack | 미충족 |
| ONNX export | `--opset 16`, `--nms-mode none` | opset 12 고정, `--nms-mode` 없음 | 미충족 |
| export 검증 | `tools/verify_export.py` | 파일 없음 | 미충족 |
| profile | `tools/profile_model.py` | 파일 없음 | 미충족 |
| 산출물 | `export_check.json`, `profile.json` | 생성 코드 없음 | 미충족 |
| baseline report | `doc/REPORT/baseline_1.3.1_*.md` | 템플릿/절차 없음 | 미충족 |

## P1 수정 항목

### 1. CLI alias 정렬

관련 코드:
- `train.py`: `--batch-size`, `--img-size`
- `train_aux.py`: `--batch-size`, `--img-size`
- `test.py`: `--batch-size`, `--img-size`
- `export.py`: `--batch-size`, `--img-size`

요구서 명령은 `--img`, `--batch`를 사용한다. 기존 옵션을 제거하지 말고 alias를 추가해야 한다.

권장:
- `parser.add_argument('--img', '--img-size', dest='img_size', ...)`
- `parser.add_argument('--batch', '--batch-size', dest='batch_size', ...)`

### 2. `best.pt` 저장 보장

관련 코드:
- `train.py`: best 갱신 시 기본 분기에서 `best_###.pt`만 저장 가능
- `train_aux.py`: 동일 패턴

요구서 통과 기준은 `best.pt`, `last.pt`, `results.txt` 존재다. 기본 학습 옵션에서 `best.pt`가 항상 갱신되어야 한다.

권장:
- optimizer 포함 저장 분기에서도 best 갱신 시 `torch.save(ckpt, best)` 추가
- strip 여부는 기존 `--model-saveoptimizer` 정책을 유지

### 3. `test.test()` 반환값 처리 수정

관련 코드:
- `test.py`는 `(results, maps, times, per_class_results)` 4개 반환
- `train.py`, `train_aux.py`의 학습 후 COCO 평가 경로는 3개 unpack 사용

권장:
- `results, _, _, _ = test.test(...)`로 수정

### 4. `export.py` 요구사항 반영

현재 상태:
- ONNX export는 `opset_version=12` 고정
- `--opset` 없음
- `--nms-mode` 없음

권장:
- `--opset` 기본값 16 추가
- `--nms-mode` 기본값 `none` 추가
- 1.3.1에서는 `none` 외 값 입력 시 명확히 unsupported 처리
- 기존 `--include-nms`, `--end2end` 경로는 건드리지 않되 기본 경로는 raw ONNX로 유지

### 5. `tools/profile_model.py` 신규 작성

필수 기능:
- weights/cfg/img/output 인자 처리
- params 계산
- thop 기반 GFLOPs 계산
- `profile.json` 저장

현재 `thop`은 설치되어 있으므로 구현 가능하다.

### 6. `tools/verify_export.py` 신규 작성

필수 기능:
- weights/onnx/img/output 인자 처리
- 동일 dummy input으로 PyTorch와 ONNX Runtime forward
- output flatten 후 `max_abs_diff`, `mean_abs_diff` 계산
- `export_check.json` 저장

현재 `onnxruntime`은 설치되어 있으므로 비교 도구 구현은 가능하다.

### 7. ONNX 의존성 처리

현재 환경:
- `onnx`: 미설치
- `onnxruntime`: 설치됨
- `thop`: 설치됨

`export.py`는 ONNX export 후 `import onnx`를 수행하므로 1.3.1 실행에는 `onnx` 설치가 필요하다.

권장:
- `requirements.txt`의 `onnx>=1.9.0` 주석 해제 또는 별도 설치 안내 추가
- missing dependency 메시지를 명확히 출력

## 구현 우선순위

1. `train.py`, `train_aux.py`, `test.py`, `export.py` CLI alias 추가
2. `train.py`, `train_aux.py`의 `best.pt` 저장 보장
3. `train.py`, `train_aux.py`의 `test.test()` unpack 수정
4. `export.py`의 `--opset`, `--nms-mode none` 추가
5. `tools/profile_model.py` 작성
6. `tools/verify_export.py` 작성
7. `onnx` 의존성 처리
8. L/W6 smoke 실행 후 `baseline_1.3.1_*.md` 작성

## 최종 판정

현재 코드는 1.3.1 요구사항 기준으로 "부분 충족"이다. 학습/평가/export의 기존 뼈대는 사용할 수 있지만, 요구서 명령 그대로 실행하면 실패하거나 산출물이 부족하다. 위 P1을 먼저 처리해야 1.3.1 기준선을 학습 서버에서 재현 가능하게 만들 수 있다.
