# 코드 리뷰 기능 회귀 이슈

검토 대상 패치는 핵심 학습 경로를 깨뜨릴 가능성이 큽니다. 표준 라벨 탐색, AUX GPU loss 할당, close-mosaic 동작, 기대되는 best 체크포인트 출력, resume 로딩, 최종 COCO 평가가 영향을 받습니다. 단순 스타일 문제가 아니라 기능 회귀입니다.

## [P1] 표준 images-to-labels 매핑 복원

- 위치: `utils/datasets.py:525`
- 일반적인 YOLO 레이아웃인 `.../images/...`와 대응되는 `.../labels/...`를 사용하는 데이터셋에서, 현재 변경은 이미지 경로를 그대로 둔 채 이미지 옆에서 라벨을 찾습니다.
- 데이터셋이 우연히 `JPEGImages`를 쓰는 경우가 아니라면 학습 시 모든 라벨이 누락된 것으로 보고됩니다.
- 새 예제 config도 여전히 `images`를 참조하므로, 일반/custom 데이터셋 경로가 즉시 깨집니다.

## [P1] mosaic 종료 시 persistent workers 비활성화

- 위치: `utils/datasets.py:92`
- `--close-mosaic`를 workers > 0과 함께 쓰면, 각 persistent worker가 `mosaic=True` 상태의 dataset 복사본을 계속 유지합니다.
- 학습 루프에서 나중에 `dataset.mosaic = False`를 해도 부모 프로세스만 바뀌고, close-mosaic 단계에서도 worker들은 계속 mosaic을 적용합니다.
- workers가 0이면 `persistent_workers=True`가 `num_workers > 0`을 요구하기 때문에 DataLoader 생성 자체도 실패합니다.

## [P1] AUX OTA matching tensor를 모델 device에 유지

- 위치: `utils/loss_aux.py:1390`
- GPU AUX 학습에서 `cost`와 `torch.topk`가 반환한 index는 CUDA tensor인데, 여기서 `matching_matrix`가 강제로 CPU로 이동됩니다.
- CPU tensor를 CUDA index로 인덱싱하고, 이후 CPU mask와 CUDA tensor를 섞으면서 target이 하나라도 있으면 device mismatch가 발생합니다.
- 이 때문에 일반적인 CUDA W6/AUX 학습에서 새 `train_aux.py` 경로를 사용할 수 없습니다.

## [P1] 기본 체크포인트 분기에서 `best.pt` 저장

- 위치: `train.py:688-689`
- 기본값인 `--model-saveoptimizer`가 꺼진 상태에서는 성능이 개선되어도 `best_###.pt`만 저장되고 `weights/best.pt`가 갱신되지 않습니다.
- 이후 명령과 최종 선택 로직은 여전히 `best.pt`를 찾기 때문에, 일반 학습 실행이 기대되는 best 체크포인트 없이 끝나고 `last.pt`로 fallback합니다.
- 같은 저장 분기가 `train_aux.py`에도 중복되어 있습니다.

## [P2] YAML-safe한 `save_dir` 경로 반환

- 위치: `utils/general.py:932`
- 재정의된 `increment_path()`가 이제 `Path`를 반환하므로, `opt.save_dir`가 `opt.yaml`에 Python 전용 `!!python/object/apply:pathlib.WindowsPath` 태그로 dump됩니다.
- resume 경로에서는 이 파일을 `yaml.SafeLoader`로 읽는데, 해당 태그를 생성할 수 없어서 이 변경 이후 생성된 run에 대해 `--resume`이 실패합니다.

## [P2] 새 `test` 반환값에 맞게 남은 호출부 업데이트

- 위치: `test.py:301`
- `test.test()`가 이제 네 개의 값을 반환하지만, `train.py`와 `train_aux.py`의 학습 후 COCO 평가 경로는 여전히 세 개의 값만 unpack합니다.
- `data/coco.yaml`과 `nc == 80`으로 끝나는 실행은 학습 완료 후 최종 speed/mAP 테스트 단계에서 crash합니다.

## [P2] `results.txt`를 `plot_results`가 파싱 가능하게 유지

- 위치: `train.py:585`
- results 파일에 `[val]` 토큰과 클래스별 텍스트가 epoch 로그에 삽입되었지만, `plot_results()`는 여전히 `np.loadtxt`로 고정된 숫자 컬럼을 읽습니다.
- 기본 plotting이 켜진 상태에서는 validation set이 하나뿐이어도 이 포맷 변경 때문에 최종 `results.png` 생성이 plotting error 경로로 빠집니다.

## [P2] 라벨 cache invalidation 재활성화

- 위치: `utils/datasets.py:585-587`
- 기존 `.cache` 파일이 있을 때 hash/version 체크가 주석 처리되어 있습니다.
- 이미지를 추가/삭제하거나 라벨을 수정해도 사용자가 cache를 직접 삭제하기 전까지 학습과 검증이 오래된 cached label을 계속 사용합니다.
- 데이터 변경 이후 잘못된 데이터셋 내용으로 조용히 학습할 수 있습니다.
