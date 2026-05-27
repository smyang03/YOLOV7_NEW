# Phase2 Rect DDP Stack Error

- 작성일: 2026-05-25
- 증상: `phase2` 진입 직후 DataLoader worker에서 `torch.stack()` shape mismatch 발생
- 오류 예: `stack expects each tensor to be equal size, but got [3, 640, 448] ... and [3, 384, 640]`

## 원인

`phase_train=on` 상태에서 epoch 291에 `phase2`로 전환되며 `phase2_rect=True`가 적용됐다. rectangular training은 dataset을 aspect ratio 기준으로 정렬하고, 같은 batch index에 같은 `batch_shapes`를 배정한다.

문제는 DDP에서 기본 `DistributedSampler`가 샘플 단위로 index를 분산/셔플하면서 한 rank의 같은 batch 안에 서로 다른 rectangular shape 이미지가 섞인 것이다. 그 결과 `LoadImagesAndLabels.collate_fn()`의 `torch.stack(img, 0)`에서 shape가 달라 실패했다.

## 영향

- 모델, loss, optimizer 문제가 아니다.
- phase1은 square/mosaic 경로라 정상 진행됐다.
- phase2에서 `rect=True`가 켜지는 순간 batch 내부 shape 일관성이 깨져 중단됐다.
- 8 GPU DDP에서 재현 가능성이 높다.

## 수정

`utils/datasets.py`에 `RectDistributedBatchSampler`를 추가했다.

- `rect=True` + DDP일 때 샘플 단위가 아니라 rectangular batch 단위로 rank에 분산한다.
- batch 내부 index는 dataset의 aspect-ratio 정렬 순서를 유지한다.
- 마지막 batch는 같은 batch 내부 index로 padding해 rank별 iteration 수를 맞춘다.
- `train.py`, `train_aux.py`에서 `dataloader.batch_sampler.set_epoch(epoch)`도 호출하도록 보강했다.

## 검증

실행한 검증:

```bash
python -m py_compile utils/datasets.py train.py train_aux.py
```

COCO128 로컬 검증:

- `rect=True`
- `augment=True`
- `allow_rect_mosaic=True`
- `world_size=8`
- `batch_size=4`

각 rank의 첫 3개 batch를 직접 collate했고, 모든 batch가 `torch.stack()`을 통과했다.

## 재검토 결과

추가 재검토에서 다음을 확인했다.

- `dataset.rect=True`와 `dataset.mosaic=True`가 동시에 유지된다.
- `RectDistributedBatchSampler`가 같은 rectangular batch id에 속한 index만 한 batch로 반환한다.
- epoch `0`, `291`, `292`에서 `world_size=8`, `batch_size=4` 기준 모든 rank batch가 `torch.stack()`을 통과한다.
- `train.py`, `train_aux.py` 모두 DDP process group 초기화 이후 dataloader를 생성하므로 서버 학습 경로에서는 `torch_distributed_zero_first()` barrier 조건이 충족된다.
- sampler 선택은 호출 인자 `rect`가 아니라 실제 생성된 `dataset.rect` 기준으로 보강했다. `image_weights`처럼 dataset 내부에서 rect가 비활성화되는 경우에는 rect 전용 batch sampler를 사용하지 않는다.

## 서버 재개 권장

기존 run은 stage 01의 epoch 291에서 중단됐으므로 수정 반영 후 같은 sequence output에 `--resume-sequence`로 재개한다. 재개가 불안정하면 stage 01부터 다시 시작한다.
