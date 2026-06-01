# Penn-Fudan Person External Evaluation Report

## 목적

CrowdHuman 학습 결과가 학습/검증 데이터에만 맞춰진 것인지 확인하기 위해 외부 person-only 데이터셋으로 현재 L 계열 모델을 평가했다. 평가는 person 클래스만 사용하며, head 클래스 라벨은 포함하지 않았다.

## 평가 데이터

- 데이터셋: Penn-Fudan Pedestrian
- 다운로드: `https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip`
- 로컬 경로: `datasets/penn_fudan_person/`
- 이미지 수: 170
- person bbox 수: 423
- 라벨 생성 방식: `PedMasks/*_mask.png`의 instance mask에서 bbox를 추출해 YOLO class `0` 라벨로 변환
- YAML: `data/penn_fudan_person.yaml`

`data/penn_fudan_person.yaml`은 현재 CrowdHuman 2-class 모델과 호환되도록 `nc: 2`, `names: ['person', 'head']`로 둔다. 실제 GT 라벨은 class `0/person`만 존재한다.

## 실행 조건

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU 4GB
- PyTorch: 1.12.0
- 이미지 크기: 640
- batch size: 1
- dataloader workers: 0
- conf threshold: 0.001
- IoU threshold: 0.65
- trace: off

Windows 로컬 4GB GPU에서는 `batch-size 16` 평가가 paging file 부족/메모리 압박으로 지연되어 `batch-size 1`로 측정했다. batch size는 metric 자체에는 영향을 주지 않는다.

## 실행 명령 예시

```bash
python test.py --data data/penn_fudan_person.yaml \
  --weights runs/newyolov7_0527/02_head_decoupled_l/weights/best.pt \
  --img-size 640 --batch-size 1 --device 0 --workers 0 \
  --project runs/eval_person --name penn_fudan_02_head_decoupled_l_bs1 \
  --exist-ok --no-trace --verbose
```

## 결과

| Stage | Images | Person labels | P | R | mAP@0.5 | mAP@0.5:0.95 | Speed ms/img |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00 baseline | 170 | 423 | 0.842 | 0.983 | 0.957 | 0.768 | 26.9 |
| 02 head_decoupled | 170 | 423 | 0.853 | 0.950 | 0.957 | 0.779 | 28.4 |
| 03 wiou_v3 | 170 | 423 | 0.872 | 0.920 | 0.954 | 0.770 | 26.0 |
| 08 weighted_sampler | 170 | 423 | 0.000 | 0.000 | 0.000 | 0.000 | 24.8 |

## 해석

1. `02_head_decoupled_l`이 외부 person-only 평가에서도 mAP@0.5:0.95 기준 가장 높다. baseline 대비 +0.011이다.
2. `03_wiou_v3_l`은 precision이 가장 높지만 recall과 mAP@0.5가 baseline/02보다 낮다. 보수적인 검출 성향으로 보인다.
3. `00_baseline_l`은 recall이 가장 높다. 누락을 줄이는 목적이면 baseline도 여전히 강하다.
4. `08_weighted_sampler_l`은 CrowdHuman 결과와 동일하게 실패 모델로 판단한다. 외부 데이터에서도 TP가 잡히지 않는다.

참고: TP가 0개인 경우에도 GT label 수가 0으로 오해되어 출력되지 않도록 `test.py`의 target count 계산을 보정했다.

## 결론

현재 후보 중 외부 person 평가 기준으로는 `02_head_decoupled_l`을 우선 모델로 보는 것이 타당하다. 다만 Penn-Fudan은 소형 보행자 데이터셋이라 crowd/occlusion 강도가 낮다. 최종 판단은 CrowdHuman validation, Penn-Fudan person-only, 추가 혼잡 person 평가셋을 분리해 보고 종합해야 한다.
