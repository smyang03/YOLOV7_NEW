# Train Type Summary - baseline_export

- stages: 1

| Stage | Model | Decision | primary_mAP | mAP50 | GFLOPs | Risk | Reason |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 00 baseline | l | blocker |  |  |  | blocker | training command failed with exit code 1 |

## Delta Notes

- 00 l: primary_mAP_delta=, GFLOPs_delta_percent=

## Next Action

- COCO128 quick이면 산출물, crash, stage 전환, report 생성 여부만 판단한다.
- target full이면 baseline 대비 성능과 비용을 같이 판단한다.
