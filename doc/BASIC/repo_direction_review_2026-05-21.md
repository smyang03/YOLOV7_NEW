# YOLOV7_NEW 방향성 검토 및 개선 제안 (2026-05-21)

## 1) 문서 기반 현재 레포 방향성 요약
- 본 레포의 실질적 방향성은 `YOLOv7_Custom_Design_Spec_v1.2` 요구서에 집중되어 있으며, 목적은 **CCTV 특화 탐지 성능(mAP) 향상과 추론 속도 유지의 동시 달성**이다.
- 모델 전략은 이원화되어 있다.
  - **YOLOv7-L**: 경량/속도 유지형(보수적 변경, AUX/P2 기본 off)
  - **YOLOv7-W6**: 성능 공격형(AUX on, P2 Anchor, SCDown, WIoU+TAL+VFL)
- 핵심 운영 철학은 단일 학습 엔트리포인트(`train.py`) 기반의 **3-Phase 자동 전환 학습 파이프라인**이다.
- 기술적 우선순위가 명확하다.
  1. W6에서 P2 Anchor + SCDown + Decoupled Head + TAL/VFL/WIoU 조합 고도화
  2. L은 속도 예산을 지키는 최소 변경 전략 유지
  3. GELAN/PSA/FCOS P2 등은 후순위 실험 항목으로 관리
- 데이터/도메인 가정은 매우 현실적이다.
  - RGB + IR 혼재(분리 불가)
  - 클래스 불균형 존재
  - 하드 네거티브(거미줄/그림자/나뭇가지) 억제가 중요

## 2) 현재 레포 관점의 갭 분석
현재 저장소에는 루트 README가 거의 비어 있어, 요구서의 방대한 결정사항이 코드/운영 문서로 충분히 투영되었는지 즉시 판단하기 어렵다.

### 확인된 갭
1. **문서-코드 연결성 부족**
   - 설계 요구서(Word) 중심이며, 개발/실험 체크리스트가 repo-native markdown으로 구조화되어 있지 않다.
2. **실험 재현성 리스크**
   - 3-Phase, 모델별 on/off, aug 확률, loss/assign 조합이 복합적이라 설정 관리 체계가 없으면 재현이 어려움.
3. **평가 기준 운영화 부족 가능성**
   - mAP/latency/NMS time/export 성공률(8.6/10.x)을 같은 실험 단위로 묶어 비교하는 표준 리포트 포맷이 필요.
4. **후순위 실험의 가드레일 부재 가능성**
   - PSA/GELAN/FCOS P2가 “후순위”임에도 실험 분기 난립 시 기준선이 쉽게 흔들릴 수 있음.

## 3) 개선 방향성 (우선순위)

### P0 (즉시)
1. **운영 README 재작성**
   - 목표/모델전략(L vs W6)/기본 실행커맨드/Phase 정의/성공 기준(TRT latency 포함) 명시.
2. **실험 매트릭스 문서화**
   - `baseline`, `w6_attack`, `ablation_p2`, `ablation_aux`, `fallback_ciou_bce_simota` 등 실험군 고정.
3. **성능 게이트 정의**
   - PR/실험 승인 조건을 최소 4축으로 고정: `mAP`, `TRT FP16 latency`, `NMS latency`, `ONNX/TRT export pass`.

### P1 (단기)
1. **설정 파일 계층화**
   - 모델/phase/loss/assign/aug를 yaml 조합으로 분리해 one-command 재현 보장.
2. **자동 리포트 템플릿 도입**
   - 실험 종료 시 핵심 지표를 한 장의 Markdown/CSV로 누적.
3. **도메인 리스크 테스트셋 분리**
   - spiderweb/shadow/backlight/small object/hard negative 전용 eval split 운영.

### P2 (중기)
1. **후순위 기능 실험 프로토콜 고정**
   - PSA/GELAN/FCOS P2는 “단일 변수 변경” 원칙으로만 진입.
2. **배포 타깃별 프로파일링 자동화**
   - TensorRT 8.6 vs 10.x, profile별 throughput/latency 변동 자동 수집.
3. **L/W6 제품 포지셔닝 고정**
   - L=실시간 엣지형, W6=정확도 우선형으로 KPI·릴리즈 노트를 분리.

## 4) 검토 의견 (의사결정 관점)
- 요구서 방향성 자체는 매우 타당하다. 특히 **L(안정) / W6(공격) 분리 전략**은 현업에서 실패비용을 줄이는 방법이다.
- 다만 현재 가장 큰 리스크는 모델링 아이디어 자체보다, **실험 관리와 증거(지표/리포트) 축적 구조**다.
- 따라서 다음 액션의 핵심은 “새 기법 추가”보다 “기준선 잠금 + 재현 파이프라인 + 지표 게이트”다.

## 5) 바로 실행 가능한 액션 아이템 (제안)
1. README를 운영 문서로 전면 개편 (목표/모델/phase/성공기준)
2. `docs/experiments/`에 baseline 및 ablation 템플릿 추가
3. `results/` 표준 스키마 정의 (`metrics.json`, `trt_profile.csv`, `export_log.txt`)
4. 실험 이름 규칙 통일 (`{date}_{model}_{phaseplan}_{lossassign}_{augpreset}`)
5. 후순위 실험 진입 조건 명문화 (기준선 대비 개선폭·비용 기준)

---
본 문서는 레포 내 요구서의 방향성을 코드 운영 관점으로 재정리한 1차 리뷰다.
