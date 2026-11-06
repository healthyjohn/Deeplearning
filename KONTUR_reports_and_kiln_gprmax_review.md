# KONTUR 학습자료 상세 + gprMax → Kiln AI 적용 검토

> 정리일: 2026-11-06
> 참고: 이 작업 환경에서는 아래 PDF 원문 다운로드가 네트워크 정책으로 막혀 있었습니다. 내용은 검색 엔진이 색인한 원문 요약/발췌로 정리했으니, 수치는 원문에서 한 번 더 확인하세요.

---

## 1. 국내 「3D-GPR 탐사장비의 시설물 적용방안 연구 및 활용 매뉴얼 개발」

| 항목 | 내용 |
|---|---|
| 보고서 번호 | **RD-19-R3-003** (2019년 12월) |
| 발행 | **한국시설안전공단** (현 국토안전관리원의 전신) |
| 원문 PDF | https://www.codil.or.kr/filebank/original/RK/OTKCRK211043/OTKCRK211043.pdf |
| 소장처 | CODIL(건설기술정보시스템) — 회원가입 없이 PDF 열람 가능 |

**목적 (보고서 요약 기준)**
1. 교량·터널 등 **시설물 종류별 3D-GPR 현장 운용지침** 작성
2. **활용 매뉴얼** 제작 (장비 세팅 → 측정 → 자료처리 → 해석)
3. 3D-GPR 탐사 **데이터 분석 기법** 개발
4. **시범 적용**으로 현장 적정성 검증

**읽는 법 (추천 순서)**
1. 서론·장비 개요 → 3D-GPR이 기존 단채널 GPR과 무엇이 다른지
2. **활용 매뉴얼 부분** → 측정 계획, 측선 간격, 속도, 캘리브레이션, 자료처리 순서(필터, 게인, 마이그레이션 등)를 노트로 정리
3. 시범 적용 사례 → 실제 이미지(B-scan/C-scan)에서 공동·철근·관로가 어떻게 보이는지 캡처해 두기 → 나중에 딥러닝 라벨링 기준으로 사용

**함께 보면 좋은 국내 자료**
- 「3D GPR 자료를 활용한 인공지능 기반 배관 탐지 시스템 개발 및 검증」 (KCI) — 3D GPR + AI 배관 탐지: https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003040919
- 「지하시설물 측량에 있어서 GPR 탐사방법의 정확도」: https://koreascience.or.kr/article/JAKO202102763721551.pdf

---

## 2-1. Kontur 제조사 발표자료 (TAM 25 Workshop, 2025년 8월)

| 항목 | 내용 |
|---|---|
| 제목 | **"GPR Technology and Process Improvements"** |
| 발표 | D. Barry, Kontur Americas (영업 담당 VP) |
| 행사 | TAM 25 Workshop (Iowa State Univ. InTrans 주최, 교통 자산관리) |
| PDF | https://www.intrans.iastate.edu/wp-content/uploads/2025/09/TAM-25-Workshop-KONTUR-presentation-D-Barry.pdf |

**주요 내용 (검색 요약 기준)**
- **3D GPR 최신 개발 현황** — 차량탑재 3D GPR로 도로 전 폭을 한 번에 스캔
- **유전율 매핑(Dielectric mapping)** — 신규 아스팔트 포장의 **품질관리(QC)**, 다짐 밀도 균일성 확인. MnRoad(미네소타 도로 시험장) 연구와 연계
- **Examiner Collect** — 포장·교량·매설물 프로젝트의 측정 계획과 데이터 수집
- **Orbit** — **머신러닝 기반 자동 처리** 소프트웨어 (자동 해석 방향)
- 연구 Pooled Fund 과제, MnRoad 연구로 3D GPR 검증

**학습 포인트**
- 업체 발표자료라 장점 위주입니다. "어떤 문제를 3D GPR로 푸는지" 전체 그림을 잡는 용도로 보세요.
- Orbit(ML)이 무엇을 자동화하는지 보면, 우리가 직접 만들 딥러닝 모델의 범위를 정하는 데 도움이 됩니다.

**Kontur 장비 사용 사례 (같이 보기)**
- Tetra Tech (캐나다 앨버타): Mk4 AIR로 약 500 lane-km 포장 조사, Examiner로 층 추적 + 코어로 보정
- Infrasense (미국 NJ I-95, 2023): **DXG 1820** 차량탑재로 아스팔트 두께 컬러 등고선 지도 작성

---

## 2-2. TxDOT Report 0-7217-1 (미국 텍사스 교통국 GPR 장비 비교)

| 항목 | 내용 |
|---|---|
| 제목 | **Synthesis: Commercial Air-Coupled Ground Penetrating Radar Systems to Be Used for Pavement Evaluations in Texas** |
| 번호 | FHWA/TX-26/0-7217-R1 (2025년 8월 제출) |
| 수행 | University of Texas at Arlington (UTA) |
| 본 보고서 PDF | https://library.ctr.utexas.edu/hostedpdfs/uta/0-7217-r1.pdf |
| 요약본(PSR) | https://library.ctr.utexas.edu/hostedpdfs/uta/0-7217-psr.pdf |
| 과제 정보 | https://rip.trb.org/View/2437685 |
| TxDOT 공기결합 GPR 매뉴얼 | https://www.txdot.gov/manuals/mnt/pdm/pavement_evaluation/non_destruct_eval_p_struct_prop-i1004628/_aircoupled_ground_penetrating_radar_gpr-i1004926.html |

**배경**
- TxDOT의 기존 공기결합 GPR 하드웨어가 오래되어 교체 필요 (현재 **PaveCheck** 소프트웨어와 연동)
- TxDOT 9개 지구 설문: **78%가 공기결합 GPR 사용 중**

**비교한 4개 장비**

| 장비 | 제조사 | 특징 (요약) |
|---|---|---|
| RoadScan | GSSI (미국) | 단채널 혼 안테나, 고속 포장 두께 측정. **최종 추천** |
| Hi-Pave | IDS GeoRadar (이탈리아) | 공기결합 혼 안테나 |
| **Kontur AIR** | **Kontur (노르웨이)** | **다채널 공기결합 → 고해상도 3D 이미지** |
| Raptor-80 | ImpulseRadar (스웨덴) | 다채널 3D 어레이 (※ Kontur 제품 아님) |

**결론**: TxDOT의 **포장층 두께 평가 업무**에는 **GSSI RoadScan이 가장 적합**하다고 추천.

**우리 입장에서 읽는 법**
- 결론은 "두께 측정만 하는 기관"의 기준입니다. Kontur는 **3D 이미지가 필요한 업무(공동·매설물·내부 손상)** 에서 강점이 있다는 점을 비교표에서 확인하세요.
- 장비별 **주파수, 채널 수, 스캔 폭, 측정 속도, 소프트웨어** 항목을 표로 옮겨 두면 장비 이해에 가장 빠릅니다.
- ⚠️ 검색 발췌가 일부 깨져 있어서 Kontur AIR에 대한 세부 평가 문장은 원문에서 확인이 필요합니다.

---

## 3. gprMax 학습데이터를 Kiln AI에 적용할 수 있는가?

"Kiln AI"를 두 가지로 해석해서 각각 검토했습니다.

### 해석 A — **Kiln AI 플랫폼** (getkiln.ai, 오픈소스 LLM 개발 도구)

**Kiln이 하는 일** (GitHub README 기준): LLM 평가(Evals), 프롬프트/모델 자동 최적화, RAG, 에이전트, **LLM 파인튜닝(60+ 모델)**, **합성 데이터 생성**, 데이터셋 관리(JSON). 이미지는 RAG 문서로만 넣을 수 있고, 텍스트로 변환되어 쓰입니다.

**결론: gprMax로 만든 B-scan 이미지로 YOLO 같은 탐지 모델을 학습하는 데는 Kiln을 쓸 수 없습니다.**
- Kiln은 **LLM(언어모델) 전용**입니다. 이미지 탐지 모델(CNN/YOLO) 학습 기능이 없습니다.
- 데이터셋 형식이 JSON 텍스트라 레이더 영상(2D/3D 배열)을 학습 데이터로 넣는 구조가 아닙니다.
- Kiln 자체의 "합성 데이터 생성"은 LLM으로 **텍스트**를 만드는 기능이라 gprMax(전자기 물리 시뮬레이션)를 대신하지 못합니다.

**대신 이렇게 나눠 쓰면 Kiln도 쓸모가 있습니다**

```
[gprMax 합성 B-scan] + [KONTUR 실측 데이터]
          │
          ▼
   YOLO / CNN 학습 (PyTorch, Ultralytics)  ← 탐지는 여기서
          │  결과: {위치, 깊이, 크기, 신뢰도, 클래스}  (JSON)
          ▼
   Kiln AI  ← 여기부터 Kiln
     ├─ 탐지 결과 JSON → "공동 위험등급 판정 + 탐사 보고서 문장" 생성 태스크
     ├─ Kiln 합성데이터로 다양한 탐지결과 예시 생성 → 판정 LLM 파인튜닝
     ├─ Evals로 판정/보고서 품질 평가
     └─ RAG: 3D-GPR 매뉴얼(RD-19-R3-003), TxDOT 보고서, 논문을 넣어 Q&A 학습 도우미
```

| 단계 | 도구 | Kiln 사용 가능? |
|---|---|---|
| 학습용 레이더 영상 생성 | gprMax | ✕ |
| 공동·관로 탐지 모델 학습 | PyTorch / Ultralytics YOLO | ✕ |
| 탐지 결과 → 등급 판정·보고서 자동 작성 | LLM | ✓ |
| 매뉴얼·논문 기반 질의응답 | RAG | ✓ |
| 판정 품질 평가 | Evals | ✓ |

### 해석 B — **우리 소성로(Kiln, 킬른) AI** (시멘트·석회 로터리 킬른 등)

**결론: 원리적으로는 가능하지만, 바로 쓰기는 어렵고 검증이 꼭 필요합니다.**

근거
- **킬른 내화물 GPR 측정 사례가 있음**: Proceq(Screening Eagle) GP8800 (계단주파수 SFCW 방식, KONTUR와 같은 원리)로 시멘트 킬른 내화 라이닝을 측정, 라이닝–철피 경계를 확인하고 **최대 5 cm 마모 구간**을 찾음 (업체 사례라 수치는 참고만)
  - https://www.screeningeagle.com/en/inspection/ndt-of-rotary-kiln-using-gpr
- **gprMax 합성데이터로 사전학습한 선례**: 터널 라이닝 결함 탐지에서 gprMax 합성 영상으로 사전학습 → 수렴 속도와 인식 정확도 향상
- **킬른 라이닝 두께 AI 연구**: RKU-Net 등 열전달 모델 기반 합성데이터로 코팅·벽돌 두께 추정 (오차 2.3% 미만, 단 합성데이터 기준 — GPR 아님)
  - https://www.proquest.com/docview/3229139614

gprMax로 킬른 모델 만들기 (개념)
- 층 구조: **철피(완전 반사체) / 내화벽돌 / 코팅층 / 내부 공기**, 벽돌의 유전율·두께를 무작위로 바꿔 수천 장 생성
- 라벨: 벽돌 잔여 두께, 마모 구간 위치 → 회귀 또는 분할 모델

주의할 점
- **철피는 전파가 통과하지 못하므로**, 측정은 킬른 **정지·냉각 후 내부에서** 해야 할 가능성이 큽니다 (운전 중 외부 측정 불가로 추정 — 현장 확인 필요).
- 고온을 겪은 내화물·코팅의 **유전율 값 자료가 부족**합니다 → 실측 몇 개로 시뮬레이션 파라미터를 맞추는 작업이 먼저입니다.
- KONTUR는 차량탑재·도로용 장비라 킬른 내부 측정에는 **핸드헬드 SFCW 장비**가 더 적합합니다.
- 현재 킬른 AI가 **온도·공정 데이터(시계열)** 기반이라면 GPR 영상과는 데이터 종류가 달라서, 별도 모델로 만든 뒤 결과(라이닝 두께)만 기존 AI의 입력으로 넣는 구조가 현실적입니다.

---

## 4. gprMax 맛보기 입력 파일 (도로 밑 공동, 2D)

KONTUR(계단주파수)는 주파수 영역에서 측정하지만, gprMax는 시간 영역 시뮬레이터라 중심주파수가 비슷한 펄스(Ricker)로 근사합니다.

```text
#title: Road cavity B-scan (KONTUR-like, 2D)
#domain: 3.0 2.0 0.005
#dx_dy_dz: 0.005 0.005 0.005
#time_window: 40e-9

#material: 6 0.005 1 0 asphalt
#material: 9 0.01 1 0 base_soil
#material: 12 0.02 1 0 subgrade

#box: 0 0 0 3.0 1.2 0.005 subgrade
#box: 0 1.2 0 3.0 1.6 0.005 base_soil
#box: 0 1.6 0 3.0 1.8 0.005 asphalt

## 공동(공기): 지표 아래 약 0.8 m, 반경 0.25 m
#cylinder: 1.5 1.0 0 1.5 1.0 0.005 0.25 free_space

#waveform: ricker 1 600e6 my_ricker
#hertzian_dipole: z 0.10 1.85 0 my_ricker
#rx: 0.15 1.85 0
#src_steps: 0.02 0 0
#rx_steps: 0.02 0 0
```

실행 (gprMax 설치 후):
```bash
python -m gprMax road_cavity.in -n 130          # 130개 A-scan 생성
python -m tools.outputfiles_merge road_cavity    # B-scan으로 합치기
python -m tools.plot_Bscan road_cavity_merged.out Ez
```
- 공동 위치·크기·깊이, 층 유전율을 무작위로 바꾸는 스크립트로 수백~수천 장을 만들면 YOLO 학습 데이터가 됩니다.
- 위 파일은 문법 예시이며 이 환경에서 실행 검증은 하지 않았습니다.
