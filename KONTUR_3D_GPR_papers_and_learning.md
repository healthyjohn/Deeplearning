# KONTUR (구 3D-Radar) 3D GPR 장비 — 논문 & 학습자료 모음

> 정리일: 2026-11-06
> 표기: ✅ = 출처에서 KONTUR/3D-Radar(GeoScope, DXG 안테나 등) 사용이 확인됨 · ⚠️ = 같은 분야 핵심 논문이지만 장비 제조사를 확인하지 못함(원문에서 "Equipment" 절 확인 필요)

---

## 0. KONTUR 장비 한눈에 보기

| 항목 | 내용 |
|---|---|
| 회사 | **Kontur AS** (노르웨이). 2001년 설립된 **3D-Radar AS**가 전신. 트론헤임/오슬로 |
| 핵심 기술 | **Step-Frequency(SF, 계단 주파수) 3D GPR** — 한 번에 넓은 주파수 대역(약 40/90 MHz ~ 3 GHz)을 스윕 |
| 하드웨어 | 레이더 본체 **GeoScope MkIV**, 지면결합형 다채널 안테나 **DXG 시리즈**(예: DXG 1820), 공기결합형 **Mk4 AIR / Raptor** 계열 |
| 소프트웨어 | **Examiner** (Collect / Specialist / Explore), ML 처리 **Orbit** |
| 주 용도 | 도로 하부 공동(싱크홀) 탐사, 포장 두께·밀도(유전율) 평가, 교량 상판, 매설관 탐지, 철도 자갈층, 고고학 |
| 국내 쓰임새 | 서울시 등 지자체 **차량탑재 3D GPR 지하공동 탐사**에 널리 사용 → 딥러닝 기반 공동 자동탐지 연구가 활발 |

**Step-Frequency GPR을 이해하려면**: 일반 임펄스 GPR은 짧은 펄스를 쏘고 시간영역에서 반사파를 받지만, SF-GPR은 주파수를 계단식으로 바꿔가며 측정한 뒤 **역푸리에변환(IFFT)으로 시간영역 신호(A-scan)를 합성**합니다. 그래서 저주파(깊이)와 고주파(해상도)를 한 장비로 얻습니다.

---

## 1. 국내 논문

### 1-1. KONTUR(3D-Radar) 데이터 + 딥러닝 (가장 추천)
세종대 안윤규 교수 / 이종재 교수 연구팀 — 서울 도심 도로에서 3D-Radar 차량탑재 장비로 수집한 데이터를 사용

1. ✅ **Kang, M.-S., Kim, N., Lee, J.-J., An, Y.-K.** "Deep learning-based automated underground cavity detection using three-dimensional ground penetrating radar." *Structural Health Monitoring*, 2019/2020.
   - 3D-Radar GeoScope MK IV로 수집한 현장 데이터. 공동 / 관로 / 맨홀 / 정상 지반을 A·B·C-scan 이미지로 분류
   - https://www.researchgate.net/publication/332081634
2. ⚠️(같은 연구팀·같은 장비 계열) **Kim, N., Kim, S., An, Y.-K., Lee, J.-J.** "Triplanar Imaging of 3-D GPR Data for Deep-Learning-Based Underground Object Detection." *IEEE JSTARS*, 2019.
   - 서울 도로 **13 km** 차량탑재 다채널 3D GPR 데이터. B-scan + C-scan + 단면을 3채널 이미지로 묶는 아이디어
   - https://dblp.uni-trier.de/rec/journals/staeors/KimKAL19.html
3. ⚠️ **Kang, M.-S., Kim, N., Im, S.B., Lee, J.-J., An, Y.-K.** "3D GPR Image-based UcNet for Enhancing Underground Cavity Detectability." *Remote Sensing* 11(21), 2545, 2019.
   - 데이터가 적어 **전이학습(transfer learning)** 사용 → 소량 데이터로 시작할 때 참고
   - https://www.mdpi.com/2072-4292/11/21/2545
4. ⚠️ **Kim, N. et al.** "A Novel 3D GPR Image Arrangement for Deep Learning-based Underground Object Classification." *Int. J. Pavement Engineering*, 2019.
5. 연구실 논문 목록: https://app.rndcircle.io/lab/7cd94cc5-9497-400b-b5cd-a8f9edc24922/papers/page/3

### 1-2. 차량탑재 3D-GPR 현장 탐사 (국문)
6. ⚠️ 이승호·장일호, "3D-GPR탐사를 이용한 도로하부 지반상태에 대한 연구", 2019 (상지대) — 테스트베드에서 심도별 공동·매설물 탐지
   - https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO201912261948983 · PDF: https://www.koreascience.or.kr/article/JAKO201912261948983.pdf
7. ⚠️ "멀티채널 GPR 장비를 이용한 도로하부 공동의 크기 변화 분석", 2020 — 공동 크기·등급의 장기 모니터링
   - https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO202015463049588
8. ⚠️ "지하공동 탐지를 위한 3차원 시간영역 유한차분 GPR 탐사 모델링", 2016 — 시뮬레이션(FDTD) 기초
   - https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO201609562998110&dbt=NART

### 1-3. GPR 공동탐지 딥러닝 (국문, 장비 확인 필요)
9. ⚠️ 최병훈 외, "도심지 도로 지하공동 탐지를 위한 딥러닝 기반 GPR 자료 해석 기법", *지구물리와 물리탐사*, 2022 — YOLOv3, 정밀도 84% / 재현율 92%
   - https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002902847
10. ⚠️ 최용욱·서상진·장한길로·윤대웅, "데이터 증강 및 앙상블 기법을 이용한 딥러닝 기반 GPR 공동 탐지 모델 성능 향상 연구", 2023
    - https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003021182
11. ⚠️ "NM-GPR을 활용한 공동 탐지 AI 모델 적용성 연구", *한국지반공학회논문집* 41(3), 2025 — YOLOv8 vs Faster R-CNN
    - https://cdn.apub.kr/journalsite/sites/kgs/2025-041-03/N0990410302/N0990410302.pdf

### 1-4. 국내 연구보고서 (학습용으로 매우 좋음)
12. **"3D-GPR 탐사장비의 시설물 적용방안 연구 및 활용 매뉴얼 개발"** (RD-19-R3-003, 2019.12) — 장비 운용·해석 매뉴얼 성격. 처음 공부할 때 1순위로 읽어볼 것
    - https://www.codil.or.kr/filebank/original/RK/OTKCRK211043/OTKCRK211043.pdf?stream=T
13. "도심지 지하공동 탐지 다중 채널 3차원 GPR 시스템 개발" (국가R&D 보고서) — 16채널 차량탑재 GPR 국산화 과정
    - https://scienceon.kisti.re.kr/srch/selectPORSrchReport.do?cn=TRKO201800014587
14. 참고: 한국 고속도로 아스팔트 덮인 콘크리트 교량 상판의 유전율 평가 (*Sensors* 2020, 1 GHz 공기결합 GPR; 장비 제조사 확인 필요)
    - https://doi.org/10.3390/s20092497

---

## 2. 해외 논문

### 2-1. GeoScope / DXG 사용 확인 논문
15. ✅ "Pavement Distress Detection Using Three-Dimension Ground Penetrating Radar and Deep Learning." *Applied Sciences* 12(11), 5738, 2022.
    - GeoScope 3D GPR 수평단면(C-scan) 이미지 + **CP-YOLOX, SViT** → 균열·공동 등 포장 내부 손상 탐지
    - https://doi.org/10.3390/app12115738
16. ✅ "Thickness Uniformity Assessment of Epoxy Asphalt Pavement Layer on Steel Bridge Decks Using Three-Dimensional GPR." *Buildings* 15(12), 2138, 2025.
    - **GeoScope MKIV + DXG 다채널 안테나**. 강상판 위 얇은 포장층 두께·유전율
    - https://doi.org/10.3390/buildings15122138
17. ✅(요약 기준) "Research on 3D GPR deep underground cavity identification algorithm in urban roads using multi-dimensional time-frequency features." *NDT & E International*, 2024.
    - GeoScope, 90–1000 MHz, 횡방향 간격 7.65 cm. 시간-주파수 특징 기반 깊은 공동 식별
    - https://www.sciencedirect.com/science/article/abs/pii/S0963869524000252
18. 그림 참고(장비 구성 사진): https://www.researchgate.net/figure/GeoScope-3D-Radar-equipment-a-radar-mainframe-b-multichannel-antenna-array-c_fig1_361106284

### 2-2. 3D-Radar 초기 연구 (노르웨이·유럽)
19. ✅ SINTEF/NTNU — 도로 포장 평가용 3D-Radar 발표·보고서(RV33 포장층 두께, 2007년 장비 도입)
    - https://www.sintef.no/en/publications/publication/1268317
    - NVF 발표자료: https://www.vegagerdin.is/nvf/nvf34.nsf/28ccef055afc64ac002572ac004ba020/69ec18fe82d8467b002573fa0038bb1f/$FILE/5-3%20Use%20of%20Ground%20Penetrating%20Radar%20(GPR).pdf
20. ✅ EGU 2015 — 3-D radar의 아스팔트 두께 예측 정확도(0.13 m 이상에서 정확)
    - https://meetingorganizer.copernicus.org/EGU2015/EGU2015-7485.pdf
21. ✅ "Step-frequency radar applied on thin road layers" (Fauchard 외, 프랑스)
    - https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_hal_primary_oai_HAL_hal_02178048v1
22. ✅ 스웨덴 Trafikverket 2018 — 철도 자갈층/노반 두께를 최대 100 km/h로 3D 레이더 측정
    - https://bibliotek.vti.se/bib/291170

### 2-3. 미국 DOT 보고서 (장비 비교에 유용)
23. ✅ TxDOT Report 0-7217-1 — 상용 공기결합 GPR 비교(GSSI, **Kontur**, IDS, ImpulseRadar)
    - https://library.ctr.utexas.edu/hostedpdfs/uta/0-7217-r1.pdf
24. ✅ Univ. of Pittsburgh — 지하 매설물 탐지 신기술 조사(차량 부착형 Kontur 포함)
    - https://www.engineering.pitt.edu/contentassets/e9b3db3b2163488aaf2e4c50a2f6c640/undeground-utilities-final-report_mod.pdf

### 2-4. 3D GPR 딥러닝 최신 연구 (장비 확인 필요 — 방법론 공부용)
25. ⚠️ "3DInvNet: A Deep Learning-Based 3D GPR Data Inversion" — https://arxiv.org/pdf/2305.05425
26. ⚠️ "Intelligent recognition of GPR road hidden defect images based on feature fusion and attention mechanism" — https://arxiv.org/pdf/2512.21452
27. ⚠️ "TriView-YOLO: Early Multi-View Fusion for GPR Cavity Detection in Soft, High-Water-Content Soils" — https://arxiv.org/pdf/2608.09522 (1-1의 Triplanar 아이디어와 비교해 읽기 좋음)
28. ⚠️ "Automatic detection and location of pavement internal distresses from GPR images based on deep learning" — https://www.sciencedirect.com/science/article/abs/pii/S0950061823042022
29. ⚠️ "3D GPR acquisition and control technology for road underground diseases with dual-band antenna arrays" — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10575334/

---

## 3. 학습자료

### 3-1. KONTUR 제조사 자료
- **Kontur 공식 문서/브로셔**: https://www.kontur.tech/brochures-and-product-sheets
- **Kontur Americas 워크숍 발표 (TAM 25, 2025.08)** — 유전율 매핑, Examiner Collect, Orbit(ML) 소개. 최신 장비 흐름 파악에 좋음
  - https://www.intrans.iastate.edu/wp-content/uploads/2025/09/TAM-25-Workshop-KONTUR-presentation-D-Barry.pdf
- 고객 사례: Tetra Tech 500 km 포장 조사(Mk4 AIR + Examiner, 코어 보정), Infrasense I-95 (DXG 1820) 등
  - 사례 목록(미러 사이트로 보임, 공식 사이트에서 다시 확인 권장): https://kontur-jet.vercel.app/customer-projects
- 이탈리아 CNR 교육자료(3D GPR 응용 사례: 터널·철도·댐·고고학): https://training.itineris.cnr.it/mdl/pluginfile.php/2725/mod_resource/content/1/Modulo%201.pdf
- Examiner 사용법 교육은 공개 튜토리얼이 거의 없음 → **국내 대리점 또는 Kontur 지원팀에 사용자 교육 요청**이 가장 확실

### 3-2. GPR 기초 (무료)
- AASHTO SHRP2 R06D GPR 웨비나 자료(도로 GPR 실무): https://shrp2.transportation.org/Documents/Renewal/R06D%20GPR%20webinar%206-28-18.pdf
- Guideline Geo "Introduction to GPR" 무료 웨비나: https://www.guidelinegeo.com/?p=7759
- 위 1-4의 국내 **3D-GPR 활용 매뉴얼**(RD-19-R3-003)

### 3-3. 딥러닝 실습용 도구·데이터
- **gprMax** (오픈소스 FDTD GPR 시뮬레이터) — 공동·관로 모델을 만들어 합성 B-scan 생성 → 학습 데이터 부족 해결: https://www.gprmax.com / https://github.com/gprMax/gprMax
- **PINN4GPR** 튜토리얼 — gprMax로 무작위 데이터셋 자동 생성 + CNN 학습 파이프라인 예제: https://pinn4gpr.readthedocs.io/en/latest/tutorial.html
- Apaydın & İşseven (2024, *Open Geosciences*) — gprMax 합성 + 실험 데이터로 Faster R-CNN / YOLOv5 / SSD 비교: https://www.degruyterbrill.com/document/doi/10.1515/geo-2022-0685/html
- Roboflow GPR 쌍곡선(hyperbola) 탐지 데이터셋 (CC BY 4.0): https://universe.roboflow.com/gpr-eywbm/gpr-hhalx

---

## 4. 추천 학습 순서 (약 6주)

| 주차 | 할 일 | 자료 |
|---|---|---|
| 1 | GPR 원리: 유전율, 전파속도, 반사, 쌍곡선이 생기는 이유 / A·B·C-scan 개념 | 3-2 웨비나, RD-19-R3-003 매뉴얼 |
| 2 | Step-Frequency GPR과 KONTUR 장비 구성(GeoScope + DXG, Examiner) | 0장, TAM 25 발표, TxDOT 보고서 |
| 3 | 실무 해석: 공동 vs 관로 vs 맨홀 신호 구분, 포장 두께·유전율 | 논문 6, 7, 16, 19 |
| 4 | 딥러닝 입문 논문 정독 | 논문 1 → 2 → 3 (같은 팀, 순서대로) |
| 5 | 실습: gprMax로 공동/관로 합성 B-scan 만들고 YOLO 학습 | 3-3 도구 |
| 6 | 최신 기법: 다중 뷰 융합, 어텐션, 3D 역산 | 논문 15, 25, 26, 27 |

**논문 읽을 때 체크 포인트**
- 장비/주파수 대역/채널 간격(KONTUR DXG는 보통 약 7.5 cm 간격)
- 입력을 B-scan만 쓰는지, C-scan(평면)까지 함께 쓰는지
- 라벨 정답 확인 방법(시추·내시경 검증 여부)
- 데이터 부족 해결법(전이학습, 증강, gprMax 합성)

---

## 5. 더 찾아볼 곳
- **RISS / KCI / ScienceON** 검색어: `3D GPR 공동`, `차량탑재 GPR`, `GeoScope`, `3D-Radar`, `지하공동 딥러닝`
- **Google Scholar** 검색어: `"GeoScope" GPR`, `"3D-Radar" step-frequency`, `"Kontur" GPR`, `"DXG" antenna array`
- 위 ⚠️ 표시 논문은 원문 "Equipment / Data acquisition" 절에서 장비명을 확인하면 KONTUR 사용 여부를 바로 알 수 있음
