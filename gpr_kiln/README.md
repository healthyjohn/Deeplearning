# gpr_kiln — gprMax 학습데이터 + 이미지 탐지 모델을 Kiln에서 쓰기

도로 밑 **공동(cavity)** 과 **관로(pipe)** 를 GPR B-scan 영상에서 찾는 YOLO 모델을,
**gprMax 시뮬레이션 데이터**로 학습시키고, 그 전체 과정을 **Kiln의 도구(MCP)** 로 불러 쓸 수 있게 만든 패키지입니다.

```
Kiln 태스크/에이전트 ──(MCP 도구 호출)──▶ gpr-kiln MCP 서버
                                           ├─ gpr_simulate_scene    시뮬레이션 1장 미리보기
                                           ├─ gpr_generate_dataset  gprMax → YOLO 데이터셋
                                           ├─ gpr_train_detector    YOLO 학습
                                           └─ gpr_detect            B-scan → 공동/관로 위치·깊이 JSON
```

## 1. 설치

Python 3.10 이상. CPU만 있어도 동작합니다 (GPU가 있으면 학습이 훨씬 빠름).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # gprMax 4.x, ultralytics, mcp, h5py, pillow 설치
pip install pytest && pytest -q tests     # 동작 확인
```

## 2. 명령어로 쓰기

```bash
# 1) 학습 데이터 만들기 (gprMax: 4코어 CPU 기준 장면당 약 45초)
gpr-kiln generate data/gpr --engine gprmax -n 200 --seed 1
#    같은 폴더에 다시 실행하면 이어서 추가됩니다 (seed만 바꾸기)
gpr-kiln generate data/gpr --engine gprmax -n 200 --seed 2

#    빠른 근사 엔진(analytic): 수천 장을 몇 초에 → 사전학습/테스트용
gpr-kiln generate data/gpr_fast --engine analytic -n 2000

# 2) 탐지 모델 학습 (Ultralytics YOLO)
gpr-kiln train data/gpr/data.yaml --epochs 100

# 3) 탐지
gpr-kiln detect runs/gpr/train/weights/best.pt my_bscan.png \
    --scan-length-m 1.32 --time-window-ns 26 --eps 9 --save out.png
```

`detect` 결과 예:
```json
{"n_detections": 1,
 "detections": [{"class": "cavity", "confidence": 0.91, "bbox_px": [160, 190, 570, 350],
                 "x_m": 0.75, "width_m": 0.85, "apex_time_ns": 7.8, "depth_m": 0.39}]}
```
- `depth_m`은 쌍곡선 꼭짓점 시간 × 전파속도(√eps로 계산) / 2 로 구한 **근사 깊이**(공동 윗면 기준)입니다.

## 3. Kiln에 연결하기

### 방법 A — MCP 서버로 연결 (권장)
Kiln의 도구(Tools) 설정에서 **로컬 MCP 서버**를 추가하고 아래처럼 입력합니다.

| 항목 | 값 |
|---|---|
| Command | `/절대경로/.venv/bin/gpr-kiln-mcp` |
| Arguments | (없음) |
| Env | `GPR_KILN_WORKDIR=/절대경로/gpr_kiln_work` |

- 다른 PC나 서버에서 돌릴 때: `gpr-kiln serve --transport streamable-http` 로 띄우고 Kiln에는 원격 MCP 서버 URL(`http://서버:8000/mcp`)로 등록
- 등록 후 Kiln 태스크의 실행 설정(Run config)에서 `gpr_detect` 등 도구를 켜면, 모델이 필요할 때 도구를 호출합니다.
- 상대경로로 넘긴 파일 경로는 `GPR_KILN_WORKDIR` 아래에서 찾습니다.

> Kiln 메뉴 이름은 버전에 따라 조금 다를 수 있습니다. 이미 다른 이미지 탐지 모델을 Kiln 도구로 쓰고 있다면, 그 모델을 등록한 같은 화면에 추가하면 됩니다.

### 방법 B — Kiln Code Tool(Python)로 직접 호출
이미 쓰는 탐지 모델이 Kiln Code Tool로 붙어 있다면, 같은 방식으로 아래 함수를 부르면 됩니다.

```python
from gpr_kiln.detector import detect

def gpr_detect(image_path: str, scan_length_m: float = None, time_window_ns: float = None, eps: float = 9.0) -> dict:
    return detect("/절대경로/best.pt", image_path, conf=0.25,
                  scan_length_m=scan_length_m, time_window_ns=time_window_ns, eps=eps)
```

### Kiln 태스크 예시
- **입력**: B-scan 이미지 경로 + 측선 정보(길이, 시간창, 지반 유전율)
- **도구**: `gpr_detect`
- **출력(구조화)**: `{등급: A~D, 위치_m, 깊이_m, 근거, 추가조사_필요}`
- Kiln Evals로 판정 결과를 검증하고, 사람이 확인한 판정은 Kiln 데이터셋에 쌓아 LLM 판정기를 파인튜닝

## 4-0. Examiner 데이터로 바로 탐지하기 (KONTUR Deep 1808)

장비 프로파일 `kontur-deep1808`: 30–1494 MHz, 트레이스 간격 55 mm, Examiner 처리 ε=7 · Time Window 42 ns (깊이 약 2.35 m).
학습 장면 1장 = 약 2.6 m × 42 ns. 긴 측선은 이 크기로 잘라가며 탐지합니다.

```bash
# 1) 이 장비에 맞춘 학습 데이터 (돌·자갈 잡음 포함)
gpr-kiln generate data/kontur --profile kontur-deep1808 -n 300 --seed 1
gpr-kiln train data/kontur/data.yaml --epochs 80

# 2-a) Examiner: Export Data → "Processed Data: SEG-Y" 로 내보낸 파일
gpr-kiln detect-long best.pt line031_ch3.sgy --from-m 75.5 --to-m 99.5 --time-window-ns 42 --eps 7

# 2-b) 또는 Examiner B-scan 화면 캡처 (데이터 영역만 자르기, 맨 위 = 0 ns)
gpr-kiln detect-long best.pt bscan.png --from-m 75.5 --to-m 99.5 --time-window-ns 42 --eps 7
```
결과: 도로 거리(m), 공동 윗면 깊이(m), 클래스, 신뢰도 + 표시된 이미지. Kiln에서는 `gpr_detect_long` 도구로 같은 기능을 씁니다.

SEG-Y 내보낼 때 권장
- Region Processing: Interference Suppression, IFFT, Background Removal은 켜고, 게인은 0 dB (탐지 코드가 학습과 같은 방식으로 다시 정규화)
- 채널(Ch)별 파일이면 그대로, 한 파일에 여러 채널이면 `--channel`, `--traces-per-channel` 지정
- SEG-Y의 시간 단위는 장비마다 달라서 자동 추정합니다. 출력 `info.time_window_ns`가 Examiner Time Window와 같은지 확인하세요.

실측 라벨 만들기: Examiner에서 표시한 "공동탐지의심" 같은 주석(Annotations)을 내보내면(텍스트/GeoJSON), 그 위치로 실측 학습 라벨을 만들 수 있습니다 (다음 단계).

## 4. 실제 KONTUR 데이터에 적용하기

1. **같은 전처리**: 실측 B-scan을 `.npy`(행=시간 샘플, 열=트레이스)로 내보내 `--dt`(샘플 간격, 초)와 함께 넣으면, 학습 때와 같은 처리(배경 제거 → 시간 게인 → 정규화)를 거쳐 탐지합니다. Examiner에서 이미지로 내보낸 경우는 PNG를 바로 넣어도 되지만, 표시 설정(게인, 배경 제거)이 학습 영상과 비슷해야 정확합니다.
2. **주파수 맞추기**: KONTUR DXG는 약 100 MHz–3 GHz 계단주파수 장비입니다. gprMax는 시간영역이라 중심주파수 펄스(Ricker)로 근사합니다. `--freq`(기본 600 MHz)를 실제로 보는 대역에 맞추세요.
3. **트레이스 간격**: 시뮬레이션 기본 2.5 cm. 실측 데이터 간격이 다르면 가로 길이를 같은 비율로 맞춰서 넣으세요.
4. **합성 → 실측 순서로 학습**: gprMax로 사전학습한 뒤, 라벨링한 실측 데이터 수십~수백 장으로 추가 학습(fine-tune)하는 것이 핵심입니다. 합성 데이터만으로는 실측 잡음·불균질 지반을 다 담지 못합니다.

## 5. 시뮬레이션 설정 (scenes.py)

| 항목 | 기본값 / 범위 |
|---|---|
| 모델 크기 | 폭 1.6 m × 깊이 1.0 m, 셀 6 mm (2D) |
| 층 구조 | 아스팔트 5–25 cm (εr 4–7) / 기층 15–35 cm (εr 6–10) / 노상 (εr 8–16) |
| 공동 | 공기, 반경 5–22 cm |
| 관로 | 금속(PEC), 반경 2–10 cm |
| 장면당 물체 | 0–2개 (10%는 물체 없음) |
| 라벨 | 장면 형상에서 쌍곡선 꼭짓점·팔 위치를 계산해 자동 생성 |

클래스를 늘리려면 `CLASS_NAMES`와 `random_scene()`에 추가하면 됩니다 (예: 물 찬 공동 → `free_space` 대신 εr 80 재질).

## 6. 검증 결과 (2026-11-06, CPU 4코어)

| 단계 | 결과 |
|---|---|
| gprMax 데이터 생성 | 60장면 (학습 43 / 검증 17), 장면당 약 45초, 총 52분 |
| YOLO11n 학습 | 60 epoch, 약 15분 (CPU) |
| 검증셋 성능 | Precision 0.95 · Recall 0.97 · **mAP50 0.99** · mAP50-95 0.57 |
| 클래스별 mAP50 | cavity 0.99 · pipe 0.995 |
| MCP 종단 시험 | MCP 클라이언트 → `gpr_simulate_scene`(새 gprMax 장면) → `gpr_detect` 정상 동작 |

학습에 쓰지 않은 새 장면 2개 결과:

| 장면 | 실제 공동 (위치 / 윗면 깊이) | 탐지 (위치 / 추정 깊이) |
|---|---|---|
| seed 777 | 0.85 m / 0.56 m | 0.69 m / 0.52 m (신뢰도 0.65) |
| seed 2024 | 0.72 m / 0.27 m | 0.74 m / 0.32 m (신뢰도 0.51) |

- 데모 모델: `gpr_kiln/models/gprmax60_yolo11n.pt` (합성 60장면으로만 학습 → **실측 데이터에 바로 쓰기엔 부족**, 출발점 용도)
- 예시 영상: `gpr_kiln/examples/` (`*_detected.png`가 탐지 결과)
- 시뮬레이션 데이터 안에서는 성능이 높지만, 실측 성능은 KONTUR 실측 데이터로 다시 확인해야 합니다. 신뢰도가 0.5–0.65로 낮은 편이라 장면 수를 수백 개 이상으로 늘리는 것을 권장합니다.

## 7. 한계
- 2D 시뮬레이션입니다. 3D(실제 KONTUR 3D 볼륨)는 gprMax로 가능하지만 장면당 계산이 수십 배 늘어납니다.
- 지반을 균질 층으로 단순화했습니다. 실제 지반의 자갈, 수분 불균질은 `#fractal_box` 등으로 추가할 수 있습니다.
- 깊이 추정은 평균 유전율 하나를 쓰는 근사값입니다.
