# 인수인계 — Track B (날짜 전용 검출기) 담당자용

작성 2026-09-07 · 예선 마감 **2026-09-15 23:59**

이 문서 하나로 시작할 수 있게 썼다. 배경 설계는 `PIPELINE.md`, 검출기 계획은
`DETECTOR_PLAN.md`, 인용 가능한 학술 근거는 `선행연구.md`.

---

## 0. 30초 오리엔테이션

**과제**: 상품 뒷면 사진에서 소비기한을 뽑아 `submission.csv`(`image_id,year,month,day,final_date`)를 만든다.

**현재 파이프라인**:
```
사진 → EXIF 보정·축소 디코딩 → 범용 텍스트 검출(수십 개 박스)
     → 기하 규칙으로 순위 → 상위 K개만 인식 → 정규식 파싱 → 규칙으로 소비기한 선별
```

**당신의 임무**: 위 흐름에서 **"범용 텍스트 검출 + 순위"** 부분을 **날짜 전용 검출기**로
바꿔보고, **같은 지표로 A/B 비교**한다. 이기면 채택, 지면 버린다.

**왜 이 부분인가**: 실패 원인을 끝까지 분해했더니 이렇다.

| 원인 | 비중 | 당신 담당? |
|---|---|---|
| 검출기가 정답 박스를 아예 못 찾음 | **16.4%** | ✅ |
| 찾았으나 상위 9위 밖 (**순위**) | **9.2%** | ✅ |
| 읽었으나 파싱 실패 | 11.2% | ❌ (다른 팀원이 진행 중) |
| 오독 | 5.2% | ❌ |
| 정답 | 57.6% | — |

검출+순위가 25.6%로 최대 덩어리다. 그리고 전용 검출기는 박스를 **한 개(또는 소수)** 만
뱉으므로 **순위 문제가 정의상 사라진다.**

---

## 1. 넘어야 할 기준선 (이미 측정해 뒀다)

ExpDate evaluation **665장**, IoU 0.3 기준. 현행 범용 텍스트 검출기(PP-OCRv4 mobile DBNet, 640px):

| 지표 | 값 |
|---|---|
| **검출 recall (IoU 0.3)** | **82.9%** ← **B1 게이트: 이걸 넘어야 계속 간다** |
| 검출 recall (IoU 0.5) | **53.1%** ← 박스를 헐겁게 잡는다. 여기 기회가 있다 |
| 장당 박스 수 | 16.1개 |

recall@K (상위 K개 안에 정답이 있을 확률):

| K | 1 | 2 | 3 | 5 | 8 | 12 |
|---|---|---|---|---|---|---|
| recall | 26.5% | 42.1% | 50.5% | 63.3% | 71.6% | 77.1% |

**이 표가 핵심이다.** 우리는 비용 때문에 상위 몇 개만 읽는데, 정답이 1위일 확률이 26.5%뿐이다.
날짜 전용 검출기가 90% 재현율로 박스 1~2개만 뱉으면 **recall@1이 90%대**가 된다.

> `results/` 는 `.gitignore` 라 클론에는 없다. **가장 먼저 이 명령으로 기준선을 만들어라**
> (665장, 약 3분). 이후 모든 비교의 기준이 된다:
> ```bash
> python -m eval.detector_ab --baseline --out results/det_baseline.json
> ```
> 위 표의 숫자가 그대로 재현되면 환경이 정상이다.

**⭐ IoU 0.5에서 53.1%로 떨어진다는 점을 주목하라.** 현행 검출기는 날짜를 "찾긴 하는데
헐겁게" 잡는다. 전용 검출기는 박스가 더 타이트할 가능성이 높고, 그러면 **크롭 품질이
좋아져 인식 정확도까지 덤으로 오른다.** 계획에 없던 추가 이득이다.

---

## 2. 환경 세팅 (5분)

```bash
git clone https://github.com/39byte/ITDA_CatchCatch.git
cd ITDA_CatchCatch

# 채점 환경과 같은 Python 3.10. uv 가 없으면 https://docs.astral.sh/uv/
uv venv --python 3.10 .venv
uv pip install --python .venv -r requirements.txt
uv pip install --python .venv pytest psutil

.venv/Scripts/python.exe -m pytest tests/ -q      # 108개 통과해야 정상
```

**데이터는 저장소에 없다**(`.gitignore`). 두 가지가 필요하다:

| 무엇 | 어디에 두나 | 규모 | 출처 |
|---|---|---|---|
| ExpDate **Products-Real** | `kist_data/{train,evaluation}/` | 1,102 + 665장 | [프로젝트 페이지](https://felizang.github.io/expdate/) → Google Drive → `Products-Real.zip` (630MB) |
| ExpDate **Products-Synth** | `kist_data_synth/` | **11,858장 / 날짜 박스 16,674개** | 같은 Drive → `Products-Synth.zip` (1.08GB) |
| 대회 배포 이미지 | `data/` | 3,352장, 라벨 없음 | 대회 제공 |

두 ExpDate 배포판은 **디렉터리 구조가 다르다**(변환기가 둘 다 처리한다):
```
kist_data/          ← Products-Real: train/·evaluation/ 아래에 images/ + annotations.json
kist_data_synth/    ← Products-Synth: 바로 아래에 images/ + annotations.json (평평하다)
```
⚠️ **합성 데이터에는 `exp` 표시도 `dmy_ann` 도 없다.** 이미지당 날짜가 평균 1.4개인데
어느 것이 소비기한인지 알 수 없다. **검출기 학습에만 쓰고, 검증에는 절대 쓰지 않는다.**

정답 파일 생성(이미 있으면 생략):
```bash
python -m eval.expdate --root kist_data/evaluation --out labels/expdate
```

---

## 3. 저장소 지도

```
itda_ocr/              ← 파이프라인 (채점 대상 코드)
  engine.py              엔진 래퍼, 스레드 고정, 박스 필터·순위   ← 당신이 갈아끼울 곳
  pipeline.py            이미지 로딩, 장당 처리, 배치 드라이버
  parse.py               정규식 패밀리 (다른 팀원 담당)
  select.py              소비기한 선별 하드 룰
eval/
  score.py               공식 산식 + 오류 분류표
  expdate.py             ExpDate → 정답 CSV/박스 JSON 어댑터
  run.py                 end-to-end 실행 + 채점
  detector_ab.py       ← **당신의 주력 도구.** 검출기 A/B 비교
tools/
  expdate_to_detection.py ← **학습 데이터 변환기** (YOLO / COCO)
bench/timing.py          장당 비용 측정 (코어·스레드 고정)
docs/
  DETECTOR_PLAN.md     ← **먼저 읽을 것.** 선행연구 + 단계별 게이트
  PIPELINE.md            전체 설계 (길다. §4 엔진 선택만 읽어도 된다)
  선행연구.md            검증된 인용 136건
```

---

## 4. 바로 시작하기

### 4-1. 학습 데이터 만들기 (1분)

```bash
# YOLO 포맷 — images/ 옆에 labels/ 를 만든다. 이미지는 복사하지 않는다(수백 MB 절약)
python -m tools.expdate_to_detection --root kist_data       --format yolo   # 실사
python -m tools.expdate_to_detection --root kist_data_synth --format yolo   # 합성

# 또는 COCO 포맷 (NanoDet / mmdetection 계열)
python -m tools.expdate_to_detection --root kist_data --format coco
```

산출:
- `<root>/**/labels/*.txt` — YOLO 라벨 (이미지 옆)
- `labels/kist_data.yaml`, `labels/kist_data_synth.yaml` — YOLOv8 데이터 설정
- `labels/detection/*_coco.json` — COCO

| 분할 | 이미지 | 날짜 박스 | 용도 |
|---|---|---|---|
| Real train | 1,102 | 1,244 | **미세조정 본체** |
| Real evaluation | 665 | 733 | **검증 전용** |
| Synth | 11,858 | 16,674 | 사전학습 (검증 금지) |

합성 YAML에는 `val` 을 일부러 비워 뒀다 — 합성으로 검증하면 실사 성능을 알 수 없다.

**클래스 처리**: train은 소비기한도 그냥 `date`, evaluation만 `exp`를 붙인다. 검출기 입장에선
같은 대상이라 **하나의 클래스로 합쳤다**. 보조 클래스(`due`/`prod`/`code`)를 같이 학습하고
싶으면 `--classes date,due,prod,code`.

### 4-2. 학습 (예: YOLOv8)

```bash
yolo detect train data=labels/kist_data.yaml model=yolov8n.pt imgsz=480 epochs=100
```

권장 설정과 근거는 `DETECTOR_PLAN.md` §5. 요약하면:
- **COCO 사전학습 필수, 백본 동결 금지** (1,102장 규모에선 scratch가 크게 진다)
- **`imgsz=480` 이상** (§6 함정 참조)
- **조명·색상 증강을 세게**, 기하 증강은 보수적으로

### 4-3. 예측 내보내기

프레임워크는 자유다. **예측 JSON 하나만** 이 포맷으로 내면 된다:

```json
{ "test_00001": [[x1, y1, x2, y2, score], ...], "test_00002": [...] }
```

- `image_id` = 확장자 뺀 파일명
- 좌표 = **원본 이미지 픽셀** (리사이즈 전). 우리 쪽에서 알아서 맞춘다
- `score` = 내림차순 정렬에만 쓴다. 스케일 무관

### 4-4. A/B 비교 — **이게 최종 산출물이다**

```bash
python -m eval.detector_ab --pred results/det_yolo.json \
                           --compare results/det_baseline.json
```

출력 예:
```
## det_yolo
  검출 recall (IoU 0.3) 91.1%   (IoU 0.5) 90.8%

    K   recall@K det_baseline       차이
    1      91.1%        26.5%   +64.7%
    3      91.1%        50.5%   +40.6%

검출 recall 차이 +8.3%
```

### 4-5. 파이프라인에 꽂아 end-to-end 확인

`itda_ocr/engine.py` 의 `Engine.detect()` 가 유일한 접점이다. 같은 시그니처
(`np.ndarray` → `(N,4,2)` 박스 배열, **입력 배열 좌표계**)로 대체하면 나머지는 그대로 돈다.

```bash
python -m eval.run --images kist_data/evaluation/images \
                   --gt labels/expdate/gt_dates.csv --out results/trackb
```

**현재 end-to-end 점수는 26.80 / 50 (완전일치 50.2%)** 이다. 이걸 넘어야 채택이다.

---

## 5. 게이트 — 통과 못 하면 즉시 중단

마감이 9/15다. 각 단계의 손절선을 **미리** 정해 뒀다(`DETECTOR_PLAN.md` §6).

| 단계 | 게이트 |
|---|---|
| **B1** (반나절) | 실사 1,102장만 미세조정 → **검출 recall > 82.9%**. 못 넘으면 **중단** |
| B2 (1일) | 합성 12k 사전학습 + 증강 → B1보다 유의미하게 개선 |
| B3 (반나절) | ONNX + INT8 → **4코어 x86에서 검출 단독 ≤ 40ms** |
| B4 (반나절) | end-to-end **> 26.80 / 50** |

**B1이 진짜 관문이다.** 가장 싼 실험으로 "이 방향이 되긴 하나?"에 답한다.
안 되면 반나절만 잃는다. 이미 쓴 시간이 아까워 계속 가는 게 가장 큰 위험이다.

---

## 6. 이미 비싸게 배운 함정 (읽고 시작하면 몇 시간 아낀다)

### ⚠️ 320px 입력은 쓸 수 없다

문헌에 "NanoDet-Plus-m @320px = 5.25ms" 같은 매력적인 수치가 있지만 **우리한테 이식되지 않는다.**
날짜 박스 높이가 이미지 높이의 2.6%뿐이라:

| 입력 | 박스 높이 p50 | 8px 미만 |
|---|---|---|
| 320px | 8.2px | **39~46%** ❌ |
| 480px | 12.6px | 2.5% |
| 640px | 16.8px | 0% |

**실질 하한은 480px.** 그 아래로 내리면 정답의 40%가 물리적으로 못 잡히는 크기가 된다.

### ⚠️ evaluation 665장을 학습에 넣지 마라

넣는 순간 이후 모든 비교가 무효가 되고, 그 사실을 눈치채기 어렵다.
학습은 `train/` 1,102장 + 합성만 쓴다.

### ⚠️ 속도는 이 개발 PC와 Colab에서 재지 마라

- 개발 PC(8물리/12논리): 2GHz 상한 + 배경 부하로 p50/min이 1.5~1.8까지 벌어진다
- `cpu_affinity([0,1,2,3])` 는 **물리 코어 2개에 4스레드**를 얹는 함정이다(6배 느려짐).
  `bench/timing.py` 는 2칸씩 건너뛰어 잡도록 고쳐 뒀다
- **Colab 무료 티어는 보통 2 vCPU** — 4코어 채점 환경의 대리가 못 된다

→ 정확도는 Colab에서 재도 되지만, **속도는 깨끗한 4코어 리눅스에서만** 낸다.

### ⚠️ x86에서 FP16 양자화 금지

AVX512-FP16이 없으면 ONNX Runtime이 Cast로 FP32 승격해서 **오히려 느려진다**
(한 실측 사례에서 캐스팅이 전체 추론의 53.95%). **INT8 동적 양자화 또는 FP32만.**

### ⚠️ 가중치를 커밋하지 마라

`.pt/.pth/.onnx/.safetensors` 는 `.gitignore` 에 있다. 대회 규정이기도 하다.
학습 결과물은 별도 공유(드라이브 등)하고 저장소엔 넣지 않는다.

### ⚠️ `engine.py` 를 만진다면: RapidOCR의 `det_limit_side_len` 은 무시된다

`limit_type="max"` 일 때 `TextDetector.get_preprocess()` 가 이미지 크기를 보고
**960/1500/2000으로 덮어쓴다.** 1008px 이미지가 1500px로 *확대*돼 검출이 5배 느려진다.
증상은 "설정을 바꿔도 박스 개수가 그대로". 우리는 `DetPreProcess` 를 직접 끼워 우회했다.

---

## 7. 알아 두면 좋은 배경

### 선행연구의 결론: **비교 논문이 없다**

"날짜 전용 단일 클래스 검출"과 "범용 텍스트 검출"을 **같은 데이터에서 통제 비교한 논문은
존재하지 않는다.** 문헌은 겹치지 않는 두 갈래로만 있다.

→ 그래서 당신이 만드는 A/B 표가 **우리 프로젝트의 방법론적 기여**가 된다.
요약서(정성 25점)에 그대로 들어간다. **지더라도 그 결과는 점수가 된다** —
"기각한 대안"으로 쓰면 되기 때문이다. 부담 없이 재고 정직하게 보고하면 된다.

### 참고할 선례

- **Gong et al. (Lincoln, 2018–2021)** — 단일 클래스 "use-by-date ROI" FCN. 우리 노선의 가장
  명확한 선례. **Global-level CNN이 이미지 품질(흐림/선명/날짜없음)을 사전 판정**하는
  브랜치가 있는데, 우리에겐 없는 아이디어라 참고할 만하다
- **도트매트릭스 가설** — 소비기한은 잉크젯 점 글자가 많고, 독립 논문 4편이 "점 글자는
  일반 검출기가 유난히 못 잡는다"고 보고한다(Wahyono & Jo 2015: 검출율 68.8%).
  **우리 recall 실패의 원인일 수 있다.** 실패 이미지에 인쇄 방식을 태깅해 보면 2시간에 확인된다

### 인용 시 주의

- **ExpDate 논문의 97.74%는 엔드투엔드 수치**다. 검출 단독 recall이 아니다
- "ExpDate 검출기가 FCOS 기반"이라는 서술은 2차 출처뿐이고 **원문 대조 실패**했다.
  원문 PDF를 구하기 전엔 인용 금지
- GPU FPS·ARM 수치를 CPU 근거로 쓰지 말 것. 지연 수치는 **하드웨어·스레드 수**를 반드시 함께 확인

---

## 8. 미해결 / 판단이 필요한 것

1. **ExpDate는 한국 제품이 아니다.** 학습한 검출기가 대회 평가셋(한국 제품, 2026년 별도 수집)에서
   무너질 수 있다 — 좁은 도메인 학습 검출기가 다르게 수집된 셋에서 가장 심하게 무너진다는 건
   확인된 경고다. **완화**: 조명·색상 랜덤화를 세게, 그리고 **기존 DBNet 폴백을 반드시 유지.**
2. **박스를 몇 개 뱉게 할 것인가.** 소비기한만 1개? 아니면 제조일자 등 날짜 계열 전부?
   후자가 우리 구조와 맞는다 — `select.py` 가 이미 소비기한 선별을 잘 한다(오류 0.5%).
   현재 변환기는 후자로 되어 있다.
3. **합성 데이터(Products-Synth 약 12,000장)를 쓸지.** B1을 통과한 뒤에 판단한다.
   같은 Drive 폴더의 `Products-Synth.zip`(1.08GB)이다.

---

## 9. 연락·규칙

- 저장소: https://github.com/39byte/ITDA_CatchCatch (Public, `main`)
- 브랜치를 따서 작업하고 PR로 합친다. `main` 은 **항상 돌아가는 상태**로 유지한다
- 커밋 전 `python -m pytest tests/ -q` (108개)
- 실험 결과는 숫자와 함께 커밋 메시지에 남긴다 — 요약서 PDF의 원재료가 된다
