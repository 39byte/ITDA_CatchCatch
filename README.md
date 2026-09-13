# ITDA OCR CHALLENGE — 소비기한 추출 · [DAT] 캐치캐치

상품 뒷면 이미지에서 소비기한을 추출해 `submission.csv`(`image_id, year, month, day, final_date`)를
생성한다.

---

## 1. 실행

```bash
git clone https://github.com/39byte/ITDA_CatchCatch.git
cd ITDA_CatchCatch
pip install -r requirements.txt
bash download_weights.sh   # 가중치 확인 (오프라인 실행 전 무결성 검증)

export ITDA_INPUT_DIR=./val_images
export ITDA_OUTPUT_PATH=./submission.csv

jupyter nbconvert --to notebook --execute predict.ipynb \
    --ExecutePreprocessor.timeout=2400 \
    --output /tmp/executed.ipynb
```

## 2. 가중치 — **오프라인 즉시 구동 (런타임 다운로드 0)**

인터넷이 차단된 채점 환경에서도 별도 다운로드 없이 즉시 실행됩니다.

1. **RapidOCR 기본 가중치** (det/cls/rec ONNX 약 16 MB): `rapidocr-onnxruntime==1.4.4` **wheel 내부에 포함**되어 있어 `pip install` 만으로 로컬에 완비됩니다.
2. **NanoDet 날짜 전용 검출기** (5.6 MB): 저장소 `weights/date_detector_ema.onnx` 에 직접 포함되어 있습니다.
3. **`download_weights.sh`**: 대회 채점 규격에 맞추어 포함되어 있으며, 실행 시 로컬 가중치 무결성을 검증하고 즉시 정상 종료(exit 0)합니다.

- 노트북 실행 중 **네트워크 호출이 전혀 발생하지 않습니다.**
- 오프라인 실행 검증 완료: 네트워크를 완전히 차단한 상태에서 공식 채점 명령(`jupyter nbconvert`)으로 완주함을 확인했습니다.

## 3. 파이프라인 개요

```
[0] 정규화     EXIF 회전 보정 → JPEG draft() 축소 디코딩 (≥4MP 코호트 318→103 ms)
[1] 검출       RapidOCR DB 검출기 1회 (기본)  또는  NanoDet 날짜 전용 검출기 (Track B)
[2] 박스 필터  종횡비·높이·면적으로 후보를 좁혀 상위 몇 개만 인식으로 넘긴다   ← 핵심
[3] 인식       크롭 배치 인식 (+ 방향 분류기로 180° 뒤집힘 처리)
[4] 파싱       박스 병합 → 정규식 패밀리 → 부분 결과 허용
[5] 선별       하드 룰 캐스케이드로 소비기한 하나를 고른다                  ← 핵심
[6] 출력       스키마 검증 후 저장 (실패해도 raise하지 않고 안전값으로 복구)
```

**[1] 검출 — NanoDet 날짜 전용 검출기 (Track B).** `weights/date_detector_ema.onnx`
(NanoDet-Plus-m @480, 단일 클래스 `date`, 5.6 MB)가 있으면 `[1]` 이 범용 텍스트
검출 대신 이 모델을 쓴다. ExpDate evaluation 665장 기준:

| | 검출 recall (IoU 0.3) | recall@1 | end-to-end Score |
|---|---|---|---|
| RapidOCR 범용 텍스트 검출 | 82.9% | 26.5% | 36.85 / 50 |
| NanoDet 날짜 전용 검출 | **97.6%** | 50.2% | 39.71 / 50 |
| + 크롭 여백 보정 (`nanodet_expand=0.10`) | 97.6% | 50.2% | 41.41 / 50 |
| **+ 검출기 점수 순서 보존** | **97.6%** | **84.7%** | **43.47 / 50** |

**검출기를 바꾸면 그 뒤 두 단계도 같이 바뀌어야 한다.** 병합 직후 두 곳이 어긋나 있었다.

*(1) 순위.* `filter_boxes` 는 종횡비 사전확률로 박스를 **재정렬**한다 — 범용 텍스트
검출기에는 "날짜다움" 점수가 없어 기하로 추측할 수밖에 없기 때문이다. NanoDet 은
단일 클래스 `date` 검출기라 박스 점수가 곧 날짜다움인데, 그 위에 종횡비 재정렬을
덮으면 소비기한 박스 recall@1 이 **84.7% → 50.2%** 로 무너진다. 지금은
`Engine.detect_and_filter()` 가 NanoDet 경로에서 재정렬을 끄고 검출기 순서를 쓴다.

*(2) 크롭 여백.* RapidOCR DB 박스는 `unclip_ratio=1.6`
으로 이미 부풀려져 나오지만 NanoDet 회귀 박스는 글자에 딱 맞아서, 그대로 자르면
`2021.08.04` 가 `2021.08.0` 이 된다 — 이 한 가지로 665장 중 88장이 퇴행했다.
여백 비율은 스윕으로 정했고 0.06~0.15 가 평탄해 중앙값을 골랐다
(`itda_ocr/nanodet_det.py: DEFAULT_EXPAND`).

근거 전문은 `docs/ERROR_ANALYSIS.md` §10(여백)·§12(순위).

`predict.ipynb` 는 이 파일의 존재를 확인해 자동으로 NanoDet 경로를 켠다.
재현 절차는 [`docs/REPRODUCE.md`](docs/REPRODUCE.md), 설계 배경은
[`docs/DETECTOR_PLAN.md`](docs/DETECTOR_PLAN.md).

**[2]와 [5]가 이 설계의 요지입니다.**

- **[2] 박스 필터** — 이미지의 글자 밀도가 높아(1000px 기준 연결성분 중앙값 88개,
  텍스트 20~60줄) 검출된 텍스트를 전부 인식하면 어떤 모델을 써도 예산을 초과합니다.
  인식은 크롭당 비용이 붙으므로, 인식기에 넘기기 전에 값싼 기하 신호로 후보를 줄입니다.
  튜닝 목표는 캐스케이드 규칙 그대로 — 싼 단계는 재현율을 거의 1로 두고 정밀도는
  뒤 단계가 회수합니다.
- **[5] 선별** — 이미지의 71%에 날짜 모양 오답(품목보고번호·바코드·전화번호·특허번호·
  로트코드)이 있습니다. 이 과제의 본질은 인식이 아니라 **선별**입니다. 가장 효과가 큰
  단일 규칙은 "날짜 숫자가 더 긴 숫자열의 일부이면 기각"이며, 이는 `20130628332176`
  처럼 앞 8자리가 유효 날짜인 품목보고번호를 한 번에 제거합니다.

설계 근거 전문은 [`docs/PIPELINE.md`](docs/PIPELINE.md), 인용 문헌은
[`docs/선행연구.md`](docs/선행연구.md)에 있습니다.

## 4. 저장소 구조

```
predict.ipynb        채점 대상 노트북 (CONFIG 셀은 템플릿 원문 그대로)
requirements.txt     == 로 고정
itda_ocr/            파이프라인 (import 시 부작용 없음)
  engine.py            엔진 래퍼, 스레드 고정, 박스 필터
  pipeline.py          이미지 로딩, 장당 처리, 배치 드라이버
  parse.py             정규식 패밀리, 박스 병합, 부분 추출
  select.py            선별 하드 룰 캐스케이드
eval/                자체 채점 하네스 (채점 대상 아님)
  score.py             공식 산식 + 오류 분류표
  expdate.py           ExpDate 데이터셋 → 정답 CSV 어댑터
  run.py               실행 → 채점 → 리포트
bench/timing.py      장당 비용 측정 (코어·스레드 고정)
tests/               단위 테스트
docs/                설계 문서와 선행연구
```

## 5. 개발용

```bash
pip install -r requirements-dev.txt             # 검증용(pytest). 채점 requirements 와 분리
python -m pytest tests/ -q                       # 단위 테스트
python -m bench.timing --images data --limit 60 # 장당 비용
python -m eval.run --images <dir> --gt <gt.csv>                              # 기준선 정확도
python -m eval.run --images <dir> --gt <gt.csv> --nanodet weights/date_detector_ema.onnx  # NanoDet 검출
python -m eval.failure_taxonomy --images <dir> --gt-dates <csv> --gt-boxes <json>  # 단계별 오답 분류
```

NanoDet 검출기 도입 전후(36.85 → 43.47)를 처음부터 재현하는 절차: [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

정답 CSV는 출처를 가리지 않습니다 — 스키마(`image_id,year,month,day,final_date`)만
같으면 ExpDate 어댑터 산출물이든 손으로 라벨링한 파일이든 그대로 채점됩니다.

## 6. 외부 데이터셋

성능 검증에 **ExpDate (Products-Real)** 를 사용합니다.

> Seker, A. C., & Ahn, S. C. (2022). A generalized framework for recognition of
> expiration dates on product packages using fully convolutional networks.
> *Expert Systems with Applications*, 203, 117310. KIST. CC BY 4.0.
> https://felizang.github.io/expdate/

데이터셋은 저장소에 포함하지 않습니다(`.gitignore`). 변환:

```bash
python -m eval.expdate --root <ExpDate 압축 해제 경로> --inspect   # 구조 먼저 확인
python -m eval.expdate --root <ExpDate 압축 해제 경로> --out labels/expdate
```
