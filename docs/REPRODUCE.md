# 재현 — NanoDet 검출기 도입 전후 (36.85 → 40.64 / 50)

## 변경 이력

| 단계 | 무엇을 | 점수 |
| --- | --- | --- |
| 0 | 기준선 (RapidOCR 범용 텍스트 검출) | 36.85 |
| 1 | Track B: NanoDet 날짜 전용 검출기 도입 (thr 0.05) | 39.71 |
| 2 | + 병합-중복 후보 제거(`_drop_overlapping_duplicates`), thr 0.10 | 40.27 |
| 3 | + 구분자 없는 6자리 패턴(`ymd6`/`dmy6`) | **40.64** |

아래 1~5 는 **현재 코드(3단계, 40.64)** 기준이다. 이전 단계 숫자를 그대로 재현하려면
해당 커밋으로 `git checkout` 한 뒤 같은 명령을 돌리면 된다.

ExpDate(Products-Real) 데이터셋을 이미 공유받았다고 가정한다. **다운로드 절차는 생략.**
아래 폴더 구조로 배치한 뒤 명령을 순서대로 실행하면 개선폭이 그대로 재현된다.

## 0. 데이터 배치

저장소 루트 기준으로 이렇게 놓는다 (`expdate/` 는 `.gitignore` 대상):

```
ITDA_CatchCatch/
├── expdate/
│   ├── evaluation/
│   │   ├── images/            # test_00001.jpg … test_00665.jpg  (665장)
│   │   └── annotations.json   # dmy_ann 이 든 주석 파일
│   └── train/                 # (선택) 검출기 재학습용. 성능 재현에는 불필요
│       ├── images/            # img_00001.jpg … img_01102.jpg
│       └── annotations.json
└── weights/
    └── date_detector_ema.onnx # 저장소에 포함됨 (force-add, 5.6MB)
```

- `evaluation/` 만 있으면 아래 1~5 를 전부 돌릴 수 있다.
- 압축 해제 결과가 `expdate/Products-Real/evaluation/...` 처럼 한 단계 더 들어가 있으면,
  `evaluation` 폴더를 `expdate/evaluation/` 로 옮기거나 아래 명령의 경로를 그에 맞게 바꾼다.

## 1. 의존성 설치

```bash
pip install -r requirements.txt
```

Python 3.10 / CPU. 새 의존성 없음 — NanoDet 추론은 이미 고정된 `onnxruntime` +
`opencv-python` + `numpy` 만 쓴다 (torch·nanodet 패키지 불필요).

## 2. 정답 CSV 생성

```bash
python -m eval.expdate --root expdate/evaluation --out labels/expdate
```

→ `labels/expdate/gt_dates.csv` (665행), `labels/expdate/gt_boxes.json` 생성.
결정적이다 — 몇 번을 돌려도 같은 파일이 나온다.

## 3. 기준선 (RapidOCR 범용 텍스트 검출)

```bash
python -m eval.run --images expdate/evaluation/images --gt labels/expdate/gt_dates.csv \
    --out results/baseline
```

**기대 출력:** `정확도 점수  36.85 / 50   (73.7%)` · `완전일치  70.7%`

## 4. 개선본 (NanoDet 날짜 전용 검출 + 병합-중복 제거 + 파싱 패턴 확장)

```bash
python -m eval.run --images expdate/evaluation/images --gt labels/expdate/gt_dates.csv \
    --nanodet weights/date_detector_ema.onnx --nanodet-score-thr 0.10 --out results/b4_nanodet
```

**기대 출력:** `정확도 점수  40.64 / 50   (81.3%)` · `완전일치  78.0%`

→ `--nanodet` 유무만 차이. **+3.79점 / 완전일치 +49장**이 개선폭이다(위 변경 이력 3단계 누적).

## 5. predict.ipynb (채점 대상 노트북)

`weights/date_detector_ema.onnx` 가 있으면 노트북이 자동으로 NanoDet 검출을 쓴다
(CFG 셀에서 파일 존재를 확인해 스스로 선택).

```bash
export ITDA_INPUT_DIR=expdate/evaluation/images
export ITDA_OUTPUT_PATH=./submission.csv
jupyter nbconvert --to notebook --execute predict.ipynb \
    --ExecutePreprocessor.timeout=2400 --output /tmp/executed.ipynb
```

실행 로그의 `engine ready ... | 검출: NanoDet 날짜검출기` 로 NanoDet 경로가 켜진 것을 확인한다.
생성된 `submission.csv` 를 채점하면 4번과 같은 점수(40.64)가 나온다:

```bash
python -m eval.run --predictions ./submission.csv --gt labels/expdate/gt_dates.csv \
    --out results/predict_check
```

---

## (선택) 검출기 자체의 recall 비교

torch·nanodet 저장소 없이 `weights/date_detector_ema.onnx` 만으로 돈다.

```bash
# 현행 RapidOCR 검출기의 박스
python -m eval.detector_ab --baseline --images expdate/evaluation/images \
    --out results/det_baseline.json
# NanoDet 검출기의 박스
python -m tools.nanodet_predict --model weights/date_detector_ema.onnx \
    --images expdate/evaluation/images --out results/det_nanodet.json
# 비교
python -m eval.detector_ab --pred results/det_nanodet.json --compare results/det_baseline.json
```

→ 검출 recall (IoU 0.3): **82.9% → ~97.7%**, recall@1: **26.5% → ~85%**.

## (선택) 단계별 오답 분류

```bash
python -m eval.failure_taxonomy --images expdate/evaluation/images \
    --gt-dates labels/expdate/gt_dates.csv --gt-boxes labels/expdate/gt_boxes.json \
    --out results/failure_taxonomy.json
```

→ 검출 실패 / 순위 실패 / 파싱 실패 / 오독 / 정답 5분류 (검출기 박스와 GT 박스 대조).
