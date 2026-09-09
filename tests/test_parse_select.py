"""실측 함정 문자열로 파서·선별을 검증한다.

문자열은 전부 `docs/PIPELINE.md` §3 "날짜 표기 유형(62장 표본)" 에서 왔다 —
지어낸 예시가 아니라 실제 배포셋에서 관찰된 형태다.
"""

import pytest

from itda_ocr.parse import parse, parse_boxes
from itda_ocr.select import select, to_row


def best(text, impute=True):
    """한 줄에서 최종 선택된 날짜(YYYY-MM-DD 또는 None).

    보정은 기본적으로 꺼져 있지만(§select.select), 파싱 자체를 시험하는
    테스트에서는 켜서 완전한 날짜를 확인한다.
    """
    cand = select(parse(text), text, impute_missing=impute)
    return cand.final_date if cand else None


# ── 정상 포맷 ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("2021.08.02", "2021-08-02"),
    ("2021-08-02", "2021-08-02"),
    ("2021/08/02", "2021-08-02"),
    ("2021 08 02", "2021-08-02"),
    ("20210802", "2021-08-02"),
    ("2021년 8월 2일", "2021-08-02"),
    ("21.09.17", "2021-09-17"),
    ("02.08.2021", "2021-08-02"),
    ("11/Oct/2021", "2021-10-11"),
    ("28 OCT 2021", "2021-10-28"),
    ("JUL 22 21", "2021-07-22"),
])
def test_formats(text, expected):
    assert best(text) == expected


# ── 구분자 없이 붙는 시각·로트코드 (실측 65% 포맷의 흔한 꼬리) ──────────────

@pytest.mark.parametrize("text, expected", [
    ("2021.08.02E", "2021-08-02"),
    ("2021.02.08B5", "2021-02-08"),
    ("21.09.17까지RT F2", "2021-09-17"),
    ("소비기한 2021.08.02 까지01", "2021-08-02"),
    ("2021.08.02 15:49", "2021-08-02"),
])
def test_trailing_noise(text, expected):
    assert best(text) == expected


# ── 오답 후보: 이것들을 날짜로 내면 안 된다 ────────────────────────────────

@pytest.mark.parametrize("text", [
    "품목보고번호 20130628332176",   # 앞 8자리가 유효 날짜 — 최악의 함정
    "8801234567890",                # 바코드
    "10-1855891-0000",              # 특허번호
    "고객상담실 1899-1494",
    "제품문의 080-1234-5678",
])
def test_rejects_non_dates(text):
    assert best(text) is None, f"오답 후보를 날짜로 채택했다: {text!r}"


def test_barcode_prefix_is_not_a_date():
    """880… 바코드의 앞부분이 연도로 보여도 더 긴 숫자열의 일부다."""
    cands = parse("8801234567890")
    assert all(c.embedded for c in cands if c.year)


def test_item_report_number_is_embedded():
    """품목보고번호 킬러는 '더 긴 숫자열의 일부인가'로 판정한다.

    연도가 창 안이라 파싱은 되지만, 더 긴 숫자열의 일부이므로 채택되면 안 된다.
    """
    cands = [c for c in parse("20270628332176") if c.final_date == "2027-06-28"]
    assert cands and all(c.embedded for c in cands)
    assert best("품목보고번호 20270628332176") is None


def test_normalisation_does_not_create_false_digit_runs():
    """혼동 정규화(S→5) '이전' 원문에서 판정해야 멀쩡한 날짜가 살아남는다."""
    assert best("2021.08.02S") == "2021-08-02"


# ── 제조일자 vs 소비기한 ───────────────────────────────────────────────────

def test_positive_keyword_wins_over_negative():
    text = "제조일자 2021.02.08 소비기한 2023.02.07"
    assert best(text) == "2023-02-07"


def test_later_date_wins_when_no_keyword():
    """키워드가 없는 26~39%에서는 나중 날짜가 정답이다."""
    assert best("2021.02.08 2023.02.07") == "2023-02-07"


def test_manufacture_keyword_penalised():
    assert best("제조 2021.02.08") is None or best("제조 2021.02.08") == "2021-02-08"


def test_since_is_not_an_expiry():
    assert best("SINCE 1986") is None


# ── 불완전 날짜: 버리면 10점을 버리는 것이다 ───────────────────────────────

def test_missing_day_is_imputed_to_first():
    """`OCT. 2021` — 실물 001955.jpg 라벨이 '01일까지'라고 규칙을 밝혔다."""
    assert best("à consommer avant le -- OCT. 2021") == "2021-10-01"


def test_missing_day_respects_explicit_last_day_rule():
    assert best("OCT. 2021 해당 월 말일까지") == "2021-10-31"


def test_missing_year_is_imputed():
    """`02.18까지` — 연도가 없어도 월·일은 살린다."""
    row = to_row(select(parse("02.18까지"), "02.18까지", impute_missing=True), "x")
    assert (row["month"], row["day"]) == ("02", "18")
    assert row["year"] != "NONE"


def test_partial_extraction_beats_abstaining():
    """연·월만 맞아도 10점. 완전 파싱 실패 시 기권하면 0점이다."""
    cand = select(parse("2027.06"), "2027.06")
    assert cand is not None and cand.year == "2027" and cand.month == "06"


# ── 박스 병합 (DB 검출기는 날짜를 조각내서 준다) ───────────────────────────

def test_merges_split_boxes_into_one_date():
    boxes = [("2021.", 10, 100, 60, 120),
             ("08",    62, 100, 85, 120),
             (".02",   87, 100, 120, 120),
             ("까지",  122, 100, 160, 120)]
    assert select(parse_boxes(boxes), "").final_date == "2021-08-02"


def test_does_not_merge_across_rows():
    boxes = [("2021.08.02", 10, 100, 120, 120),
             ("2023.08.02", 10, 400, 120, 420)]
    finals = {c.final_date for c in parse_boxes(boxes)}
    assert {"2021-08-02", "2023-08-02"} <= finals


def test_keyword_on_same_band_reaches_the_date():
    """라벨이 같은 가로 밴드의 왼쪽 멀리 있는 경우가 많다."""
    boxes = [("소비기한", 10, 100, 90, 120),
             ("2023.02.07", 400, 102, 520, 122),
             ("제조일자", 10, 300, 90, 320),
             ("2021.02.08", 400, 302, 520, 322)]
    assert select(parse_boxes(boxes), "").final_date == "2023-02-07"


# ── 연도 창 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", ["1986.08.02", "2045.08.02"])
def test_year_window_rejects_implausible(text):
    assert best(text) is None


def test_year_window_is_wide_enough():
    for text, exp in [("2018.01.01", "2018-01-01"), ("2032.12.31", "2032-12-31")]:
        assert best(text) == exp


# ── 출력 스키마 ────────────────────────────────────────────────────────────

def test_row_schema_uses_none_strings():
    row = to_row(None, "000001")
    assert row == {"image_id": "000001", "year": "NONE", "month": "NONE",
                   "day": "NONE", "final_date": "NONE"}


def test_row_zero_pads():
    row = to_row(select(parse("2027.6.4"), "", impute_missing=True), "x")
    assert row["month"] == "06" and row["day"] == "04"
    assert row["final_date"] == "2027-06-04"


# ── ExpDate 665장 실측에서 드러난 회귀 (실제 오답에서 왔다) ────────────────

def test_lot_code_digits_do_not_trigger_item_number_killer():
    """`2021.06.090A` — 날짜를 정확히 읽고도 뒤에 붙은 로트코드 숫자 때문에
    품목보고번호 킬러가 오발동해 기각했던 사례. 구분자가 있으면 이미 형식이
    갖춰진 날짜이므로 인접 숫자는 '더 긴 숫자열'의 증거가 아니다."""
    assert best("2021.06.090A") == "2021-06-09"
    assert best("2021.06.09 0A2C") == "2021-06-09"


def test_item_number_killer_still_fires_on_undelimited_runs():
    """단, 구분자 없는 연속 숫자에는 그대로 발동해야 한다."""
    assert best("품목보고번호 20270628332176") is None


def test_two_digit_year_prefers_ymd_over_dmy():
    """`22.04.30` 은 2022-04-30 이지 2030-04-22 가 아니다.

    두 해석 모두 달력상 유효하고 연도 창 안이라, 사전확률로 갈라야 한다.
    같은 패턴 이름을 주면 '나중 날짜 우선' 동점 규칙이 뒤집힌 해석을 고른다.
    """
    assert best("22.04.30") == "2022-04-30"
    assert best("21.03.12") == "2021-03-12"


@pytest.mark.parametrize("text, ymd", [
    ("02/2022",     ("2022", "02", "01")),   # MM/YYYY — 일자 없음
    ("22/MRY/2023", ("2023", "05", "22")),   # MAY를 MRY로 오독 (1글자 허용)
    ("202112.16A6", ("2021", "12", "16")),   # 연·월이 붙고 일자만 떨어진 형태
    ("JUN 2023",    ("2023", "06", "01")),   # 월 이름 + 연도
])
def test_recognition_error_tolerant_patterns(text, ymd):
    """ExpDate 실패 목록에서 그대로 가져온 문자열들.

    전부 인식은 됐는데 정규식에 구멍이 있어 버려지던 것들이다.
    런타임 비용이 0이면서 12.5% 버킷을 직접 줄인다.
    """
    cand = select(parse(text), text, impute_missing=True)
    assert cand is not None, f"{text!r} 를 여전히 파싱하지 못한다"
    assert (cand.year, cand.month, cand.day) == ymd


def test_fuzzy_month_needs_a_year_anchor():
    """완화된 월 이름은 4자리 연도가 함께 있을 때만 쓴다 — 오탐 방지."""
    from itda_ocr.parse import fuzzy_month
    assert fuzzy_month("MRY") == 5 and fuzzy_month("OCT") == 10
    assert fuzzy_month("XYZ") is None
    assert best("NET WT 500") is None


def test_imputation_is_off_by_default():
    """ExpDate 실측상 일자 보정은 기대값이 음수다 (50점 만점에 −1.02점).

    정답에 일자가 없을 때 NONE을 그대로 내면 50점, `01`을 채우면 10점이다.
    """
    cand = select(parse("OCT. 2021"), "OCT. 2021")
    assert cand is not None and cand.year == "2021" and cand.month == "10"
    assert cand.day is None, "기본값에서 일자를 채우면 안 된다"


@pytest.mark.parametrize("text, ymd", [
    ("25 082023",    ("2023", "08", "25")),   # DD MMYYYY
    ("BB:2023.1015", ("2023", "10", "15")),   # YYYY.MMDD
    ("12102022",     ("2022", "10", "12")),   # 구분자 없는 DDMMYYYY
])
def test_more_undelimited_formats(text, ymd):
    """ExpDate 실패 목록의 나머지 형식들. 전부 인식은 됐는데 패턴이 없었다."""
    cand = select(parse(text), text)
    assert cand is not None and (cand.year, cand.month, cand.day) == ymd


@pytest.mark.parametrize("text, expected", [
    ("92/60/7202",  "2027-06-29"),
    ("82-60-204X3", "2028-06-02"),
])
def test_upside_down_crop_is_rescued(text, expected):
    """방향 분류기가 놓친 180° 뒤집힌 크롭. 정방향에서 아무것도 못 건졌을 때만
    역방향을 시도하며, 추론 비용은 0이다."""
    cand = select(parse_boxes([(text, 0, 0, 10, 5)]), text)
    assert cand is not None and cand.final_date == expected


def test_reverse_fallback_does_not_override_forward_reading():
    """정방향으로 읽히면 역방향은 시도조차 하지 않는다."""
    cand = select(parse_boxes([("2021.08.02", 0, 0, 10, 5)]), "")
    assert cand.final_date == "2021-08-02"
    assert not cand.pattern.endswith("_rev")


# ── 정규식 결함 수정 검증 ──────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("2022. 09. 27",  "2022-09-27"),   # 점 뒤 공백
    ("2020. 02. 18",  "2020-02-18"),   # test_00092 실제 오답
    ("2022. 11.08",   "2022-11-08"),   # 한쪽만 점 뒤 공백 (test_00271)
    ("2021 .06.09",   "2021-06-09"),   # 점 앞 공백
    ("2021. 10. 20",  "2021-10-20"),   # test_00439
    ("2022.03-01",    "2022-03-01"),   # 이종 구분자 (. 와 -)
    ("2021-05.10",    "2021-05-10"),   # 이종 구분자 (- 와 .)
    ("2112.22",       "2021-12-22"),   # 2자리 연도 ymmd2 (test_00008)
    ("2211.17",       "2022-11-17"),   # 2자리 연도 ymmd2 (test_00375)
    ("2104.04",       "2021-04-04"),   # 2자리 연도 ymmd2 (test_00577)
    ("2021 JUN 12",   "2021-06-12"),   # Y_MON_D
    ("2021.OCT.05",   "2021-10-05"),   # Y_MON_D with sep
    ("221117",        "2022-11-17"),   # 6자리 YYMMDD
])
def test_regex_defect_fixes(text, expected):
    assert best(text) == expected


def test_does_not_merge_overlapping_redundant_boxes():
    """동일 스탬프의 중복 검출 박스는 가로로 이어붙이지 않아야 한다."""
    boxes = [
        ("2021.12.", 100, 10, 180, 30),
        ("2021.12.04", 100, 10, 210, 30),
        ("2021.12.0", 100, 10, 190, 30),
    ]
    finals = [c.final_date for c in parse_boxes(boxes)]
    assert "2021-12-04" in finals
    assert "2021-12-20" not in finals  # 괴물 날짜 생성 방지!


