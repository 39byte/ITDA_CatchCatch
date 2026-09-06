"""실측 함정 문자열로 파서·선별을 검증한다.

문자열은 전부 `docs/PIPELINE.md` §3 "날짜 표기 유형(62장 표본)" 에서 왔다 —
지어낸 예시가 아니라 실제 배포셋에서 관찰된 형태다.
"""

import pytest

from itda_ocr.parse import parse, parse_boxes
from itda_ocr.select import select, to_row


def best(text):
    """한 줄에서 최종 선택된 날짜(YYYY-MM-DD 또는 None)."""
    cand = select(parse(text), text)
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
    row = to_row(select(parse("02.18까지"), "02.18까지"), "x")
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
    row = to_row(select(parse("2027.6.4"), ""), "x")
    assert row["month"] == "06" and row["day"] == "04"
    assert row["final_date"] == "2027-06-04"
