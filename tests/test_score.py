"""채점기 검증 — 데이터도 모델도 없이 산식 구현이 맞는지 증명한다."""

import pytest

from eval.score import MAX_SCORE, classify, score, score_row

GT = {"year": "2027", "month": "06", "day": "24", "final_date": "2027-06-24"}
NONE_ROW = dict.fromkeys(("year", "month", "day", "final_date"), "NONE")


def _pred(**overrides):
    return {**GT, **overrides}


@pytest.mark.parametrize(
    "pred, expected, why",
    [
        (_pred(), 50, "전부 정답"),
        (_pred(day="25", final_date="2027-06-25"), 10, "일 하나 틀림 → 5+5"),
        (_pred(month="07", final_date="2027-07-24"), 10, "월 하나 틀림"),
        (_pred(year="2028", final_date="2028-06-24"), 10, "연 하나 틀림"),
        (_pred(month="07", day="25", final_date="2027-07-25"), 5, "둘 틀림 → 5"),
        (_pred(year="2028", month="07", day="25", final_date="2028-07-25"), 0, "셋 틀림"),
        (NONE_ROW, 0, "전부 NONE"),
        # 부분 추출 경로: 일자만 못 읽어도 10점이 남는다 — 기권하면 0점이다.
        (_pred(day="NONE", final_date="NONE"), 10, "연·월만"),
        (_pred(month="NONE", day="NONE", final_date="NONE"), 5, "연만"),
    ],
)
def test_known_score_table(pred, expected, why):
    assert score_row(pred, GT) == expected, why


def test_final_date_dominates():
    """final_date 하나가 35점 — 나머지 셋을 합친 것보다 크다."""
    ymd_only = _pred(final_date="NONE")
    final_only = {**NONE_ROW, "final_date": GT["final_date"]}
    assert score_row(ymd_only, GT) == 15
    assert score_row(final_only, GT) == 35


def test_normalisation_of_missing_cells():
    """빈 문자열·nan·소문자 none은 전부 NONE과 같게 취급한다."""
    for blank in ("", "  ", "nan", "none", None):
        row = {**NONE_ROW, "year": blank}
        assert score_row(row, NONE_ROW) == MAX_SCORE


def test_zero_padding_is_significant():
    """'6' != '06'. 제출 스키마는 2자리 문자열을 요구한다."""
    assert score_row(_pred(month="6"), GT) == 45


# ── 오류 분류 ──────────────────────────────────────────────────────────────

def test_classify_correct():
    assert classify(_pred(), GT, [{"text": "2027.06.24", "final_date": "2027-06-24"}]) == "correct"


def test_classify_no_candidate():
    assert classify(NONE_ROW, GT, []) == "no_candidate"


def test_classify_wrong_candidate():
    """정답이 후보에 있었는데 제조일자를 골랐다 → 선별 룰 문제."""
    candidates = [
        {"text": "2025.06.24", "final_date": "2025-06-24"},
        {"text": "2027.06.24", "final_date": "2027-06-24"},
    ]
    pred = _pred(year="2025", final_date="2025-06-24")
    assert classify(pred, GT, candidates) == "wrong_candidate"


def test_classify_parse_fail():
    """원문에는 정답이 있는데 파싱이 못 했다 → 정규식 문제."""
    candidates = [{"text": "2027.06.24E", "final_date": None}]
    assert classify(NONE_ROW, GT, candidates) == "parse_fail"


def test_classify_parse_fail_two_digit_year():
    candidates = [{"text": "27.06.24까지", "final_date": None}]
    assert classify(NONE_ROW, GT, candidates) == "parse_fail"


def test_classify_misread():
    """후보는 있었지만 정답 숫자가 어디에도 없다 → 인식 문제."""
    candidates = [{"text": "2027.05.24", "final_date": "2027-05-24"}]
    assert classify(NONE_ROW, GT, candidates) == "misread"


def test_classify_partial_beats_other_buckets():
    """연·월만 맞은 건 실패가 아니라 10점짜리 성공이다."""
    pred = _pred(day="NONE", final_date="NONE")
    assert classify(pred, GT, [{"text": "2027.06.2", "final_date": None}]) == "partial"


# ── 집계 ───────────────────────────────────────────────────────────────────

def test_score_aggregates_and_counts_missing_as_zero():
    gt = {"a": GT, "b": GT, "c": GT}
    preds = {"a": _pred(), "b": _pred(day="25", final_date="2027-06-25")}  # 'c' 누락
    result = score(preds, gt)

    assert result["n"] == 3
    assert result["score"] == pytest.approx((50 + 10 + 0) / 3)
    assert result["exact_match"] == pytest.approx(1 / 3)
    assert result["missing_predictions"] == 1


def test_score_ignores_extra_predictions():
    result = score({"a": _pred(), "zzz": _pred()}, {"a": GT})
    assert result["n"] == 1 and result["score"] == 50
