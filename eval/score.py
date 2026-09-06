"""ITDA 공식 정확도 산식과 오류 분류.

산식(운영진 확정): ``year 5 + month 5 + day 5 + final_date 35`` = 장당 최대 50점.
최종 점수는 장당 점수의 **평균**이다. 따라서 장당 점수표는

    전부 정답 50 / 하나 틀림 10 / 둘 틀림 5 / 셋 이상 틀림 0

완전일치가 70%(35/50)로 지배적이지만 부분점수 15점(30%)이 실재한다 —
연·월만 맞혀도 10점이므로 파이프라인은 절대 기권하지 않고 부분 결과를 낸다.

GT는 ExpDate 어댑터가 만들든 손으로 라벨링하든 같은 CSV 스키마를 쓴다:
``image_id,year,month,day,final_date`` (미인식은 문자열 ``NONE``).
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

FIELDS = ("year", "month", "day", "final_date")
WEIGHTS = {"year": 5, "month": 5, "day": 5, "final_date": 35}
MAX_SCORE = 50
NONE = "NONE"

#: 오류 분류 → 그 분류가 지시하는 조치. 리포트에 그대로 찍는다.
TAXONOMY = {
    "correct": "—",
    "partial": "부분 추출 경로 (연·월만 정답, 10점 확보)",
    "no_candidate": "검출 / 박스 필터 재현율",
    "misread": "인식기 · 크롭 전처리",
    "wrong_candidate": "선별 하드 룰 (품목보고번호 · 제조일자)",
    "parse_fail": "정규식 패밀리",
}


def _norm(value) -> str:
    """CSV 셀을 비교 가능한 문자열로. 결측·NaN·빈칸은 전부 NONE으로 접는다.

    pandas로 읽으면 ``"05"``가 정수 5가 되는 사고가 잦아 str 강제 후 비교한다.
    """
    if value is None:
        return NONE
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return NONE
    return text


def _digits(value) -> str:
    return re.sub(r"\D", "", _norm(value))


def score_row(pred, gt) -> int:
    """장당 점수(0~50). ``pred``/``gt``는 FIELDS를 키로 갖는 매핑."""
    return sum(w for f, w in WEIGHTS.items() if _norm(pred.get(f)) == _norm(gt.get(f)))


def _text_holds_gt(text: str, gt) -> bool:
    """후보 원문이 정답 날짜를 담고 있는가 (파싱만 실패했는지 판별용).

    ``2021-10-01`` 은 ``20211001`` 로도, 2자리 연도 ``211001`` 로도 인쇄된다.
    """
    hay = _digits(text)
    full = _digits(gt.get("final_date"))
    if not hay or not full:
        return False
    return full in hay or (len(full) == 8 and full[2:] in hay)


def classify(pred, gt, candidates=()) -> str:
    """틀린 원인을 하나로 분류한다.

    ``candidates`` 는 Stage 4가 만든 날짜 후보 전부:
    ``[{"text": <인식 원문>, "final_date": <파싱 결과 또는 None>}, ...]``.
    GT 박스가 없어도 동작한다 — 정답이 *후보 목록*에 있었는지만 보면
    "못 찾았다"와 "찾고도 잘못 골랐다"가 갈리기 때문이다.
    """
    if score_row(pred, gt) == MAX_SCORE:
        return "correct"

    if (
        _norm(pred.get("day")) == NONE
        and _norm(pred.get("year")) == _norm(gt.get("year"))
        and _norm(pred.get("month")) == _norm(gt.get("month"))
    ):
        return "partial"

    if not candidates:
        return "no_candidate"

    gt_final = _norm(gt.get("final_date"))
    if any(_norm(c.get("final_date")) == gt_final for c in candidates):
        return "wrong_candidate"

    if any(_text_holds_gt(c.get("text", ""), gt) for c in candidates):
        return "parse_fail"

    return "misread"


def load_csv(path) -> dict:
    """제출 스키마 CSV → ``{image_id: row}``. 전 셀을 문자열로 유지한다."""
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        return {r["image_id"]: r for r in csv.DictReader(fh)}


def score(predictions, ground_truth, candidates_by_id=None) -> dict:
    """GT에 있는 이미지 전부에 대해 채점한다.

    GT에 있으나 예측에 없는 이미지는 **전부 NONE으로 간주해 0점** 처리한다 —
    채점팀도 그렇게 본다. 예측에만 있고 GT에 없는 행은 무시한다.
    """
    candidates_by_id = candidates_by_id or {}
    per_image, buckets = {}, Counter()

    for image_id, gt in ground_truth.items():
        pred = predictions.get(image_id, {})
        per_image[image_id] = score_row(pred, gt)
        buckets[classify(pred, gt, candidates_by_id.get(image_id, ()))] += 1

    n = len(ground_truth)
    total = sum(per_image.values())
    return {
        "n": n,
        "score": total / n if n else 0.0,          # 만점 50
        "exact_match": buckets["correct"] / n if n else 0.0,
        "missing_predictions": sum(1 for i in ground_truth if i not in predictions),
        "buckets": dict(buckets),
        "per_image": per_image,
    }


def format_report(result: dict) -> str:
    """사람이 읽는 리포트. 오류 분류표가 다음에 뭘 고칠지를 그대로 지시한다."""
    n = result["n"]
    lines = [
        f"이미지 {n}장",
        f"정확도 점수  {result['score']:.2f} / 50   ({result['score'] / 50:.1%})",
        f"완전일치     {result['exact_match']:.1%}",
    ]
    if result["missing_predictions"]:
        lines.append(f"⚠️ 예측 누락 {result['missing_predictions']}장 (0점 처리)")

    lines += ["", f"{'분류':<17}{'장수':>6}{'비중':>8}   지시하는 조치"]
    for name in TAXONOMY:
        count = result["buckets"].get(name, 0)
        if count:
            lines.append(f"{name:<17}{count:>6}{count / n:>8.1%}   {TAXONOMY[name]}")
    return "\n".join(lines)
