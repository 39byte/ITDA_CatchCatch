"""후보 날짜 중 **소비기한 하나**를 고른다 — 이 과제 정확도의 핵심.

이미지의 71%에 날짜 모양 오답이 있다(품목보고번호·바코드·전화번호·특허번호·
로트코드·`SINCE 1986`). 즉 이 과제의 본질은 인식이 아니라 **선별**이다.

규칙은 **순서가 곧 정확도**다. 학습 모델이 아니라 손으로 짠 캐스케이드를 쓰는 이유:
심사자가 판정 근거를 그대로 읽을 수 있고(정성 25점), 300장 규모에서 LightGBM은
노이즈를 학습한다. 선행 근거로 CloudScan(ICDAR 2017)에서 수작업 특징 위의 로지스틱
회귀가 326k 문서에서 LSTM과 0.4점 차였고, SROIE 2019 우승 방법도 규칙 기반이었다.
"""

from __future__ import annotations

import calendar
import re

from .parse import Candidate

#: 이 키워드 근처의 날짜는 소비기한이 **아니다**.
NEGATIVE = (
    "품목보고번호", "품목보고", "보고번호", "제조일자", "제조년월", "제조",
    "생산일자", "생산", "등록번호", "허가번호", "전화", "상담", "고객", "문의",
    "LOT", "L0T", "MFG", "MFD", "PROD", "BATCH", "SINCE", "TEL", "FAX",
)

#: 이 키워드 근처의 날짜는 소비기한일 **가능성이 높다**.
POSITIVE = (
    "소비기한", "유통기한", "품질유지기한", "까지", "기한", "유통", "소비",
    "EXP", "EXPIRY", "EXPIRES", "BBE", "BBD", "BEST BEFORE", "BEST BY",
    "USE BY", "USEBY", "정확한", "표시일",
)

#: 패턴별 신뢰도 사전 — 4자리 연도가 앵커된 형태가 가장 믿을 만하다.
PATTERN_PRIOR = {
    "korean": 12, "ymd4": 10, "dmy4": 8, "mon_d_y": 8, "d_mon_y": 8,
    "ymd8": 4,          # 구분자가 없어 품목보고번호와 가장 헷갈린다
    "ymmd": 4,          # 202112.16 — 연·월이 붙은 형태
    # 완화 패턴들. 엄격한 해석이 하나도 없을 때에만 이기도록 낮게 둔다.
    "d_fuzz_y": 1, "fuzz_y": -3, "m_y": -1,
    # 2자리 연도는 앞뒤 해석이 모두 유효할 때가 많다(`22.04.30`).
    # 국내·아시아권 인쇄는 YY.MM.DD 가 지배적이므로 그쪽을 확실히 선호한다.
    "ymd2": 3, "dmy2": -1,
    "mon_y": 2, "ym4": 1,   # 일자 없음 — 부분 점수용
    "md": -2, "dm": -4,     # 연도 없음 — 가장 약하다
}

#: 실측상 촬영이 가을·겨울(2025-10/12)에 몰려 있다. 연도가 안 찍힌 날짜는
#: 촬영 시점보다 뒤이므로, 상반기 날짜는 이듬해로 보는 편이 기대값이 높다.
ANCHOR_YEAR = 2026

#: 일자가 없을 때(`OCT. 2021`) 라벨이 스스로 규칙을 밝히는 경우가 있다.
#: 실물 `001955.jpg` 는 "월/년의 01일까지" 라고 적혀 있었다 — "말일" 가정은 반증됐다.
_DAY_RULE_LAST = re.compile(r"말일|마지막\s*날|end\s+of\s+(the\s+)?month", re.I)
_DAY_RULE_FIRST = re.compile(r"0?1\s*일\s*까지|1st\s+of", re.I)


def _has(text: str, words) -> bool:
    upper = text.upper()
    return any(w.upper() in upper for w in words)


def score(cand: Candidate, full_text: str = "") -> float:
    """후보 점수. 값이 클수록 소비기한일 가능성이 높다.

    가중치는 규칙의 **순위**를 표현한다 — 1번이 나머지 전부를 이기도록 설계했다.
    """
    s = 0.0

    # 1. 품목보고번호 킬러 — 단일 최대 효과 규칙.
    #    20130628332176 처럼 앞 8자리가 유효 날짜인 번호를 한 번에 제거한다.
    #    다른 어떤 신호로도 뒤집히지 않도록 압도적인 음수를 준다.
    if cand.embedded:
        s -= 1000

    # 2·3. 키워드 — 후보가 나온 '줄 전체'가 문맥이다. 검출 박스를 가로로 병합해
    #      두었으므로 같은 가로 밴드의 멀리 떨어진 라벨도 여기 들어온다.
    if _has(cand.context, NEGATIVE):
        s -= 60
    if _has(cand.context, POSITIVE):
        s += 40

    # 4. 형식 신뢰도
    s += PATTERN_PRIOR.get(cand.pattern, 0)

    # 5. 완전한 날짜를 불완전한 것보다 선호 (35점짜리 final_date가 걸려 있다)
    if cand.complete:
        s += 15

    return s


def _impute_day(cand: Candidate, full_text: str) -> str:
    """일자가 없을 때(`OCT. 2021`) 채워 넣는다. NONE이면 0점, 채우면 최소 10점."""
    if _DAY_RULE_LAST.search(full_text):
        return f"{calendar.monthrange(int(cand.year), int(cand.month))[1]:02d}"
    return "01"   # 기본값. '말일' 가정은 실물로 반증됐다 — §PIPELINE 보정표


def _impute_year(cand: Candidate) -> str:
    """연도가 없을 때(`02.18까지`) 추정한다.

    EXIF 촬영일시 폴백은 쓰지 않는다 — 확인된 결측 사례 2건(`000995`,`001955`)
    **모두 EXIF에 촬영일시가 없었다.** 발동하지 않는 분기다.
    """
    month = int(cand.month) if cand.month else 12
    return str(ANCHOR_YEAR + 1 if month <= 6 else ANCHOR_YEAR)


def impute(cand: Candidate, full_text: str = "") -> Candidate:
    """빠진 필드를 채운다. 기권(NONE)은 0점이고 보정은 최소 10점이다."""
    year, month, day = cand.year, cand.month, cand.day
    if year and month and not day:
        day = _impute_day(cand, full_text)
    elif month and day and not year:
        year = _impute_year(cand)
    if (year, month, day) == (cand.year, cand.month, cand.day):
        return cand
    # 보정한 날짜가 달력상 불가능하면(2월 31일) 원본을 그대로 둔다.
    if year and month and day and int(day) > calendar.monthrange(int(year), int(month))[1]:
        return cand
    return Candidate(**{**cand.__dict__, "year": year, "month": month, "day": day})


def rank(candidates, full_text: str = "") -> list[tuple[float, Candidate]]:
    """점수 내림차순. 동점이면 **나중 날짜 우선**(제조/소비 쌍은 약 10%)."""
    scored = [(score(c, full_text), c) for c in candidates]
    scored.sort(key=lambda p: (p[0], p[1].final_date or "", p[1].year or ""), reverse=True)
    return scored


#: 이 점수 아래는 채택하지 않는다. 실질적으로 `embedded`(품목보고번호 계열)만
#: 걸리도록 잡았다 — 나머지 감점을 전부 합쳐도 여기까지 내려가지 않는다.
MIN_SCORE = -500

#: 크롭을 더 읽을지 말지의 기준. 이 점수 이상이면 "충분히 확신한다"로 보고 멈춘다.
#:
#: ⚠️ **"완전한 날짜가 하나라도 나오면 멈춘다"로 두면 안 된다.** 완화 패턴이
#: 쓰레기를 완전한 날짜로 파싱해 조기 종료를 유발하고, 진짜 날짜가 든 크롭을
#: 읽기 전에 멈춰버린다(ExpDate 실측에서 완전일치는 올랐는데 총점이 내려갔다).
#: 완전(+15) + 강한 형식 사전확률(≥4)만 이 문턱을 넘는다.
STOP_SCORE = 19


def select(candidates, full_text: str = "", impute_missing: bool = False) -> Candidate | None:
    """소비기한 하나를 고르고 빠진 필드를 보정한다. 후보가 없으면 None.

    **불확실하다고 기권하지는 않는다.** 산식이 ``year 5 + month 5 + day 5 +
    final 35`` 이므로 연도 하나만 맞아도 5점이고 추측의 기대값은 양수다.

    ``impute_missing`` 은 **False로 확정됐다.** 인쇄물에 일자가 없으면 정답도
    ``NONE`` 이라는 규칙이 확인됐다(대회 요구사항). 따라서 빈 칸을 지어내면 안 된다.

    실측도 같은 방향이다 (ExpDate 665장): 정답에 일자가 없을 때 NONE을 그대로 내면
    네 칼럼이 모두 맞아 **50점**인데 ``01`` 을 채우면 일자·final이 함께 틀려 **10점**이
    된다(−40). 정답에 일자가 있으면 채우든 말든 10점으로 같고 우연히 맞을 확률은
    1/31뿐이다. 불완전 정답이 6.5%(43/665)로 손익분기(≈1/30)를 크게 넘어,
    보정이 **50점 만점에 1.02점을 깎고 있었다.** 규칙과 실측이 일치한다.

    다만 ``embedded``(더 긴 숫자열의 일부 = 품목보고번호·바코드 계열)는 *불확실한*
    후보가 아니라 *거의 확실히 틀린* 후보다. 내도 0점, 안 내도 0점이지만 —
    내면 오류 분류표가 "선별 실패"를 "인식 실패"로 오독하게 만든다. 하드 기각한다.
    """
    ranked = rank(candidates, full_text)
    if not ranked or ranked[0][0] < MIN_SCORE:
        return None
    best = ranked[0][1]
    return impute(best, full_text) if impute_missing else best


def to_row(cand: Candidate | None, image_id: str) -> dict:
    """제출 스키마 한 행. 미인식은 문자열 ``NONE``."""
    if cand is None:
        return {"image_id": image_id, "year": "NONE", "month": "NONE",
                "day": "NONE", "final_date": "NONE"}
    return {
        "image_id": image_id,
        "year": cand.year or "NONE",
        "month": cand.month or "NONE",
        "day": cand.day or "NONE",
        "final_date": cand.final_date or "NONE",
    }
