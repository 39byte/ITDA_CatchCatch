"""인식된 텍스트 → 날짜 후보.

두 가지 설계 결정이 이 모듈의 정확도를 지배한다.

1. **혼동 문자 정규화는 길이를 보존한다.** ``O→0`` 같은 치환을 1:1로만 하면
   정규화본에서 찾은 매치 위치를 **원문에 그대로 되짚을 수 있다.** 품목보고번호
   판정(§select)이 원문에서 이뤄져야 하기 때문에 이 성질이 필요하다 —
   ``2021.08.02S`` 를 정규화하면 끝의 ``S`` 가 ``5`` 가 되어 "더 긴 숫자열"로
   오판되고, 멀쩡한 날짜가 버려진다.

2. **부분 결과를 버리지 않는다.** 채점 산식이 ``year 5 + month 5 + day 5 +
   final 35`` 이므로 ``OCT. 2021`` 처럼 일자가 없어도 연·월만으로 10점이다.
   완전 매치만 인정하면 그 10점이 0점이 된다.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass

#: 길이 보존 1:1 치환만 담는다. 이 불변식이 깨지면 원문 인덱스가 어긋난다.
CONFUSION = str.maketrans({
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "l": "1", "I": "1", "i": "1", "|": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2", "z": "2",
    "、": ".", "·": ".", ",": ".", "•": ".",
})

MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}
MONTHS["SEPT"] = 9
_MON = "|".join(sorted(MONTHS, key=len, reverse=True))

#: 유효 연도 창. 배포셋의 연도 분포로 좁히면 과적합이다 — 평가셋은 2026~2028 쏠림.
YEAR_MIN, YEAR_MAX = 2018, 2032


def normalize(text: str) -> str:
    """OCR 혼동 문자를 되돌린다. **길이를 바꾸지 않는다.**

    ⚠️ 무조건 치환하면 안 된다 — ``O→0`` 을 그냥 적용하면 ``OCT`` 가 ``0CT`` 가
    되어 **월 이름 패턴이 영원히 매치되지 않는다.** 글자 사이에 낀 글자는 글자로
    두고, 숫자 이웃을 가진 글자만 숫자로 되돌린다.
    """
    chars = list(text)
    last = len(text) - 1
    for i, ch in enumerate(text):
        mapped = CONFUSION.get(ord(ch))
        if mapped is None:
            continue
        prev_alpha = i > 0 and text[i - 1].isalpha()
        next_alpha = i < last and text[i + 1].isalpha()
        if prev_alpha or next_alpha:
            continue          # OCT / DEC / SEP 의 글자를 지키는 분기
        chars[i] = mapped
    return "".join(chars)


@dataclass(frozen=True)
class Candidate:
    """텍스트 한 조각에서 나온 날짜 후보 하나."""

    year: str | None
    month: str | None
    day: str | None
    text: str          # 매치된 원문 조각 (정규화 이전)
    context: str       # 후보가 나온 줄 전체 (키워드 탐색용)
    pattern: str
    embedded: bool     # 더 긴 숫자열의 일부인가 → 품목보고번호 계열
    span: tuple[int, int]
    source: int = 0    # 후보를 만든 박스 인덱스
    #: 검출기confidence × 인식기confidence (0~1). 박스 좌표·confidence 없이
    #: 순수 텍스트로 호출되면(단위테스트 등) 1.0 — 전부 동률이라 기존 순위에
    #: 영향 없다. select.py 에서 **동점자 정리용**으로만 쓴다.
    conf: float = 1.0

    @property
    def final_date(self) -> str | None:
        if self.year and self.month and self.day:
            return f"{self.year}-{self.month}-{self.day}"
        return None

    @property
    def complete(self) -> bool:
        return self.final_date is not None


def fuzzy_month(name: str) -> int | None:
    """월 이름을 1글자 오독까지 허용해 해석한다 (``MRY`` → ``MAY``).

    인식기가 세 글자 중 하나를 놓치는 일이 잦다. 4자리 연도가 함께 앵커된
    자리에서만 쓰므로 오탐 위험은 낮다.
    """
    key = name.upper()[:3]
    if key in MONTHS:
        return MONTHS[key]
    best, best_cost = None, 2
    for cand, num in MONTHS.items():
        cand = cand[:3]
        cost = sum(a != b for a, b in zip(key, cand))
        if len(key) == len(cand) and cost < best_cost:
            best, best_cost = num, cost
    return best


def _year4(value: str) -> str | None:
    """2자리/4자리 연도를 4자리로. 창 밖이면 None."""
    y = int(value)
    if len(value) == 2:
        y += 2000
    return str(y) if YEAR_MIN <= y <= YEAR_MAX else None


def _valid_md(month: int, day: int | None) -> bool:
    if not 1 <= month <= 12:
        return False
    if day is None:
        return True
    return 1 <= day <= 31


def _build(year, month, day, *, raw, span, context, pattern, source, conf=1.0):
    """검증 후 Candidate 생성. 달력상 불가능하면 None."""
    if month is not None and not _valid_md(month, day):
        return None
    if year is not None and day is not None and month is not None:
        # 윤년·소월 검증은 연도가 있어야 가능하다.
        if day > calendar.monthrange(int(year), month)[1]:
            return None
    return Candidate(
        year=year,
        month=f"{month:02d}" if month else None,
        day=f"{day:02d}" if day else None,
        text=raw[span[0]:span[1]],
        context=raw,
        pattern=pattern,
        embedded=_embedded(raw, *span),
        span=span,
        source=source,
        conf=conf,
    )


def _embedded(raw: str, start: int, end: int) -> bool:
    """매치가 **더 긴 숫자열의 일부**인가 — 품목보고번호 킬러의 판정 근거.

    ``20130628332176`` 에서 ``20130628`` 을 잡으면 뒤가 ``3`` 이므로 True.
    **반드시 원문(정규화 이전)에서 판정한다.**

    ⚠️ **구분자가 있는 매치에는 적용하지 않는다.** ExpDate 실측에서
    ``2021.06.090A`` (날짜 뒤에 로트코드가 구분자 없이 붙은 형태)를 정확히 읽고도
    뒤의 ``0`` 때문에 기각해 버렸다. 점이 찍힌 ``2021.06.09`` 는 이미 형식이 갖춰진
    날짜이고, 뒤에 붙은 숫자는 시각·로트코드이지 14자리 번호의 증거가 아니다.
    품목보고번호는 **구분자 없는 연속 숫자**라는 점이 이 둘을 가른다.
    """
    if any(sep in raw[start:end] for sep in ".-/ "):
        return False
    before = raw[start - 1] if start > 0 else ""
    after = raw[end] if end < len(raw) else ""
    return before.isdigit() or after.isdigit()


# ── 패턴 정의 ──────────────────────────────────────────────────────────────
# 구분자는 역참조(\2)로 **일관성을 강제**한다. 그러지 않으면 영양성분표의
# 무작위 숫자쌍이 전부 날짜로 잡힌다.
_SEP = r"[.\-/ ]"

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ymd4",   re.compile(rf"(20\d{{2}})({_SEP})(\d{{1,2}})\2(\d{{1,2}})")),
    ("dmy4",   re.compile(rf"(\d{{1,2}})({_SEP})(\d{{1,2}})\2(20\d{{2}})")),
    ("ymd8",   re.compile(r"(20\d{2})(\d{2})(\d{2})")),
    # `BB:2023.1015` — 연도 뒤에 월·일이 붙어 있는 형태.
    ("y_mmdd", re.compile(r"(20\d{2})[.\-/ ](\d{2})(\d{2})(?!\d)")),
    # `25 082023` — 일자 뒤에 월·연이 붙어 있는 형태.
    ("d_mmy",  re.compile(r"(?<!\d)(\d{1,2})[.\-/ ](\d{2})(20\d{2})(?!\d)")),
    # `12102022` — 구분자 없는 8자리 DDMMYYYY. 20으로 시작하지 않아 ymd8이 못 잡는다.
    ("dmy8",   re.compile(r"(\d{2})(\d{2})(20\d{2})")),
    ("korean", re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")),
    ("ymd2",   re.compile(rf"(\d{{2}})({_SEP})(\d{{1,2}})\2(\d{{1,2}})")),
    ("d_mon_y", re.compile(rf"(\d{{1,2}})\s*{_SEP}?\s*({_MON})\w*\s*{_SEP}?\s*(\d{{2,4}})",
                           re.I)),
    ("mon_d_y", re.compile(rf"({_MON})\w*\.?\s*(\d{{1,2}})\s*[,. ]\s*(\d{{2,4}})", re.I)),
    # ── 여기서부터 불완전 날짜: 버리지 않는다 (부분 점수 10점) ──
    ("mon_y",  re.compile(rf"({_MON})\w*\.?\s*(20\d{{2}})", re.I)),
    # ── 인식 오류를 흡수하는 완화 패턴 (엄격한 것들이 먼저 시도된 뒤) ──
    # `22/MRY/2023` — MAY를 MRY로 읽는 식의 1글자 오독. 4자리 연도가 앵커라 안전하다.
    ("d_fuzz_y", re.compile(r"(\d{1,2})\s*[./\- ]\s*([A-Za-z]{3,4})\s*[./\- ]\s*(\d{2,4})")),
    ("fuzz_y",  re.compile(r"(?<![A-Za-z])([A-Za-z]{3,4})\.?\s*(20\d{2})")),
    # `202112.16A6` — 연·월이 붙고 일자만 구분자로 떨어진 형태.
    ("ymmd",   re.compile(r"(20\d{2})(\d{2})[./\- ](\d{1,2})(?![\d])")),
    ("ym4",    re.compile(rf"(20\d{{2}})({_SEP})(\d{{1,2}})(?!{_SEP}?\d)")),
    # `02/2022` — 월/연 (일자 없음). 4자리 연도가 뒤에 오는 형태.
    ("m_y",    re.compile(r"(?<!\d)(\d{1,2})\s*[./\-]\s*(20\d{2})(?!\d)")),
    # 연도 없음. 앞에 '숫자+구분자'가 오면 더 긴 날짜의 꼬리이므로 잡지 않는다
    # — 그러지 않으면 1986.08.02 의 '08.02'를 연도 없는 날짜로 오인하고
    #   보정 단계가 엉뚱한 연도를 붙여 되살린다.
    ("md",     re.compile(rf"(?<!\d)(?<![\d][.\-/])(\d{{1,2}})({_SEP})(\d{{1,2}})(?!{_SEP}?\d)")),
]


def _interpret(name, m, raw, source, conf=1.0):
    """패턴 매치 → 가능한 해석들. 모호하면 여러 개를 내고 달력 검증으로 거른다.

    ⚠️ **텍스트에 연도가 찍혀 있는데 창([2018,2032]) 밖이면 후보 자체를 버린다.**
    연도만 None으로 두고 남기면 보정 단계가 그럴듯한 연도를 붙여 되살리기 때문에,
    ``1986.08.02`` 이나 품목보고번호가 유효한 소비기한으로 둔갑한다.
    연도 None이 허용되는 건 텍스트에 애초에 연도가 없는 ``md`` 뿐이다.
    """
    g, span = m.groups(), m.span()
    kw = dict(raw=raw, span=span, context=raw, pattern=name, source=source, conf=conf)

    def dated(year, month, day, pattern=None):
        """연도가 창 밖이면(=None) 후보를 만들지 않는다."""
        if not year:
            return []
        return [_build(year, month, day, **{**kw, "pattern": pattern or name})]

    if name == "ymd4":
        return dated(_year4(g[0]), int(g[2]), int(g[3]))
    if name == "dmy4":
        return dated(_year4(g[3]), int(g[2]), int(g[0]))
    if name == "ymd8":
        return dated(_year4(g[0]), int(g[1]), int(g[2]))
    if name == "y_mmdd":
        return dated(_year4(g[0]), int(g[1]), int(g[2]))
    if name == "d_mmy":
        return dated(_year4(g[2]), int(g[1]), int(g[0]))
    if name == "dmy8":
        return dated(_year4(g[2]), int(g[1]), int(g[0]))
    if name == "korean":
        return dated(_year4(g[0]), int(g[1]), int(g[2]))
    if name == "ymd2":
        # 21.09.17 → YY.MM.DD 와 DD.MM.YY 둘 다 시도, 달력·연도창이 걸러준다.
        # ⚠️ 둘 다 유효할 때가 많다(`22.04.30` → 2022-04-30 / 2030-04-22).
        # 국내 표기는 YY.MM.DD가 지배적이므로 **서로 다른 패턴 이름**을 붙여
        # select.py의 사전확률이 앞쪽을 선호하게 한다. 같은 이름을 쓰면
        # "나중 날짜 우선" 동점 규칙이 뒤집힌 해석을 골라버린다.
        return dated(_year4(g[0]), int(g[2]), int(g[3]), "ymd2") + \
               dated(_year4(g[3]), int(g[2]), int(g[0]), "dmy2")
    if name == "d_mon_y":
        return dated(_year4(g[2]), MONTHS[g[1].upper()[:3]], int(g[0]))
    if name == "mon_d_y":
        return dated(_year4(g[2]), MONTHS[g[0].upper()[:3]], int(g[1]))
    if name == "mon_y":
        return dated(_year4(g[1]), MONTHS[g[0].upper()[:3]], None)
    if name == "d_fuzz_y":
        month = fuzzy_month(g[1])
        return dated(_year4(g[2]), month, int(g[0])) if month else []
    if name == "fuzz_y":
        month = fuzzy_month(g[0])
        return dated(_year4(g[1]), month, None) if month else []
    if name == "ymmd":
        return dated(_year4(g[0]), int(g[1]), int(g[2]))
    if name == "m_y":
        return dated(_year4(g[1]), int(g[0]), None)
    if name == "ym4":
        return dated(_year4(g[0]), int(g[2]), None)
    if name == "md":
        # 연도 없음 (02.18까지). MM.DD 를 DD.MM 보다 선호한다(같은 이유).
        return [_build(None, int(g[0]), int(g[2]), **kw),
                _build(None, int(g[2]), int(g[0]), **{**kw, "pattern": "dm"})]
    return []


def parse(raw: str, source: int = 0, conf: float = 1.0) -> list[Candidate]:
    """한 줄에서 날짜 후보를 전부 뽑는다.

    정규식은 **정규화본**에 돌리고(``O→0`` 보정을 받기 위해),
    ``embedded`` 판정과 ``text`` 는 **원문**에서 가져온다(위 설계 결정 1).

    ``conf`` 는 이 텍스트를 만든 검출·인식 confidence(0~1). 직접 문자열을
    넘기는 호출(단위테스트 등)에는 없으므로 기본 1.0 — 모든 후보가 동일하게
    받아 상대 순위에 영향이 없다.
    """
    if not raw:
        return []
    norm = normalize(raw)
    out, claimed = [], []

    for name, pattern in _PATTERNS:
        for m in pattern.finditer(norm):
            start, end = m.span()
            # 더 구체적인 패턴이 이미 차지한 구간은 건너뛴다 (ymd4 > ym4 > md).
            if any(s <= start and end <= e for s, e in claimed):
                continue
            found = [c for c in _interpret(name, m, raw, source, conf) if c is not None]
            if found:
                claimed.append((start, end))
                out.extend(found)
    return out


def merge_lines(items, y_tol: float = 0.6) -> list[tuple[str, int, float]]:
    """가로로 인접한 검출 박스를 한 줄로 잇는다.

    DB 검출기는 ``2021.``, ``08``, ``.02``, ``까지`` 를 **따로** 준다. 병합하지
    않으면 어떤 정규식도 매치하지 못한다 — 빠뜨리기 쉬운 필수 단계다.

    ``items``: ``[(text, x0, y0, x1, y1[, conf]), ...]`` — conf 없으면 1.0.
    반환: ``[(줄 텍스트, 대표 박스 인덱스, 병합 confidence), ...]`` — 공백
    있음/없음 두 형태를 모두 낸다. 인쇄물은 ``2021. 08. 02`` 와 ``2021.08.02``
    를 오간다. 병합 confidence 는 **구성 박스 중 최솟값** — 여러 박스를
    이어붙인 결과는 그중 가장 못 미더운 박스만큼만 믿을 수 있다.
    """
    if not items:
        return []

    indexed = list(enumerate(items))
    heights = [max(1.0, b[4] - b[2]) for _, b in indexed]
    band = y_tol * (sum(heights) / len(heights))

    rows: list[list[tuple[int, tuple]]] = []
    for idx, box in sorted(indexed, key=lambda p: (p[1][2], p[1][1])):
        cy = (box[2] + box[4]) / 2
        for row in rows:
            ref = row[0][1]
            if abs(cy - (ref[2] + ref[4]) / 2) <= band:
                row.append((idx, box))
                break
        else:
            rows.append([(idx, box)])

    lines = []
    for row in rows:
        row.sort(key=lambda p: p[1][1])
        texts = [b[0] for _, b in row]
        confs = [b[5] if len(b) > 5 else 1.0 for _, b in row]
        head = row[0][0]
        if len(row) > 1:
            merged_conf = min(confs)
            lines.append((" ".join(texts), head, merged_conf))
            lines.append(("".join(texts), head, merged_conf))
    return lines


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


#: 겹침-중복 제거가 발동하는 최소 confidence 격차. 동률/근소 차이(≤0.1)면
#: **아무것도 지우지 않는다** — confidence 로 우열을 가릴 근거가 약할 때
#: 임의로 하나를 골라 지우면, 구조 점수(PATTERN_PRIOR 등)로는 이겼을 정답
#: 후보가 순위 계산에 도달하기도 전에 사라질 수 있다(ExpDate val_split
#: img_00940 에서 실측: 정답 d_mon_y 후보가 conf 동률인 ymd8 쓰레기와
#: 함께 지워짐 — 원래대로 두면 pattern prior(8 vs 4)만으로 정답이 이겼다).
#: 격차가 뚜렷할 때만 지우므로, 그 경우엔 어차피 낮은 쪽이 랭킹에서도 진다 —
#: 이 함수는 "이길 후보가 뻔한데 랭킹 전에 미리 치운다" 수준으로만 보수적으로 쓴다.
_DEDUP_MIN_GAP = 0.1


def _drop_overlapping_duplicates(cands: list[Candidate]) -> list[Candidate]:
    """병합이 만든 부분-중복 재인식을, confidence 격차가 뚜렷할 때만 제거한다.

    NanoDet이 같은 물리적 날짜 스탬프에 겹치는 박스 두 개를 내면(예: 하나는
    전체, 하나는 일부만 걸친 크롭), 인식·병합 결과가 서로의 부분/전체를
    포함하는 **서로 다른** (연,월,일) 후보로 파싱될 수 있다 — 예:
    ``23.02.14``(정답) 와, 그 박스가 이웃과 병합돼 생긴 ``2023.02.23.02.14``
    (겹친 숫자가 재조합되어 엉뚱한 날짜로 읽힘). 원문 숫자열이 포함 관계이고
    confidence 격차가 ``_DEDUP_MIN_GAP`` 보다 크면 낮은 쪽을 버린다. 동일
    (연,월,일) 중복은 이 아래 ``parse_boxes`` 의 키 기반 정리가 이미 처리하므로,
    여기선 **다른** (연,월,일)로 갈라진 경우만 다룬다.
    """
    digits = [_digits_only(c.text) for c in cands]
    drop: set[int] = set()
    for i in range(len(cands)):
        if i in drop or not digits[i]:
            continue
        for j in range(i + 1, len(cands)):
            if j in drop or not digits[j] or digits[i] == digits[j]:
                continue
            # 완전한 날짜(연+월+일)는 불완전한 후보(연·월뿐 등)에 의해 지워지지
            # 않는다 — completeness 는 이미 select.score()에서 +15로 강하게
            # 대접받는 신호라, confidence 만으로 뒤집으면 안 된다. 실측:
            # ExpDate val_split img_00940 에서 완전한 정답 후보가, 겹치는
            # 불완전 후보(day 없음)의 confidence 가 우연히 더 높다는 이유로
            # 지워졌다 — 원래대로 뒀으면 완전함 보너스로 랭킹에서 이겼다.
            if cands[i].complete != cands[j].complete:
                continue
            if digits[i] in digits[j] or digits[j] in digits[i]:
                gap = cands[i].conf - cands[j].conf
                if abs(gap) > _DEDUP_MIN_GAP:
                    drop.add(i if gap < 0 else j)
    return [c for k, c in enumerate(cands) if k not in drop]


def parse_boxes(items) -> list[Candidate]:
    """검출 박스 목록에서 후보 전부를 뽑는다 — 개별 박스 **와** 병합 줄 양쪽에서.

    ``items``: ``[(text, x0, y0, x1, y1[, conf]), ...]`` — conf 없으면 1.0.
    """
    def box_conf(box):
        return box[5] if len(box) > 5 else 1.0

    out = []
    for i, box in enumerate(items):
        out.extend(parse(box[0], source=i, conf=box_conf(box)))
    for line, head, conf in merge_lines(items):
        out.extend(parse(line, source=head, conf=conf))

    # 180° 뒤집힌 크롭 구제. 방향 분류기(cls)가 놓치면 인식 결과가 통째로
    # 뒤집혀 나온다 — 실측에서 `92/60/7202`(= 2027/06/29), `82-60-204X3`(= 2028-06-02).
    # 정방향에서 아무것도 못 건졌을 때만 시도하고, 패턴 이름에 `_rev` 를 붙여
    # select 가 낮은 사전확률을 주도록 한다. 추론 비용은 0이다.
    if not out:
        for i, box in enumerate(items):
            for cand in parse(box[0][::-1], source=i, conf=box_conf(box)):
                out.append(Candidate(**{**cand.__dict__,
                                        "pattern": cand.pattern + "_rev",
                                        "context": box[0]}))

    out = _drop_overlapping_duplicates(out)

    # 같은 (연,월,일)이 여러 경로로 나오면 하나만 남긴다. confidence 가 높은 쪽,
    # 동률이면 원문이 긴 쪽을 남겨야 키워드 문맥이 보존된다.
    best: dict[tuple, Candidate] = {}
    for c in out:
        key = (c.year, c.month, c.day)
        if key not in best or (c.conf, len(c.context)) > (best[key].conf, len(best[key].context)):
            best[key] = c
    return list(best.values())
