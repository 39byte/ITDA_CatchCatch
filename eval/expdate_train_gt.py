"""ExpDate **train** 주석 → 날짜 정답 CSV (val_split threshold 튜닝 전용).

train 은 `dmy_ann`/`exp` 가 없고 `transcription` 문자열만 있어서 정식 어댑터
(`eval/expdate.py`)를 못 쓴다. 여기서는 **transcription(사람이 단 깨끗한 정답
문자열)** 만 파싱한다. `itda_ocr.parse` 와는 별개의 파서 — 노이즈 OCR 출력이 아니라
정답 문자열을 읽는 것이라 "자기 오류를 자기가 채점" 순환에 해당하지 않는다.

    python -m eval.expdate_train_gt --root expdate/train --out labels/expdate_train
    python -m eval.expdate_train_gt --root expdate/train --ids labels/detection/val_split_coco.json \
        --out labels/expdate_train

⚠️ **date 주석이 정확히 1개인 이미지만** 정답으로 삼는다 (train 에 exp 표시가 없어
2개 이상이면 어느 게 소비기한인지 모호). 그 외는 건너뛴다.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import re
from pathlib import Path

_MON = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def _y4(y: str) -> int | None:
    n = int(y)
    if n < 100:
        n += 2000
    return n if 2015 <= n <= 2035 else None


def parse_transcription(text: str):
    """깨끗한 정답 문자열 → (year, month, day) 또는 None.

    ExpDate train transcription 에서 관측된 형식만 다룬다:
      YYYY.MM.DD / YYYY/MM/DD / YYYY-MM-DD / YYYY MM DD
      DD MM YYYY / DD.MM.YYYY / DD/MM/YY
      MON DD YY / DD MON YYYY / MON YYYY
    """
    t = text.strip().upper()
    SEP = r"[.\-/\s]"

    # MON DD YY  (MAR 22 21 / FEB/21/21)
    m = re.fullmatch(rf"([A-Z]{{3}}){SEP}+(\d{{1,2}}){SEP}+(\d{{2,4}})", t)
    if m and m.group(1) in _MON:
        y = _y4(m.group(3))
        return (y, _MON[m.group(1)], int(m.group(2))) if y else None

    # DD MON YYYY  (11 OCT 2021)
    m = re.fullmatch(rf"(\d{{1,2}}){SEP}+([A-Z]{{3}}){SEP}+(\d{{2,4}})", t)
    if m and m.group(2) in _MON:
        y = _y4(m.group(3))
        return (y, _MON[m.group(2)], int(m.group(1))) if y else None

    # MON YYYY  (OCT 2021) — 일자 없음
    m = re.fullmatch(rf"([A-Z]{{3}}){SEP}+(\d{{4}})", t)
    if m and m.group(1) in _MON:
        y = _y4(m.group(2))
        return (y, _MON[m.group(1)], None) if y else None

    nums = re.findall(r"\d+", t)

    def valid(y, mo, d):
        return (y and 1 <= mo <= 12 and (d is None or
                (1 <= d <= 31 and d <= calendar.monthrange(y, mo)[1])))

    if len(nums) == 3:
        a, b, c = nums
        cands = []
        if len(a) == 4:                       # YYYY.MM.DD
            cands.append((_y4(a), int(b), int(c)))
        elif len(c) == 4:                     # DD.MM.YYYY
            cands.append((_y4(c), int(b), int(a)))
        else:                                 # 전부 2자리 — 두 해석 다 유효하고 다르면 모호 → skip
            yy_mm_dd = (_y4(a), int(b), int(c))
            dd_mm_yy = (_y4(c), int(b), int(a))
            ok = [c for c in (yy_mm_dd, dd_mm_yy) if valid(*c)]
            if len(ok) == 2 and ok[0] != ok[1]:
                return None                    # 모호 (예: 20/07/21) — 튜닝 GT 로 안 씀
            return ok[0] if ok else None
        for y, mo, d in cands:
            if valid(y, mo, d):
                return (y, mo, d)
        return None

    if len(nums) == 2:
        a, b = nums
        if len(a) == 4 and 1 <= int(b) <= 12:      # YYYY.MM
            return (_y4(a), int(b), None)
        if len(b) == 4 and 1 <= int(a) <= 12:      # MM/YYYY
            return (_y4(b), int(a), None)
    return None


def build(root: str, want_ids: set | None):
    raw = json.loads(Path(root, "annotations.json").read_text(encoding="utf-8"))
    rows, skip = [], {"date 주석 != 1개": 0, "transcription 파싱 실패": 0, "id 목록 밖": 0}
    for name, entry in raw.items():
        iid = Path(name).stem
        if want_ids is not None and iid not in want_ids:
            skip["id 목록 밖"] += 1
            continue
        dates = [a for a in entry.get("ann", [])
                 if str(a.get("cls", "")).lower() in ("date", "exp") and a.get("transcription")]
        if len(dates) != 1:
            skip["date 주석 != 1개"] += 1
            continue
        got = parse_transcription(dates[0]["transcription"])
        if got is None:
            skip["transcription 파싱 실패"] += 1
            continue
        y, mo, d = got
        final = f"{y:04d}-{mo:02d}-{d:02d}" if d else "NONE"
        rows.append({"image_id": iid, "year": f"{y:04d}",
                     "month": f"{mo:02d}", "day": f"{d:02d}" if d else "NONE",
                     "final_date": final})
    rows.sort(key=lambda r: r["image_id"])
    return rows, skip


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="labels/expdate_train")
    ap.add_argument("--ids", help="이 COCO json 의 이미지만 (예: val_split_coco.json)")
    args = ap.parse_args(argv)

    want = None
    if args.ids:
        want = {Path(i["file_name"]).stem
                for i in json.loads(Path(args.ids).read_text())["images"]}

    rows, skip = build(args.root, want)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "gt_dates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "year", "month", "day", "final_date"])
        w.writeheader()
        w.writerows(rows)
    print(f"정답 {len(rows)}장 → {csv_path}")
    print("건너뜀:", skip)


if __name__ == "__main__":
    main()
