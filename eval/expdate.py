"""ExpDate(Products-Real) → 우리 채점 스키마 어댑터.

ExpDate: Seker & Ahn, *Expert Systems with Applications* 2022, KIST. CC BY 4.0.
https://felizang.github.io/expdate/  (Products-Real.zip, 630 MB, Google Drive)
사용 시 KIST 귀속 표기와 논문 인용이 필요하다.

공개된 주석 스키마::

    { "<image name>": { "height": int, "width": int,
        "ann": [ { "cls": str, "bbox": [x1,y1,x2,y2], "transcription": str,
                   "dmy_ann": [ {"cls","bbox","transcription"} ] } ] } }

클래스 4종: date / due mark / production mark / code mark.
**Products-Real 테스트셋은 소비기한 인스턴스에 `exp` 라벨을 추가로 붙여** 다른
날짜와 구분한다 — 즉 665장이 우리의 *선별* 문제를 그대로 라벨링해 둔 셈이다.

⚠️ 공개돼 있지 **않은** 것: JSON 파일명, 디렉터리 트리, `cls` 문자열의 실제 표기,
`exp` 를 어떻게 표시하는지. 그래서 이 모듈은 파일명을 하드코딩하지 않고 글롭하며,
클래스 문자열을 정규화하고, ``--inspect`` 로 실물 구조를 먼저 보여준다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def norm_key(value) -> str:
    """``"due mark"`` / ``"due_mark"`` / ``"Due-Mark"`` 를 하나로 접는다."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_annotation_files(root) -> list[Path]:
    """주석 JSON을 찾는다. 파일명이 공개돼 있지 않으므로 글롭한다."""
    root = Path(root)
    found = [p for p in root.rglob("*.json") if p.stat().st_size > 0]
    return sorted(found, key=lambda p: -p.stat().st_size)


def load_annotations(path) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: 최상위가 dict가 아니다 (이미지명 키를 기대했다)")
    return data


def find_image_dirs(root) -> list[Path]:
    dirs = Counter()
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMAGE_SUFFIXES:
            dirs[p.parent] += 1
    return [d for d, _ in dirs.most_common()]


# ── 값 정규화 ──────────────────────────────────────────────────────────────

def parse_year(text) -> str | None:
    digits = re.sub(r"\D", "", str(text))
    if len(digits) == 4:
        return digits
    if len(digits) == 2:
        return f"20{digits}"          # 소비기한은 20xx대다
    return None


def parse_month(text) -> str | None:
    raw = str(text).strip().upper()
    for name, num in MONTHS.items():
        if raw.startswith(name):
            return f"{num:02d}"
    digits = re.sub(r"\D", "", raw)
    if digits and 1 <= int(digits) <= 12:
        return f"{int(digits):02d}"
    return None


def parse_day(text) -> str | None:
    digits = re.sub(r"\D", "", str(text))
    if digits and 1 <= int(digits) <= 31:
        return f"{int(digits):02d}"
    return None


def _is_expiry(ann: dict) -> bool:
    """이 인스턴스가 소비기한인가.

    `exp` 표시 방식이 문서화돼 있지 않아 세 가지 가능성을 모두 본다:
    별도 cls 값, 불리언 필드, 혹은 어떤 값에든 'exp'가 박혀 있는 경우.
    """
    for key, value in ann.items():
        if key == "dmy_ann":
            continue
        nk = norm_key(key)
        if nk in {"exp", "isexp", "expiry"} and value:
            return True
        if isinstance(value, str) and "exp" in norm_key(value):
            return True
    return False


def extract_date(ann: dict) -> dict | None:
    """``dmy_ann`` 에서 연·월·일을 뽑는다.

    ``dmy_ann`` 이 없으면 **건너뛴다.** 여기서 우리 파서(`itda_ocr.parse`)를
    폴백으로 쓰면 정답이 우리 파서의 오류를 그대로 물려받아, 자기 오류를 자기가
    채점하는 순환이 된다.
    """
    parts = ann.get("dmy_ann") or []
    if not parts:
        return None
    got = {}
    for part in parts:
        kind = norm_key(part.get("cls", ""))
        text = part.get("transcription", "")
        if kind.startswith("y"):
            got["year"] = parse_year(text)
        elif kind.startswith("mo") or kind == "m":
            got["month"] = parse_month(text)
        elif kind.startswith("d"):
            got["day"] = parse_day(text)
    if not got:
        return None
    year, month, day = got.get("year"), got.get("month"), got.get("day")
    final = f"{year}-{month}-{day}" if year and month and day else None
    return {"year": year or "NONE", "month": month or "NONE",
            "day": day or "NONE", "final_date": final or "NONE"}


def pick_expiry_ann(anns: list[dict]) -> dict | None:
    """이미지의 인스턴스들 중 소비기한 하나를 고른다."""
    marked = [a for a in anns if _is_expiry(a)]
    if marked:
        return marked[0]
    dates = [a for a in anns if norm_key(a.get("cls", "")) == "date"]
    if len(dates) == 1:
        return dates[0]
    # 날짜가 여럿인데 exp 표시가 없으면(=학습셋) 정답이 모호하다 → 건너뛴다.
    return None


# ── 산출물 ────────────────────────────────────────────────────────────────

def convert(root, out_dir) -> dict:
    """``gt_dates.csv`` 와 ``gt_boxes.json`` 을 만든다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_annotation_files(root)
    if not files:
        raise SystemExit(f"{root} 아래에서 JSON 주석을 찾지 못했다. --inspect 로 구조를 확인할 것.")

    merged: dict[str, dict] = {}
    for path in files:
        try:
            merged.update(load_annotations(path))
        except (ValueError, json.JSONDecodeError):
            continue

    rows, boxes, skipped = [], {}, Counter()
    for name, entry in merged.items():
        anns = entry.get("ann") or []
        image_id = Path(str(name)).stem
        boxes[image_id] = [{"bbox": a.get("bbox"), "cls": norm_key(a.get("cls", "")),
                            "exp": _is_expiry(a)} for a in anns if a.get("bbox")]
        chosen = pick_expiry_ann(anns)
        if chosen is None:
            skipped["소비기한 인스턴스를 특정할 수 없음"] += 1
            continue
        date = extract_date(chosen)
        if date is None:
            skipped["dmy_ann 없음"] += 1
            continue
        rows.append({"image_id": image_id, **date})

    rows.sort(key=lambda r: r["image_id"])
    csv_path = out_dir / "gt_dates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["image_id", "year", "month", "day", "final_date"])
        writer.writeheader()
        writer.writerows(rows)
    boxes_path = out_dir / "gt_boxes.json"
    boxes_path.write_text(json.dumps(boxes, ensure_ascii=False), encoding="utf-8")

    return {"annotation_files": [str(p) for p in files], "images": len(merged),
            "labelled": len(rows), "skipped": dict(skipped),
            "gt_dates": str(csv_path), "gt_boxes": str(boxes_path),
            "image_dirs": [str(d) for d in find_image_dirs(root)[:5]]}


def inspect(root) -> str:
    """실물 구조를 출력한다. 포맷이 예상과 다르면 여기서 즉시 드러난다."""
    lines = [f"# ExpDate 구조 점검: {root}", ""]

    dirs = find_image_dirs(root)
    lines.append("## 이미지 디렉터리")
    if not dirs:
        lines.append("  (없음 — 압축을 풀었는지 확인)")
    for d in dirs[:8]:
        n = sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        lines.append(f"  {d}  ({n}장)")

    files = find_annotation_files(root)
    lines += ["", "## 주석 JSON"]
    if not files:
        lines.append("  (없음)")
    for p in files[:8]:
        lines.append(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")

    for path in files[:3]:
        try:
            data = load_annotations(path)
        except (ValueError, json.JSONDecodeError) as exc:
            lines.append(f"\n  ⚠️ {path.name}: {exc}")
            continue

        cls_top, cls_dmy, keys, dmy_keys = Counter(), Counter(), Counter(), Counter()
        samples = defaultdict(list)
        exp_hits, has_dmy = 0, 0
        for entry in data.values():
            for ann in entry.get("ann") or []:
                keys.update(ann.keys())
                cls_top[str(ann.get("cls"))] += 1
                if _is_expiry(ann):
                    exp_hits += 1
                if ann.get("dmy_ann"):
                    has_dmy += 1
                if len(samples[str(ann.get("cls"))]) < 4:
                    samples[str(ann.get("cls"))].append(ann.get("transcription"))
                for part in ann.get("dmy_ann") or []:
                    dmy_keys.update(part.keys())
                    cls_dmy[str(part.get("cls"))] += 1

        lines += [
            "", f"## {path.name}",
            f"  이미지 {len(data)}장 / ann 키 {sorted(keys)}",
            f"  dmy_ann 키 {sorted(dmy_keys)}",
            f"  dmy_ann 보유 인스턴스 {has_dmy}",
            f"  exp 로 판정된 인스턴스 {exp_hits}   ← 0이면 _is_expiry 를 고쳐야 한다",
            "  상위 cls: " + ", ".join(f"{k!r}×{v}" for k, v in cls_top.most_common(8)),
            "  dmy cls: " + ", ".join(f"{k!r}×{v}" for k, v in cls_dmy.most_common(8)),
            "  전사 샘플:",
        ]
        for cls, texts in list(samples.items())[:6]:
            lines.append(f"    {cls!r}: {texts}")

        first = next(iter(data.items()), None)
        if first:
            lines += ["  첫 항목 원본:",
                      "    " + json.dumps({first[0]: first[1]}, ensure_ascii=False)[:600]]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="ExpDate → 채점용 정답 파일")
    ap.add_argument("--root", required=True, help="압축을 푼 ExpDate 디렉터리")
    ap.add_argument("--out", default="labels/expdate", help="산출 디렉터리")
    ap.add_argument("--inspect", action="store_true",
                    help="변환하지 않고 실물 구조만 출력한다 (먼저 이걸 실행할 것)")
    args = ap.parse_args(argv)

    if args.inspect:
        print(inspect(args.root))
        return
    info = convert(args.root, args.out)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"\n정답 {info['labelled']}장 생성 → {info['gt_dates']}")
    if info["skipped"]:
        print("건너뜀:", info["skipped"])


if __name__ == "__main__":
    main()
