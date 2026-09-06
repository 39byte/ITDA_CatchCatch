"""ExpDate 어댑터 검증 — 실제 데이터가 도착하기 **전에**.

공개된 스키마를 그대로 흉내 낸 픽스처로 변환 경로를 통째로 돌린다. 데이터가
없다고 코드를 미검증 상태로 두면, 630 MB를 내려받은 뒤에야 버그를 만나게 된다.

`exp` 표시 방식이 공개돼 있지 않으므로 **가능한 세 표기를 모두** 시험한다.
"""

import csv
import json

import pytest

from eval.expdate import (convert, extract_date, inspect, parse_day,
                          parse_month, parse_year, pick_expiry_ann)


def _ann(cls, box, text, dmy=None, **extra):
    ann = {"cls": cls, "bbox": box, "transcription": text}
    if dmy is not None:
        ann["dmy_ann"] = dmy
    ann.update(extra)
    return ann


def _dmy(day, month, year):
    return [{"cls": "day", "bbox": [0, 0, 1, 1], "transcription": day},
            {"cls": "month", "bbox": [0, 0, 1, 1], "transcription": month},
            {"cls": "year", "bbox": [0, 0, 1, 1], "transcription": year}]


@pytest.fixture
def dataset(tmp_path):
    """공개 스키마를 그대로 따르는 Products-Real 축소판."""
    root = tmp_path / "Products-Real"
    (root / "test" / "images").mkdir(parents=True)
    for name in ("a", "b", "c", "d"):
        (root / "test" / "images" / f"{name}.jpg").write_bytes(b"\xff\xd8\xff\xe0stub")

    data = {
        # exp 가 별도 cls 값인 경우
        "a.jpg": {"height": 800, "width": 600, "ann": [
            _ann("production mark", [10, 10, 50, 20], "MFG"),
            _ann("date", [10, 30, 90, 45], "01 FEB 24", dmy=_dmy("01", "FEB", "24")),
            _ann("exp", [10, 60, 90, 75], "29 OCT 23", dmy=_dmy("29", "OCT", "23")),
        ]},
        # exp 가 불리언 필드인 경우
        "b.jpg": {"height": 800, "width": 600, "ann": [
            _ann("date", [5, 5, 80, 20], "2023 10 29",
                 dmy=_dmy("29", "10", "2023"), exp=True),
            _ann("code mark", [5, 40, 80, 55], "L095E"),
        ]},
        # 날짜가 하나뿐이면 exp 표시가 없어도 그것이 정답이다
        "c.jpg": {"height": 400, "width": 400, "ann": [
            _ann("date", [1, 1, 40, 12], "15/03/2025", dmy=_dmy("15", "03", "2025")),
        ]},
        # dmy_ann 이 없으면 건너뛴다 (우리 파서로 정답을 만들면 순환이 된다)
        "d.jpg": {"height": 400, "width": 400, "ann": [
            _ann("date", [1, 1, 40, 12], "2025.03.15"),
        ]},
    }
    (root / "test" / "annotations.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return root


def test_convert_produces_submission_schema(dataset, tmp_path):
    info = convert(dataset, tmp_path / "out")
    rows = list(csv.DictReader((tmp_path / "out" / "gt_dates.csv").open(encoding="utf-8")))
    by_id = {r["image_id"]: r for r in rows}

    assert list(rows[0]) == ["image_id", "year", "month", "day", "final_date"]
    assert by_id["a"]["final_date"] == "2023-10-29", "별도 cls 로 표시된 exp"
    assert by_id["b"]["final_date"] == "2023-10-29", "불리언 필드로 표시된 exp"
    assert by_id["c"]["final_date"] == "2025-03-15", "날짜가 하나뿐인 경우"
    assert "d" not in by_id, "dmy_ann 이 없으면 정답을 지어내지 않는다"
    assert info["skipped"] == {"dmy_ann 없음": 1}


def test_expiry_beats_other_dates(dataset, tmp_path):
    """제조일자(01 FEB 24)가 아니라 소비기한(29 OCT 23)을 정답으로 삼아야 한다."""
    convert(dataset, tmp_path / "out")
    rows = {r["image_id"]: r for r in
            csv.DictReader((tmp_path / "out" / "gt_dates.csv").open(encoding="utf-8"))}
    assert rows["a"]["final_date"] != "2024-02-01"


def test_boxes_are_exported_with_exp_flag(dataset, tmp_path):
    convert(dataset, tmp_path / "out")
    boxes = json.loads((tmp_path / "out" / "gt_boxes.json").read_text(encoding="utf-8"))
    assert len(boxes["a"]) == 3
    assert sum(b["exp"] for b in boxes["a"]) == 1
    assert boxes["a"][0]["cls"] == "productionmark", "cls 표기 정규화"


def test_finds_annotations_without_hardcoded_filename(dataset, tmp_path):
    """파일명이 공개돼 있지 않으므로 글롭으로 찾아야 한다."""
    (dataset / "test" / "annotations.json").rename(dataset / "test" / "whatever_v2.json")
    info = convert(dataset, tmp_path / "out")
    assert info["labelled"] == 3


def test_inspect_runs_and_reports_exp_detection(dataset):
    text = inspect(dataset)
    assert "exp 로 판정된 인스턴스" in text
    assert "images" in text


def test_inspect_survives_missing_data(tmp_path):
    assert "없음" in inspect(tmp_path)


# ── 값 정규화: ExpDate는 13가지 날짜 포맷을 원문 그대로 담는다 ──────────────

@pytest.mark.parametrize("raw, expected", [("23", "2023"), ("2023", "2023"), ("x", None)])
def test_parse_year(raw, expected):
    assert parse_year(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("OCT", "10"), ("Oct", "10"), ("OCTOBER", "10"), ("10", "10"), ("3", "03"), ("13", None),
])
def test_parse_month(raw, expected):
    assert parse_month(raw) == expected


@pytest.mark.parametrize("raw, expected", [("29", "29"), ("1", "01"), ("00", None), ("32", None)])
def test_parse_day(raw, expected):
    assert parse_day(raw) == expected


def test_partial_dmy_still_yields_partial_ground_truth():
    """일자 라벨이 없으면 연·월만 정답으로 남긴다 (부분 점수 대응)."""
    got = extract_date({"dmy_ann": [{"cls": "month", "transcription": "OCT"},
                                    {"cls": "year", "transcription": "2021"}]})
    assert got == {"year": "2021", "month": "10", "day": "NONE", "final_date": "NONE"}


def test_ambiguous_multi_date_image_is_skipped():
    """exp 표시가 없는데 날짜가 여럿이면 정답이 모호하다 — 지어내지 않는다."""
    assert pick_expiry_ann([{"cls": "date", "bbox": [0, 0, 1, 1]},
                            {"cls": "date", "bbox": [0, 2, 1, 3]}]) is None
