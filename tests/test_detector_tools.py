"""Track B 도구 검증 — 변환기와 A/B 하네스.

인수인계 대상 코드라 특히 조용한 실패가 위험하다. 좌표 규약이 어긋나면
비교 표는 멀쩡해 보이는데 결론이 통째로 뒤집힌다.
"""

import json

import pytest

from eval.detector_ab import evaluate, iou, load_gt
from tools.expdate_to_detection import class_of, convert_split, norm


@pytest.fixture
def split(tmp_path):
    """ExpDate 한 분할의 축소판. evaluation 처럼 `exp` 표기를 쓴다."""
    d = tmp_path / "evaluation"
    (d / "images").mkdir(parents=True)
    for name in ("a", "b"):
        (d / "images" / f"{name}.jpg").write_bytes(b"\xff\xd8stub")
    (d / "ann.json").write_text(json.dumps({
        "a.jpg": {"height": 100, "width": 200, "ann": [
            {"cls": "exp", "bbox": [20, 40, 120, 60], "transcription": "2021.08.03"},
            {"cls": "due", "bbox": [5, 5, 15, 15]},
        ]},
        "b.jpg": {"height": 50, "width": 50, "ann": [
            {"cls": "date", "bbox": [0, 0, 25, 10]},
        ]},
    }), encoding="utf-8")
    return d


# ── 변환기 ────────────────────────────────────────────────────────────────

def test_exp_and_date_merge_into_one_class():
    """train은 `date`, evaluation은 `exp` 를 쓴다. 검출기에겐 같은 대상이다."""
    assert class_of("exp", ["date"]) == 0
    assert class_of("date", ["date"]) == 0
    assert class_of("due", ["date"]) is None          # 기본값에선 제외
    assert class_of("due", ["date", "due"]) == 1      # 요청하면 포함


def test_class_name_normalisation():
    """`due mark` / `due_mark` / `Due-Mark` 가 같은 것으로 접혀야 한다."""
    assert norm("due mark") == norm("due_mark") == norm("Due-Mark") == "duemark"


def test_yolo_labels_are_normalised_centre_format(split, tmp_path):
    convert_split(split, ["date"], "yolo", tmp_path / "out")
    line = (split / "labels" / "a.txt").read_text(encoding="utf-8").strip()
    cls, cx, cy, w, h = line.split()
    # bbox [20,40,120,60] on 200x100 → 중심 (70,50), 크기 (100,20)
    assert cls == "0"
    assert float(cx) == pytest.approx(70 / 200)
    assert float(cy) == pytest.approx(50 / 100)
    assert float(w) == pytest.approx(100 / 200)
    assert float(h) == pytest.approx(20 / 100)


def test_yolo_labels_sit_beside_images(split, tmp_path):
    """YOLO는 경로의 /images/ 를 /labels/ 로 바꿔 라벨을 찾는다.
    이미지를 복사하지 않고 이 규약을 쓰는 것이 640MB를 아끼는 이유다."""
    convert_split(split, ["date"], "yolo", tmp_path / "out")
    assert (split / "labels").is_dir()
    assert {p.stem for p in (split / "labels").glob("*.txt")} == \
           {p.stem for p in (split / "images").glob("*.jpg")}


def test_non_date_marks_are_excluded_by_default(split, tmp_path):
    """`due` 마크는 날짜가 아니므로 기본 학습 대상에서 빠져야 한다."""
    res = convert_split(split, ["date"], "yolo", tmp_path / "out")
    assert res["stats"]["date"] == 2 and "due" not in res["stats"]


def test_coco_bbox_is_xywh(split, tmp_path):
    """COCO는 [x, y, w, h] 다 — ExpDate의 [x1,y1,x2,y2] 와 다르다."""
    res = convert_split(split, ["date"], "coco", tmp_path / "out")
    data = json.loads(open(res["json"], encoding="utf-8").read())
    ann = next(a for a in data["annotations"] if a["image_id"] == 1)
    assert ann["bbox"] == [20, 40, 100, 20]
    assert data["categories"] == [{"id": 0, "name": "date"}]


# ── A/B 하네스 ─────────────────────────────────────────────────────────────

def test_iou_basics():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_load_gt_prefers_exp_over_other_dates(tmp_path):
    """evaluation 에는 소비기한 외의 날짜(`date`)도 있다. 채점 대상은 `exp` 뿐이다."""
    p = tmp_path / "gt.json"
    p.write_text(json.dumps({
        "img": [{"bbox": [0, 0, 1, 1], "cls": "date", "exp": False},
                {"bbox": [9, 9, 10, 10], "cls": "exp", "exp": True}]}), encoding="utf-8")
    assert load_gt(p) == {"img": [[9, 9, 10, 10]]}


def test_load_gt_falls_back_to_date_when_no_exp(tmp_path):
    """train 분할에는 `exp` 표시가 없다."""
    p = tmp_path / "gt.json"
    p.write_text(json.dumps({
        "img": [{"bbox": [1, 2, 3, 4], "cls": "date", "exp": False}]}), encoding="utf-8")
    assert load_gt(p) == {"img": [[1, 2, 3, 4]]}


def test_recall_at_k_is_monotonic_and_uses_score_order():
    """K를 늘리면 recall 은 절대 줄지 않는다. 순서는 score 내림차순."""
    gt = {"a": [[0, 0, 10, 10]], "b": [[0, 0, 10, 10]]}
    pred = {
        "a": [[50, 50, 60, 60, 0.9], [0, 0, 10, 10, 0.1]],   # 정답이 2위
        "b": [[0, 0, 10, 10, 0.9]],                          # 정답이 1위
    }
    res = evaluate(pred, gt, max_k=3)
    curve = res["recall_at_k@iou0.3"]
    assert curve[0] == pytest.approx(0.5), "1위만 보면 b만 맞는다"
    assert curve[1] == pytest.approx(1.0), "2위까지 보면 둘 다 맞는다"
    assert curve == sorted(curve), "recall@K 는 단조 증가여야 한다"


def test_missing_predictions_count_as_misses():
    """예측 파일에 빠진 이미지는 실패로 센다 — 조용히 제외하면 점수가 부풀려진다."""
    gt = {"a": [[0, 0, 10, 10]], "b": [[0, 0, 10, 10]]}
    res = evaluate({"a": [[0, 0, 10, 10, 1.0]]}, gt)
    assert res["missing"] == 1
    assert res["recall@iou0.3"] == pytest.approx(0.5)
