"""날짜 검출기 A/B — 범용 텍스트 검출(현행) vs 날짜 전용 검출(Track B).

선행연구에 이 통제 비교가 **존재하지 않는다**(`docs/DETECTOR_PLAN.md` §3).
그래서 이 스크립트가 그 비교를 대신한다. 우리 프로젝트의 방법론적 기여이기도 하다.

## 프레임워크에 묶이지 않는다

새 검출기를 이 저장소에 이식할 필요가 없다. 어떤 프레임워크로 학습하든
**예측 JSON 하나만** 내보내면 된다:

    { "<image_id>": [[x1, y1, x2, y2, score], ...], ... }

- `image_id` 는 확장자를 뺀 파일명 (`test_00001`)
- 좌표는 **원본 이미지 픽셀 기준** (리사이즈 전). 우리 쪽에서 알아서 맞춘다
- `score` 는 내림차순 정렬에만 쓴다. 스케일은 상관없다

## 사용

    # 1) 현행 기준선을 같은 포맷으로 뽑는다
    python -m eval.detector_ab --baseline --out results/det_baseline.json

    # 2) 새 검출기 예측과 비교한다
    python -m eval.detector_ab --pred results/det_yolo.json \
                               --compare results/det_baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GT_DEFAULT = "labels/expdate/gt_boxes.json"
IMAGES_DEFAULT = "kist_data/evaluation/images"


def iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_gt(path) -> dict:
    """image_id → 소비기한 박스 목록 (원본 좌표)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    gt = {}
    for image_id, boxes in raw.items():
        want = [b["bbox"] for b in boxes if b.get("exp")]
        if not want:                      # train 쪽은 exp 표시가 없다 → date 사용
            want = [b["bbox"] for b in boxes if b.get("cls") == "date"]
        if want:
            gt[image_id] = want
    return gt


def evaluate(pred: dict, gt: dict, thresholds=(0.3, 0.5), max_k: int = 12) -> dict:
    """검출 recall 과 recall@K. 우리 파이프라인에서 중요한 건 이 둘뿐이다.

    mAP 는 내지 않는다 — 우리는 박스 하나만 인식으로 넘기므로
    "정답이 상위 K 안에 있는가"가 유일하게 점수로 연결되는 지표다.
    """
    out = {"n_images": len(gt), "n_predicted": 0, "missing": 0}
    boxes_per_image = []
    for thr in thresholds:
        hits_any = 0
        hits_at = [0] * (max_k + 1)
        for image_id, targets in gt.items():
            boxes = pred.get(image_id)
            if boxes is None:
                continue
            ranked = sorted(boxes, key=lambda b: -(b[4] if len(b) > 4 else 0.0))
            if thr == thresholds[0]:
                boxes_per_image.append(len(ranked))
            if any(iou(b[:4], t) >= thr for b in ranked for t in targets):
                hits_any += 1
            for k in range(1, max_k + 1):
                if any(iou(b[:4], t) >= thr for b in ranked[:k] for t in targets):
                    hits_at[k] += 1
        n = len(gt) or 1
        out[f"recall@iou{thr}"] = hits_any / n
        out[f"recall_at_k@iou{thr}"] = [hits_at[k] / n for k in range(1, max_k + 1)]

    out["missing"] = sum(1 for i in gt if i not in pred)
    out["n_predicted"] = len(pred)
    out["boxes_per_image_mean"] = (
        sum(boxes_per_image) / len(boxes_per_image) if boxes_per_image else 0.0)
    return out


def format_report(name: str, res: dict, other=None, other_name="") -> str:
    lines = [f"## {name}", f"  이미지 {res['n_images']}장 / 예측 있는 이미지 {res['n_predicted']}장"
             + (f"  ⚠️ 누락 {res['missing']}장" if res["missing"] else ""),
             f"  장당 박스 수 평균 {res['boxes_per_image_mean']:.1f}",
             f"  검출 recall (IoU 0.3) {res['recall@iou0.3']:.1%}"
             f"   (IoU 0.5) {res['recall@iou0.5']:.1%}", ""]
    header = f"  {'K':>3}{'recall@K':>11}"
    if other:
        header += f"{other_name[:12]:>13}{'차이':>9}"
    lines.append(header)
    cur = res["recall_at_k@iou0.3"]
    ref = other["recall_at_k@iou0.3"] if other else None
    for k in (1, 2, 3, 5, 8, 12):
        row = f"  {k:>3}{cur[k-1]:>11.1%}"
        if ref:
            delta = cur[k-1] - ref[k-1]
            row += f"{ref[k-1]:>13.1%}{delta:>+9.1%}"
        lines.append(row)
    return "\n".join(lines)


def run_baseline(images, out_path, det_side=640, draft_to=720, limit=None) -> dict:
    """현행 범용 텍스트 검출기를 같은 예측 포맷으로 내보낸다.

    ⚠️ 우리 파이프라인은 축소 디코딩(draft)한 배열에서 검출하므로,
    비교 가능하도록 **원본 좌표계로 되돌려** 저장한다.
    """
    from PIL import Image

    from itda_ocr.engine import Engine
    from itda_ocr.pipeline import iter_images, load_image

    engine = Engine(det_side=det_side, threads=4)
    paths = iter_images(images)
    if limit:
        paths = paths[:limit]

    pred = {}
    for path in paths:
        with Image.open(path) as probe:
            ow, oh = probe.size
        img = load_image(path, draft_to)
        h, w = img.shape[:2]
        sx, sy = ow / w, oh / h            # 축소본 → 원본
        kept = engine.detect_and_filter(img)
        boxes = []
        for rank, (score, _, box) in enumerate(kept):
            xs, ys = box[:, 0], box[:, 1]
            boxes.append([float(xs.min()*sx), float(ys.min()*sy),
                          float(xs.max()*sx), float(ys.max()*sy),
                          float(-rank)])   # 필터 순위를 점수로 (내림차순 유지)
        pred[path.stem] = boxes
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(pred), encoding="utf-8")
    return pred


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="날짜 검출기 A/B")
    ap.add_argument("--pred", help="새 검출기 예측 JSON")
    ap.add_argument("--compare", help="비교 대상 예측 JSON (보통 기준선)")
    ap.add_argument("--baseline", action="store_true",
                    help="현행 검출기를 돌려 예측 JSON을 만든다")
    ap.add_argument("--gt", default=GT_DEFAULT)
    ap.add_argument("--images", default=IMAGES_DEFAULT)
    ap.add_argument("--out", default="results/det_baseline.json")
    ap.add_argument("--det-side", type=int, default=640)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    gt = load_gt(args.gt)
    if not gt:
        raise SystemExit(f"{args.gt} 에서 정답 박스를 읽지 못했다")

    if args.baseline:
        pred = run_baseline(args.images, args.out, args.det_side, limit=args.limit)
        print(f"기준선 예측 {len(pred)}장 → {args.out}\n")
        print(format_report(f"현행 범용 텍스트 검출 (det_side={args.det_side})",
                            evaluate(pred, gt)))
        return

    if not args.pred:
        ap.error("--pred 또는 --baseline 이 필요하다")

    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    res = evaluate(pred, gt)
    other = other_name = None
    if args.compare:
        ref = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        other, other_name = evaluate(ref, gt), Path(args.compare).stem
    print(format_report(Path(args.pred).stem, res, other, other_name or ""))

    if other:
        gain = res["recall@iou0.3"] - other["recall@iou0.3"]
        print(f"\n검출 recall 차이 {gain:+.1%}")
        print("→ 게이트: 기준선을 넘지 못하면 Track B를 중단한다 "
              "(docs/DETECTOR_PLAN.md §6 B1)")


if __name__ == "__main__":
    main()
