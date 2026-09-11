"""현재 파이프라인의 오답을 **파이프라인 단계별 원인**으로 분류한다.

eval/score.py 의 taxonomy 와 다른 점: 여기서는 검출기 박스와 GT 박스를 대조해
`no_candidate` 를 **검출 실패**(박스 자체가 없음)와 **순위 실패**(박스는 찾았으나
top-K 밖이라 인식 안 됨)로 쪼갠다.

    .venv/bin/python -m eval.failure_taxonomy \
        --images expdate/evaluation/images \
        --gt-dates labels/expdate/gt_dates.csv \
        --gt-boxes labels/expdate/gt_boxes.json \
        --out results/failure_taxonomy.json

5개 카테고리:
  정답        4필드 전부 정답 (score_row == 50)
  검출 실패    GT 날짜 박스와 IoU>=0.3 로 겹치는 검출 박스가 (필터 통과분 중) 없음
  순위 실패    GT 박스는 검출됐으나 최종 답이 틀림 —
              (a) 정답으로 파싱된 후보가 있었는데 다른 걸 선택했거나,
              (b) GT 박스가 top-K 밖이라 인식 대상조차 안 됨
  파싱 실패    GT 박스를 인식했고 텍스트에 정답 날짜 숫자가 들어있으나 날짜로 파싱 실패
  오독        GT 박스를 인식했으나 글자를 잘못 읽음 (텍스트에 정답 숫자 없음)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval.score import _norm, _text_holds_gt, score_row
from itda_ocr.engine import Engine, filter_boxes
from itda_ocr.parse import parse_boxes
from itda_ocr.pipeline import Config, iter_images, load_image
from itda_ocr.select import STOP_SCORE, score as sel_score, select, to_row


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _box_xyxy(box):
    xs, ys = box[:, 0], box[:, 1]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def run_one(engine: Engine, path, cfg: Config):
    """process_image 를 재현하되 박스 좌표·크롭·후보를 전부 붙잡는다.

    ⚠️ engine.detect() / crop 이 도는 좌표계는 **draft 축소된 배열** 프레임이다.
    GT 박스는 원본 해상도 좌표이므로, 반환하는 좌표(kept_xyxy/recog_xyxy)를
    원본 프레임으로 되돌려 놓는다 (eval/run.py recall_at_k 와 동일한 처리).
    """
    from PIL import Image
    with Image.open(path) as probe:
        ow, oh = probe.size
    img = load_image(path, cfg.draft_to)
    h, w = img.shape[:2]
    sx, sy = ow / w, oh / h                              # 축소본 -> 원본

    def to_orig(g):
        return (g[0] * sx, g[1] * sy, g[2] * sx, g[3] * sy)

    det_boxes, det_scores = engine.detect(img)
    kept = filter_boxes(det_boxes, det_scores, img.shape)   # (prior, idx, box, score) 내림차순
    kept_xyxy = [to_orig(_box_xyxy(b)) for _, _, b, _ in kept]

    recog_xyxy, recog_texts, recog_confs, candidates = [], [], [], []
    items = []
    for start in range(0, min(len(kept), cfg.max_k), cfg.top_k):
        crops, geoms, dconfs = [], [], []
        for _, _, box, det_conf in kept[start:start + cfg.top_k]:
            patch = engine.crop(img, box)
            if patch.size:
                crops.append(patch)
                geoms.append(to_orig(_box_xyxy(box)))
                dconfs.append(float(det_conf))
        if not crops:
            continue
        texts = engine.recognize(crops)
        for (t, rec_sc), g, dc in zip(texts, geoms, dconfs):
            recog_texts.append(str(t))
            recog_xyxy.append(g)
            recog_confs.append(dc * rec_sc)
        items = [(t, g[0], g[1], g[2], g[3], c)
                 for t, g, c in zip(recog_texts, recog_xyxy, recog_confs)]
        candidates = parse_boxes(items)
        if any(sel_score(c, "") >= STOP_SCORE for c in candidates):
            break

    full_text = " ".join(recog_texts)
    winner = select(candidates, full_text, cfg.impute_missing)
    row = to_row(winner, Path(path).stem)
    return {
        "row": row,
        "kept_xyxy": kept_xyxy,
        "recog_xyxy": recog_xyxy,
        "recog_texts": recog_texts,
        "candidates": [{"text": c.text, "final_date": c.final_date, "source": c.source}
                       for c in candidates],
    }


def classify(diag, gt_row, gt_box, iou_thr=0.3) -> str:
    row = diag["row"]
    if score_row(row, gt_row) == 50:
        return "정답"

    # 1. 검출 실패 — 필터 통과 박스 중 GT 와 겹치는 게 없다
    if not any(iou(b, gt_box) >= iou_thr for b in diag["kept_xyxy"]):
        return "검출 실패"

    gt_final = _norm(gt_row.get("final_date"))

    # 2a. 순위 실패 — 정답으로 파싱된 후보가 있었는데 다른 걸 선택
    if gt_final != "NONE" and any(_norm(c["final_date"]) == gt_final for c in diag["candidates"]):
        return "순위 실패"

    # GT 박스가 실제로 인식된 크롭 중에 있었나
    gt_crops = [i for i, b in enumerate(diag["recog_xyxy"]) if iou(b, gt_box) >= iou_thr]
    if gt_crops:
        gt_text = " ".join(diag["recog_texts"][i] for i in gt_crops)
        # 3. 파싱 실패 — 텍스트는 맞게 읽었는데 날짜로 해석 못 함
        if _text_holds_gt(gt_text, gt_row):
            return "파싱 실패"
        # 4. 오독 — 글자를 잘못 읽음
        return "오독"

    # 2b. 순위 실패 — GT 박스는 검출됐으나 top-K 밖이라 인식조차 안 됨
    return "순위 실패"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt-dates", required=True)
    ap.add_argument("--gt-boxes", required=True)
    ap.add_argument("--out", default="results/failure_taxonomy.json")
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    import csv
    gt_dates = {r["image_id"]: r for r in csv.DictReader(open(args.gt_dates, encoding="utf-8-sig"))}
    gt_boxes_raw = json.loads(Path(args.gt_boxes).read_text(encoding="utf-8"))
    gt_box = {}
    for iid, boxes in gt_boxes_raw.items():
        exp = [b["bbox"] for b in boxes if b.get("exp")]
        if exp:
            gt_box[iid] = tuple(exp[0])

    cfg = Config()
    engine = Engine(det_side=cfg.det_side, threads=cfg.threads,
                    box_thresh=cfg.box_thresh, unclip_ratio=cfg.unclip_ratio)

    paths = iter_images(args.images)
    if args.limit:
        paths = paths[:args.limit]

    counts = Counter()
    per_image = {}
    for i, path in enumerate(paths):
        iid = path.stem
        if iid not in gt_dates or iid not in gt_box:
            continue
        try:
            diag = run_one(engine, path, cfg)
            cat = classify(diag, gt_dates[iid], gt_box[iid], args.iou_thr)
        except Exception as e:                              # noqa: BLE001
            cat = "검출 실패"
            diag = {"error": repr(e)}
        counts[cat] += 1
        per_image[iid] = cat
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)

    n = sum(counts.values())
    order = ["정답", "검출 실패", "순위 실패", "파싱 실패", "오독"]
    print(f"\n{'원인':<10}{'건수':>8}{'비중':>10}")
    for k in order:
        c = counts.get(k, 0)
        print(f"{k:<10}{c:>8}{c / n:>9.1%}")
    print(f"{'합계':<10}{n:>8}{1.0:>9.1%}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"n": n, "counts": dict(counts), "per_image": per_image,
         "iou_thr": args.iou_thr}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
