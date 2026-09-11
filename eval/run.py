"""파이프라인을 돌리고 채점한다.

    python -m eval.expdate --root <ExpDate> --out labels/expdate --inspect
    python -m eval.expdate --root <ExpDate> --out labels/expdate
    python -m eval.run --images <ExpDate>/.../images --gt labels/expdate/gt_dates.csv
    python -m eval.run --images ... --mode boxes --gt labels/expdate/gt_boxes.json

정답 CSV는 출처를 가리지 않는다 — ExpDate 어댑터가 만든 것이든 손으로 라벨링한
50장이든 스키마만 같으면 그대로 채점된다.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from eval.score import format_report, load_csv, score
from itda_ocr.pipeline import Config, iter_images, run, write_rows


def _timing_report(diagnostics) -> str:
    """장당 비용을 p50/p90/p99와 함께 보고한다.

    총 처리량만 보면 꼬리가 숨고, 지연 백분위만 보면 2400초 예산을 못 본다
    — 우리 문제는 실제로 둘 다이므로 둘 다 낸다.
    """
    if not diagnostics:
        return ""
    stages = ("load", "detect", "recognize", "parse", "total")
    lines = ["", f"{'단계':<12}{'p50':>9}{'p90':>9}{'p99':>9}   (ms)"]
    for stage in stages:
        vals = sorted(d["ms"][stage] for d in diagnostics)
        q = lambda f: vals[min(int(len(vals) * f), len(vals) - 1)]  # noqa: E731
        lines.append(f"{stage:<12}{statistics.median(vals):>9.1f}{q(.90):>9.1f}{q(.99):>9.1f}")
    boxes = sorted(d["n_filtered"] for d in diagnostics)
    lines.append(f"\n인식 후보 박스 수  p50 {statistics.median(boxes):.0f} / "
                 f"p90 {boxes[int(len(boxes) * .9)]}")
    return "\n".join(lines)


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def recall_at_k(images, gt_boxes_path, cfg: Config, limit=None, max_k=20) -> str:
    """박스 필터의 recall@K — top_k 를 **재서** 정하기 위한 진단.

    Viola & Jones의 캐스케이드 규칙대로 싼 단계는 재현율이 거의 1이어야 한다.
    여기서 정답 박스를 놓치면 뒤에서 회수할 방법이 없다.

    ⚠️ GT 박스는 원본 좌표계이고 우리는 축소 디코딩(draft)한 배열에서 검출하므로,
    비교 전에 실제 배열 크기로 스케일을 맞춘다.
    """
    from PIL import Image

    from itda_ocr.engine import Engine
    from itda_ocr.pipeline import load_image

    gt = json.loads(Path(gt_boxes_path).read_text(encoding="utf-8"))
    engine = Engine(det_side=cfg.det_side, threads=cfg.threads,
                    box_thresh=cfg.box_thresh, unclip_ratio=cfg.unclip_ratio,
                    nanodet_onnx=cfg.nanodet_onnx,
                    nanodet_score_thr=cfg.nanodet_score_thr,
                    nanodet_expand=cfg.nanodet_expand,
                    nanodet_nms_iou=cfg.nanodet_nms_iou)

    paths = iter_images(images)
    if limit:
        paths = paths[:limit]

    hits = [0] * (max_k + 1)
    evaluated = 0
    for path in paths:
        targets = [b["bbox"] for b in gt.get(path.stem, []) if b.get("exp")] or \
                  [b["bbox"] for b in gt.get(path.stem, []) if b.get("cls") == "date"]
        if not targets:
            continue
        with Image.open(path) as probe:
            ow, oh = probe.size
        img = load_image(path, cfg.draft_to)
        h, w = img.shape[:2]
        sx, sy = w / ow, h / oh
        targets = [(t[0] * sx, t[1] * sy, t[2] * sx, t[3] * sy) for t in targets]

        kept = engine.detect_and_filter(img)
        evaluated += 1
        for k in range(1, max_k + 1):
            found = False
            for _, _, box in kept[:k]:
                xs, ys = box[:, 0], box[:, 1]
                cand = (xs.min(), ys.min(), xs.max(), ys.max())
                if any(_iou(cand, t) >= 0.3 for t in targets):
                    found = True
                    break
            hits[k] += found

    if not evaluated:
        return "GT 박스와 매칭되는 이미지가 없다 — gt_boxes.json 과 --images 가 맞는지 확인할 것."
    lines = [f"박스 필터 recall@K  (이미지 {evaluated}장, IoU ≥ 0.3)", "",
             f"{'K':>4}{'recall':>10}   비고"]
    for k in range(1, max_k + 1):
        note = "← 인식 비용 ≈ K × 20ms" if k == cfg.top_k else ""
        lines.append(f"{k:>4}{hits[k] / evaluated:>10.1%}   {note}")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="파이프라인 실행 + 채점")
    ap.add_argument("--images", help="입력 이미지 디렉터리")
    ap.add_argument("--gt", help="정답 파일 (gt_dates.csv 또는 gt_boxes.json)")
    ap.add_argument("--predictions", help="이미 만든 submission.csv 를 채점만 한다")
    ap.add_argument("--out", default="results", help="산출 디렉터리")
    ap.add_argument("--mode", choices=["score", "boxes"], default="score")
    ap.add_argument("--limit", type=int, help="앞 N장만")
    ap.add_argument("--det-side", type=int, default=Config.det_side)
    ap.add_argument("--top-k", type=int, default=Config.top_k)
    ap.add_argument("--max-k", type=int, default=Config.max_k)
    ap.add_argument("--threads", type=int, default=Config.threads)
    ap.add_argument("--nanodet", help="날짜 전용 검출기 ONNX 경로 (지정 시 detect 를 대체)")
    ap.add_argument("--nanodet-score-thr", type=float, default=Config.nanodet_score_thr)
    def _parse_expand(val):
        if isinstance(val, (tuple, list)):
            return tuple(val)
        if "," in str(val):
            return tuple(float(x.strip()) for x in str(val).split(","))
        return float(val)

    ap.add_argument("--nanodet-expand", type=_parse_expand, default=Config.nanodet_expand,
                    help="NanoDet 박스를 인식 전에 넓히는 비율 (0.15,0.08 또는 단일 float)")
    ap.add_argument("--nanodet-nms-iou", type=float, default=Config.nanodet_nms_iou,
                    help="NanoDet NMS IoU 임계값 (기본 0.60)")
    args = ap.parse_args(argv)

    cfg = Config(det_side=args.det_side, top_k=args.top_k, max_k=args.max_k,
                 threads=args.threads,
                 nanodet_onnx=args.nanodet, nanodet_score_thr=args.nanodet_score_thr,
                 nanodet_expand=args.nanodet_expand,
                 nanodet_nms_iou=args.nanodet_nms_iou)

    if args.mode == "boxes":
        if not (args.images and args.gt):
            ap.error("--mode boxes 에는 --images 와 --gt(gt_boxes.json) 가 필요하다")
        print(recall_at_k(args.images, args.gt, cfg, args.limit))
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = []

    if args.predictions:
        predictions = load_csv(args.predictions)
    else:
        if not args.images:
            ap.error("--images 또는 --predictions 가 필요하다")
        source = args.images
        if args.limit:                      # 앞 N장만 임시 디렉터리에 링크하지 않고
            paths = iter_images(source)[:args.limit]   # 직접 목록으로 처리한다
            rows, diagnostics = [], []
            from itda_ocr.engine import Engine
            from itda_ocr.pipeline import process_image
            engine = Engine(det_side=cfg.det_side, threads=cfg.threads,
                            box_thresh=cfg.box_thresh, unclip_ratio=cfg.unclip_ratio,
                            nanodet_onnx=cfg.nanodet_onnx,
                            nanodet_score_thr=cfg.nanodet_score_thr,
                            nanodet_expand=cfg.nanodet_expand,
                            nanodet_nms_iou=cfg.nanodet_nms_iou)
            for p in paths:
                try:
                    row = process_image(engine, p, cfg)
                    diagnostics.append({"image_id": row["image_id"], **row.pop("_diag")})
                    rows.append(row)
                except Exception:                      # noqa: BLE001
                    rows.append({"image_id": p.stem, "year": "NONE", "month": "NONE",
                                 "day": "NONE", "final_date": "NONE"})
            write_rows(out_dir / "submission.csv", rows)
            result = {"n": len(rows), "rows": rows, "diagnostics": diagnostics}
        else:
            result = run(source, out_dir / "submission.csv", cfg,
                         collect_diag=True)
            diagnostics = result["diagnostics"]
        predictions = {r["image_id"]: r for r in result["rows"]}
        print(f"처리 {len(predictions)}장 → {out_dir / 'submission.csv'}")

    if not args.gt:
        print("(--gt 가 없어 채점은 건너뛴다)")
        print(_timing_report(diagnostics))
        return

    gt = load_csv(args.gt)
    cands = {d["image_id"]: d.get("candidates", []) for d in diagnostics}
    result = score(predictions, gt, cands)
    print()
    print(format_report(result))
    print(_timing_report(diagnostics))

    (out_dir / "report.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "per_image"},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # 틀린 것만 따로 남긴다 — 눈으로 볼 목록이 곧 다음 작업이다.
    wrong = [{"image_id": i, "score": s, **{k: predictions.get(i, {}).get(k, "")
                                            for k in ("final_date",)},
              "gt": gt[i]["final_date"]}
             for i, s in sorted(result["per_image"].items(), key=lambda p: p[1])
             if s < 50]
    (out_dir / "failures.json").write_text(
        json.dumps(wrong[:200], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n실패 {len(wrong)}건 → {out_dir / 'failures.json'}")


if __name__ == "__main__":
    main()
