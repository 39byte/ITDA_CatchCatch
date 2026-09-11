"""NanoDet(Track B) 검출기로 이미지 폴더에 추론 → eval.detector_ab 예측 JSON 생성.

HANDOVER.md §4-3 포맷:  {"<image_id>": [[x1, y1, x2, y2, score], ...], ...}
좌표는 **원본 이미지 픽셀** 기준.

ONNX 경로(`itda_ocr.nanodet_det.NanoDetDetector`)만 쓴다 — torch·nanodet 저장소 불필요.
저장소에 포함된 venv(`pip install -r requirements.txt`)로 바로 돌아간다.

    python -m tools.nanodet_predict \
        --model  weights/date_detector_ema.onnx \
        --images expdate/evaluation/images \
        --out    results/det_nanodet.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from itda_ocr.nanodet_det import NanoDetDetector
from itda_ocr.pipeline import Config, iter_images, load_image


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/date_detector_ema.onnx")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="results/det_nanodet.json")
    ap.add_argument("--score-thr", type=float, default=0.05,
                    help="이 점수 미만 박스는 버린다 (recall 평가는 낮게, 파이프라인 기본 0.05)")
    ap.add_argument("--draft-to", type=int, default=Config.draft_to,
                    help="0 이면 축소 디코딩 없이 원본에서 검출")
    ap.add_argument("--threads", type=int, default=Config.threads)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    det = NanoDetDetector(args.model, threads=args.threads, score_thr=args.score_thr)

    paths = iter_images(args.images)
    if args.limit:
        paths = paths[:args.limit]

    pred: dict[str, list] = {}
    n_boxes, t0 = 0, time.time()
    for i, path in enumerate(paths):
        with Image.open(path) as probe:
            ow, oh = probe.size
        img = load_image(path, args.draft_to)                 # 파이프라인과 같은 프레임
        h, w = img.shape[:2]
        sx, sy = ow / w, oh / h                               # 축소본 -> 원본
        quads, scores = det.detect(img)                       # (N,4,2), (N,) — 점수 내림차순
        boxes = []
        for q, sc in zip(quads, scores):
            x0, y0 = q[0]
            x1, y1 = q[2]
            boxes.append([float(x0 * sx), float(y0 * sy),
                          float(x1 * sx), float(y1 * sy), float(sc)])
        pred[path.stem] = boxes
        n_boxes += len(boxes)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(paths)}  ({(time.time() - t0) / (i + 1) * 1000:.0f} ms/img)",
                  flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(pred), encoding="utf-8")
    empty = sum(1 for v in pred.values() if not v)
    print(f"\n{len(pred)}장 → {args.out}")
    print(f"박스 총 {n_boxes}개 / 장당 평균 {n_boxes / max(len(pred), 1):.1f} / 검출 0인 이미지 {empty}장")


if __name__ == "__main__":
    main()
