"""NanoDet(Track B) 검출기로 이미지 폴더에 추론 → eval.detector_ab 예측 JSON 생성.

HANDOVER.md §4-3 포맷:  {"<image_id>": [[x1, y1, x2, y2, score], ...], ...}
좌표는 **원본 이미지 픽셀** 기준 (nanodet post_process 가 warp_matrix 역변환으로 복원).

이 스크립트는 nanodet 저장소 코드를 import 하므로, nanodet 전용 venv + PYTHONPATH 로 돌린다:

    NANODET=~/Desktop/nanodet
    PYTHONPATH=$NANODET  $NANODET/.venv/bin/python tools/nanodet_predict.py \
        --config  $NANODET/config/nanodet-plus-m_480_date.yml \
        --model   weights/date_detector_ema.pth \
        --images  expdate/evaluation/images \
        --out     results/det_nanodet.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import torch

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class Predictor:
    """demo/demo.py 의 Predictor 를 그대로 옮긴 것 (시각화 부분만 제거)."""

    def __init__(self, cfg, model_path, logger, device="cpu"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location="cpu")
        load_model_weight(model, ckpt, logger)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            return {}
        h, w = img.shape[:2]
        img_info = {"id": 0, "file_name": Path(img_path).name, "height": h, "width": w}
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        # results: {img_id: {class_id: [[x0,y0,x1,y1,score], ...]}}  — 원본 좌표
        return results[0]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="results/det_nanodet.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--score-thr", type=float, default=0.05,
                    help="이 점수 미만 박스는 버린다 (nanodet 기본 0.05)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    load_config(cfg, args.config)
    logger = Logger(-1, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device=args.device)

    paths = sorted(p for p in Path(args.images).glob("*.*")
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        paths = paths[:args.limit]

    pred: dict[str, list] = {}
    n_boxes = 0
    t0 = time.time()
    for i, path in enumerate(paths):
        dets = predictor.inference(path)
        boxes = []
        for _cls, arr in (dets or {}).items():        # 단일 클래스라 _cls==0
            for x0, y0, x1, y1, score in arr:
                if score >= args.score_thr:
                    boxes.append([float(x0), float(y0), float(x1), float(y1), float(score)])
        boxes.sort(key=lambda b: -b[4])
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
