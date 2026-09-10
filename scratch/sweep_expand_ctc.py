import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import cv2
import numpy as np
from eval.score import load_csv, score_row
from itda_ocr.engine import Engine
from itda_ocr.pipeline import Config, process_image

gt = load_csv('labels/expdate/gt_dates.csv')
sub = load_csv('results/merge_fixed/submission.csv')
diff_ids = [img_id for img_id, r in gt.items() if score_row(sub.get(img_id, {}), r) < 50]
print(f'Total errors: {len(diff_ids)}')

def apply_ctc_mask(engine):
    chars = engine._rec.postprocess_op.character
    allowed_set = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-/:, ()[]~年月日')
    disallowed_indices = np.array([i for i, c in enumerate(chars) if i != 0 and c != 'blank' and c not in allowed_set], dtype=np.int64)

    orig_op = engine._rec.postprocess_op
    class MaskedPostProcess:
        def __init__(self, op, disallowed):
            self.op = op
            self.disallowed = disallowed
            self.character = op.character
        def __getattr__(self, name):
            return getattr(self.op, name)
        def __call__(self, preds, *args, **kwargs):
            preds = preds.copy()
            preds[:, :, self.disallowed] = -np.inf
            return self.op(preds, *args, **kwargs)

    engine._rec.postprocess_op = MaskedPostProcess(orig_op, disallowed_indices)

# (expand_x, expand_y) 조합 테스트
configs = [
    (0.10, 0.10),  # baseline
    (0.12, 0.08),
    (0.15, 0.08),
    (0.15, 0.10),
    (0.18, 0.08),
]

for exp_x, exp_y in configs:
    cfg = Config(max_k=20, nanodet_onnx='weights/date_detector_ema.onnx', nanodet_expand=exp_x)
    engine = Engine(nanodet_onnx='weights/date_detector_ema.onnx',
                    nanodet_nms_iou=0.60,
                    nanodet_score_thr=cfg.nanodet_score_thr,
                    nanodet_expand=exp_x)
    apply_ctc_mask(engine)
    
    # nanodet detector detect 메서드의 expand 비대칭 패치
    def custom_detect(img, orig_det=engine._nanodet.detect, ex=exp_x, ey=exp_y):
        h, w = img.shape[:2]
        S = engine._nanodet.input_size
        resized = engine._nanodet._cv2.resize(img, (S, S), interpolation=engine._nanodet._cv2.INTER_LINEAR)
        x = (resized.astype(np.float32) - np.array([103.53, 116.28, 123.675], dtype=np.float32)) / np.array([57.375, 57.12, 58.395], dtype=np.float32)
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])

        out = engine._nanodet._sess.run(None, {engine._nanodet._inp: x})[0][0]
        scores = out[:, 0]
        m = scores >= engine._nanodet.score_thr
        if not m.any():
            return np.empty((0, 4, 2), dtype=np.float32)

        from itda_ocr.nanodet_det import _softmax, _nms, _REG_MAX
        priors = engine._nanodet._priors[m]
        reg = out[m, 1:].reshape(-1, 4, _REG_MAX + 1)
        dist = (_softmax(reg, axis=-1) * engine._nanodet._proj).sum(-1) * priors[:, 2:3]
        cx, cy = priors[:, 0], priors[:, 1]
        boxes = np.stack([cx - dist[:, 0], cy - dist[:, 1],
                          cx + dist[:, 2], cy + dist[:, 3]], axis=-1)
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, S)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, S)
        sc = scores[m]

        keep = _nms(boxes, sc, engine._nanodet.nms_iou)[: engine._nanodet.max_det]
        boxes, sc = boxes[keep], sc[keep]

        boxes[:, 0::2] *= w / S
        boxes[:, 1::2] *= h / S

        order = sc.argsort()[::-1]
        quad = np.empty((len(order), 4, 2), dtype=np.float32)
        for j, i in enumerate(order):
            x0, y0, x1, y1 = boxes[i]
            dx = (x1 - x0) * ex
            dy = (y1 - y0) * ey
            x0, y0, x1, y1 = x0 - dx, y0 - dy, x1 + dx, y1 + dy
            quad[j] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        return quad

    engine._nanodet.detect = custom_detect

    score_delta = 0
    fixed = 0
    regressed = 0
    for img_id in diff_ids:
        img_path = Path('kist_data/evaluation/images') / f'{img_id}.jpg'
        res = process_image(engine, img_path, cfg)
        s_orig = score_row(sub.get(img_id, {}), gt[img_id])
        s_new = score_row(res, gt[img_id])
        delta = s_new - s_orig
        score_delta += delta
        if delta > 0:
            fixed += 1
        elif delta < 0:
            regressed += 1
    print(f"Expand (x={exp_x:.2f}, y={exp_y:.2f}): Delta = {score_delta:+d} pt (Fixed: {fixed}, Regressed: {regressed})")
