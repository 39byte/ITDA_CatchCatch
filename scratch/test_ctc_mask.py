import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import cv2
import numpy as np
from pathlib import Path
from eval.score import load_csv, score_row
from itda_ocr.engine import Engine
from itda_ocr.pipeline import Config, process_image

# 오답 목록 로드
gt = load_csv('labels/expdate/gt_dates.csv')
sub = load_csv('results/merge_fixed/submission.csv')
diff_ids = [img_id for img_id, r in gt.items() if score_row(sub.get(img_id, {}), r) < 50]
print(f'Total errors: {len(diff_ids)}')

cfg = Config(max_k=20, nanodet_onnx='weights/date_detector_ema.onnx')
engine = Engine(nanodet_onnx='weights/date_detector_ema.onnx',
                nanodet_nms_iou=0.60,
                nanodet_score_thr=cfg.nanodet_score_thr,
                nanodet_expand=cfg.nanodet_expand)

# 마스킹 적용
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

fixed_cnt = 0
regressed_cnt = 0
score_delta = 0

for img_id in diff_ids:
    img_path = Path('kist_data/evaluation/images') / f'{img_id}.jpg'
    res = process_image(engine, img_path, cfg)
    pred_row = res
    gt_row = gt[img_id]
    
    s_orig = score_row(sub.get(img_id, {}), gt_row)
    s_new = score_row(pred_row, gt_row)
    delta = s_new - s_orig
    score_delta += delta
    
    if delta > 0:
        fixed_cnt += 1
        print(f"FIXED {img_id}: GT={gt_row['final_date']} | OLD={sub.get(img_id,{}).get('final_date')} ({s_orig}pt) -> NEW={pred_row['final_date']} ({s_new}pt, +{delta})")
    elif delta < 0:
        regressed_cnt += 1
        print(f"REGRESSED {img_id}: GT={gt_row['final_date']} | OLD={sub.get(img_id,{}).get('final_date')} ({s_orig}pt) -> NEW={pred_row['final_date']} ({s_new}pt, {delta})")

print(f"Total delta: {score_delta} pt. Fixed: {fixed_cnt}, Regressed: {regressed_cnt}")
