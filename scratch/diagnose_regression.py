import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.score import load_csv, score_row
from itda_ocr.engine import Engine
from itda_ocr.pipeline import Config, process_image

reg_ids = ['test_00002', 'test_00008', 'test_00020', 'test_00064', 'test_00111', 'test_00134', 'test_00137', 'test_00153', 'test_00161', 'test_00207']
gt = load_csv('labels/expdate/gt_dates.csv')

# 1. CTC masking O, expand=0.10
cfg_1 = Config(max_k=20, nanodet_onnx='weights/date_detector_ema.onnx', nanodet_expand=0.10)
eng_1 = Engine(nanodet_onnx='weights/date_detector_ema.onnx', nanodet_nms_iou=0.60, nanodet_expand=0.10)

# 2. CTC masking X, expand=0.10
eng_0 = Engine(nanodet_onnx='weights/date_detector_ema.onnx', nanodet_nms_iou=0.60, nanodet_expand=0.10)
eng_0._rec.postprocess_op = eng_0._rec.postprocess_op.original_op

# 3. BOTH (expand 0.15, 0.08 + CTC mask)
cfg_both = Config(max_k=20, nanodet_onnx='weights/date_detector_ema.onnx', nanodet_expand=(0.15, 0.08))
eng_both = Engine(nanodet_onnx='weights/date_detector_ema.onnx', nanodet_nms_iou=0.60, nanodet_expand=(0.15, 0.08))

print("ID | GT | BASE(0.10, NoMask) | MASK_ONLY(0.10, Mask) | BOTH(0.15, Mask)")
for img_id in reg_ids:
    p = Path('kist_data/evaluation/images') / f'{img_id}.jpg'
    r_base = process_image(eng_0, p, cfg_1)
    r_mask = process_image(eng_1, p, cfg_1)
    r_both = process_image(eng_both, p, cfg_both)
    
    gt_val = gt[img_id]['final_date']
    b_val = r_base['final_date']
    m_val = r_mask['final_date']
    both_val = r_both['final_date']
    print(f"{img_id} | GT={gt_val} | BASE={b_val} | MASK_ONLY={m_val} | BOTH={both_val}")
