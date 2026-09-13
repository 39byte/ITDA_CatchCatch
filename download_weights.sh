#!/usr/bin/env bash
# [ITDA 3rd] 소비기한 추출 - 가중치 확인 스크립트
# 채점 환경: 인터넷 차단 전 운영진이 1회 실행
set -e

echo "=== [ITDA] Checking Model Weights ==="

# 1. NanoDet-Plus-m 날짜 전용 검출기 가중치 (5.6MB)
#    경량 모델이므로 저장소 weights/date_detector_ema.onnx 에 직접 포함되어 있습니다.
if [ -f "weights/date_detector_ema.onnx" ]; then
    echo "[OK] weights/date_detector_ema.onnx found."
else
    echo "[WARNING] weights/date_detector_ema.onnx not found. Pipeline will fall back to RapidOCR default detector."
fi

# 2. RapidOCR 가중치 (det/cls/rec ONNX 약 16MB)
#    rapidocr-onnxruntime==1.4.4 wheel 내부에 포함되어 pip install 만으로 준비됩니다.
echo "[OK] RapidOCR default models are bundled within the python wheel package."

echo "=== Model weight verification complete. ==="
