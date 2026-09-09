"""NanoDet-Plus-m 날짜 전용 검출기 (ONNX) — Track B.

`Engine.detect()` 와 **같은 계약**을 지킨다: 입력 `img` (BGR np.ndarray, draft 축소된
파이프라인 프레임), 반환은 그 프레임 좌표계의 박스 ``(N, 4, 2) float32``.
점수 내림차순으로 정렬해서 돌려준다. 범용 검출기와 달리 이 점수는 "날짜다움"
그 자체라 **이 순서가 곧 최종 순위여야 한다** — ``filter_boxes`` 의 기본 종횡비
재정렬을 덮어씌우면 소비기한 박스 recall@1 이 84.7% → 50.2% 로 무너진다.
그래서 ``Engine.detect_and_filter()`` 가 NanoDet 경로에서는 재정렬을 끈다.

디코딩은 nanodet `NanoDetPlusHead.get_bboxes` 를 numpy 로 옮긴 것:
Integral(분포→거리) → distance2bbox → sigmoid → NMS. ONNX 는 head raw 출력
``(1, 4789, 33)`` 까지만 담고 있어 후처리는 여기서 한다.
"""

from __future__ import annotations

import numpy as np

_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)   # BGR (cv2.imread 순서)
_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
_STRIDES = (8, 16, 32, 64)
_REG_MAX = 7

#: 반환 박스를 각 변으로 넓히는 비율. **0 이면 안 된다.**
#:
#: RapidOCR DB 검출기는 ``unclip_ratio=1.6`` 으로 이미 부풀린 박스를 주는데
#: NanoDet 회귀 박스는 글자에 딱 맞게 나온다. ``Engine.crop()`` 은 박스를 그대로
#: 자르므로, 검출기만 바꾸면 크롭 경계에서 첫·마지막 글자가 잘려나간다
#: (``'2021.08.04'`` → ``'2021.08.0'``). 실제로 병합 직후 665장 중 88장이
#: 이렇게 퇴행했고 그중 37장이 "연·월은 맞고 일자만 틀림"이었다.
#:
#: 값은 ExpDate 665장 스윕으로 정했다 (0.00~0.30, 10점):
#: 0.00 39.71 · 0.02 40.50 · 0.04 39.94 · 0.06 40.83 · 0.08 41.15 ·
#: **0.10 41.41** · 0.12 41.33 · 0.15 41.13 · 0.20 40.92 · 0.30 41.07.
#: 0.06~0.15 는 서로 통계적으로 구분되지 않으므로(평균 41.17 ± 0.23, 부트스트랩
#: 95% CI가 전부 0을 포함) **최고점이 아니라 평탄 구간의 중앙**을 골랐다.
#: 날짜 문자열이 10자 안팎이라 10% ≈ 글자 한 칸 — 기전과도 맞는다.
DEFAULT_EXPAND = 0.10
DEFAULT_NMS_IOU = 0.60


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _center_priors(size: int) -> np.ndarray:
    """``(sum(h*w), 4)`` — (cx, cy, stride, stride). nanodet get_bboxes 와 동일."""
    out = []
    for stride in _STRIDES:
        h = w = -(-size // stride)                       # ceil(size / stride)
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        px = (xs.reshape(-1) * stride).astype(np.float32)
        py = (ys.reshape(-1) * stride).astype(np.float32)
        s = np.full_like(px, stride)
        out.append(np.stack([px, py, s, s], axis=-1))
    return np.concatenate(out, axis=0)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ovr <= iou_thr]
    return keep


class NanoDetDetector:
    """ONNX NanoDet-Plus-m. ``detect(img) -> (N, 4, 2)`` (점수 내림차순)."""

    def __init__(self, onnx_path: str, input_size: int = 480, threads: int = 4,
                 score_thr: float = 0.35, nms_iou: float = DEFAULT_NMS_IOU, max_det: int = 20,
                 expand: float = DEFAULT_EXPAND):
        import cv2
        import onnxruntime as ort

        self._cv2 = cv2
        self.input_size = input_size
        self.score_thr = score_thr
        self.nms_iou = nms_iou
        self.max_det = max_det
        self.expand = expand                                   # §DEFAULT_EXPAND

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(onnx_path, sess_options=so,
                                          providers=["CPUExecutionProvider"])
        self._inp = self._sess.get_inputs()[0].name
        self._priors = _center_priors(input_size)                # (4789, 4)
        self._proj = np.arange(_REG_MAX + 1, dtype=np.float32)   # [0..7]

    # ── Engine.detect 와 같은 시그니처 ────────────────────────────────────
    def detect(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        S = self.input_size
        resized = self._cv2.resize(img, (S, S), interpolation=self._cv2.INTER_LINEAR)
        x = (resized.astype(np.float32) - _MEAN) / _STD          # BGR, nanodet val 파이프라인과 동일
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])     # (1, 3, S, S)

        out = self._sess.run(None, {self._inp: x})[0][0]          # (4789, 33)
        # ⚠️ nanodet _forward_onnx 는 cls 채널에 이미 sigmoid 를 적용해 내보낸다.
        #    여기서 다시 sigmoid 를 씌우면 [0,1] -> [0.5,0.73] 로 뭉개진다.
        scores = out[:, 0]                                        # 이미 확률 (단일 클래스)
        m = scores >= self.score_thr
        if not m.any():
            return np.empty((0, 4, 2), dtype=np.float32)

        priors = self._priors[m]
        reg = out[m, 1:].reshape(-1, 4, _REG_MAX + 1)
        dist = (_softmax(reg, axis=-1) * self._proj).sum(-1) * priors[:, 2:3]   # (n, 4) 픽셀
        cx, cy = priors[:, 0], priors[:, 1]
        boxes = np.stack([cx - dist[:, 0], cy - dist[:, 1],
                          cx + dist[:, 2], cy + dist[:, 3]], axis=-1)
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, S)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, S)
        sc = scores[m]

        keep = _nms(boxes, sc, self.nms_iou)[: self.max_det]
        boxes, sc = boxes[keep], sc[keep]

        # 480 프레임 -> 입력 img 프레임
        boxes[:, 0::2] *= w / S
        boxes[:, 1::2] *= h / S

        order = sc.argsort()[::-1]                                # 점수 내림차순
        quad = np.empty((len(order), 4, 2), dtype=np.float32)
        for j, i in enumerate(order):
            x0, y0, x1, y1 = boxes[i]
            if self.expand:
                dx, dy = (x1 - x0) * self.expand, (y1 - y0) * self.expand
                x0, y0, x1, y1 = x0 - dx, y0 - dy, x1 + dx, y1 + dy
            quad[j] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        return quad
