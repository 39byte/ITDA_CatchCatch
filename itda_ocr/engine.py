"""OCR 엔진 래퍼 — 검출과 인식을 **분리해서** 부른다.

기성 파이프라인을 그대로 쓰면 안 되는 이유는 속도다. PaddleOCR 계열의 기본 동작은
**검출된 텍스트를 전부 인식**하는 것이고, 우리 데이터는 1000px 기준 연결성분 중앙값이
88개(텍스트 20~60줄)라 그것만으로 이미지당 1.3~1.8초가 든다 — 예산의 9~13배다.
검출과 인식 사이에 필터를 넣으려면 두 단계를 따로 불러야 한다.

⚠️ **RapidOCR의 `det_limit_side_len` 은 `limit_type="max"` 에서 무시된다.**
`TextDetector.get_preprocess()` 가 이미지 크기를 보고 960/1500/2000 중 하나로
**덮어쓰기** 때문이다. 1008px 이미지가 1500px로 *확대*되어 검출이 5배 느려진다.
그래서 여기서는 `preprocess_op` 를 직접 만들어 끼우고 `__call__` 을 쓰지 않는다.
(`DetPreProcess.resize` 자체는 축소만 하므로 안전하다 — 문제는 오직 그 오버라이드다.)
"""

from __future__ import annotations

import os

import numpy as np

from .nanodet_det import DEFAULT_EXPAND, DEFAULT_NMS_IOU

#: onnxruntime / OpenMP 는 **import 시점에** 스레드 수를 읽는다. 그래서 이 설정은
#: 반드시 아래 import 보다 먼저 와야 한다. 측정 재현성의 전제이기도 하다
#: (Mytkowicz et al., ASPLOS 2009 — "Producing Wrong Data Without Doing Anything
#: Obviously Wrong": 스레드 수를 명시하지 않은 벤치마크는 편향된다).
DEFAULT_THREADS = 4


def pin_threads(n: int = DEFAULT_THREADS) -> None:
    """스레드 수를 고정한다. **cv2/onnxruntime import 전에** 부를 것."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(n))


class DateCTCLabelDecode:
    """CTC 디코딩 단계에서 날짜 인식에 불필요한 외래 문자/한자 6,000여 개를 마스킹.

    Scheidl et al. (ICFHR 2018, Word Beam Search) 사전 제약 디코딩 원리.
    PaddleOCR 기본 사전 6,625개 중 CJK 한자(6,280개) 등 노이즈 토큰의 logit을 -inf로
    슬라이싱하여 '专84.2031' 같은 환각을 0ms에 원천 차단한다.
    """

    def __init__(self, original_op, allowed_chars: str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-/:, ()[]~年月日"):
        self.original_op = original_op
        self.character = original_op.character
        allowed_set = set(allowed_chars)
        self.disallowed = np.array([
            i for i, c in enumerate(self.character)
            if i != 0 and c != 'blank' and c not in allowed_set
        ], dtype=np.int64)

    def __getattr__(self, name):
        return getattr(self.original_op, name)

    def __call__(self, preds, *args, **kwargs):
        preds[:, :, self.disallowed] = -np.inf
        return self.original_op(preds, *args, **kwargs)


class Engine:
    """검출·방향분류·인식을 따로 호출할 수 있는 얇은 래퍼."""

    def __init__(self, det_side: int = 640, threads: int = DEFAULT_THREADS,
                 box_thresh: float = 0.5, unclip_ratio: float = 1.6,
                 text_score: float = 0.0, nanodet_onnx: str | None = None,
                 nanodet_score_thr: float = 0.05,
                 nanodet_expand: float | tuple[float, float] = DEFAULT_EXPAND,
                 nanodet_nms_iou: float = DEFAULT_NMS_IOU):
        pin_threads(threads)
        import cv2
        from rapidocr_onnxruntime import RapidOCR
        from rapidocr_onnxruntime.ch_ppocr_det.utils import DetPreProcess

        cv2.setNumThreads(threads)
        self._cv2 = cv2
        self.det_side = det_side

        #: Track B — 날짜 전용 검출기(ONNX). 지정하면 detect() 가 이걸 쓴다.
        #: 인식·방향분류는 그대로 RapidOCR 모듈을 쓴다.
        self._nanodet = None
        if nanodet_onnx:
            from .nanodet_det import NanoDetDetector
            self._nanodet = NanoDetDetector(nanodet_onnx, threads=threads,
                                            score_thr=nanodet_score_thr,
                                            nms_iou=nanodet_nms_iou,
                                            expand=nanodet_expand)

        self._ocr = RapidOCR(
            intra_op_num_threads=threads,
            inter_op_num_threads=1,
            det_box_thresh=box_thresh,
            det_unclip_ratio=unclip_ratio,
            text_score=text_score,
        )
        self._det = self._ocr.text_det
        self._cls = self._ocr.text_cls
        self._rec = self._ocr.text_rec
        #: CTC Logit Masking: 6,280개 CJK 한자 및 잡음 기호 차단
        self._rec.postprocess_op = DateCTCLabelDecode(self._rec.postprocess_op)

        # get_preprocess() 를 우회하는 고정 전처리기. 이것이 이 클래스의 존재 이유다.
        self._pre = DetPreProcess(det_side, "max", self._det.mean, self._det.std)

    # ── 검출 ───────────────────────────────────────────────────────────────
    def detect(self, img: np.ndarray) -> np.ndarray:
        """텍스트 박스를 찾는다. 반환 좌표는 **입력 `img` 의 좌표계**다.

        검출은 `det_side` 로 줄여서 싸게 하고, 박스는 원래 크기로 되돌아온다
        — "검출 해상도 ≠ 인식 해상도" 원칙이 여기서 구현된다.
        """
        if self._nanodet is not None:
            return self._nanodet.detect(img)
        tensor = self._pre(img)
        if tensor is None:
            return np.empty((0, 4, 2), dtype=np.float32)
        preds = self._det.infer(tensor)[0]
        boxes, _ = self._det.postprocess_op(preds, img.shape[:2])
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 4, 2), dtype=np.float32)
        return self._det.filter_tag_det_res(boxes, img.shape[:2])

    def detect_and_filter(self, img: np.ndarray):
        """검출 → 박스 필터 → **순위**까지. 순위 정책이 여기 한 곳에만 있다.

        ⚠️ 이 메서드가 존재하는 이유가 순위다. ``filter_boxes`` 의 기본 정렬은
        종횡비 사전확률(`_date_prior`)인데, 그건 **범용 텍스트 검출기용**이다 —
        RapidOCR DB 박스에는 "날짜다움" 점수가 없으니 기하로 대신 추측할 수밖에 없다.

        NanoDet 은 다르다. 단일 클래스 `date` 검출기라 **박스 점수가 곧 날짜다움**이고,
        ``detect()`` 가 점수 내림차순으로 돌려준다. 여기에 종횡비 재정렬을 덮으면
        그 신호가 통째로 버려진다 — ExpDate 665장 실측 소비기한 박스 recall@1 이
        **84.7% → 50.2%** 로 무너졌다(recall@3 95.9% → 81.8%).
        그래서 NanoDet 경로에서는 **거르기만 하고 다시 줄 세우지 않는다.**
        """
        boxes = self.detect(img)
        return filter_boxes(boxes, img.shape, rerank=self._nanodet is None)

    # ── 인식 ───────────────────────────────────────────────────────────────
    def recognize(self, crops: list[np.ndarray], use_cls: bool = True):
        """크롭들을 **한 번에 배치로** 인식한다. 반환 ``[(text, score), ...]``.

        기본 패키징의 95% 이상이 정방향(0°)이므로, 1차는 cls 없이 바로 인식하고
        숫자가 전혀 검출되지 않을 때에만 조건부(Lazy)로 방향 분류기(_cls)를 호출한다.
        """
        if not crops:
            return []
        rec_res = self._rec(crops)[0]
        texts = [str(t) for t, _ in rec_res]
        # 크롭들 중 최소 하나라도 숫자 3개 이상(연/월/일 파편)이 잡히면 정상 방향으로 판단
        if any(sum(c.isdigit() for c in t) >= 3 for t in texts) or not use_cls:
            return [(str(t), float(s)) for t, s in rec_res]
        # 180도 역방향 크롭 구제: 숫자가 전혀 안 잡힐 때에만 _cls 실행 후 재인식
        oriented_crops = self._cls(crops)[0]
        rec_res2 = self._rec(oriented_crops)[0]
        return [(str(t), float(s)) for t, s in rec_res2]

    # ── 크롭 ───────────────────────────────────────────────────────────────
    #: 인식기의 입력 높이. 크롭을 이보다 크게 키우는 건 순수한 낭비다 —
    #: 인식기가 어차피 48로 되돌린다.
    REC_HEIGHT = 48

    def crop(self, img: np.ndarray, box: np.ndarray) -> np.ndarray:
        """박스를 잘라낸다. 인식기 입력 높이보다 작을 때만 확대한다.

        640×640 레터박스 코호트(33%)는 날짜 글자가 ~14px이라 확대 없이는
        인식기가 읽지 못한다 — 되돌릴 원본이 없는 코호트다. 반대로 이미 48보다
        큰 크롭을 더 키우면 인식 텐서만 넓어져 **느려지기만 한다**(측정에서
        단일 크롭 500ms를 만든 원인).
        """
        h, w = img.shape[:2]
        xs, ys = box[:, 0], box[:, 1]
        x0, x1 = max(int(xs.min()), 0), min(int(np.ceil(xs.max())), w)
        y0, y1 = max(int(ys.min()), 0), min(int(np.ceil(ys.max())), h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return np.empty((0, 0, 3), dtype=img.dtype)

        patch = img[y0:y1, x0:x1]
        ph = patch.shape[0]
        if ph < self.REC_HEIGHT:
            scale = min(4.0, self.REC_HEIGHT / max(ph, 1))
            patch = self._cv2.resize(
                patch, (max(int(patch.shape[1] * scale), 1), max(int(ph * scale), 1)),
                interpolation=self._cv2.INTER_LINEAR)
        # 인식기 입력 폭 상한 (가로세로비 W/H <= 6.67, 320px) 제한.
        # 비정상적인 극단적 가로 비율 노이즈 박스로 인한 CTC 타임스텝 폭증(p90/p99 꼬리 지연) 차단.
        MAX_REC_WIDTH = 320
        if patch.shape[1] > MAX_REC_WIDTH:
            scale_w = MAX_REC_WIDTH / patch.shape[1]
            patch = self._cv2.resize(
                patch, (MAX_REC_WIDTH, max(int(patch.shape[0] * scale_w), 16)),
                interpolation=self._cv2.INTER_AREA)
        return np.ascontiguousarray(patch)


# ── Stage 2: 박스 필터 ─────────────────────────────────────────────────────
# 이 설계가 0.15초 예산 안에서 성립하는 **유일한 이유**. 인식은 크롭당 비용이
# 붙으므로, 인식기에 넘기기 전에 값싼 기하 신호로 후보를 줄인다.
#
# 튜닝 목표는 Viola & Jones(CVPR 2001)의 캐스케이드 규칙 그대로 —
# **싼 단계는 재현율을 거의 1로 두고 오탐률은 느슨하게 둔다.** 정밀도는
# Stage 5(select.py)가 회수한다. 여기서 정답 박스를 놓치면 회수할 방법이 없다.

def box_metrics(box: np.ndarray) -> tuple[float, float, float]:
    """(너비, 높이, 종횡비) — 회전 박스는 외접 사각형으로 근사한다."""
    xs, ys = box[:, 0], box[:, 1]
    w = float(xs.max() - xs.min())
    h = float(ys.max() - ys.min())
    return w, h, (w / h if h > 0 else 0.0)


def filter_boxes(boxes: np.ndarray, img_shape, *, min_height: float = 6.0,
                 min_ratio: float = 1.2, max_ratio: float = 25.0,
                 max_width_frac: float = 0.95, rerank: bool = True):
    """날짜일 수 **없는** 박스를 떨어뜨리고, 나머지를 그럴듯한 순으로 정렬한다.

    세로 위치 prior는 쓰지 않는다 — 실측 텍스트 세로 분포가 균일해서
    ("하단만 보기" 류의) 위치 휴리스틱이 통하지 않는다.
    """
    if len(boxes) == 0:
        return []
    ih, iw = img_shape[:2]
    kept = []
    for i, box in enumerate(boxes):
        w, h, ratio = box_metrics(box)
        if h < min_height or w < min_height:
            continue                      # 인식기가 읽을 수 없는 크기
        if not (min_ratio <= ratio <= max_ratio):
            continue                      # 날짜는 가로로 긴 한 줄이다
        if w > iw * max_width_frac:
            continue                      # 페이지 폭 전체 = 문단, 날짜 아님
        kept.append((_date_prior(w, h, ratio), i, box))
    # ``rerank=False`` 는 "입력 순서가 이미 더 좋은 순위다" 라는 뜻이다.
    # 날짜 전용 검출기의 점수 순서가 그렇다 — §Engine.detect_and_filter
    if rerank:
        kept.sort(key=lambda t: -t[0])
    return kept


#: 날짜 박스 종횡비의 실측 중앙값. ExpDate 1,767장(train+eval)에서 잰 값으로,
#: p10 3.8 / p50 **5.4** / p90 7.4~8.2 로 분포가 매우 좁다.
#: ⚠️ 이 상수 하나가 순위를 크게 좌우한다. 눈대중으로 7.0을 쓰고 있었는데
#: 5.4로 바꾸자 recall@3 이 45.0%→52.8%, recall@5 가 58.2%→66.2% 로 올랐다.
DATE_ASPECT = 5.4


def _date_prior(w: float, h: float, ratio: float) -> float:
    """날짜 스탬프다움 — 인식 없이 얻을 수 있는 값싼 사전확률.

    ``YYYY.MM.DD`` 는 10자 안팎이라 종횡비가 좁은 구간에 몰린다(위 실측).
    큰 글자일수록 인쇄된 스탬프일 확률이 높다(잉크젯 날짜는 보통 라벨 본문보다 크다).

    상대 크기(높이/이미지높이 ≈ 0.026)도 실측상 분포가 좁지만, 항으로 넣어 보면
    K가 커질수록 오히려 나빠져 채택하지 않았다 — 종횡비만으로 충분하다.
    """
    ratio_fit = -abs(ratio - DATE_ASPECT) / DATE_ASPECT
    return ratio_fit + min(h, 60.0) / 120.0
