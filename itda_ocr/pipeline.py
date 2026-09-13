"""이미지 로딩과 장당 파이프라인, 그리고 배치 드라이버.

배치 드라이버는 **anytime 알고리즘**으로 설계했다(Russell & Zilberstein, IJCAI 1991).
시작 직후부터 완전히 유효한 CSV가 디스크에 존재하고, 이후 장마다 개선된다.
nbconvert 타임아웃은 커널을 강제 종료해서 ``finally`` 가 돌지 않을 수 있으므로,
**선기록이 빈 결과 파일에 대한 유일한 구조적 보증**이다.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .nanodet_det import DEFAULT_EXPAND, DEFAULT_NMS_IOU
from .parse import parse_boxes
from .select import STOP_SCORE, score as sel_score, select, to_row

FIELDNAMES = ["image_id", "year", "month", "day", "final_date"]

#: PIL이 열 수 있는 확장자만. 채점 디렉터리에 비이미지가 섞여도 죽지 않아야 한다.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Config:
    """심사자가 읽는 설정. 임계값을 코드에 흩뿌리지 않고 여기 모은다."""

    #: DCT 단계 축소 디코딩. ≥4MP 코호트(19.8%)가 318ms → 103ms.
    #: 예산의 28%가 디코딩이라 이건 최적화가 아니라 필수 요건이다.
    draft_to: int = 720
    #: 검출 입력 상한(long side). ExpDate 665장 실측 검출 recall:
    #: 480px 75.2% / 640px 83.6% / 960px 86.8%. 480→640은 +8.4pp를 ~23ms에 산다.
    det_side: int = 640
    #: 인식 1회분 배치 크기. 이만큼씩 읽고 유효 날짜가 나오면 멈춘다.
    #: NanoDet 도입 후 recall@2가 94.7%에 달하므로 2개씩 점진 탐색해 불필요한 연산을 줄인다.
    top_k: int = 2
    #: 조기 종료가 없을 때 읽을 크롭 수 상한.
    max_k: int = 9
    threads: int = 4
    box_thresh: float = 0.5
    unclip_ratio: float = 1.6
    #: 빠진 일자·연도를 채울 것인가. **False로 확정.** 인쇄물에 일자가 없으면
    #: 정답도 NONE이라는 규칙이 확인됐다. 실측도 같은 방향이다 — 보정은 ExpDate에서
    #: 50점 만점에 1.02점을 깎았다(정답이 NONE일 때 NONE은 50점, `01`은 10점).
    impute_missing: bool = False
    #: 장당 예산(초). 초과가 예상되면 top_k 를 줄인다.
    per_image_budget: float = 0.15
    flush_every: int = 50
    #: Track B — 날짜 전용 검출기 ONNX 경로. 지정하면 detect() 가 RapidOCR 대신 이걸 쓴다.
    nanodet_onnx: str | None = None
    nanodet_score_thr: float = 0.05
    #: NanoDet 박스를 인식 전에 각 변으로 넓히는 비율. 근거는 §nanodet_det.DEFAULT_EXPAND
    #: — 검출기만 바꾸면 크롭 경계에서 끝 글자가 잘린다 (39.71 → 41.41 / 50).
    nanodet_expand: float | tuple[float, float] = DEFAULT_EXPAND
    nanodet_nms_iou: float = DEFAULT_NMS_IOU


def load_image(path, draft_to: int = 720) -> np.ndarray:
    """BGR 배열로 읽는다. **EXIF 회전 보정이 무조건 먼저.**

    표본 조사에서 "90° 회전 문제"로 지목된 이미지가 전부 EXIF Orientation=6
    이었다(293장, 8.7%). 이 한 줄이 전체 이미지 회전 TTA를 불필요하게 만든다.

    ``draft()`` 는 JPEG를 DCT 단계에서 1/2·1/4·1/8로 **디코딩하며** 줄인다.
    전부 디코딩한 뒤 리사이즈하는 것보다 훨씬 싸다.
    """
    im = Image.open(path)
    if draft_to:
        im.draft("RGB", (draft_to, draft_to))
    # EXIF orientation fast-path: orientation이 없거나 1(정상)이면 transpose 연산 생략
    exif = im.getexif()
    orientation = exif.get(0x0112) if exif else None
    if orientation not in (1, None):
        im = ImageOps.exif_transpose(im)
    arr = np.asarray(im if im.mode == "RGB" else im.convert("RGB"))
    return arr[:, :, ::-1]


def iter_images(input_dir) -> list[Path]:
    """채점 템플릿과 같은 규칙으로 파일을 모은다(``*.*`` 정렬).

    ``image_id`` 는 확장자를 뺀 basename **그대로**다 — 배포셋에 ``000001.jpg`` 와
    ``3350.jpeg`` 가 섞여 있으므로 제로패딩을 정규화하면 안 된다.
    """
    return sorted(p for p in Path(input_dir).glob("*.*")
                  if p.suffix.lower() in IMAGE_SUFFIXES)


def process_image(engine, path_or_img, cfg: Config, top_k: int | None = None,
                  max_k: int | None = None, image_id: str | None = None) -> dict:
    """이미지 1장 → 제출 행 + 진단 정보.

    반환에는 후보 목록이 함께 담긴다. 채점 하네스가 "정답이 후보에 있었는가"로
    선별 실패와 인식 실패를 가르기 때문이다(`eval/score.py`).
    """

    t0 = time.perf_counter()
    if isinstance(path_or_img, np.ndarray):
        img = path_or_img
        image_id = image_id or "image"
        t_load = time.perf_counter()
    else:
        image_id = image_id or Path(path_or_img).stem
        img = load_image(path_or_img, cfg.draft_to)
        t_load = time.perf_counter()

    kept = engine.detect_and_filter(img)
    t_det = time.perf_counter()

    batch = top_k if top_k is not None else cfg.top_k
    limit = max_k if max_k is not None else cfg.max_k

    # 배치로 읽되 유효 날짜가 나오면 멈춘다. 쉬운 이미지는 1배치에서 끝나고,
    # 어려운 이미지만 깊이 들어간다 — 순위를 신뢰할 수 없다는 실측의 귀결이다.
    texts, geoms, items, candidates = [], [], [], []
    for start in range(0, min(len(kept), limit), batch):
        crops, batch_geoms = [], []
        for _, _, box in kept[start:start + batch]:
            patch = engine.crop(img, box)
            if patch.size:
                crops.append(patch)
                xs, ys = box[:, 0], box[:, 1]
                batch_geoms.append((float(xs.min()), float(ys.min()),
                                    float(xs.max()), float(ys.max())))
        if not crops:
            continue
        texts.extend(engine.recognize(crops))
        geoms.extend(batch_geoms)
        items = [(t, g[0], g[1], g[2], g[3]) for (t, _), g in zip(texts, geoms)]
        candidates = parse_boxes(items)
        # **확신할 때만** 멈춘다. "완전한 날짜가 하나라도 나오면"으로 두면
        # 완화 패턴이 만든 쓰레기 날짜가 조기 종료를 유발해, 진짜 날짜가 든
        # 크롭을 읽기 전에 멈춰버린다 (§select.STOP_SCORE).
        if any(sel_score(c, "") >= STOP_SCORE for c in candidates):
            break
    t_rec = time.perf_counter()
    full_text = " ".join(t for t, _ in texts)
    winner = select(candidates, full_text, cfg.impute_missing)
    t_end = time.perf_counter()

    row = to_row(winner, image_id)
    row["_diag"] = {
        "n_filtered": len(kept),
        "n_recognized": len(texts),   # 배치 전체 합계 (마지막 배치가 아니라)
        "texts": [t for t, _ in texts],
        "candidates": [{"text": c.text, "final_date": c.final_date} for c in candidates],
        "ms": {
            "load": (t_load - t0) * 1000,
            "detect": (t_det - t_load) * 1000,
            "recognize": (t_rec - t_det) * 1000,
            "parse": (t_end - t_rec) * 1000,
            "total": (t_end - t0) * 1000,
        },
    }
    return row


def write_rows(path, rows) -> None:
    """제출 스키마로 저장한다. 인덱스 컬럼은 생기지 않는다."""
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def blank_rows(paths) -> list[dict]:
    return [{"image_id": Path(p).stem, "year": "NONE", "month": "NONE",
             "day": "NONE", "final_date": "NONE"} for p in paths]


def run(input_dir, output_path, cfg: Config | None = None, engine=None,
        deadline: float | None = None, progress_every: int = 200,
        collect_diag: bool = False) -> dict:
    """디렉터리 전체를 처리한다. **어떤 경우에도 raise하지 않는다.**

    ``deadline`` 은 절대 시각(``time.time()`` 기준). 남은 예산이 부족해지면
    ``top_k`` 를 낮춰 장당 비용을 떨어뜨리고, 그래도 모자라면 남은 이미지를
    NONE으로 둔 채 종료한다 — 예산을 쉬운 입력과 어려운 입력에 **고르지 않게**
    쓰는 budgeted batch 설계(Huang et al., ICLR 2018)의 구현이다.
    """
    cfg = cfg or Config()
    paths = iter_images(input_dir)
    rows = blank_rows(paths)
    write_rows(output_path, rows)          # ← 1초 시점부터 유효한 산출물이 있다

    if engine is None:
        from .engine import Engine
        engine = Engine(det_side=cfg.det_side, threads=cfg.threads,
                        box_thresh=cfg.box_thresh, unclip_ratio=cfg.unclip_ratio,
                        nanodet_onnx=cfg.nanodet_onnx,
                        nanodet_score_thr=cfg.nanodet_score_thr,
                        nanodet_expand=cfg.nanodet_expand,
                        nanodet_nms_iou=cfg.nanodet_nms_iou)

    diags, degraded, skipped = [], 0, 0
    started = time.time()
    for i, path in enumerate(paths):
        remaining = len(paths) - i
        top_k = cfg.top_k
        max_k = cfg.max_k
        if deadline is not None:
            budget = (deadline - time.time()) / max(remaining, 1)
            if budget <= 0:
                skipped = remaining
                break
            if budget < cfg.per_image_budget * 0.6:
                top_k, max_k, degraded = 1, 1, degraded + 1   # 단계 하향

        try:
            row = process_image(engine, path, cfg, top_k=top_k, max_k=max_k)
            diag = row.pop("_diag")
            if collect_diag:
                diags.append({"image_id": row["image_id"], **diag})
            rows[i] = row
        except Exception:                  # noqa: BLE001 — 한 장 때문에 전체를 잃지 않는다
            pass                           # rows[i] 는 NONE 행으로 남는다

        if (i + 1) % cfg.flush_every == 0:
            write_rows(output_path, rows)
        if progress_every and (i + 1) % progress_every == 0:
            rate = (time.time() - started) / (i + 1)
            print(f"  {i + 1}/{len(paths)}  {rate * 1000:.0f} ms/img", flush=True)

    write_rows(output_path, rows)
    elapsed = time.time() - started
    return {
        "n": len(paths),
        "elapsed": elapsed,
        "ms_per_image": elapsed / len(paths) * 1000 if paths else 0.0,
        "coverage": sum(r["final_date"] != "NONE" for r in rows) / len(paths) if paths else 0.0,
        "degraded": degraded,
        "skipped": skipped,
        "rows": rows,
        "diagnostics": diags,
    }
