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

from .parse import parse_boxes
from .select import select, to_row

FIELDNAMES = ["image_id", "year", "month", "day", "final_date"]

#: PIL이 열 수 있는 확장자만. 채점 디렉터리에 비이미지가 섞여도 죽지 않아야 한다.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Config:
    """심사자가 읽는 설정. 임계값을 코드에 흩뿌리지 않고 여기 모은다."""

    #: DCT 단계 축소 디코딩. ≥4MP 코호트(19.8%)가 318ms → 103ms.
    #: 예산의 28%가 디코딩이라 이건 최적화가 아니라 필수 요건이다.
    draft_to: int = 720
    #: 검출 입력 상한(long side). 1600px 경로는 0.15초 예산에서 삭제했다.
    det_side: int = 480
    #: 인식으로 넘길 박스 수. 인식은 크롭당 ~20ms라 이 값이 예산을 지배한다.
    top_k: int = 2
    threads: int = 4
    box_thresh: float = 0.5
    unclip_ratio: float = 1.6
    #: 장당 예산(초). 초과가 예상되면 top_k 를 줄인다.
    per_image_budget: float = 0.15
    flush_every: int = 50


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
    im = ImageOps.exif_transpose(im)
    return np.asarray(im.convert("RGB"))[:, :, ::-1]


def iter_images(input_dir) -> list[Path]:
    """채점 템플릿과 같은 규칙으로 파일을 모은다(``*.*`` 정렬).

    ``image_id`` 는 확장자를 뺀 basename **그대로**다 — 배포셋에 ``000001.jpg`` 와
    ``3350.jpeg`` 가 섞여 있으므로 제로패딩을 정규화하면 안 된다.
    """
    return sorted(p for p in Path(input_dir).glob("*.*")
                  if p.suffix.lower() in IMAGE_SUFFIXES)


def process_image(engine, path, cfg: Config, top_k: int | None = None) -> dict:
    """이미지 1장 → 제출 행 + 진단 정보.

    반환에는 후보 목록이 함께 담긴다. 채점 하네스가 "정답이 후보에 있었는가"로
    선별 실패와 인식 실패를 가르기 때문이다(`eval/score.py`).
    """
    from .engine import filter_boxes

    t0 = time.perf_counter()
    image_id = Path(path).stem
    img = load_image(path, cfg.draft_to)
    t_load = time.perf_counter()

    boxes = engine.detect(img)
    t_det = time.perf_counter()

    kept = filter_boxes(boxes, img.shape)
    k = top_k if top_k is not None else cfg.top_k
    chosen = kept[:k]
    crops, geoms = [], []
    for _, _, box in chosen:
        patch = engine.crop(img, box)
        if patch.size:
            crops.append(patch)
            xs, ys = box[:, 0], box[:, 1]
            geoms.append((float(xs.min()), float(ys.min()),
                          float(xs.max()), float(ys.max())))
    texts = engine.recognize(crops) if crops else []
    t_rec = time.perf_counter()

    items = [(text, g[0], g[1], g[2], g[3]) for (text, _), g in zip(texts, geoms)]
    candidates = parse_boxes(items)
    full_text = " ".join(t for t, _ in texts)
    winner = select(candidates, full_text)
    t_end = time.perf_counter()

    row = to_row(winner, image_id)
    row["_diag"] = {
        "n_boxes": int(len(boxes)),
        "n_filtered": len(kept),
        "n_recognized": len(crops),
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
                        box_thresh=cfg.box_thresh, unclip_ratio=cfg.unclip_ratio)

    diags, degraded, skipped = [], 0, 0
    started = time.time()
    for i, path in enumerate(paths):
        remaining = len(paths) - i
        top_k = cfg.top_k
        if deadline is not None:
            budget = (deadline - time.time()) / max(remaining, 1)
            if budget <= 0:
                skipped = remaining
                break
            if budget < cfg.per_image_budget * 0.6:
                top_k, degraded = 1, degraded + 1   # 단계 하향

        try:
            row = process_image(engine, path, cfg, top_k=top_k)
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
