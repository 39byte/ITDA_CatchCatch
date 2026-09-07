"""ExpDate 주석 → 검출기 학습 포맷 (YOLO / COCO).

Track B(날짜 전용 검출기) 학습을 바로 시작할 수 있게 만드는 변환기다.
설계 배경은 `docs/DETECTOR_PLAN.md`, 인수인계는 `docs/HANDOVER.md`.

**이미지를 복사하지 않는다.** YOLO 규약대로 기존 `images/` 옆에 `labels/` 를
만들 뿐이라 640MB를 두 번 쓰지 않는다. YOLOv8은 경로의 `/images/` 를
`/labels/` 로 바꿔 라벨을 찾으므로 이대로 학습이 된다.

    python -m tools.expdate_to_detection --root kist_data --format yolo
    python -m tools.expdate_to_detection --root kist_data --format coco

⚠️ **train 과 evaluation 의 클래스 이름이 다르다.**
train 은 소비기한도 그냥 `date` 이고, evaluation 만 소비기한에 `exp` 를 붙인다.
검출기 입장에서 둘은 같은 대상이므로 기본값에서 **같은 클래스로 합친다**.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

#: ExpDate 의 원래 클래스. 기본 학습 대상은 날짜 계열뿐이다.
#: `due`/`prod`/`code` 는 마크(키워드·로트코드)라 날짜가 아니지만,
#: 보조 클래스로 함께 학습해 볼 수 있게 열어 둔다.
ALL_CLASSES = ("date", "due", "prod", "code")

#: 소비기한을 뜻하는 표기들. train 은 `date`, evaluation 은 `exp` 를 쓴다.
DATE_ALIASES = {"date", "exp"}


def norm(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def load(split_dir: Path) -> dict:
    files = sorted(split_dir.glob("*.json"), key=lambda p: -p.stat().st_size)
    if not files:
        raise SystemExit(f"{split_dir} 에 주석 JSON이 없다")
    return json.loads(files[0].read_text(encoding="utf-8"))


def class_of(raw_cls: str, wanted: list[str]) -> int | None:
    """주석의 cls 문자열 → 클래스 인덱스. 대상이 아니면 None."""
    key = norm(raw_cls)
    if key in {norm(a) for a in DATE_ALIASES}:
        key = "date"
    return wanted.index(key) if key in wanted else None


def convert_split(split_dir: Path, wanted: list[str], fmt: str, out_dir: Path) -> dict:
    data = load(split_dir)
    images_dir = split_dir / "images"
    if not images_dir.is_dir():
        raise SystemExit(f"{images_dir} 가 없다")

    stats = Counter()
    if fmt == "yolo":
        labels_dir = split_dir / "labels"        # images/ 의 형제 — YOLO 규약
        labels_dir.mkdir(exist_ok=True)
        for name, entry in data.items():
            H, W = entry.get("height"), entry.get("width")
            lines = []
            for ann in entry.get("ann") or []:
                idx = class_of(ann.get("cls", ""), wanted)
                box = ann.get("bbox")
                if idx is None or not box or not H or not W:
                    continue
                x0, y0, x1, y1 = box
                bw, bh = (x1 - x0) / W, (y1 - y0) / H
                cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
                if bw <= 0 or bh <= 0:
                    stats["잘못된 박스"] += 1
                    continue
                lines.append(f"{idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                stats[wanted[idx]] += 1
            (labels_dir / f"{Path(name).stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8")
            stats["이미지"] += 1
            if not lines:
                stats["라벨 없는 이미지"] += 1
        return {"labels": str(labels_dir), "stats": dict(stats)}

    # COCO (NanoDet / mmdetection 계열)
    images, annotations, ann_id = [], [], 1
    for img_id, (name, entry) in enumerate(sorted(data.items()), start=1):
        H, W = entry.get("height"), entry.get("width")
        images.append({"id": img_id, "file_name": name, "height": H, "width": W})
        stats["이미지"] += 1
        found = 0
        for ann in entry.get("ann") or []:
            idx = class_of(ann.get("cls", ""), wanted)
            box = ann.get("bbox")
            if idx is None or not box:
                continue
            x0, y0, x1, y1 = box
            w, h = x1 - x0, y1 - y0
            if w <= 0 or h <= 0:
                stats["잘못된 박스"] += 1
                continue
            annotations.append({"id": ann_id, "image_id": img_id,
                                "category_id": idx, "bbox": [x0, y0, w, h],
                                "area": w * h, "iscrowd": 0, "segmentation": []})
            ann_id += 1
            found += 1
            stats[wanted[idx]] += 1
        if not found:
            stats["라벨 없는 이미지"] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split_dir.name}_coco.json"
    path.write_text(json.dumps({
        "images": images, "annotations": annotations,
        "categories": [{"id": i, "name": c} for i, c in enumerate(wanted)],
    }, ensure_ascii=False), encoding="utf-8")
    return {"json": str(path), "images_dir": str(images_dir), "stats": dict(stats)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="ExpDate → 검출기 학습 포맷")
    ap.add_argument("--root", default="kist_data", help="train/ 과 evaluation/ 의 부모")
    ap.add_argument("--format", choices=["yolo", "coco"], default="yolo")
    ap.add_argument("--classes", default="date",
                    help=f"쉼표 구분. 가능: {','.join(ALL_CLASSES)} (기본: date 단일 클래스)")
    ap.add_argument("--out", default="labels/detection", help="COCO json 출력 위치")
    args = ap.parse_args(argv)

    wanted = [norm(c) for c in args.classes.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in ALL_CLASSES]
    if unknown:
        ap.error(f"알 수 없는 클래스: {unknown}. 가능: {ALL_CLASSES}")

    root = Path(args.root)
    # 배포 형태가 두 가지다:
    #   Products-Real  → root/{train,evaluation}/{images, annotations.json}
    #   Products-Synth → root/{images, annotations.json}   (평평하다)
    if (root / "images").is_dir():
        splits = [root]
    else:
        splits = [root / s for s in ("train", "evaluation") if (root / s).is_dir()]
    if not splits:
        raise SystemExit(
            f"{root} 에서 images/ 를 찾지 못했다. Products-Real 이면 train/·evaluation/ 이,\n"
            f"Products-Synth 면 root 바로 아래에 images/ 가 있어야 한다.")

    report = {}
    for d in splits:
        report[d.name] = convert_split(d, wanted, args.format, Path(args.out))
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.format == "yolo":
        yaml_path = Path(args.out).parent / f"{root.name}.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        names = "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(wanted))
        header = ("# YOLOv8 학습용. 이미지는 복사하지 않았다 — labels/ 만 새로 만들었다.\n"
                  f"path: {root.resolve().as_posix()}\n")
        if len(splits) == 1:
            # Products-Synth 같은 평평한 배포. 검증셋이 없으므로 val 을 비워 둔다 —
            # 검증은 반드시 Products-Real 의 evaluation 으로 한다.
            body = ("train: images\n"
                    "# val 없음: 합성 데이터로 검증하면 실사 성능을 알 수 없다.\n"
                    "#   합성 사전학습 → 실사 미세조정 순서로 쓰고, 검증은\n"
                    "#   Products-Real 의 evaluation 으로만 한다.\n")
        else:
            body = "train: train/images\nval: evaluation/images\n"
        yaml_path.write_text(header + body + names, encoding="utf-8")
        print(f"\nYOLO 데이터 설정 → {yaml_path}")
        print(f"  yolo detect train data={yaml_path} model=yolov8n.pt imgsz=480 epochs=100")

    print("\n⚠️ Products-Real 의 evaluation 665장은 **검증 전용**이다. "
          "학습에 넣으면 이후 모든 비교가 무효가 된다.")


if __name__ == "__main__":
    main()
