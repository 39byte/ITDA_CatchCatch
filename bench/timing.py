"""속도 측정 — 정량 10점을 정직하게 방어하기 위한 도구.

세 가지를 지킨다. 전부 벤치마킹 문헌이 지적하는 실수를 피하기 위한 것이다.

1. **스레드와 코어를 명시적으로 고정한다.** 명시하지 않은 측정은 편향된다
   (Mytkowicz et al., ASPLOS 2009, *"Producing Wrong Data Without Doing Anything
   Obviously Wrong"*). 12코어 개발 PC에서 그냥 재면 4코어 채점 환경과 무관한 숫자다.
2. **startup(모델 로드 1회)과 장당 비용(N회)을 분리해 보고한다**
   (Georges et al., OOPSLA 2007). N이 작으면 고정비가 지배한다.
3. **장당 백분위와 총 처리량을 함께 낸다.** 2400초에 N장을 넣는 문제는 실제로
   처리량 문제이므로 지연 백분위만 보고하는 건 틀린 프레이밍이고, 반대로 평균만
   보면 꼬리가 숨는다.

⚠️ 이 개발 PC는 타이밍 노이즈가 크다(같은 작업의 p50 대비 max가 4배까지 벌어진다).
그래서 p50과 **min** 을 함께 낸다 — min은 방해가 가장 적었던 실행이라 모델의
본래 비용에 가장 가깝다.
"""

from __future__ import annotations

import argparse
import statistics
import time


def physical_cores(n: int = 4) -> list[int]:
    """서로 다른 **물리** 코어 n개의 논리 번호를 고른다.

    하이퍼스레딩 환경에서 논리 0~3을 그냥 잡으면 물리 코어 2개에 4스레드를 얹게 되어
    (실측) 오히려 3~5배 느려진다 — 4코어 채점 환경의 대리가 되지 못한다.
    """
    import psutil
    logical = psutil.cpu_count(logical=True) or n
    physical = psutil.cpu_count(logical=False) or logical
    # SMT가 있으면 형제 스레드는 보통 인접 번호로 붙는다(0,1 = 물리 0). 하이브리드
    # 배치(P코어만 SMT)에서도 앞쪽을 2칸씩 건너뛰면 서로 다른 물리 코어를 잡는다.
    # `logical // physical` 로 계산하면 8물리/12논리에서 stride=1이 되어
    # 물리 2개에 4스레드를 얹는 최악의 경우가 나온다 — 실측 6배 느려졌다.
    stride = 2 if logical > physical else 1
    return [i * stride for i in range(n) if i * stride < logical]


def pct(values, q: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * q), len(ordered) - 1)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="장당 추론 비용 측정")
    ap.add_argument("--images", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--det-side", type=int, default=640)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-affinity", action="store_true",
                    help="코어 고정을 끈다 (다른 부하가 있을 때 비교용)")
    args = ap.parse_args(argv)

    from itda_ocr.engine import pin_threads
    pin_threads(args.threads)              # import 보다 먼저 (엔진 안에서도 보장한다)

    pinned = None
    if not args.no_affinity:
        try:
            import psutil
            pinned = physical_cores(args.threads)
            psutil.Process().cpu_affinity(pinned)
        except Exception as exc:           # noqa: BLE001 — 측정 편의 기능일 뿐이다
            print(f"(코어 고정 실패, 계속 진행: {exc})")
            pinned = None

    from itda_ocr.engine import Engine
    from itda_ocr.pipeline import Config, iter_images, process_image

    cfg = Config(det_side=args.det_side, top_k=args.top_k, threads=args.threads)

    t0 = time.perf_counter()
    engine = Engine(det_side=cfg.det_side, threads=cfg.threads,
                    box_thresh=cfg.box_thresh, unclip_ratio=cfg.unclip_ratio)
    startup = time.perf_counter() - t0

    paths = iter_images(args.images)[:args.limit]
    if not paths:
        raise SystemExit(f"{args.images} 에 이미지가 없다")

    for p in paths[:args.warmup]:          # ORT는 입력 shape마다 메모리 계획을 새로 짠다
        try:
            process_image(engine, p, cfg)
        except Exception:                  # noqa: BLE001
            pass

    stages = {k: [] for k in ("load", "detect", "recognize", "parse", "total")}
    covered = 0
    wall = time.perf_counter()
    for p in paths:
        try:
            row = process_image(engine, p, cfg)
        except Exception:                  # noqa: BLE001
            continue
        diag = row["_diag"]
        for k in stages:
            stages[k].append(diag["ms"][k])
        covered += row["final_date"] != "NONE"
    wall = time.perf_counter() - wall

    n = len(stages["total"])
    print(f"\n이미지 {n}장 / 스레드 {args.threads} / "
          f"코어 고정 {pinned if pinned else '없음'}")
    print(f"det_side={args.det_side}  top_k={args.top_k}")
    print(f"\nstartup (모델 로드, 1회)   {startup * 1000:>8.0f} ms   ← 장당 비용에 포함되지 않는다")
    print(f"\n{'단계':<12}{'min':>9}{'p50':>9}{'p90':>9}{'p99':>9}   (ms)")
    for k, vals in stages.items():
        print(f"{k:<12}{min(vals):>9.1f}{statistics.median(vals):>9.1f}"
              f"{pct(vals, .90):>9.1f}{pct(vals, .99):>9.1f}")

    total = stages["total"]
    noise = statistics.median(total) / min(total) if min(total) else 0
    print(f"\n총 벽시계          {wall:.1f}s")
    print(f"처리량             {n / wall:.1f} img/s   ({wall / n * 1000:.0f} ms/img)")
    print(f"커버리지           {covered / n:.1%}")
    print(f"\n3352장 추정        {3352 * wall / n:.0f}s / 2400s "
          f"{'✅' if 3352 * wall / n < 2400 else '❌ 초과'}")
    print(f"장당 목표 150ms    p50 {statistics.median(total):.0f} ms "
          f"{'✅' if statistics.median(total) <= 150 else '❌ 초과'}")
    print(f"\n노이즈 지표        p50/min = {noise:.2f}  "
          f"({'안정적' if noise < 1.4 else '⚠️ 이 머신은 측정 노이즈가 크다 — min을 함께 볼 것'})")


if __name__ == "__main__":
    main()
