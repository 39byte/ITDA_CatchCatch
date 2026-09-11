"""B3 게이트 실측 — NanoDet 검출 단독 지연시간 (§docs 히스토리 "단계와 게이트").

게이트: **4코어 x86, 검출 단독 p50 ≤ 40ms.** 이 저장소의 개발 PC(Apple Silicon)는
x86이 아니라 대리가 못 된다 — 반드시 x86 러너(예: GitHub Actions ubuntu-latest)에서
돌린다. 코어 고정은 `bench/timing.py::physical_cores()` 와 같은 이유로 필요하다:
하이퍼스레딩 환경에서 논리 코어 0~3을 그냥 잡으면 물리 코어 2개에 스레드 4개가
몰려 3~6배 느려진 값이 나온다 — 2칸씩 건너뛰어 서로 다른 물리 코어를 잡는다.

실제 평가셋(``expdate/``)은 저장소에 없고(용량·라이선스) CI에도 올리지 않는다.
검출 지연은 NanoDet 내부에서 항상 480x480 로 리사이즈된 뒤 결정되므로(원본 해상도는
초기 리사이즈 비용에만 영향), 대신 **고정 시드 합성 이미지**로 측정한다 — 실제
사진과 박스 개수(따라서 NMS 비용)가 다를 수 있지만, 그 항은 전체 지연의 일부일
뿐이고 컨볼루션 추론이 지배적이므로 이 대체가 정당하다.

FP32 로 먼저 재고, 게이트를 넘으면 그걸로 끝. 못 넘으면 INT8 **동적** 양자화
(x86 FP16 금지 — Cast로 FP32 승격되어 오히려 느려진다, HANDOVER 경고)를 적용해
재측정한다.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def physical_cores(n: int = 4) -> list[int]:
    """서로 다른 물리 코어 n개의 논리 번호. `bench/timing.py` 와 동일 로직."""
    import psutil
    logical = psutil.cpu_count(logical=True) or n
    physical = psutil.cpu_count(logical=False) or logical
    stride = 2 if logical > physical else 1
    return [i * stride for i in range(n) if i * stride < logical]


def pin(threads: int) -> list[int] | None:
    import psutil
    from itda_ocr.engine import pin_threads
    pin_threads(threads)                    # OpenMP/ORT 스레드 수 — import 전에.
    try:
        cores = physical_cores(threads)
        psutil.Process().cpu_affinity(cores)
        return cores
    except Exception as exc:                # noqa: BLE001 — 코어 고정은 측정 편의 기능
        print(f"(코어 고정 실패, 계속 진행: {exc})")
        return None


def synth_images(n: int, seed: int = 0) -> list[np.ndarray]:
    """실제 제품 사진 대신 쓸 고정 시드 합성 이미지. BGR, 세로형(휴대폰 촬영 비율)."""
    rng = np.random.default_rng(seed)
    sizes = [(1280, 960), (1440, 1080), (1600, 1200)]   # (h, w) — 실측 분포와 유사한 스케일
    out = []
    for i in range(n):
        h, w = sizes[i % len(sizes)]
        out.append(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))
    return out


def pct(values, q: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * q), len(ordered) - 1)]


def bench_one(model_path: str, images: list[np.ndarray], threads: int,
              warmup: int, label: str) -> dict:
    from itda_ocr.nanodet_det import NanoDetDetector

    t0 = time.perf_counter()
    det = NanoDetDetector(model_path, threads=threads)
    startup_ms = (time.perf_counter() - t0) * 1000

    for img in images[:warmup]:
        det.detect(img)                     # ORT가 shape별 메모리 계획을 새로 짠다

    times = []
    n_boxes = []
    for img in images:
        t = time.perf_counter()
        boxes = det.detect(img)
        times.append((time.perf_counter() - t) * 1000)
        n_boxes.append(len(boxes))

    result = {
        "label": label,
        "model": model_path,
        "n": len(times),
        "startup_ms": startup_ms,
        "min": min(times),
        "p50": statistics.median(times),
        "p90": pct(times, 0.90),
        "p99": pct(times, 0.99),
        "max": max(times),
        "mean_boxes": statistics.mean(n_boxes),
    }
    return result


def report(r: dict, gate_ms: float) -> bool:
    passed = r["p50"] <= gate_ms
    print(f"\n=== {r['label']} ({r['model']}) ===")
    print(f"이미지 {r['n']}장, startup {r['startup_ms']:.0f}ms(1회, 장당 비용 제외), "
          f"박스 평균 {r['mean_boxes']:.1f}개")
    print(f"{'min':>8}{'p50':>8}{'p90':>8}{'p99':>8}{'max':>8}   (ms)")
    print(f"{r['min']:>8.2f}{r['p50']:>8.2f}{r['p90']:>8.2f}{r['p99']:>8.2f}{r['max']:>8.2f}")
    verdict = "✅ 통과" if passed else "❌ 미달"
    print(f"게이트(검출 단독 p50 ≤ {gate_ms:.0f}ms): {verdict} "
          f"(p50={r['p50']:.2f}ms, p99={r['p99']:.2f}ms)")
    return passed


class DynamicQuantUnsupported(RuntimeError):
    """이 그래프에는 동적 양자화가 런타임에서 아예 실행되지 않는다(ConvInteger 미구현 등)."""


def quantize(src: str, dst: str) -> None:
    """INT8 **동적** 양자화. x86 FP16 은 절대 쓰지 않는다(Cast로 FP32 승격 — 더 느려짐).

    ⚠️ NanoDet-Plus-m 은 백본이 전부 Conv(Gemm/MatMul 은 0개) 인데, onnxruntime의
    동적 양자화는 Conv 를 ``ConvInteger`` 로 바꾼다 — 이 op은 CPU EP 에 구현이
    없는 빌드가 있다(1.23.2 에서 실측 확인: ``NOT_IMPLEMENTED``). 그 경우 이 함수는
    양자화 자체는 성공하지만 **로드 시점에 실패**하므로, 호출부에서 반드시 로드까지
    시도해 `DynamicQuantUnsupported` 로 잡아야 한다. 근본 해결은 정적 양자화(QDQ,
    calibration 필요) — `QLinearConv` 는 CPU EP 에서 폭넓게 지원된다.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(
        model_input=src,
        model_output=dst,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],   # CNN 백본이라 Conv 를 포함해야 의미 있다
        reduce_range=True,     # 구형 x86(VNNI 없음)에서 오버플로 방지 — 대신 정밀도를 조금 낸다
    )
    # 양자화 변환 자체는 항상 "성공"한다 — 실제 실행 가능 여부는 로드해봐야 안다.
    import onnxruntime as ort
    try:
        ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    except Exception as exc:                                    # noqa: BLE001
        raise DynamicQuantUnsupported(
            f"{dst} 를 onnxruntime이 로드하지 못한다 — 이 그래프에서 동적 양자화가 "
            f"지원되지 않는다는 뜻이다. 정적 양자화(QDQ)로 전환이 필요하다. 원인: {exc}"
        ) from exc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="B3 게이트 실측: NanoDet 검출 단독 지연")
    ap.add_argument("--model", default="weights/date_detector_ema.onnx")
    ap.add_argument("--n", type=int, default=300, help="합성 이미지 수")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--gate-ms", type=float, default=40.0)
    ap.add_argument("--quantize", action="store_true",
                     help="FP32 가 게이트를 못 넘으면 INT8 동적 양자화 후 재측정")
    ap.add_argument("--force-quantize", action="store_true",
                     help="FP32 통과 여부와 무관하게 INT8 도 항상 재측정 (비교용)")
    ap.add_argument("--out", help="JSON 결과 저장 경로")
    args = ap.parse_args(argv)

    import platform
    import psutil
    print(f"플랫폼: {platform.machine()} / {platform.system()}  "
          f"(x86_64 러너에서 돌려야 게이트 판정이 유효하다)")
    print(f"논리 코어 {psutil.cpu_count(logical=True)} / "
          f"물리 코어 {psutil.cpu_count(logical=False)}")

    cores = pin(args.threads)
    print(f"코어 고정: {cores if cores else '실패(정보용 측정만)'}")

    images = synth_images(args.n)

    results = {}
    r_fp32 = bench_one(args.model, images, args.threads, args.warmup, "FP32")
    passed_fp32 = report(r_fp32, args.gate_ms)
    results["fp32"] = r_fp32

    need_int8 = args.force_quantize or (args.quantize and not passed_fp32)
    overall_pass = passed_fp32
    if need_int8:
        int8_path = str(Path(args.model).with_suffix("")) + "_int8.onnx"
        print(f"\nFP32 가 게이트 미달 — INT8 동적 양자화 적용 중... -> {int8_path}")
        try:
            quantize(args.model, int8_path)
            r_int8 = bench_one(int8_path, images, args.threads, args.warmup, "INT8 동적양자화")
            passed_int8 = report(r_int8, args.gate_ms)
            results["int8"] = r_int8
            overall_pass = passed_fp32 or passed_int8
        except DynamicQuantUnsupported as exc:
            print(f"\n⚠️ INT8 동적 양자화를 이 모델에서 실행할 수 없다: {exc}")
            print("   NanoDet-Plus-m은 백본이 전부 Conv(MatMul/Gemm 0개)라 동적 양자화가 "
                  "Conv를 ConvInteger로 바꾸는데, 이 op이 CPU EP에 없다. "
                  "정적 양자화(QDQ, calibration 필요)로 전환해야 한다 — 다음 단계로 남긴다.")
            results["int8_error"] = str(exc)

    if args.out:
        import json
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\n-> {args.out}")

    print(f"\n=== 최종 판정: {'✅ B3 게이트 통과' if overall_pass else '❌ B3 게이트 미달'} ===")
    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
