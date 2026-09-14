from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import pynvml
import torch

from ml.energy.benchmark import coefficient_of_variation
from ml.energy.gpu_meter import NVMLGpuMeter


def gpu_metadata(device_index: int) -> dict[str, object]:
    pynvml.nvmlInit()

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(
            device_index
        )

        name = pynvml.nvmlDeviceGetName(handle)
        driver = pynvml.nvmlSystemGetDriverVersion()

        if isinstance(name, bytes):
            name = name.decode()

        if isinstance(driver, bytes):
            driver = driver.decode()

        return {
            "gpu_name": name,
            "driver_version": driver,
            "device_index": device_index,
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "platform": platform.platform(),
        }

    finally:
        pynvml.nvmlShutdown()


def run_gpu_workload(
    duration_s: float,
    *,
    device: torch.device,
    matrix_size: int,
) -> int:
    generator = torch.Generator(
        device=device
    ).manual_seed(20260915)

    a = torch.randn(
        matrix_size,
        matrix_size,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    b = torch.randn(
        matrix_size,
        matrix_size,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    out = torch.empty_like(a)

    torch.cuda.synchronize(device)

    deadline = time.perf_counter() + duration_s
    iterations = 0

    with torch.inference_mode():
        while time.perf_counter() < deadline:
            for _ in range(10):
                torch.mm(a, b, out=out)
                iterations += 1

            torch.cuda.synchronize(device)

    return iterations


def measure_idle(
    duration_s: float,
    *,
    device_index: int,
    interval_s: float,
) -> dict[str, object]:
    meter = NVMLGpuMeter(
        device_index=device_index,
        interval_s=interval_s,
    )

    meter.start()
    time.sleep(duration_s)
    summary = meter.stop()

    return summary.to_dict()


def measure_workload(
    duration_s: float,
    *,
    device_index: int,
    interval_s: float,
    matrix_size: int,
) -> tuple[dict[str, object], int]:
    device = torch.device(f"cuda:{device_index}")

    meter = NVMLGpuMeter(
        device_index=device_index,
        interval_s=interval_s,
    )

    meter.start()

    iterations = run_gpu_workload(
        duration_s,
        device=device,
        matrix_size=matrix_size,
    )

    summary = meter.stop()

    return summary.to_dict(), iterations


def repeatability_summary(
    values: list[float],
) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values),
        "cv": coefficient_of_variation(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--baseline-seconds", type=float, default=10.0)
    parser.add_argument("--workload-seconds", type=float, default=20.0)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--matrix-size", type=int, default=2048)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/research/energy_meter_calibration.json"
        ),
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    interval_s = args.interval_ms / 1000.0

    print("G2C energy-meter calibration")
    print(
        "GPU:",
        torch.cuda.get_device_name(args.device),
    )
    print(
        f"Sampling interval: {args.interval_ms:.1f} ms"
    )

    print("\nWarm-up workloads")

    for index in range(args.warmups):
        count = run_gpu_workload(
            3.0,
            device=torch.device(
                f"cuda:{args.device}"
            ),
            matrix_size=args.matrix_size,
        )

        print(
            f"  warmup {index + 1}/{args.warmups}: "
            f"{count} iterations"
        )

    runs: list[dict[str, object]] = []

    for index in range(args.repetitions):
        print(
            f"\nPair {index + 1}/{args.repetitions}"
        )

        time.sleep(args.cooldown_seconds)

        idle = measure_idle(
            args.baseline_seconds,
            device_index=args.device,
            interval_s=interval_s,
        )

        idle_duration = float(idle["duration_s"])
        idle_energy = float(
            idle["instant_energy_j"]
        )

        idle_power = idle_energy / idle_duration

        print(
            "  idle: "
            f"{idle_energy:.3f} J, "
            f"{idle_power:.3f} W"
        )

        workload, iterations = measure_workload(
            args.workload_seconds,
            device_index=args.device,
            interval_s=interval_s,
            matrix_size=args.matrix_size,
        )

        workload_duration = float(
            workload["duration_s"]
        )

        gross_energy = float(
            workload["instant_energy_j"]
        )

        baseline_equivalent = (
            idle_power * workload_duration
        )

        incremental_energy = (
            gross_energy - baseline_equivalent
        )

        print(
            "  workload: "
            f"{gross_energy:.3f} J, "
            f"{gross_energy / workload_duration:.3f} W, "
            f"{iterations} iterations"
        )

        runs.append(
            {
                "pair_index": index + 1,
                "idle": idle,
                "workload": workload,
                "iterations": iterations,
                "derived": {
                    "idle_power_w": idle_power,
                    "gross_workload_energy_j": gross_energy,
                    "baseline_equivalent_energy_j": (
                        baseline_equivalent
                    ),
                    "incremental_workload_energy_j": (
                        incremental_energy
                    ),
                    "gross_energy_per_iteration_j": (
                        gross_energy / iterations
                    ),
                    "incremental_energy_per_iteration_j": (
                        incremental_energy / iterations
                    ),
                },
            }
        )

    gross = [
        float(
            run["derived"][
                "gross_workload_energy_j"
            ]
        )
        for run in runs
    ]

    incremental = [
        float(
            run["derived"][
                "incremental_workload_energy_j"
            ]
        )
        for run in runs
    ]

    gross_per_iteration = [
        float(
            run["derived"][
                "gross_energy_per_iteration_j"
            ]
        )
        for run in runs
    ]

    incremental_per_iteration = [
        float(
            run["derived"][
                "incremental_energy_per_iteration_j"
            ]
        )
        for run in runs
    ]

    result = {
        "schema_version": (
            "aeroxai.energy_meter_calibration.v1"
        ),
        "purpose": (
            "measurement calibration only; "
            "no MetroPT or TCN result"
        ),
        "energy_provenance": {
            "gpu_power": "MEASURED_NVML",
            "gpu_energy": (
                "DERIVED_FROM_MEASURED_POWER_"
                "TRAPEZOIDAL_INTEGRATION"
            ),
            "cpu_energy": "UNKNOWN",
        },
        "hardware": gpu_metadata(args.device),
        "measurement": {
            "sampling_interval_ms": args.interval_ms,
            "baseline_seconds": args.baseline_seconds,
            "workload_seconds": args.workload_seconds,
            "cooldown_seconds": args.cooldown_seconds,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "matrix_size": args.matrix_size,
            "primary_power_field": (
                "NVML_FI_DEV_POWER_INSTANT"
            ),
            "secondary_power_field": (
                "NVML_FI_DEV_POWER_AVERAGE"
            ),
        },
        "runs": runs,
        "repeatability": {
            "gross_energy_j": (
                repeatability_summary(gross)
            ),
            "incremental_energy_j": (
                repeatability_summary(incremental)
            ),
            "gross_energy_per_iteration_j": (
                repeatability_summary(
                    gross_per_iteration
                )
            ),
            "incremental_energy_per_iteration_j": (
                repeatability_summary(
                    incremental_per_iteration
                )
            ),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    repeatability = result["repeatability"]

    print("\nRepeatability")

    print(
        "  gross energy CV:       "
        f"{100 * repeatability['gross_energy_j']['cv']:.3f}%"
    )

    print(
        "  incremental energy CV: "
        f"{100 * repeatability['incremental_energy_j']['cv']:.3f}%"
    )

    print(
        "  gross J/iteration CV:  "
        f"{100 * repeatability['gross_energy_per_iteration_j']['cv']:.3f}%"
    )

    print(
        "  incr. J/iteration CV:  "
        f"{100 * repeatability['incremental_energy_per_iteration_j']['cv']:.3f}%"
    )

    print(
        f"\nWrote {args.output}"
    )


if __name__ == "__main__":
    main()
