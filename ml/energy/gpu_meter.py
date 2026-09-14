from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import pairwise
from threading import Event, Thread
from time import perf_counter

import pynvml


@dataclass(frozen=True)
class GpuPowerSample:
    time_s: float
    instant_w: float
    average_w: float


@dataclass(frozen=True)
class GpuPowerSummary:
    duration_s: float
    sample_count: int
    instant_energy_j: float
    average_field_energy_j: float
    mean_instant_w: float
    mean_average_w: float
    peak_instant_w: float
    peak_average_w: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def integrate_trapezoid(
    samples: Iterable[GpuPowerSample],
    field: str = "instant_w",
) -> float:
    points = list(samples)

    if len(points) < 2:
        return 0.0

    if field not in {"instant_w", "average_w"}:
        raise ValueError("field must be 'instant_w' or 'average_w'")

    energy_j = 0.0

    for left, right in pairwise(points):
        dt = right.time_s - left.time_s

        if dt < 0:
            raise ValueError("sample timestamps must be nondecreasing")

        p_left = getattr(left, field)
        p_right = getattr(right, field)

        energy_j += 0.5 * (p_left + p_right) * dt

    return energy_j


def summarize_samples(
    samples: Iterable[GpuPowerSample],
) -> GpuPowerSummary:
    points = list(samples)

    if len(points) < 2:
        raise ValueError("at least two power samples are required")

    duration_s = points[-1].time_s - points[0].time_s
    instant = [sample.instant_w for sample in points]
    average = [sample.average_w for sample in points]

    return GpuPowerSummary(
        duration_s=duration_s,
        sample_count=len(points),
        instant_energy_j=integrate_trapezoid(points, "instant_w"),
        average_field_energy_j=integrate_trapezoid(
            points,
            "average_w",
        ),
        mean_instant_w=sum(instant) / len(instant),
        mean_average_w=sum(average) / len(average),
        peak_instant_w=max(instant),
        peak_average_w=max(average),
    )


class NVMLGpuMeter:
    """In-process NVIDIA GPU power sampler.

    NVML power fields are reported in milliwatts and converted
    to watts. Energy is derived from measured power by
    trapezoidal numerical integration.
    """

    def __init__(
        self,
        device_index: int = 0,
        interval_s: float = 0.1,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")

        self.device_index = device_index
        self.interval_s = interval_s

        self._samples: list[GpuPowerSample] = []
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._thread_error: BaseException | None = None
        self._started_at: float | None = None
        self._handle = None

    @staticmethod
    def _field_value_w(value) -> float:
        if value.nvmlReturn != pynvml.NVML_SUCCESS:
            raise RuntimeError(
                f"NVML field read failed: {value.nvmlReturn}"
            )

        return float(value.value.uiVal) / 1000.0

    def _read_sample(self) -> GpuPowerSample:
        assert self._handle is not None
        assert self._started_at is not None

        values = pynvml.nvmlDeviceGetFieldValues(
            self._handle,
            [
                pynvml.NVML_FI_DEV_POWER_INSTANT,
                pynvml.NVML_FI_DEV_POWER_AVERAGE,
            ],
        )

        return GpuPowerSample(
            time_s=perf_counter() - self._started_at,
            instant_w=self._field_value_w(values[0]),
            average_w=self._field_value_w(values[1]),
        )

    def _sample_loop(self) -> None:
        try:
            while not self._stop_event.wait(self.interval_s):
                self._samples.append(self._read_sample())
        except Exception as exc: # noqa: BLE001
            self._thread_error = exc
            self._stop_event.set()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("meter is already running")

        pynvml.nvmlInit()

        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(
                self.device_index
            )

            self._samples = []
            self._thread_error = None
            self._stop_event.clear()
            self._started_at = perf_counter()

            # First synchronous sample verifies both fields.
            self._samples.append(self._read_sample())

            self._thread = Thread(
                target=self._sample_loop,
                daemon=True,
            )
            self._thread.start()

        except Exception:
            pynvml.nvmlShutdown()
            self._handle = None
            self._started_at = None
            raise

    def stop(self) -> GpuPowerSummary:
        if self._thread is None:
            raise RuntimeError("meter is not running")

        self._stop_event.set()
        self._thread.join()
        self._thread = None

        try:
            if self._thread_error is not None:
                raise RuntimeError(
                    "NVML sampling thread failed"
                ) from self._thread_error

            # Final synchronous sample closes the integration interval.
            self._samples.append(self._read_sample())

            return summarize_samples(self._samples)

        finally:
            pynvml.nvmlShutdown()
            self._handle = None
            self._started_at = None

    @property
    def samples(self) -> tuple[GpuPowerSample, ...]:
        return tuple(self._samples)
