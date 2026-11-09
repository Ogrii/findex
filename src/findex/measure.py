"""Tiny measurement helpers: wall-clock time and peak Python memory.

``tracemalloc`` slows the program down noticeably, so time and memory are measured in
*separate* runs: ``run_timed`` without tracing, ``run_traced`` with it. Elapsed time
from ``run_traced`` is therefore inflated - never compare it with ``run_timed``.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from typing import NamedTuple


class Measured[T](NamedTuple):
    result: T
    elapsed: float  # seconds, time.perf_counter()
    peak_bytes: int  # 0 for run_timed (memory was not traced)


def run_timed[T](fn: Callable[..., T], /, *args, **kwargs) -> Measured[T]:
    """Run ``fn`` and measure wall-clock time only."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return Measured(result, time.perf_counter() - start, 0)


def run_traced[T](fn: Callable[..., T], /, *args, **kwargs) -> Measured[T]:
    """Run ``fn`` under ``tracemalloc`` and report the peak of traced allocations."""
    tracemalloc.start()
    try:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return Measured(result, elapsed, peak)


def mib(n_bytes: int) -> float:
    return n_bytes / (1024 * 1024)
