"""Eager-vs-lazy measurement: prints the Markdown table for the README.

    python -m findex.bench data/              # whole corpus
    python -m findex.bench data/ --limit 300  # a slice, if the eager run is too heavy

For every version: elapsed time = median of ``--repeats`` runs *without* tracemalloc,
peak memory = one separate run *with* tracemalloc (tracing would distort the timings).
"""

from __future__ import annotations

import argparse
import gc
import logging
import platform
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

from findex.eager import compute_stats_eager
from findex.measure import mib, run_timed, run_traced
from findex.stats import CorpusStats, compute_stats

VERSIONS: dict[str, Callable[..., CorpusStats]] = {
    "eager (lists)": compute_stats_eager,
    "lazy (generators)": compute_stats,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m findex.bench", description=__doc__.split("\n")[0]
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, metavar="N", help="only the first N documents")
    parser.add_argument("--repeats", type=int, default=3, help="timing runs per version")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    if not args.root.exists():
        print(f"error: {args.root} does not exist", file=sys.stderr)
        return 2

    rows = []
    results: list[CorpusStats] = []
    for name, fn in VERSIONS.items():
        timings = []
        for _ in range(args.repeats):
            gc.collect()
            timings.append(run_timed(fn, args.root, limit=args.limit).elapsed)
        gc.collect()
        traced = run_traced(fn, args.root, limit=args.limit)
        results.append(traced.result)
        rows.append((name, traced.result, traced.peak_bytes, statistics.median(timings)))

    if results[0] != results[1]:
        print("error: eager and lazy versions disagree!", file=sys.stderr)
        return 1

    print(
        f"Python {platform.python_version()} on {platform.system()} {platform.machine()}; "
        f"time = median of {args.repeats} runs without tracemalloc, "
        f"memory = tracemalloc peak\n"
    )
    print("| Version           | Documents | Tokens    | Peak memory | Elapsed |")
    print("| ----------------- | --------- | --------- | ----------- | ------- |")
    for name, stats, peak, elapsed in rows:
        print(
            f"| {name:<17} | {stats.documents:>9,} | {stats.tokens:>9,} | "
            f"{mib(peak):>7.1f} MiB | {elapsed:>5.2f} s |"
        )
    eager_peak, lazy_peak = rows[0][2], rows[1][2]
    print(f"\nEager uses {eager_peak / lazy_peak:.1f}x more peak memory than lazy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
