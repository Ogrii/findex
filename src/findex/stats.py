"""Corpus statistics in ONE lazy pass.

    iter_documents -> (islice) -> tokenize -> Counter

Each stage pulls from the previous one, item by item. The only structure that grows
is the ``Counter`` - the sink that legitimately holds the answer (the term table).
Everything upstream is a stream, so peak memory is about "one document + the Counter".

Usage::

    python -m findex.stats data/                # whole corpus
    python -m findex.stats data/ --limit 100    # first 100 documents only
"""

from __future__ import annotations

import argparse
import heapq
import logging
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.measure import mib, run_timed, run_traced
from findex.tokenize import tokenize


@dataclass
class CorpusStats:
    documents: int
    tokens: int
    vocabulary: int
    top_terms: list[tuple[str, int]]


class _Tally:
    """Mutable counter for the side-effect of counting documents inside the stream."""

    documents = 0


def top_terms(counts: Counter[str], n: int) -> list[tuple[str, int]]:
    """The n most frequent terms; ties are broken alphabetically (deterministic).

    ``heapq.nsmallest`` keeps only n candidates, so no sorted copy of the whole
    vocabulary is built.
    """
    return heapq.nsmallest(n, counts.items(), key=lambda item: (-item[1], item[0]))


def token_stream(docs: Iterable[Document], tally: _Tally) -> Iterator[str]:
    """Flatten a stream of documents into a stream of tokens (counting documents)."""
    for doc in docs:
        tally.documents += 1
        yield from tokenize(doc.text)


def compute_stats(root: Path | str, *, limit: int | None = None, top_n: int = 50) -> CorpusStats:
    """Documents, tokens, vocabulary and top terms - single pass, constant memory."""
    docs: Iterable[Document] = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)  # lazy "first N" - never reads the rest
    tally = _Tally()
    counts = Counter(token_stream(docs, tally))
    return CorpusStats(
        documents=tally.documents,
        tokens=counts.total(),
        vocabulary=len(counts),
        top_terms=top_terms(counts, top_n),
    )


def print_report(stats: CorpusStats, elapsed: float, peak_bytes: int | None) -> None:
    print(f"Documents:  {stats.documents:>12,}")
    print(f"Tokens:     {stats.tokens:>12,}")
    print(f"Vocabulary: {stats.vocabulary:>12,}")
    print(f"\nTop {len(stats.top_terms)} terms:")
    for rank, (term, count) in enumerate(stats.top_terms, start=1):
        print(f"  {rank:>3}. {term:<20} {count:>10,}")
    print(f"\nElapsed:     {elapsed:.2f} s")
    if peak_bytes is None:
        print("Peak memory: not measured (--no-memory)")
    else:
        print(f"Peak memory: {mib(peak_bytes):.1f} MiB (tracemalloc, slows the run)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m findex.stats", description=__doc__.split("\n")[0]
    )
    parser.add_argument("root", type=Path, help="directory of text files or a .jsonl file")
    parser.add_argument("--limit", type=int, metavar="N", help="only the first N documents")
    parser.add_argument("--top", type=int, default=50, metavar="K", help="how many top terms")
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="skip tracemalloc (it slows the run down) to get a clean elapsed time",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    # iter_documents is a generator and would only fail on the first next(), so check here.
    if not args.root.exists():
        print(
            f"error: {args.root} does not exist (see README: scripts/fetch_corpus.py)",
            file=sys.stderr,
        )
        return 2

    if args.no_memory:
        measured = run_timed(compute_stats, args.root, limit=args.limit, top_n=args.top)
        peak = None
    else:
        measured = run_traced(compute_stats, args.root, limit=args.limit, top_n=args.top)
        peak = measured.peak_bytes
    if measured.result.documents == 0:
        print(f"warning: no documents found under {args.root}", file=sys.stderr)
    print_report(measured.result, measured.elapsed, peak)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
