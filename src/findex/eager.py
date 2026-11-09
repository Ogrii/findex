"""The deliberately GREEDY version - written to be measured, not to be used.

Same result as ``findex.stats.compute_stats``, but built the way people coming from
other languages usually write it: read everything into a list, turn that into another
list, and only then count. Both big lists stay alive until the function returns.
"""

from __future__ import annotations

from collections import Counter
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.stats import CorpusStats, top_terms
from findex.tokenize import tokenize


def compute_stats_eager(
    root: Path | str, *, limit: int | None = None, top_n: int = 50
) -> CorpusStats:
    # 1) the whole corpus text in memory at once
    docs = list(islice(iter_documents(root), limit))
    # 2) every token of every document as a separate str object, in lists of lists
    token_lists = [list(tokenize(doc.text)) for doc in docs]
    # 3) only now count
    counts: Counter[str] = Counter()
    for tokens in token_lists:
        counts.update(tokens)
    return CorpusStats(
        documents=len(docs),
        tokens=sum(len(tokens) for tokens in token_lists),
        vocabulary=len(counts),
        top_terms=top_terms(counts, top_n),
    )
