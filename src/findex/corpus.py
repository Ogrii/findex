"""Lazy document stream.

``iter_documents(root)`` is a *generator*: it opens one file at a time, yields it,
and only then moves on to the next one. At any moment at most one document's text
is held by this module, so the corpus can be far larger than RAM.

Two corpus layouts are supported:

* a directory tree of text files (``.txt``, ``.rst``, ``.md`` by default);
* a single JSON-lines file (``.jsonl`` / ``.ndjson``): one JSON object per line,
  the document text is taken from the ``"text"`` field.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)

DEFAULT_SUFFIXES: tuple[str, ...] = (".txt", ".rst", ".md")
JSONL_SUFFIXES: tuple[str, ...] = (".jsonl", ".ndjson")
ENCODING = "utf-8"


class Document(NamedTuple):
    """One document of the corpus."""

    doc_id: int  # 0, 1, 2, ... in the (deterministic) order of iteration
    path: Path  # file the text came from (for JSONL: the .jsonl file itself)
    text: str


def iter_documents(
    root: Path | str,
    *,
    suffixes: Iterable[str] = DEFAULT_SUFFIXES,
) -> Iterator[Document]:
    """Yield the documents under ``root`` one by one, never all at once.

    The order is deterministic (sorted by name inside every directory), so a
    document keeps the same ``doc_id`` between runs - the index will rely on it.

    Because this is a generator, nothing runs (and a missing ``root`` does not
    raise) until the first ``next()``. Callers that want to fail early should
    check ``Path(root).exists()`` themselves - see ``findex.stats``.

    Files that are not valid UTF-8 are read with ``errors="replace"`` and a warning
    is logged; unreadable files and broken JSON lines are logged and skipped.
    """
    root = Path(root)
    source = (
        _iter_jsonl(root)
        if root.is_file() and root.suffix.lower() in JSONL_SUFFIXES
        else _iter_text_files(root, frozenset(s.lower() for s in suffixes))
    )
    for doc_id, (path, text) in enumerate(source):
        yield Document(doc_id, path, text)


def _walk(root: Path, suffixes: frozenset[str]) -> Iterator[Path]:
    """Lazily walk a tree in sorted order, skipping hidden entries.

    Only one directory listing is held at a time, which keeps the walk lazy
    *and* deterministic (``Path.rglob`` is lazy but its order is not defined).
    """
    if root.is_file():
        yield root
        return
    for entry in sorted(root.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            if not entry.is_symlink():  # avoid symlink loops
                yield from _walk(entry, suffixes)
        elif entry.suffix.lower() in suffixes:
            yield entry


def _iter_text_files(root: Path, suffixes: frozenset[str]) -> Iterator[tuple[Path, str]]:
    for path in _walk(root, suffixes):
        text = _read_text(path)
        if text is not None:
            yield path, text


def _read_text(path: Path) -> str | None:
    """Read one file; ``None`` means "skip it" (already logged)."""
    try:
        return path.read_text(encoding=ENCODING)
    except UnicodeDecodeError:
        log.warning("%s: not valid %s, decoding with replacement characters", path, ENCODING)
        try:
            return path.read_text(encoding=ENCODING, errors="replace")
        except OSError as exc:
            log.warning("%s: skipped (%s)", path, exc)
    except OSError as exc:
        log.warning("%s: skipped (%s)", path, exc)
    return None


def _iter_jsonl(path: Path) -> Iterator[tuple[Path, str]]:
    # A file object is itself a lazy iterator over lines: one line in memory at a time.
    with path.open(encoding=ENCODING, errors="replace") as lines:
        for lineno, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                text = record["text"]
            except (json.JSONDecodeError, KeyError, TypeError):
                log.warning('%s:%d: not a JSON object with a "text" field, skipped', path, lineno)
                continue
            if isinstance(text, str):
                yield path, text
            else:
                log.warning('%s:%d: "text" is not a string, skipped', path, lineno)
