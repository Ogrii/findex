"""Streaming tokenizer with Unicode normalization.

``tokenize(text)`` yields tokens one at a time (it is a generator).

Pipeline for one document::

    NFC  ->  casefold  ->  NFC again  ->  unify apostrophes  ->  re.finditer

Token rules (these are the documented policy; ``tests/test_tokenize.py`` pins them):

* **Letters and digits** (any script - ``\\w`` is Unicode-aware) make up a token.
* **Case**: ``str.casefold()``, not ``lower()`` - ``"Straße"`` and ``"STRASSE"`` match.
* **Unicode**: text is NFC-normalized, so ``"café"`` is one token whether it was typed
  as ``U+00E9`` or as ``e`` + ``U+0301``. NFC is applied again after ``casefold`` because
  casefolding can emit decomposed sequences (``"İ"`` -> ``"i"`` + ``U+0307``).
* **Apostrophes** inside a word are kept and unified to ASCII ``'``: ``don't``,
  ``don’t`` -> ``don't``; ``п'ять``, ``п’ять``, ``пʼять`` -> ``п'ять``. An apostrophe at the
  start or end of a word is *not* part of the token (``'quoted'`` -> ``quoted``,
  ``users'`` -> ``users``). Possessives are not stripped (``python's`` stays whole);
  that is a job for stemming later.
* **Hyphens separate tokens**: ``built-in`` -> ``built``, ``in``. Better recall for
  compounds, simple rule; a dash is never part of a token.
* **Digits**: digit runs are tokens, and a single ``.`` *between digits* is kept, so
  versions and decimals stay whole (``3.12``, ``3.14``, ``1.2.3``). A trailing dot is
  sentence punctuation (``3.12.`` -> ``3.12``). A comma is always a separator
  (``1,000`` -> ``1``, ``000``). Mixed alphanumerics stay whole (``3rd``, ``x86``).
* **Underscores** are word characters (``snake_case`` is one token) but are trimmed from
  both ends (``__init__`` -> ``init``); a run of only underscores yields nothing.

Known limits: scripts that rely on combining marks without precomposed forms
(Devanagari, Thai, ...) are split at the marks, and CJK text comes out as long runs.
The stretch goal of Lab 1 (the third-party ``regex`` module with ``\\p{L}``) addresses that.

Memory note: the *normalized copy of one document* exists while it is being tokenized,
but tokens themselves are produced lazily, so memory is O(one document), not O(corpus).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

# Typographic / modifier apostrophes that all mean "the apostrophe inside a word".
_APOSTROPHES = str.maketrans(
    {
        "\u2019": "'",  # ’ right single quotation mark (most common in Ukrainian/English text)
        "\u2018": "'",  # ‘ left single quotation mark
        "\u02bc": "'",  # ʼ modifier letter apostrophe (the "correct" Ukrainian apostrophe)
        "\u00b4": "'",  # ´ acute accent, often typed instead of an apostrophe
    }
)

# A word character, plus the Combining Diacritical Marks block. NFC already composes
# the common cases (é, й, ї); this keeps a leftover mark (e.g. i + U+0307) in its word.
_W = r"[\w\u0300-\u036f]"

# word-chars, then any number of ( "'" | "." between two digits ) followed by word-chars.
_TOKEN_RE = re.compile(rf"{_W}+(?:(?:'|(?<=\d)\.(?=\d)){_W}+)*")


def normalize(text: str) -> str:
    """NFC -> casefold -> NFC -> unify apostrophes. Returns a *new* string."""
    text = unicodedata.normalize("NFC", text)
    text = text.casefold()
    text = unicodedata.normalize("NFC", text)  # casefold may leave non-NFC sequences
    return text.translate(_APOSTROPHES)


def tokenize(text: str) -> Iterator[str]:
    """Yield the tokens of ``text`` lazily, in order of appearance."""
    # finditer (not findall): matches are produced one by one, never as one big list.
    for match in _TOKEN_RE.finditer(normalize(text)):
        token = match.group().strip("_'")
        if token:  # e.g. "____" is only underscores
            yield token
