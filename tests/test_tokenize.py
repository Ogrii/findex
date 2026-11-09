"""Checks that pin the documented tokenizer policy (see findex/tokenize.py)."""

import inspect

from findex.tokenize import tokenize


def toks(text: str) -> list[str]:
    return list(tokenize(text))


def test_is_a_lazy_generator():
    assert inspect.isgeneratorfunction(tokenize)
    stream = tokenize("one two three")
    assert iter(stream) is stream  # an iterator, not a list
    assert next(stream) == "one"  # produced on demand


def test_mixed_case_is_casefolded():
    assert toks("Hello WORLD hElLo") == ["hello", "world", "hello"]
    assert toks("Straße") == toks("STRASSE") == ["strasse"]  # casefold, not lower()


def test_cyrillic():
    assert toks("Привіт, світе! Київ — столиця України.") == [
        "привіт",
        "світе",
        "київ",
        "столиця",
        "україни",
    ]


def test_combining_accent_is_normalized_to_nfc():
    composed, decomposed = "caf\u00e9", "cafe\u0301"
    assert composed != decomposed  # different code points...
    assert toks(composed) == toks(decomposed) == ["caf\u00e9"]  # ...same single token
    assert toks("Й") == toks("И\u0306") == ["й"]  # Ukrainian/Russian short i


def test_punctuation_is_dropped():
    assert toks('Hello, world! (test) "quoted"; end...') == [
        "hello",
        "world",
        "test",
        "quoted",
        "end",
    ]


def test_empty_and_whitespace_only():
    assert toks("") == []
    assert toks("  \n\t  ") == []
    assert toks("... --- !!!") == []


def test_apostrophes_are_kept_inside_words_and_unified():
    assert toks("don't don\u2019t") == ["don't", "don't"]
    assert toks("п'ять п\u2019ять п\u02bcять") == ["п'ять"] * 3
    assert toks("'quoted' users' python's") == ["quoted", "users", "python's"]


def test_hyphen_separates_tokens():
    assert toks("built-in well-known") == ["built", "in", "well", "known"]
    assert toks("-5 and 5-") == ["5", "and", "5"]  # a dash is never part of a number


def test_digits_and_versions():
    assert toks("Python 3.12 and 3.12.1.") == ["python", "3.12", "and", "3.12.1"]
    assert toks("pi=3.14.") == ["pi", "3.14"]
    assert toks("1,000") == ["1", "000"]  # comma always separates
    assert toks("3rd x86 py3") == ["3rd", "x86", "py3"]  # mixed alphanumerics stay whole
    assert toks("a.5 5.b") == ["a", "5", "5", "b"]  # a dot joins only digit-dot-digit


def test_underscores():
    assert toks("snake_case __init__ _private ____") == ["snake_case", "init", "private"]


def test_casefold_expansion_stays_one_token():
    # "İ" casefolds to "i" + U+0307 (combining dot); it must not split the word.
    assert len(toks("İstanbul")) == 1
