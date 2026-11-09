import inspect
import json
import logging
from pathlib import Path

import findex.corpus as corpus
from findex.corpus import Document, iter_documents


def make_tree(root: Path) -> None:
    (root / "b").mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    (root / "b" / "c.rst").write_text("gamma", encoding="utf-8")
    (root / "b" / "b.md").write_text("beta", encoding="utf-8")
    (root / "ignored.png").write_bytes(b"\x89PNG")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "x.txt").write_text("hidden", encoding="utf-8")


def test_is_a_generator_function():
    assert inspect.isgeneratorfunction(iter_documents)


def test_walks_tree_in_deterministic_order(tmp_path):
    make_tree(tmp_path)
    docs = list(iter_documents(tmp_path))
    assert [d.text for d in docs] == ["alpha", "beta", "gamma"]
    assert [d.doc_id for d in docs] == [0, 1, 2]
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].path == tmp_path / "a.txt"


def test_reads_one_file_at_a_time(tmp_path, monkeypatch):
    make_tree(tmp_path)
    reads: list[Path] = []
    real = corpus._read_text

    def spy(path):
        reads.append(path)
        return real(path)

    monkeypatch.setattr(corpus, "_read_text", spy)
    stream = iter_documents(tmp_path)
    assert reads == []  # creating the generator reads nothing
    next(stream)
    assert len(reads) == 1  # first document -> exactly one file read
    next(stream)
    assert len(reads) == 2


def test_bad_encoding_is_logged_not_fatal(tmp_path, caplog):
    (tmp_path / "bad.txt").write_bytes(b"caf\xe9 ok")  # latin-1 byte, invalid UTF-8
    (tmp_path / "good.txt").write_text("fine", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        docs = list(iter_documents(tmp_path))
    assert [d.text for d in docs] == ["caf\ufffd ok", "fine"]
    assert any("bad.txt" in r.message for r in caplog.records)


def test_jsonl_one_document_per_line(tmp_path, caplog):
    path = tmp_path / "docs.jsonl"
    lines = [
        json.dumps({"text": "перший"}, ensure_ascii=False),
        "not json",
        "",
        json.dumps({"no_text": 1}),
        json.dumps({"text": "second"}),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        docs = list(iter_documents(path))
    assert [d.text for d in docs] == ["перший", "second"]
    assert [d.doc_id for d in docs] == [0, 1]
    assert len(caplog.records) == 2  # the two broken lines were reported


def test_empty_directory_yields_nothing(tmp_path):
    assert list(iter_documents(tmp_path)) == []
