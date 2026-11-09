from findex.eager import compute_stats_eager
from findex.stats import compute_stats


def make_corpus(root):
    (root / "1.txt").write_text("The cat sat. The cat ran! Привіт, привіт.", encoding="utf-8")
    (root / "2.txt").write_text("A cat, a dog; the end. Python 3.12", encoding="utf-8")
    (root / "3.txt").write_text("", encoding="utf-8")


def test_counts(tmp_path):
    make_corpus(tmp_path)
    stats = compute_stats(tmp_path, top_n=3)
    assert stats.documents == 3
    # doc1: the cat sat the cat ran привіт привіт (8); doc2: a cat a dog the end python 3.12 (8)
    assert stats.tokens == 16
    assert stats.vocabulary == 10
    # cat and the both occur 3 times: ties are broken alphabetically
    assert stats.top_terms == [("cat", 3), ("the", 3), ("a", 2)]


def test_limit_uses_only_first_documents(tmp_path):
    make_corpus(tmp_path)
    stats = compute_stats(tmp_path, limit=1)
    assert stats.documents == 1
    assert stats.tokens == 8  # the cat sat the cat ran привіт привіт


def test_eager_and_lazy_agree(tmp_path):
    make_corpus(tmp_path)
    assert compute_stats_eager(tmp_path) == compute_stats(tmp_path)
    assert compute_stats_eager(tmp_path, limit=2) == compute_stats(tmp_path, limit=2)
