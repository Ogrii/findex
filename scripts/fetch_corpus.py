"""Download the findex corpus into data/ (data/ is git-ignored).

Corpus = the CPython documentation (Doc/**/*.rst) + all Python Enhancement Proposals
(peps/**/*.rst). Both come from public GitHub repositories; only the needed folder is
checked out (sparse clone, depth 1) and only .rst files are copied. The exact commits
are recorded in corpus-sources.lock (committed to git) so the corpus is reproducible.

Requires git. Usage:  uv run python scripts/fetch_corpus.py
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOCK = ROOT / "corpus-sources.lock"  # outside data/ so it is not read as a document

# (target folder, repository, folder inside the repository)
SOURCES = [
    ("cpython-docs", "https://github.com/python/cpython.git", "Doc"),
    ("peps", "https://github.com/python/peps.git", "peps"),
]


def git(*args: str, cwd: Path | None = None) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def fetch(name: str, url: str, subdir: str) -> str:
    target = DATA / name
    if target.exists():
        shutil.rmtree(target)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        clone = Path(tmp) / "repo"
        print(f"[{name}] cloning {url} (folder {subdir}/ only)...")
        git("clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(clone))
        git("sparse-checkout", "set", subdir, cwd=clone)
        commit = git("rev-parse", "HEAD", cwd=clone)
        shutil.copytree(
            clone / subdir,
            target,
            ignore=lambda _dir, names: [n for n in names if "." in n and not n.endswith(".rst")],
        )
    n_files = sum(1 for _ in target.rglob("*.rst"))
    print(f"[{name}] {n_files} .rst files @ {commit[:10]}")
    return f"{name}: {url} folder={subdir} commit={commit}"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    lines = [fetch(*source) for source in SOURCES]
    LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("done ->", DATA, "| versions pinned in", LOCK.name)


if __name__ == "__main__":
    main()
