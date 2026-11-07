"""Upstream git checkouts: one bare clone per repo, one detached worktree per commit."""
import subprocess
from pathlib import Path


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def checkout(repo: str, commit: str, root: Path) -> Path:
    short = repo.split("/", 1)[1]
    root = Path(root).resolve()  # git worktree add resolves relative paths against the bare repo
    bare = root / f"{short}.git"
    wt = root / "wt" / f"{short}_{commit}"
    if wt.exists():
        return wt
    if not bare.exists():
        root.mkdir(parents=True, exist_ok=True)
        _git("clone", "--bare", f"https://github.com/{repo}.git", str(bare))
    if _git("-C", str(bare), "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode:
        _git("-C", str(bare), "fetch", "origin", commit)
    wt.parent.mkdir(parents=True, exist_ok=True)
    _git("-C", str(bare), "worktree", "add", "--detach", str(wt), commit)
    return wt
