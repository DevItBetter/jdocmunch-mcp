"""Small Git helpers for local indexing metadata."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from ..storage.doc_store import normalize_commit_sha


def _git(cwd: Path, args: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            check=True,
        )
    except Exception:
        return False, ""
    return True, proc.stdout.strip()


def local_git_state(folder_path: Path, scope_path: Optional[Path] = None) -> tuple[Optional[str], bool]:
    """Return (HEAD sha, dirty) for a local Git worktree.

    Non-Git folders return ``(None, False)``. Once a worktree is detected,
    failure to prove clean status is treated as dirty so callers never emit an
    immutable repo@sha handle for an unknown state.
    """
    folder_path = folder_path.resolve()
    ok, inside = _git(folder_path, ["rev-parse", "--is-inside-work-tree"])
    if not ok or inside != "true":
        return None, False

    ok, head = _git(folder_path, ["rev-parse", "HEAD"])
    head_sha = normalize_commit_sha(head)
    if not ok or not head_sha:
        return None, False

    ok, root = _git(folder_path, ["rev-parse", "--show-toplevel"])
    git_root = Path(root).resolve() if ok and root else folder_path

    status_args = ["status", "--porcelain"]
    if scope_path is not None:
        try:
            rel = scope_path.resolve().relative_to(git_root).as_posix()
        except ValueError:
            rel = scope_path.resolve().as_posix()
        status_args.extend(["--", rel or "."])

    ok, status = _git(git_root, status_args)
    if not ok:
        return head_sha, True
    return head_sha, bool(status)


def stable_local_git_state(
    before: tuple[Optional[str], bool],
    after: tuple[Optional[str], bool],
) -> tuple[Optional[str], bool]:
    """Combine pre/post read Git state; SHA movement makes the index dirty."""
    before_sha, before_dirty = before
    after_sha, after_dirty = after
    moved = before_sha != after_sha and bool(before_sha or after_sha)
    return after_sha or before_sha, bool(before_dirty or after_dirty or moved)
