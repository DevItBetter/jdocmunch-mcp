"""Small Git helpers for local indexing metadata."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from ..storage.doc_store import normalize_commit_sha

DEFAULT_GIT_TIMEOUT = 10.0


def _git_timeout() -> Optional[float]:
    """Resolve the per-call git subprocess ceiling (seconds).

    Bounded by default so a blocked git (index.lock contention, credential
    prompt, LFS smudge) can never hang the synchronous tool path. Override with
    ``JDOCMUNCH_GIT_TIMEOUT``; a value <= 0 disables the ceiling entirely.
    """
    raw = os.environ.get("JDOCMUNCH_GIT_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_GIT_TIMEOUT
    try:
        val = float(raw)
    except ValueError:
        return DEFAULT_GIT_TIMEOUT
    return val if val > 0 else None


def _git(cwd: Path, args: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=_git_timeout(),
            check=True,
        )
    except Exception:
        return False, ""
    return True, proc.stdout.strip()


def _git_bytes(cwd: Path, args: list[str]) -> tuple[bool, bytes]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_git_timeout(),
            check=True,
        )
    except Exception:
        return False, b""
    return True, proc.stdout


def _git_root(folder_path: Path) -> Optional[Path]:
    ok, inside = _git(folder_path, ["rev-parse", "--is-inside-work-tree"])
    if not ok or inside != "true":
        return None
    ok, root = _git(folder_path, ["rev-parse", "--show-toplevel"])
    return Path(root).resolve() if ok and root else None


def local_git_head(folder_path: Path) -> Optional[str]:
    """Return the current HEAD SHA for a local Git worktree, if available."""
    folder_path = folder_path.resolve()
    if _git_root(folder_path) is None:
        return None
    ok, head = _git(folder_path, ["rev-parse", "HEAD"])
    return normalize_commit_sha(head) if ok else None


def local_git_state(folder_path: Path, scope_path: Optional[Path] = None) -> tuple[Optional[str], bool]:
    """Return (HEAD sha, dirty) for a local Git worktree.

    "Dirty" means tracked content in scope differs from HEAD. Untracked files
    are deliberately ignored (``--untracked-files=no``): they do not affect
    whether the *indexed corpus* is reproducible at HEAD, and that corpus is
    independently proven fully tracked by ``local_git_paths_tracked``. The
    explicit ``no`` mode also overrides any ``status.showUntrackedFiles`` repo
    config so the result is deterministic.

    Non-Git folders return ``(None, False)``. Once a worktree is detected,
    failure to prove clean status is treated as dirty so callers never emit an
    immutable repo@sha handle for an unknown state.
    """
    folder_path = folder_path.resolve()
    git_root = _git_root(folder_path)
    if git_root is None:
        return None, False

    head_sha = local_git_head(folder_path)
    if not head_sha:
        return None, False

    status_args = ["status", "--porcelain", "--untracked-files=no"]
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


def _indexed_git_paths(folder_path: Path, indexed_paths: Iterable[str]) -> tuple[Optional[Path], Optional[set[str]]]:
    folder_path = folder_path.resolve()
    git_root = _git_root(folder_path)
    if git_root is None:
        return None, None

    wanted: set[str] = set()
    for rel_path in indexed_paths:
        if not isinstance(rel_path, str) or not rel_path:
            return git_root, None
        try:
            git_rel = (folder_path / rel_path).resolve().relative_to(git_root).as_posix()
        except ValueError:
            return git_root, None
        wanted.add(git_rel)
    return git_root, wanted


def local_git_paths_dirty(folder_path: Path, indexed_paths: Iterable[str]) -> bool:
    """Return True when tracked indexed paths differ from HEAD."""
    git_root, wanted = _indexed_git_paths(folder_path, indexed_paths)
    if git_root is None:
        return False
    if wanted is None:
        return True
    if not wanted:
        return False

    ordered = sorted(wanted)
    chunk_size = 200
    for i in range(0, len(ordered), chunk_size):
        ok, status = _git(
            git_root,
            ["status", "--porcelain", "--untracked-files=no", "--", *ordered[i:i + chunk_size]],
        )
        if not ok or status:
            return True
    return False


def local_git_paths_tracked(folder_path: Path, indexed_paths: Iterable[str]) -> bool:
    """Return True when every indexed path is tracked by Git."""
    git_root, wanted = _indexed_git_paths(folder_path, indexed_paths)
    if git_root is None or wanted is None:
        return False

    if not wanted:
        return True

    tracked: set[str] = set()
    ordered = sorted(wanted)
    chunk_size = 200
    for i in range(0, len(ordered), chunk_size):
        ok, output = _git_bytes(git_root, ["ls-files", "-z", "--", *ordered[i:i + chunk_size]])
        if not ok:
            return False
        tracked.update(
            p for p in output.decode("utf-8", errors="surrogateescape").split("\0") if p
        )
    return wanted <= tracked


def stable_local_git_state(
    before: tuple[Optional[str], bool],
    after: tuple[Optional[str], bool],
) -> tuple[Optional[str], bool]:
    """Combine pre/post read Git state; SHA movement makes the index dirty."""
    before_sha, before_dirty = before
    after_sha, after_dirty = after
    moved = before_sha != after_sha and bool(before_sha or after_sha)
    return after_sha or before_sha, bool(before_dirty or after_dirty or moved)
