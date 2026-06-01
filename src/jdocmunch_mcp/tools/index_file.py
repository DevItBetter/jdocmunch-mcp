"""Re-index a single file within an existing index."""

import os
import time
from pathlib import Path
from typing import Optional

from ..parser import parse_file, preprocess_content, ALL_EXTENSIONS
from ..storage import DocStore
from ..summarizer import summarize_sections
from ..embeddings import embed_sections
from ._git import local_git_state


def _find_owning_index(
    file_path: Path,
    store: DocStore,
) -> Optional[tuple[str, str, str, Path]]:
    """Find which index owns a given file path.

    Walks up the directory tree from the file, checking each ancestor
    folder name against existing local indexes.  When a match is found,
    verifies that the relative path exists in the index's doc_paths.

    Returns (owner, name, rel_path, source_root) or None.
    """
    file_path = file_path.resolve()
    parts = file_path.parts

    # Try each ancestor directory as a potential source root
    import re
    for i in range(len(parts) - 1, 0, -1):
        candidate_root = Path(*parts[:i])
        folder_name = parts[i - 1]

        # Skip invalid folder names (drive roots, dots, etc.)
        if not folder_name or not re.fullmatch(r"[A-Za-z0-9._-]+", folder_name):
            continue

        # Check if a local index with this name exists
        try:
            index = store.load_index("local", folder_name)
        except ValueError:
            continue
        if index is None:
            continue

        # Check if the file's relative path is in this index
        try:
            rel_path = file_path.relative_to(candidate_root).as_posix()
        except ValueError:
            continue

        if rel_path in index.doc_paths or rel_path in index.file_hashes:
            return ("local", folder_name, rel_path, candidate_root)

        # The file might be new (not yet in doc_paths) but under this root
        # Accept if the root contains other indexed files
        if index.doc_paths:
            return ("local", folder_name, rel_path, candidate_root)

    return None


def index_file(
    file_path: str,
    storage_path: Optional[str] = None,
    use_ai_summaries: bool = True,
) -> dict:
    """Re-index a single file within an existing index.

    Finds which index owns the file, re-parses it, and updates the
    index in place using incremental_save.

    Returns dict with results.
    """
    t0 = time.perf_counter()
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}", "exit_code": 2}
    if not path.is_file():
        return {"success": False, "error": f"Not a file: {file_path}", "exit_code": 2}

    ext = path.suffix.lower()
    if ext not in ALL_EXTENSIONS:
        return {"success": False, "error": f"Not a doc file ({ext}): {file_path}", "exit_code": 2}

    store = DocStore(base_path=storage_path)
    match = _find_owning_index(path, store)

    if match is None:
        return {"success": False, "error": f"File not in any index: {file_path}", "exit_code": 1}

    owner, name, rel_path, detected_root = match
    repo_id = f"{owner}/{name}"

    # Read and parse the file
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        content = preprocess_content(content, rel_path)
    except Exception as e:
        return {"success": False, "error": f"Failed to read: {e}", "exit_code": 2}

    try:
        new_sections = parse_file(content, rel_path, repo_id)
    except Exception as e:
        return {"success": False, "error": f"Failed to parse: {e}", "exit_code": 2}

    if not new_sections:
        new_sections = []

    new_sections = summarize_sections(new_sections, use_ai=use_ai_summaries)

    # Determine if this is a new or changed file
    index = store.load_index(owner, name)
    is_new = rel_path not in (index.file_hashes if index else {})
    source_root = detected_root
    if index is not None and getattr(index, "source_root", ""):
        candidate = Path(index.source_root).expanduser()
        if candidate.exists():
            source_root = candidate.resolve()
    head_sha, source_dirty = local_git_state(source_root, scope_path=path)

    # Preserve embedding parity: if the existing index has embeddings, embed new sections too.
    if index is not None and index._has_embeddings():
        new_sections = embed_sections(new_sections)

    # Use incremental_save to update just this file
    updated = store.incremental_save(
        owner=owner,
        name=name,
        changed_files=[] if is_new else [rel_path],
        new_files=[rel_path] if is_new else [],
        deleted_files=[],
        new_sections=new_sections,
        raw_files={rel_path: content},
        doc_types={ext: 1},
        head_sha=head_sha,
        source_dirty=source_dirty,
        source_root=str(source_root),
    )

    latency_ms = int((time.perf_counter() - t0) * 1000)
    result = {
        "success": True,
        "repo": repo_id,
        "file": rel_path,
        "is_new": is_new,
        "sections": len(new_sections),
        "total_sections": len(updated.sections) if updated else 0,
        "exit_code": 0,
        "_meta": {"latency_ms": latency_ms},
    }
    if updated and updated.head_sha:
        result["head_sha"] = updated.head_sha
    if updated:
        result["source_dirty"] = bool(updated.source_dirty)
    if updated and updated.repo_at_sha:
        result["repo_at_sha"] = updated.repo_at_sha
    return result


def index_file_cli(file_path: str) -> dict:
    """CLI entry point for index-file subcommand."""
    return index_file(file_path)
