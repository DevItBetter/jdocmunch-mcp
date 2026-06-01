"""Index local folder tool — walk, parse, summarize, save."""

import os
import time
from pathlib import Path
from typing import Optional

import pathspec

from ..parser import parse_file, preprocess_content, ALL_EXTENSIONS
from ..retrieval.roles import annotate_sections as _annotate_roles
from ..retrieval.glossary import extract_glossary, write_terms
from ..security import (
    validate_path,
    is_symlink_escape,
    is_secret_file,
    should_exclude_file,
    DEFAULT_MAX_FILE_SIZE,
)
from ..storage import DocStore
from ..storage.doc_store import normalize_commit_sha
from ..summarizer import summarize_sections
from ..embeddings import embed_sections, get_provider_name, should_embed
from ._git import local_git_state, stable_local_git_state
from ._constants import SKIP_PATTERNS


def _load_gitignore(folder_path: Path) -> Optional[pathspec.PathSpec]:
    gitignore_path = folder_path / ".gitignore"
    if gitignore_path.is_file():
        try:
            content = gitignore_path.read_text(encoding="utf-8", errors="replace")
            return pathspec.PathSpec.from_lines("gitignore", content.splitlines())
        except Exception:
            pass
    return None


def _add_commit_fields(result: dict, index) -> None:
    if not index:
        return
    if index.head_sha:
        result["head_sha"] = index.head_sha
    result["source_dirty"] = bool(index.source_dirty)
    if index.repo_at_sha:
        result["repo_at_sha"] = index.repo_at_sha


def _should_skip(rel_path: str) -> bool:
    normalized = "/" + rel_path.replace("\\", "/")
    for pat in SKIP_PATTERNS:
        if ("/" + pat) in normalized:
            return True
    return False


_DISCOVERY_HARD_CEILING_MULT = 20  # safety: stop counting at max_files * this


def discover_doc_files(
    folder_path: Path,
    max_files: int = 10_000,
    max_size: int = DEFAULT_MAX_FILE_SIZE,
    extra_ignore_patterns: Optional[list] = None,
    follow_symlinks: bool = False,
    sort_by: str = "newest",
) -> tuple:
    """Discover doc files (.md, .txt, .rst) with security filtering.

    Returns ``(files, warnings, discovered_count)``. ``files`` is capped at
    ``max_files``; ``discovered_count`` is the total that matched all filters
    (capped at ``max_files * _DISCOVERY_HARD_CEILING_MULT`` so a pathological
    directory tree cannot run forever). When ``discovered_count > max_files``
    the caller is responsible for surfacing truncation (jdoc#15).

    ``sort_by`` (jdoc#16) controls truncation order:
      * ``"newest"`` (default): when the cap is hit, the indexed subset is
        the ``max_files`` files with the most recent mtime. So a freshly-
        edited file is always in the index regardless of where it sits in
        the filesystem walk.
      * ``"walk_order"``: take the first ``max_files`` in filesystem-walk
        order (the pre-jdoc#16 behavior). Useful for deterministic
        reproducible builds where mtimes can shift.
    """
    discovered_items: list = []  # [(file_path, mtime_or_zero), ...]
    warnings = []
    hard_ceiling = max_files * _DISCOVERY_HARD_CEILING_MULT
    root = folder_path.resolve()

    gitignore_spec = _load_gitignore(root)
    extra_spec = None
    if extra_ignore_patterns:
        try:
            extra_spec = pathspec.PathSpec.from_lines("gitignore", extra_ignore_patterns)
        except Exception:
            pass

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dir_path = Path(dirpath)
        try:
            dir_rel = dir_path.relative_to(root).as_posix()
        except ValueError:
            dirnames.clear()
            continue

        # Prune skipped directories in-place so os.walk won't descend into them
        dirnames[:] = [
            d for d in dirnames
            if not _should_skip(f"{dir_rel}/{d}/".lstrip("./"))
            and not (gitignore_spec and gitignore_spec.match_file(f"{dir_rel}/{d}/".lstrip("./")))
            and not (extra_spec and extra_spec.match_file(f"{dir_rel}/{d}/".lstrip("./")))
        ]

        for filename in filenames:
            file_path = dir_path / filename

            if not follow_symlinks and file_path.is_symlink():
                continue
            if file_path.is_symlink() and is_symlink_escape(root, file_path):
                warnings.append(f"Skipped symlink escape: {file_path}")
                continue

            if not validate_path(root, file_path):
                warnings.append(f"Skipped path traversal: {file_path}")
                continue

            rel_path = f"{dir_rel}/{filename}".lstrip("./") if dir_rel != "." else filename

            if _should_skip(rel_path):
                continue

            if gitignore_spec and gitignore_spec.match_file(rel_path):
                continue

            if extra_spec and extra_spec.match_file(rel_path):
                continue

            if is_secret_file(rel_path):
                warnings.append(f"Skipped secret file: {rel_path}")
                continue

            ext = file_path.suffix.lower()
            if ext not in ALL_EXTENSIONS:
                continue

            try:
                st = file_path.stat()
                if st.st_size > max_size:
                    continue
                mtime = st.st_mtime
            except OSError:
                continue

            discovered_items.append((file_path, mtime))

        # Stop walking entirely when the safety ceiling is reached so an
        # adversarial / runaway directory tree can't churn forever.
        if len(discovered_items) >= hard_ceiling:
            break

    discovered = len(discovered_items)
    if sort_by == "newest" and discovered > max_files:
        # Only sort on the truncation path; the un-truncated case
        # preserves walk order so callers see no behavior change.
        discovered_items.sort(key=lambda item: item[1], reverse=True)
    files = [fp for fp, _ in discovered_items[:max_files]]
    return files, warnings, discovered


def _resolve_explicit_paths(
    folder_path: Path,
    paths: list,
    max_files: int,
    follow_symlinks: bool,
) -> tuple:
    """Resolve a caller-supplied list of paths into the doc-file shape that the
    downstream pipeline expects. Each entry may be:

      * an absolute path under ``folder_path``, or
      * a path relative to ``folder_path``,
      * a directory (recursed via ``discover_doc_files`` against that subtree),
      * a file (validated and added when its extension is known).

    Returns ``(files, warnings)``. Mirrors ``discover_doc_files`` semantics for
    security: rejects symlink escapes, path-traversal attempts, and entries
    outside ``folder_path``. Skips entries with unknown extensions silently
    (caller gets a `warnings` entry per skip).
    """
    files: list = []
    warnings: list = []
    seen: set = set()

    for raw in paths:
        if not isinstance(raw, str) or not raw.strip():
            warnings.append(f"Skipped empty/non-string path: {raw!r}")
            continue
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (folder_path / p)
        try:
            p = p.resolve()
        except OSError as e:
            warnings.append(f"Skipped unresolvable path {raw!r}: {e}")
            continue

        try:
            p.relative_to(folder_path)
        except ValueError:
            warnings.append(f"Skipped path outside folder: {raw!r}")
            continue

        if not p.exists():
            warnings.append(f"Skipped non-existent path: {raw!r}")
            continue

        if p.is_dir():
            sub_files, sub_warnings, _sub_discovered = discover_doc_files(
                p,
                max_files=max_files - len(files),
                follow_symlinks=follow_symlinks,
            )
            warnings.extend(sub_warnings)
            for f in sub_files:
                fr = f.resolve()
                if fr not in seen:
                    seen.add(fr)
                    files.append(f)
                    if len(files) >= max_files:
                        break
        elif p.is_file():
            if not follow_symlinks and p.is_symlink():
                warnings.append(f"Skipped symlink (follow_symlinks=False): {raw!r}")
                continue
            if p.is_symlink() and is_symlink_escape(folder_path, p):
                warnings.append(f"Skipped symlink escape: {raw!r}")
                continue
            if not validate_path(folder_path, p):
                warnings.append(f"Skipped path traversal: {raw!r}")
                continue
            ext = p.suffix.lower()
            if ext not in ALL_EXTENSIONS:
                warnings.append(f"Skipped unsupported extension: {raw!r}")
                continue
            pr = p.resolve()
            if pr not in seen:
                seen.add(pr)
                files.append(p)
        else:
            warnings.append(f"Skipped non-file/non-dir entry: {raw!r}")

        if len(files) >= max_files:
            break

    return files[:max_files], warnings


def index_local(
    path: str,
    name: Optional[str] = None,
    use_ai_summaries: bool = True,
    use_embeddings="auto",
    storage_path: Optional[str] = None,
    extra_ignore_patterns: Optional[list] = None,
    follow_symlinks: bool = False,
    incremental: bool = True,
    max_files: int = 10_000,
    sort_by: str = "newest",
    autotune: bool = False,
    paths: Optional[list] = None,
) -> dict:
    """Index a local folder containing documentation files.

    Args:
        path: Path to local folder.
        name: Optional repo identifier override. Use when two folders share the same
              name (e.g. two libraries both with a 'docs' folder). Defaults to the
              folder name.
        use_ai_summaries: Whether to use AI for section summaries.
        use_embeddings: True/False/"auto". "auto" (default) enables embeddings when
                        an embedding provider is configured (GOOGLE_API_KEY,
                        OPENAI_API_KEY, openai-compatible
                        + JDOCMUNCH_OPENAI_COMPAT_URL + JDOCMUNCH_OPENAI_COMPAT_MODEL,
                        or sentence-transformers installed).
        storage_path: Custom storage path (default: ~/.doc-index/).
        extra_ignore_patterns: Additional gitignore-style patterns to exclude.
        follow_symlinks: Whether to follow symlinks.
        incremental: When True and an existing index exists, only re-index changed files.
        max_files: Maximum number of doc files to index. Default 10000.
                   When hit, response includes truncated/discovered/indexed
                   top-level fields (jdoc#15).
        sort_by: "newest" (default) or "walk_order". Controls which subset
                 is indexed when discovered > max_files. "newest" sorts by
                 mtime descending so recently-edited files always make it
                 into the index regardless of filesystem-walk position
                 (jdoc#16). "walk_order" preserves the pre-1.65 behavior
                 for callers needing deterministic reproducible builds.
                 No effect when the corpus fits under the cap.
        paths: Optional list of explicit paths to index. When provided, the tree
            walk is skipped; only these files (and the contents of any directories
            in the list) are indexed. Each entry may be absolute or relative to
            ``path``. Useful for batch-indexing exactly the files an agent already
            knows about — e.g. the doc files git just touched.

    Returns:
        Dict with indexing results.
    """
    t0 = time.perf_counter()
    folder_path = Path(path).expanduser().resolve()

    if not folder_path.exists():
        return {"success": False, "error": f"Folder not found: {path}"}
    if not folder_path.is_dir():
        return {"success": False, "error": f"Path is not a directory: {path}"}

    use_embeddings = should_embed(use_embeddings)
    warnings = []

    try:
        if paths:
            doc_files, discover_warnings = _resolve_explicit_paths(
                folder_path,
                list(paths),
                max_files=max_files,
                follow_symlinks=follow_symlinks,
            )
            discovered_count = len(doc_files)
        else:
            doc_files, discover_warnings, discovered_count = discover_doc_files(
                folder_path,
                max_files=max_files,
                extra_ignore_patterns=extra_ignore_patterns,
                follow_symlinks=follow_symlinks,
                sort_by=sort_by,
            )
        warnings.extend(discover_warnings)

        if not doc_files:
            err: dict = {"success": False, "error": "No documentation files found"}
            if warnings:
                err["warnings"] = warnings
            return err

        repo_name = name if name else folder_path.name
        owner = "local"
        repo_id = f"{owner}/{repo_name}"
        initial_git_state = local_git_state(folder_path, scope_path=folder_path)
        store = DocStore(base_path=storage_path)
        existing_index = store.load_index(owner, repo_name)

        # Read all discovered files
        current_files: dict = {}
        for file_path in doc_files:
            if not validate_path(folder_path, file_path):
                continue
            try:
                rel_path = file_path.relative_to(folder_path).as_posix()
            except ValueError:
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                parsed_content = preprocess_content(content, rel_path)
                current_files[rel_path] = parsed_content
            except Exception as e:
                warnings.append(f"Failed to read {file_path}: {e}")

        final_git_state = local_git_state(folder_path, scope_path=folder_path)
        head_sha, source_dirty = stable_local_git_state(initial_git_state, final_git_state)

        # --- Incremental path ---
        if incremental and existing_index is not None:
            changed, new, deleted = store.detect_changes(owner, repo_name, current_files)

            if not changed and not new and not deleted:
                updated = existing_index
                if (
                    normalize_commit_sha(existing_index.head_sha) != head_sha
                    or bool(existing_index.source_dirty) != bool(source_dirty)
                    or getattr(existing_index, "source_root", "") != str(folder_path)
                ):
                    updated = store.incremental_save(
                        owner=owner, name=repo_name,
                        changed_files=[], new_files=[], deleted_files=[],
                        new_sections=[], raw_files={}, doc_types={},
                        head_sha=head_sha,
                        source_dirty=source_dirty,
                        source_root=str(folder_path),
                    ) or existing_index
                latency_ms = int((time.perf_counter() - t0) * 1000)
                nochange_result: dict = {
                    "success": True,
                    "message": "No changes detected",
                    "repo": f"{owner}/{repo_name}",
                    "folder_path": str(folder_path),
                    "incremental": True,
                    "changed": 0, "new": 0, "deleted": 0,
                    "_meta": {"latency_ms": latency_ms},
                }
                _add_commit_fields(nochange_result, updated)
                # jdoc#15: report truncation even when nothing changed,
                # since the visible-corpus boundary is unchanged.
                if discovered_count > max_files:
                    nochange_result["truncated"] = True
                    nochange_result["discovered"] = discovered_count
                    nochange_result["indexed"] = len(doc_files)
                else:
                    nochange_result["truncated"] = False
                return nochange_result

            files_to_parse = set(changed) | set(new)
            new_sections = []
            raw_subset: dict = {}
            doc_types: dict = {}

            for rel_path in files_to_parse:
                content = current_files[rel_path]
                raw_subset[rel_path] = content
                ext = rel_path.rsplit(".", 1)[-1].lower() if "." in rel_path else ""
                try:
                    sections = parse_file(content, rel_path, repo_id)
                    if sections:
                        new_sections.extend(sections)
                        doc_types[f".{ext}"] = doc_types.get(f".{ext}", 0) + 1
                except Exception as e:
                    warnings.append(f"Failed to parse {rel_path}: {e}")

            new_sections = summarize_sections(new_sections, use_ai=use_ai_summaries)
            _annotate_roles(new_sections)
            if use_embeddings:
                new_sections = embed_sections(
                    new_sections,
                    owner=owner, name=repo_name, storage_path=storage_path,
                )

            updated = store.incremental_save(
                owner=owner, name=repo_name,
                changed_files=changed, new_files=new, deleted_files=deleted,
                new_sections=new_sections, raw_files=raw_subset, doc_types=doc_types,
                head_sha=head_sha,
                source_dirty=source_dirty,
                source_root=str(folder_path),
            )

            latency_ms = int((time.perf_counter() - t0) * 1000)
            result = {
                "success": True,
                "repo": f"{owner}/{repo_name}",
                "folder_path": str(folder_path),
                "incremental": True,
                "changed": len(changed), "new": len(new), "deleted": len(deleted),
                "section_count": len(updated.sections) if updated else 0,
                "indexed_at": updated.indexed_at if updated else "",
                "semantic_search": use_embeddings and get_provider_name() is not None,
                "_meta": {"latency_ms": latency_ms},
            }
            _add_commit_fields(result, updated)
            # jdoc#15: surface truncation on the incremental path too.
            if discovered_count > max_files:
                result["truncated"] = True
                result["discovered"] = discovered_count
                result["indexed"] = len(doc_files)
                warnings.append(
                    f"max_files cap hit: indexed {len(doc_files)} of "
                    f"{discovered_count} discovered files. Raise max_files "
                    f"to capture the rest."
                )
            else:
                result["truncated"] = False
            if warnings:
                result["warnings"] = warnings
            return result

        # --- Full index path ---
        all_sections = []
        doc_types = {}
        raw_files: dict = {}
        parsed_files = []

        for rel_path, content in current_files.items():
            ext = f".{rel_path.rsplit('.', 1)[-1].lower()}" if "." in rel_path else ""
            try:
                sections = parse_file(content, rel_path, repo_id)
                if sections:
                    all_sections.extend(sections)
                    doc_types[ext] = doc_types.get(ext, 0) + 1
                    raw_files[rel_path] = content
                    parsed_files.append(rel_path)
            except Exception as e:
                warnings.append(f"Failed to parse {rel_path}: {e}")

        if not all_sections:
            return {"success": False, "error": "No sections extracted from files"}

        all_sections = summarize_sections(all_sections, use_ai=use_ai_summaries)
        _annotate_roles(all_sections)
        if use_embeddings:
            all_sections = embed_sections(
                all_sections,
                owner=owner, name=repo_name, storage_path=storage_path,
            )

        # v1.19.0: glossary sidecar built from final section content.
        try:
            entries = extract_glossary(all_sections)
            write_terms(storage_path, owner, repo_name, entries)
        except Exception:
            pass  # glossary is best-effort; never fail indexing

        # v1.24.0: related-graph adjacency list sidecar.
        try:
            from ..retrieval.related_persist import write as _write_related
            _write_related(storage_path, owner, repo_name, all_sections)
        except Exception:
            pass

        # v1.24.0: boilerplate detector sidecar.
        try:
            from ..retrieval.boilerplate import write as _write_boilerplate
            _write_boilerplate(storage_path, owner, repo_name, all_sections)
        except Exception:
            pass

        # v1.34.0: section near-duplicate detector sidecar.
        try:
            from ..retrieval.dedup import write as _write_dedup
            _write_dedup(storage_path, owner, repo_name,
                         [s.to_dict() | {"content": getattr(s, "content", "") or ""} for s in all_sections])
        except Exception:
            pass

        # v1.29.0: opt-in autotune. Runs the v1.23 weight tuner on this
        # repo's accumulated ranking events; no-op when telemetry isn't
        # enabled. Failures swallowed.
        autotune_result = None
        if autotune:
            try:
                from .tune_weights import tune_weights as _tune_weights
                autotune_result = _tune_weights(
                    repo=f"{owner}/{repo_name}",
                    storage_path=storage_path,
                )
            except Exception:
                autotune_result = None

        saved = store.save_index(
            owner=owner,
            name=repo_name,
            sections=all_sections,
            raw_files=raw_files,
            doc_types=doc_types,
            head_sha=head_sha,
            source_dirty=source_dirty,
            source_root=str(folder_path),
        )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        result = {
            "success": True,
            "repo": f"{owner}/{repo_name}",
            "folder_path": str(folder_path),
            "indexed_at": saved.indexed_at,
            "file_count": len(parsed_files),
            "section_count": len(all_sections),
            "doc_types": doc_types,
            "files": parsed_files[:20],
            "semantic_search": use_embeddings and get_provider_name() is not None,
            "_meta": {"latency_ms": latency_ms},
        }
        _add_commit_fields(result, saved)
        if autotune_result is not None:
            result["autotune"] = autotune_result

        # jdoc#15: surface truncation as structured top-level fields so
        # callers can detect it programmatically, not just from a free-text
        # note string. `truncated` is False when the corpus fit entirely
        # under the cap; True when the cap was hit. `discovered` is the
        # full match count (capped at max_files * safety ceiling).
        if discovered_count > max_files:
            result["truncated"] = True
            result["discovered"] = discovered_count
            result["indexed"] = len(doc_files)
            warnings.append(
                f"max_files cap hit: indexed {len(doc_files)} of "
                f"{discovered_count} discovered files. Raise max_files to "
                f"capture the rest."
            )
            result["note"] = (
                f"Folder has many files; indexed first {max_files} of "
                f"{discovered_count}. Raise max_files to include the rest."
            )
        else:
            result["truncated"] = False

        if warnings:
            result["warnings"] = warnings

        return result

    except Exception as e:
        return {"success": False, "error": f"Indexing failed: {str(e)}"}
