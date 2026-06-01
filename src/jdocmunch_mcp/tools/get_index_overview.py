"""High-level repo snapshot — composition tool (v1.56.0).

`get_doc_health` is health-focused (broken links, orphans, freshness).
`list_docs` is doc-level inventory. `get_all_tags` and `get_all_roles`
are axis-by-axis discovery. Agents asking "what is this repo, briefly?"
had to call all four.

This tool fuses the *brief* slice of each into a single snapshot:

  - doc_count, section_count, total_byte_size
  - format breakdown (extension → counts)
  - top_tags (top-N by section_count)
  - top_roles (top-N by section_count)
  - indexed_at

Single-pass over `index.sections` + cached file stats. Pure additive;
underlying tools remain available for deep-dive use.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from ..storage import DocStore


def get_index_overview(
    repo: str,
    top_n: int = 5,
    storage_path: Optional[str] = None,
) -> dict:
    """Return a single-call repo snapshot.

    Args:
        repo: Repository identifier.
        top_n: How many top tags / top roles to surface (default 5).
            ``0`` omits both lists; full distributions remain available
            via `get_all_tags` / `get_all_roles`.
        storage_path: Custom storage path.

    Returns:
        ``{repo, doc_count, section_count, total_byte_size,
        format_breakdown:[{format, doc_count, section_count}, ...],
        top_tags:[{tag, section_count}, ...],
        top_roles:[{role, section_count}, ...],
        indexed_at, _meta}``.
    """
    t0 = time.perf_counter()
    if top_n < 0:
        return {"error": "top_n must be non-negative"}
    store = DocStore(base_path=storage_path)
    owner, name = store._resolve_repo(repo)
    index = store.load_index(owner, name)

    if not index:
        return {"error": f"Repo not found: {repo}"}

    # Per-doc + per-format counts.
    doc_section_counts: dict[str, int] = {}
    for sec in index.sections:
        dp = sec.get("doc_path") or ""
        if dp:
            doc_section_counts[dp] = doc_section_counts.get(dp, 0) + 1

    format_doc: dict[str, int] = {}
    format_sec: dict[str, int] = {}
    total_bytes = 0
    content_dir = store._content_dir(owner, name)
    for dp, n_secs in doc_section_counts.items():
        ext = (os.path.splitext(dp)[1] or "<none>").lower()
        format_doc[ext] = format_doc.get(ext, 0) + 1
        format_sec[ext] = format_sec.get(ext, 0) + n_secs
        try:
            cached = store._safe_content_path(content_dir, dp)
            if cached and cached.exists():
                total_bytes += cached.stat().st_size
        except OSError:
            pass
    format_breakdown = [
        {"format": fmt, "doc_count": format_doc[fmt],
         "section_count": format_sec[fmt]}
        for fmt in sorted(format_doc.keys())
    ]

    # Tag distribution.
    tag_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    for sec in index.sections:
        for t in (sec.get("tags") or []):
            if isinstance(t, str) and t.strip():
                key = t.strip().lower()
                tag_counts[key] = tag_counts.get(key, 0) + 1
        role = ((sec.get("metadata") or {}).get("role") or "").strip().lower()
        if role:
            role_counts[role] = role_counts.get(role, 0) + 1

    def _topn(counts: dict[str, int], key: str) -> list:
        if top_n == 0:
            return []
        rows = [{key: k, "section_count": v} for k, v in counts.items()]
        rows.sort(key=lambda r: (-r["section_count"], r[key]))
        return rows[:top_n]

    payload = {
        "repo": f"{owner}/{name}",
        "doc_count": len(doc_section_counts),
        "section_count": len(index.sections),
        "total_byte_size": total_bytes,
        "format_breakdown": format_breakdown,
        "top_tags": _topn(tag_counts, "tag"),
        "top_roles": _topn(role_counts, "role"),
        "indexed_at": index.indexed_at,
        "_meta": {
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "top_n": top_n,
        },
    }
    if index.head_sha:
        payload["head_sha"] = index.head_sha
    payload["source_dirty"] = bool(index.source_dirty)
    payload["sha_certified"] = bool(index.sha_certified)
    if index.repo_at_sha:
        payload["repo_at_sha"] = index.repo_at_sha
    return payload
