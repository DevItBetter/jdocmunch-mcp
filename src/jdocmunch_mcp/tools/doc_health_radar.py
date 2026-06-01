"""doc_health_radar — six-axis health snapshot for a doc index (v1.62.0).

State-snapshot tool: gathers signals already aggregated by
`get_doc_health` and feeds them through the pure-function
`health_radar.compute_radar` core. Pairs with `diff_doc_health_radar`
for snapshot deltas.

Mirrors jcm's `health_radar` (six axes + diff) and jData's
`data_health_radar` — third leg of the suite-wide radar pattern.
"""

from __future__ import annotations

import time
from typing import Optional

from .get_doc_health import get_doc_health
from .health_radar import compute_radar


def doc_health_radar(
    repo: str,
    storage_path: Optional[str] = None,
) -> dict:
    """Compute the six-axis doc-health radar for a repo.

    Args:
        repo: ``owner/name`` or bare name resolving via DocStore.
        storage_path: Custom doc-index root.

    Returns:
        ``{result: {repo, section_count, doc_count, radar}, _meta}`` on
        success. The ``radar`` payload is the standard
        ``{axes, composite, grade, omitted_axes}`` shape used across the
        suite. ``{error, ...}`` on refusal.
    """
    t0 = time.perf_counter()
    health = get_doc_health(repo=repo, storage_path=storage_path)
    if "error" in health:
        return health

    fresh_counts = health.get("freshness") or {}
    role_dist = health.get("role_distribution") or {}
    section_count = int(health.get("section_count") or 0)
    broken = int(health.get("broken_link_count") or 0)
    if broken < 0:
        broken = 0  # delegate failed; don't double-count as healthy
    orphans = int(health.get("orphan_section_count") or 0)
    if orphans < 0:
        orphans = 0
    embedded = int((health.get("embeddings") or {}).get("covered_sections") or 0)

    drift = health.get("drift") or {}
    has_canary = bool(drift.get("has_canary"))
    drift_alarm: Optional[bool] = drift.get("alarm")

    radar = compute_radar(
        fresh=int(fresh_counts.get("fresh") or 0),
        edited=int(fresh_counts.get("edited_uncommitted") or 0),
        stale=int(fresh_counts.get("stale_index") or 0),
        broken_links=broken,
        orphan_count=orphans,
        embedded_sections=embedded,
        section_count=section_count,
        role_distribution=role_dist,
        has_canary=has_canary,
        drift_alarm=drift_alarm,
    )

    result = {
        "repo": health.get("repo", repo),
        "section_count": section_count,
        "doc_count": int(health.get("doc_count") or 0),
        "radar": radar,
    }
    if health.get("repo_at_sha"):
        result["repo_at_sha"] = health["repo_at_sha"]

    return {
        "result": result,
        "_meta": {
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        },
    }
