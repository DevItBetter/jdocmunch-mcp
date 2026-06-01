"""Claude Code hook handlers for jDocMunch enforcement.

PreToolUse  -- intercept Read on large doc files, suggest jDocMunch tools.
PostToolUse -- auto-reindex after Edit/Write on doc files to keep the index fresh.
PreCompact  -- emit a session snapshot so doc orientation survives context compaction.

All read JSON from stdin and write JSON to stdout per the Claude Code hooks spec.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Doc extensions that benefit from jDocMunch structured retrieval.
# Mirrors parser.ALL_EXTENSIONS.
_DOC_EXTENSIONS: set[str] = {
    ".md", ".markdown", ".mdx",
    ".txt",
    ".rst",
    ".adoc", ".asciidoc", ".asc",
    ".ipynb",
    ".html", ".htm",
    ".yaml", ".yml",
    ".json", ".jsonc",
    ".xml", ".svg", ".xhtml",
    ".tscn", ".tres",
}

# Minimum file size (bytes) to trigger the jDocMunch suggestion.
# Override with JDOCMUNCH_HOOK_MIN_SIZE env var.
_MIN_SIZE_BYTES = int(os.environ.get("JDOCMUNCH_HOOK_MIN_SIZE", "2048"))


def run_pretooluse() -> int:
    """PreToolUse hook: intercept Read calls on large doc files.

    Reads hook JSON from stdin.  If the target is a doc file above the
    size threshold, prints a stderr hint directing Claude to use
    jDocMunch tools instead.

    Small files, non-doc files, and unreadable paths are silently allowed.

    Returns exit code (always 0 -- errors are swallowed to avoid blocking).
    """
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    file_path: str = data.get("tool_input", {}).get("file_path", "")
    if not file_path:
        return 0

    _, ext = os.path.splitext(file_path)
    if ext.lower() not in _DOC_EXTENSIONS:
        return 0

    try:
        size = os.path.getsize(file_path)
    except OSError:
        return 0

    if size < _MIN_SIZE_BYTES:
        return 0

    # Targeted reads (offset/limit set) are likely pre-edit -- allow silently.
    tool_input = data.get("tool_input", {})
    if tool_input.get("offset") is not None or tool_input.get("limit") is not None:
        return 0

    # Full-file exploratory read on a large doc file -- warn but allow.
    # Hard deny breaks the Edit workflow (Claude Code requires Read before Edit).
    print(
        f"jDocMunch hint: this is a {size:,}-byte doc file. "
        "Prefer search_sections + get_section for exploration. "
        "Use Read only when you need exact line numbers for Edit.",
        file=sys.stderr,
    )
    return 0


def run_posttooluse() -> int:
    """PostToolUse hook: auto-reindex doc files after Edit/Write.

    Reads hook JSON from stdin, extracts the file path, and spawns
    ``jdocmunch-mcp index-local --path <dir>`` as a fire-and-forget
    background process to keep the index fresh.

    Non-doc files are skipped.  Errors are swallowed silently.

    Returns exit code (always 0).
    """
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    file_path: str = data.get("tool_input", {}).get("file_path", "")
    if not file_path:
        return 0

    _, ext = os.path.splitext(file_path)
    if ext.lower() not in _DOC_EXTENSIONS:
        return 0

    # Fire-and-forget: spawn index-file for the single edited file.
    resolved = str(Path(file_path).resolve())
    try:
        kwargs: dict = dict(
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        subprocess.Popen(
            ["jdocmunch-mcp", "index-file", resolved],
            **kwargs,
        )
    except (OSError, FileNotFoundError):
        pass  # jdocmunch-mcp not in PATH -- skip silently

    return 0


def run_precompact() -> int:
    """PreCompact hook: generate session snapshot before context compaction.

    Reads hook JSON from stdin. Builds a compact snapshot of the current
    doc session state and returns it as a systemMessage for context injection.

    Returns exit code (always 0 -- errors are swallowed to avoid blocking).
    """
    try:
        json.load(sys.stdin)  # Validate stdin is valid JSON
    except (json.JSONDecodeError, ValueError):
        return 0

    try:
        snapshot = _build_snapshot()
    except Exception:
        return 0

    if not snapshot:
        return 0

    result = {"systemMessage": snapshot}
    json.dump(result, sys.stdout)
    return 0


def _build_snapshot() -> str:
    """Build a compact session snapshot from indexed doc repos."""
    from ..tools.list_repos import list_repos

    repos_result = list_repos()
    repos = repos_result.get("repos", [])

    if not repos:
        return ""

    lines = ["## jDocMunch Session Snapshot", ""]
    lines.append(f"Indexed doc repos: {len(repos)}")
    for r in repos:
        name = r.get("repo_at_sha", r.get("name", r.get("repo", "?")))
        sections = r.get("section_count", r.get("sections", "?"))
        docs = r.get("doc_count", r.get("documents", "?"))
        source = r.get("source_root", r.get("source", ""))
        lines.append(f"- **{name}**: {docs} docs, {sections} sections ({source})")

    lines.append("")
    lines.append(
        "Use `search_sections` + `get_section` for doc navigation. "
        "Use `Read` only when you need exact line numbers for `Edit`."
    )
    return "\n".join(lines)
