# jdocmunch-mcp

**Version:** 1.68.0 | **Tests:** `pytest tests/ -q` (1293 passed)

## v1.68.0 - `doc_index_repo` name override for named GitHub doc indexes (PR #25, closes #24)
Contributed by @DevItBetter; follow-up to #17 and #23. Adds an optional
`name` arg to `doc_index_repo`/`index_repo` so a GitHub doc index can be
stored under a caller-chosen safe handle (`owner/name`) while preserving
the upstream GitHub identity. Without `name`, behavior is unchanged. With
it, the stored index is `owner/name`, the upstream source is persisted on
a new `DocIndex.source_repo` field, and indexing responses + `list_repos`
surface `source_repo` (and `source_repo_at_sha` for certified indexes)
alongside the stored `repo` / `repo_at_sha`.

`name` is a storage-name override, not a moving alias: it must be a single
safe component (`[A-Za-z0-9._-]+`; `/`, `\`, `@`, `.`, `..`, empty, and
non-strings rejected before any network fetch). The SHA fast-path and
incremental paths stay keyed to the stored name while fetching from the
upstream source, and both now guard on `existing_source_repo ==
source_repo_id` so reusing one stored name for a different upstream source
can't serve stale content via the fast path. Legacy indexes (empty
`source_repo`) fall back to `repo` and get backfilled on next touch.

Fully additive: `INDEX_VERSION` stays 3 (new field uses `.get` default on
load + omit-when-empty on write). Out of scope (deliberately): branch/tag/
ref selection, moving aliases, multi-alias, snapshot retention.

## v1.67.0 - certified repo@sha handles for citeable doc snapshots (PR #23, closes #22)
Contributed by @DevItBetter; follow-up to #17. Adds an immutable
`owner/repo@40hexsha` handle so downstream workflows can cite the exact
doc snapshot used for retrieval. New `DocIndex` fields `head_sha`,
`source_dirty`, `sha_certified`, `source_root`; derived `repo_at_sha`
property (never stored, emitted only when SHA is 40-hex AND not dirty
AND certified). Surfaced in `list_repos`, `search_sections`,
`get_doc_health`, `get_index_overview`, `doc_health_radar`, session
snapshot. `search_sections` (and every read tool) accepts a strict
`repo@sha` alias that resolves only when the stored index matches that
commit and is certified clean; a miss resolves to an uncreatable
sentinel name so it cannot collide with a real repo.

Certification rules (conservative by design):
- GitHub: `index_repo` resolves HEAD->SHA once, then fetches tree,
  `.gitignore`, and all content pinned at that SHA (`?ref=<sha>`),
  closing the old per-call HEAD drift window. Certified only when the
  SHA actually resolved.
- Local Git: "clean" means the *indexed corpus* is reproducible at
  HEAD, not that the worktree is pristine. Dirty files outside the
  indexed scope don't block; gitignored-but-explicitly-indexed paths
  do (separate `git ls-files` tracked-ness check). New `tools/_git.py`
  helpers; git probes bounded by `JDOCMUNCH_GIT_TIMEOUT` (default 10s,
  `<=0` disables); a timed-out/failed probe falls to dirty so an
  immutable handle is never emitted for an unknown state.
- `index_file` is sticky: surgical updates never upgrade a dirty,
  moved-HEAD, untracked-path, or no-longer-Git-backed index to
  certified. Run `index_local` to recertify the full corpus.

Fully additive: `INDEX_VERSION` stays 3 (new fields use `.get` defaults
on load + omit-when-empty on write, so old<->new indexes round-trip).
`load_index`/`delete_index` now catch `ValueError` from the sentinel
name. Suite-parity follow-up candidate for jcm/jdata if citeable
handles earn their keep.

## v1.66.3 - openai-compatible: probe actual dim at init (jdoc#20)
Hardens v1.66.0's openai-compatible provider. Backing-model swap
behind the same URL/model env vars (e.g. Ollama retagging) used to
silently mix vectors of different dims in the on-disk cache because
`_provider_identity` returned `dim=None`. Fix:
`_OpenAICompatibleProvider.__init__` now probes the endpoint with a
one-token canary and stores the discovered dim on `self.dim`;
`_provider_identity` reads it from the cached singleton. Cache layer's
strict dim check now engages; a model swap forces clean re-embed.
Probe failure is non-fatal (dim falls back to None → wildcard
behavior of v1.66.0). When jcm/jdata pick up openai-compatible (#302,
jdata#2), this pattern should ship there too.

## v1.66.2 - warm sentence-transformers before stdio (jdoc#19)
Reported by @rknighton. First semantic `search_sections` hung when
sentence-transformers was configured: lazy model load exceeded MCP
timeouts and leaked progress chatter to stdout, corrupting JSON-RPC
framing. Fix: `provider.warmup()` (provider-type-gated; only
sentence-transformers needs warming) runs in `run_server()` before
`stdio_server()` takes over, wrapped in `redirect_stdout(sys.stderr)`
so noisy library writes land safely. Network providers skip warmup
to avoid avoidable startup round-trips. Warmup failure is non-fatal.

## v1.66.1 - `should_embed("false")` parses as False (jdoc#18)
Reported by @rknighton. `should_embed()` ran any non-empty string
through `bool()`, so `use_embeddings="false"` enabled embeddings.
MCP tool inputs hit this whenever a client sent the flag as a JSON
string. Fix: recognise common string booleans before the bool()
fallback (`"true"/"false"`, `"1"/"0"`, `"yes"/"no"`, `"on"/"off"`,
`"t"/"f"`, `"y"/"n"`, case-insensitive, whitespace-trimmed).
Unknown strings still fall through to `bool()` so 1.x compat holds
(`"flase"` typo remains truthy as it did before).

## v1.66.0 - openai-compatible embeddings (PR #17)
Opt-in `openai-compatible` embedding provider for Ollama, vLLM,
LiteLLM, llama.cpp, LM Studio, and any other OpenAI-API-shaped
endpoint. Four env vars: `JDOCMUNCH_OPENAI_COMPAT_{URL,MODEL,API_KEY,
BATCH_SIZE}`. Explicit-only activation (never auto-detected).
Credential isolation: default API key is the literal `"local"`, never
falls through to `OPENAI_API_KEY`. Cache signature includes URL +
model + first-8 of compat key + batch size; ambient `OPENAI_API_KEY`
is excluded. Contributed by @DevItBetter. Follow-ups: jdoc#20
(canary-pin actual dim), jcm#302 + jdata#2 (sibling parity).

## v1.63.3 - `jdocmunch_guide` sibling-parity tool
Adds `jdocmunch_guide` -- doc-MCP sibling of `jcodemunch_guide` (jcm since
v1.84.0). Returns the version-current CLAUDE.md / AGENT.md policy snippet
for jdocmunch-mcp so an agent can keep a one-line CLAUDE.md
(`"Call jdocmunch_guide and strictly follow its instructions."`) instead
of pasting a static block that drifts. Tool count 59 -> 60. Companion
release of jdatamunch-mcp v1.12.2 ships `jdatamunch_guide` on the same
shape. Backstory: issue #296 (Codex compatibility report by @rknighton).

## v1.63.2 - drift-proof __version__ via importlib.metadata
`__version__` is now derived from `importlib.metadata.version("jdocmunch-mcp")`
in `__init__.py`, mirroring jcodemunch-mcp's pattern. pyproject.toml is
the single source of truth; the hardcoded literal can no longer drift.
v1.63.1's `tests/test_version_sync.py` regex guard retired as redundant.
Source-checkout callers without pip install see `__version__ = "unknown"`;
the replay runner's `_resolve_version()` already handles this via pyproject
fallback.

## v1.63.1 - CI green (fixture query rename + full-history checkout)
Patch release. Two independent CI fixes, no installed-user behavior change.
(1) `self_v1_11_0` fixture query renamed `wiki stats` -> `wiki benchmark`:
the old query collided with `### Stats` subheadings added to CHANGELOG.md
in v1.62.0/v1.63.0 and demoted the target wiki-benchmark page from
rank 1 to rank 4 (MRR 1.0 -> 0.925). (2) Both CI workflows now set
`fetch-depth: 0` so `scripts/generate_changelog.py` can walk every
`release:` commit; the shallow default broke `test_runs_against_real_repo`
on any push whose HEAD wasn't a `release:` commit.

## v1.63.0 — `get_doc_pr_risk_profile` (Phase-2 sibling-parity COMPLETE)
Composite doc-PR risk: volume + blast_radius + backlink_burden + tutorial_disruption
+ role_weight → 0-1 risk_score + level (low/medium/high/critical) + top-5 blockers +
recommended_action. Caller supplies the change list (paired with git diff or
get_recent_changes). Mirrors jcm's get_pr_risk_profile. Tool count 58 → 59.

## v1.62.0 — `doc_health_radar` + `diff_doc_health_radar` (Phase-2 sibling-parity)
Six-axis 0-100 doc-index health scorecard: freshness, link_integrity, orphan_health,
embedding_coverage, role_coverage, drift_health (omitted when no canary). Pure-function
diff helper alongside. Third leg of the suite-wide radar pattern (jcm + jData).
Tool count 56 → 58.

## v1.61.0 — explicit-paths indexing
`index_local(paths=[...])` skips the directory walk and indexes exactly the
listed files / subdirs. Each entry is validated against the root the same
way walked files are (path-traversal, symlink-escape, unsupported-extension
all warn-and-skip). CLI: `jdocmunch-mcp index-local --path <dir> --paths-from FILE`
(use `-` for stdin) — composes with `find` / `fd` / `rg`. Additive: omitting
`paths` preserves every existing call shape. Helper `_load_paths_from_arg`
in `server.py` parses the file/stdin (strips blanks + `#` comments).

## Purpose
Documentation section indexing for the jMunch suite. Companion to jcodemunch-mcp (which owns code symbols). Do NOT add code/docstring parsing here.

## Supported Formats
`.md/.mdx`, `.rst`, `.adoc`, `.ipynb`, `.html`, `.txt`, `.yaml/.yml` (OpenAPI only), `.json/.jsonc`, `.xml/.svg/.xhtml`, `.tscn/.tres` (Godot scenes/resources)

## Key Modules
- `storage/doc_store.py` — DocIndex, DocStore, detect_changes, incremental_save
- `parser/` — one file per format (markdown, rst, asciidoc, notebook, html, text, openapi, json, xml)
- `tools/` — index_local, index_repo, index_file, get_toc, get_toc_tree, search_sections, get_section, get_sections, list_repos, delete_index, get_broken_links, get_doc_coverage, get_backlinks, get_stale_pages, get_wiki_stats, check_section_delete_safe, get_section_blast_radius, find_similar_sections
- `cli/hooks.py` — PreToolUse (Read interceptor) + PostToolUse (auto-reindex) + PreCompact (session snapshot) hook handlers for Claude Code
- `cli/init.py` — `jdocmunch-mcp init` full onboarding: client detection, config patching, CLAUDE.md policy, Cursor/Windsurf rules, hooks, index; `claude-md` subcommand
- `embeddings/` — provider.py (Gemini + OpenAI), cosine_similarity, embed_sections, embed_query

## CLI Subcommands
| Subcommand | Purpose |
|------------|---------|
| `serve` (default) | Run the MCP server (stdio) |
| `init` | One-command onboarding: detect clients, write config, install policy, hooks, index |
| `claude-md` | Print or install the Doc Exploration Policy (`--install global\|project`) |
| `index-local --path <dir>` | Index a local folder (CLI, no MCP session needed) |
| `index-file <path>` | Re-index a single file within an existing index |
| `hook-pretooluse` | PreToolUse hook: intercept Read on large doc files (reads stdin) |
| `hook-posttooluse` | PostToolUse hook: auto-reindex doc files after Edit/Write (reads stdin) |
| `hook-precompact` | PreCompact hook: session snapshot before context compaction (reads stdin) |

## 1.x compatibility contract (license-binding)

Existing 1.x licensees must be able to upgrade between any two 1.x versions
with zero surprise. This is a hard constraint, not a guideline.

**Never on 1.x:**
- Remove or rename an MCP tool. Aliases for any rename must stay in place forever.
- Remove a `Section` field from `to_dict` output (additive only; new fields use the "omit when empty" convention).
- Drop a runtime dependency that an existing user might rely on (e.g. tiktoken stays optional; bytes/4 fallback stays).
- Force a reindex without auto-migrating on load. `INDEX_VERSION` bumps are allowed when the loader silently migrates v(N-1) → v(N) on first read.
- Change the JSON wire format of any tool response in a way that breaks an existing consumer. New keys are fine; renames + removals are not.
- Make a previously-default behavior raise. If we deprecate a flag value, keep it accepted (with a deprecation note in `_meta`) until a 2.x is approved.

**Acceptable on 1.x:**
- Add new tools, fields, response keys, env vars, kwargs (all defaulted to backwards-compat values).
- Tighten internal behavior (faster algorithms, better defaults) when no public output changes.
- Add new error returns for inputs that previously errored differently.
- Add new opt-in code paths gated by env var or kwarg.

**Reserved for 2.x (won't ship until a major-version license revision is planned):**
- See `todo.md` § "Reserved for 2.x" for the canonical list.

## Architecture
- INDEX_VERSION=3; version mismatch triggers auto-migration on first load (NEVER a forced reindex on 1.x)
- O(1) section lookup via `DocIndex.__post_init__` id dict
- `pyyaml>=6.0` required (hard dep)
- Hybrid search (v1.9.0): `search_sections` fuses BM25 + semantic cosine when embeddings exist. `use_embeddings` defaults to `"auto"` (embed when provider configured). `search_sections` params: `semantic` (None/auto, True, False), `semantic_only`, `semantic_weight` (0.0–1.0, default 0.5). `_meta.search_mode` reports `hybrid`/`semantic_only`/`lexical`.
- Embedding providers: GOOGLE_API_KEY (Gemini, text-embedding-004), OPENAI_API_KEY (text-embedding-3-small), openai-compatible + JDOCMUNCH_OPENAI_COMPAT_URL + JDOCMUNCH_OPENAI_COMPAT_MODEL, or sentence-transformers; override with JDOCMUNCH_EMBEDDING_PROVIDER env var
- Summarizer providers: ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, MINIMAX_API_KEY, ZHIPUAI_API_KEY; override with JDOCMUNCH_SUMMARIZER_PROVIDER env var (values: anthropic, gemini, openai, minimax, glm, none)
