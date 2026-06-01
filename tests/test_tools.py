"""Tests for tool functions."""

import os
import shutil
import stat
import subprocess

import pytest
from pathlib import Path

from jdocmunch_mcp.tools.index_local import _should_skip

from jdocmunch_mcp.tools.index_local import index_local
from jdocmunch_mcp.tools.index_file import index_file
from jdocmunch_mcp.tools.list_repos import list_repos
from jdocmunch_mcp.tools.delete_index import delete_index
from jdocmunch_mcp.tools.get_toc import get_toc
from jdocmunch_mcp.tools.get_toc_tree import get_toc_tree
from jdocmunch_mcp.tools.get_document_outline import get_document_outline
from jdocmunch_mcp.tools.search_sections import search_sections
from jdocmunch_mcp.tools.get_section import get_section
from jdocmunch_mcp.tools.get_sections import get_sections
from jdocmunch_mcp.tools.get_section_context import get_section_context
import jdocmunch_mcp.tools._git as git_helpers

FIXTURES = Path(__file__).parent / "fixtures"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()


def _rmtree_force(path: Path) -> None:
    """Remove trees containing read-only files, as Git can create on Windows."""
    def onerror(func, file_path, _exc_info):
        os.chmod(file_path, stat.S_IWRITE)
        func(file_path)

    shutil.rmtree(path, onerror=onerror)


@pytest.fixture
def indexed_repo(tmp_path):
    """Index the fixture docs folder and return the repo identifier."""
    result = index_local(
        path=str(FIXTURES / "docs"),
        use_ai_summaries=False,
        storage_path=str(tmp_path),
    )
    assert result["success"], f"Indexing failed: {result}"
    return result["repo"], str(tmp_path)


class TestShouldSkip:
    def test_skips_build(self):
        assert _should_skip("build/output.md") is True

    def test_skips_node_modules(self):
        assert _should_skip("node_modules/pkg/README.md") is True

    def test_does_not_skip_rebuild(self):
        """'rebuild/' should not be caught by the 'build/' pattern."""
        assert _should_skip("rebuild/output.md") is False

    def test_does_not_skip_normal_file(self):
        assert _should_skip("docs/guide.md") is False

    def test_skips_nested_git(self):
        assert _should_skip("submodule/.git/config") is True

    def test_does_not_skip_partial_match_in_filename(self):
        """'build_notes.md' has 'build' but no 'build/' component."""
        assert _should_skip("docs/build_notes.md") is False


class TestIndexLocal:
    def test_success(self, tmp_path):
        result = index_local(
            path=str(FIXTURES / "docs"),
            use_ai_summaries=False,
            storage_path=str(tmp_path),
        )
        assert result["success"] is True
        assert result["file_count"] >= 1
        assert result["section_count"] >= 1
        assert "_meta" in result

    def test_invalid_path(self, tmp_path):
        result = index_local(path="/nonexistent/path", storage_path=str(tmp_path))
        assert result["success"] is False
        assert "error" in result

    def test_not_a_dir(self, tmp_path):
        f = tmp_path / "file.md"
        f.write_text("# Hello")
        result = index_local(path=str(f), storage_path=str(tmp_path))
        assert result["success"] is False

    def test_includes_txt(self, tmp_path):
        result = index_local(
            path=str(FIXTURES / "text"),
            use_ai_summaries=False,
            storage_path=str(tmp_path),
        )
        assert result["success"] is True
        assert ".txt" in result["doc_types"]

    def test_clean_git_repo_emits_repo_at_sha_and_alias_works(self, tmp_path):
        repo_dir = tmp_path / "repo"
        store_dir = tmp_path / "store"
        repo_dir.mkdir()
        (repo_dir / "README.md").write_text("# Committed\n\nBody", encoding="utf-8")
        _git(repo_dir, "init")
        _git(repo_dir, "add", "README.md")
        _git(repo_dir, "-c", "user.name=Test", "-c", "user.email=t@example.com", "commit", "-m", "docs")
        sha = _git(repo_dir, "rev-parse", "HEAD")

        result = index_local(
            path=str(repo_dir),
            name="gitdocs",
            use_ai_summaries=False,
            use_embeddings=False,
            storage_path=str(store_dir),
        )

        assert result["success"] is True
        assert result["head_sha"] == sha
        assert result["source_dirty"] is False
        assert result["sha_certified"] is True
        assert result["repo_at_sha"] == f"local/gitdocs@{sha}"

        repos = list_repos(storage_path=str(store_dir))
        row = next(r for r in repos["repos"] if r["repo"] == "local/gitdocs")
        assert row["repo_at_sha"] == result["repo_at_sha"]
        assert row["source_dirty"] is False
        assert row["sha_certified"] is True

        search = search_sections(
            repo=result["repo_at_sha"],
            query="Committed",
            storage_path=str(store_dir),
        )
        assert "error" not in search
        assert search["repo_at_sha"] == result["repo_at_sha"]

    def test_dirty_git_repo_does_not_emit_repo_at_sha(self, tmp_path):
        repo_dir = tmp_path / "repo"
        store_dir = tmp_path / "store"
        repo_dir.mkdir()
        readme = repo_dir / "README.md"
        readme.write_text("# Clean\n\nBody", encoding="utf-8")
        _git(repo_dir, "init")
        _git(repo_dir, "add", "README.md")
        _git(repo_dir, "-c", "user.name=Test", "-c", "user.email=t@example.com", "commit", "-m", "docs")
        sha = _git(repo_dir, "rev-parse", "HEAD")
        readme.write_text("# Dirty\n\nChanged", encoding="utf-8")

        result = index_local(
            path=str(repo_dir),
            name="gitdocs",
            use_ai_summaries=False,
            use_embeddings=False,
            storage_path=str(store_dir),
        )

        assert result["success"] is True
        assert result["head_sha"] == sha
        assert result["source_dirty"] is True
        assert result["sha_certified"] is False
        assert "repo_at_sha" not in result

    def test_dirty_files_outside_indexed_subdir_do_not_block_repo_at_sha(self, tmp_path):
        repo_dir = tmp_path / "repo"
        docs_dir = repo_dir / "docs"
        src_dir = repo_dir / "src"
        store_dir = tmp_path / "store"
        docs_dir.mkdir(parents=True)
        src_dir.mkdir()
        (docs_dir / "README.md").write_text("# Docs\n\nCommitted", encoding="utf-8")
        app = src_dir / "app.py"
        app.write_text("print('clean')\n", encoding="utf-8")
        _git(repo_dir, "init")
        _git(repo_dir, "add", ".")
        _git(repo_dir, "-c", "user.name=Test", "-c", "user.email=t@example.com", "commit", "-m", "initial")
        sha = _git(repo_dir, "rev-parse", "HEAD")
        app.write_text("print('dirty')\n", encoding="utf-8")

        result = index_local(
            path=str(docs_dir),
            use_ai_summaries=False,
            use_embeddings=False,
            storage_path=str(store_dir),
        )

        assert result["success"] is True
        assert result["head_sha"] == sha
        assert result["source_dirty"] is False
        assert result["sha_certified"] is True
        assert result["repo_at_sha"] == f"local/docs@{sha}"

    def test_dirty_tracked_unindexed_file_in_root_does_not_block_repo_at_sha(self, tmp_path):
        repo_dir = tmp_path / "repo"
        store_dir = tmp_path / "store"
        repo_dir.mkdir()
        readme = repo_dir / "README.md"
        helper = repo_dir / "helper.py"
        readme.write_text("# Docs\n\nCommitted", encoding="utf-8")
        helper.write_text("print('clean')\n", encoding="utf-8")
        _git(repo_dir, "init")
        _git(repo_dir, "add", ".")
        _git(repo_dir, "-c", "user.name=Test", "-c", "user.email=t@example.com", "commit", "-m", "initial")
        sha = _git(repo_dir, "rev-parse", "HEAD")
        helper.write_text("print('dirty')\n", encoding="utf-8")

        result = index_local(
            path=str(repo_dir),
            use_ai_summaries=False,
            use_embeddings=False,
            storage_path=str(store_dir),
        )

        assert result["success"] is True
        assert result["head_sha"] == sha
        assert result["source_dirty"] is False
        assert result["sha_certified"] is True
        assert result["repo_at_sha"] == f"local/repo@{sha}"

    def test_untracked_file_hidden_by_git_config_blocks_repo_at_sha(self, tmp_path):
        repo_dir = tmp_path / "repo"
        store_dir = tmp_path / "store"
        repo_dir.mkdir()
        (repo_dir / "README.md").write_text("# Readme\n\nCommitted", encoding="utf-8")
        (repo_dir / "UNTRACKED.md").write_text("# Untracked\n\nNot committed", encoding="utf-8")
        _git(repo_dir, "init")
        _git(repo_dir, "config", "status.showUntrackedFiles", "no")
        _git(repo_dir, "add", "README.md")
        _git(repo_dir, "-c", "user.name=Test", "-c", "user.email=t@example.com", "commit", "-m", "docs")
        sha = _git(repo_dir, "rev-parse", "HEAD")

        result = index_local(
            path=str(repo_dir),
            use_ai_summaries=False,
            use_embeddings=False,
            storage_path=str(store_dir),
        )

        assert result["success"] is True
        assert result["head_sha"] == sha
        assert result["source_dirty"] is True
        assert result["sha_certified"] is False
        assert "repo_at_sha" not in result

    def test_explicit_ignored_untracked_file_blocks_repo_at_sha(self, tmp_path):
        repo_dir = tmp_path / "repo"
        store_dir = tmp_path / "store"
        repo_dir.mkdir()
        (repo_dir / ".gitignore").write_text("IGNORED.md\n", encoding="utf-8")
        (repo_dir / "README.md").write_text("# Readme\n\nCommitted", encoding="utf-8")
        (repo_dir / "IGNORED.md").write_text("# Ignored\n\nNot committed", encoding="utf-8")
        _git(repo_dir, "init")
        _git(repo_dir, "add", ".gitignore", "README.md")
        _git(repo_dir, "-c", "user.name=Test", "-c", "user.email=t@example.com", "commit", "-m", "docs")
        sha = _git(repo_dir, "rev-parse", "HEAD")

        result = index_local(
            path=str(repo_dir),
            paths=["IGNORED.md"],
            use_ai_summaries=False,
            use_embeddings=False,
            storage_path=str(store_dir),
        )

        assert result["success"] is True
        assert result["head_sha"] == sha
        assert result["source_dirty"] is True
        assert result["sha_certified"] is False
        assert "repo_at_sha" not in result

    def test_explicit_ignored_tracked_file_can_emit_repo_at_sha(self, tmp_path):
        repo_dir = tmp_path / "repo"
        store_dir = tmp_path / "store"
        repo_dir.mkdir()
        (repo_dir / ".gitignore").write_text("IGNORED.md\n", encoding="utf-8")
        (repo_dir / "IGNORED.md").write_text("# Ignored\n\nCommitted", encoding="utf-8")
        _git(repo_dir, "init")
        _git(repo_dir, "add", ".gitignore")
        _git(repo_dir, "add", "-f", "IGNORED.md")
        _git(repo_dir, "-c", "user.name=Test", "-c", "user.email=t@example.com", "commit", "-m", "docs")
        sha = _git(repo_dir, "rev-parse", "HEAD")

        result = index_local(
            path=str(repo_dir),
            paths=["IGNORED.md"],
            use_ai_summaries=False,
            use_embeddings=False,
            storage_path=str(store_dir),
        )

        assert result["success"] is True
        assert result["head_sha"] == sha
        assert result["source_dirty"] is False
        assert result["sha_certified"] is True
        assert result["repo_at_sha"] == f"local/repo@{sha}"

    def test_untracked_non_indexed_clutter_does_not_block_repo_at_sha(self, tmp_path):
        # A stray, unsupported, untracked file in the docs folder has no bearing
        # on whether the indexed corpus is reproducible at HEAD, so it must not
        # withhold the repo@sha handle (corpus-reproducible, not pristine-folder).
        repo_dir = tmp_path / "repo"
        store_dir = tmp_path / "store"
        repo_dir.mkdir()
        (repo_dir / "README.md").write_text("# Committed\n\nBody", encoding="utf-8")
        _git(repo_dir, "init")
        _git(repo_dir, "add", "README.md")
        _git(repo_dir, "-c", "user.name=Test", "-c", "user.email=t@example.com", "commit", "-m", "docs")
        sha = _git(repo_dir, "rev-parse", "HEAD")
        # Unsupported extension → discovered-but-not-indexed → untracked clutter.
        (repo_dir / "scratch.tmp").write_text("transient junk", encoding="utf-8")

        result = index_local(
            path=str(repo_dir),
            use_ai_summaries=False,
            use_embeddings=False,
            storage_path=str(store_dir),
        )

        assert result["success"] is True
        assert result["head_sha"] == sha
        assert result["source_dirty"] is False
        assert result["sha_certified"] is True
        assert result["repo_at_sha"] == f"local/repo@{sha}"

    def test_git_status_timeout_does_not_emit_repo_at_sha(self, tmp_path, monkeypatch):
        # A blocked `git status` (e.g. index.lock contention) must hit the
        # configured ceiling and fall to the safe "dirty / uncertified" side,
        # never an unbounded hang. This also guards the regression where the
        # subprocess timeout was dropped: production must pass it through.
        repo_dir = tmp_path / "repo"
        store_dir = tmp_path / "store"
        repo_dir.mkdir()
        (repo_dir / "README.md").write_text("# Clean\n\nBody", encoding="utf-8")
        _git(repo_dir, "init")
        _git(repo_dir, "add", "README.md")
        _git(repo_dir, "-c", "user.name=Test", "-c", "user.email=t@example.com", "commit", "-m", "docs")
        sha = _git(repo_dir, "rev-parse", "HEAD")

        monkeypatch.setenv("JDOCMUNCH_GIT_TIMEOUT", "5")
        real_run = subprocess.run

        def flaky_run(args, *pargs, **kwargs):
            if list(args[:2]) == ["git", "status"]:
                # Production must enforce a bounded timeout on every git call;
                # simulate the configured ceiling firing on a hung status.
                assert kwargs.get("timeout") == 5.0
                raise subprocess.TimeoutExpired(args, kwargs["timeout"])
            return real_run(args, *pargs, **kwargs)

        monkeypatch.setattr(git_helpers.subprocess, "run", flaky_run)

        result = index_local(
            path=str(repo_dir),
            use_ai_summaries=False,
            use_embeddings=False,
            storage_path=str(store_dir),
        )

        assert result["success"] is True
        assert result["head_sha"] == sha
        assert result["source_dirty"] is True
        assert result["sha_certified"] is False
        assert "repo_at_sha" not in result

    def test_git_timeout_resolves_default_override_and_disable(self, monkeypatch):
        monkeypatch.delenv("JDOCMUNCH_GIT_TIMEOUT", raising=False)
        assert git_helpers._git_timeout() == git_helpers.DEFAULT_GIT_TIMEOUT

        monkeypatch.setenv("JDOCMUNCH_GIT_TIMEOUT", "30")
        assert git_helpers._git_timeout() == 30.0

        # <= 0 explicitly disables the ceiling.
        monkeypatch.setenv("JDOCMUNCH_GIT_TIMEOUT", "0")
        assert git_helpers._git_timeout() is None
        monkeypatch.setenv("JDOCMUNCH_GIT_TIMEOUT", "-5")
        assert git_helpers._git_timeout() is None

        # Unparseable / blank values fall back to the bounded default.
        monkeypatch.setenv("JDOCMUNCH_GIT_TIMEOUT", "soon")
        assert git_helpers._git_timeout() == git_helpers.DEFAULT_GIT_TIMEOUT
        monkeypatch.setenv("JDOCMUNCH_GIT_TIMEOUT", "   ")
        assert git_helpers._git_timeout() == git_helpers.DEFAULT_GIT_TIMEOUT

    def test_sha_movement_during_local_index_is_dirty(self):
        head_sha, dirty = git_helpers.stable_local_git_state(
            ("a" * 40, False),
            ("b" * 40, False),
        )
        assert head_sha == "b" * 40
        assert dirty is True

    def test_index_file_marks_clean_repo_dirty_after_edit(self, tmp_path):
        repo_dir = tmp_path / "repo"
        store_dir = tmp_path / "store"
        repo_dir.mkdir()
        readme = repo_dir / "README.md"
        readme.write_text("# Clean\n\nBody", encoding="utf-8")
        _git(repo_dir, "init")
        _git(repo_dir, "add", "README.md")
        _git(repo_dir, "-c", "user.name=Test", "-c", "user.email=t@example.com", "commit", "-m", "docs")

        indexed = index_local(
            path=str(repo_dir),
            use_ai_summaries=False,
            use_embeddings=False,
            storage_path=str(store_dir),
        )
        assert "repo_at_sha" in indexed

        readme.write_text("# Dirty\n\nChanged", encoding="utf-8")
        updated = index_file(
            file_path=str(readme),
            use_ai_summaries=False,
            storage_path=str(store_dir),
        )

        assert updated["success"] is True
        assert updated["source_dirty"] is True
        assert updated["sha_certified"] is False
        assert "repo_at_sha" not in updated

        repos = list_repos(storage_path=str(store_dir))
        row = next(r for r in repos["repos"] if r["repo"] == "local/repo")
        assert row["source_dirty"] is True
        assert "repo_at_sha" not in row

    def test_index_file_marks_dirty_when_sibling_doc_has_uncommitted_change(self, tmp_path):
        repo_dir = tmp_path / "repo"
        store_dir = tmp_path / "store"
        repo_dir.mkdir()
        readme = repo_dir / "README.md"
        guide = repo_dir / "GUIDE.md"
        readme.write_text("# Readme\n\nClean", encoding="utf-8")
        guide.write_text("# Guide\n\nClean", encoding="utf-8")
        _git(repo_dir, "init")
        _git(repo_dir, "add", ".")
        _git(repo_dir, "-c", "user.name=Test", "-c", "user.email=t@example.com", "commit", "-m", "docs")

        indexed = index_local(
            path=str(repo_dir),
            use_ai_summaries=False,
            use_embeddings=False,
            storage_path=str(store_dir),
        )
        assert "repo_at_sha" in indexed

        guide.write_text("# Guide\n\nDirty", encoding="utf-8")
        updated = index_file(
            file_path=str(readme),
            use_ai_summaries=False,
            storage_path=str(store_dir),
        )

        assert updated["success"] is True
        assert updated["source_dirty"] is True
        assert updated["sha_certified"] is False
        assert "repo_at_sha" not in updated

        repos = list_repos(storage_path=str(store_dir))
        row = next(r for r in repos["repos"] if r["repo"] == "local/repo")
        assert row["source_dirty"] is True
        assert "repo_at_sha" not in row

    def test_index_file_marks_dirty_when_head_moves_for_sibling_doc(self, tmp_path):
        repo_dir = tmp_path / "repo"
        store_dir = tmp_path / "store"
        repo_dir.mkdir()
        readme = repo_dir / "README.md"
        guide = repo_dir / "GUIDE.md"
        readme.write_text("# Readme\n\nClean", encoding="utf-8")
        guide.write_text("# Guide\n\nVersion one", encoding="utf-8")
        _git(repo_dir, "init")
        _git(repo_dir, "add", ".")
        _git(repo_dir, "-c", "user.name=Test", "-c", "user.email=t@example.com", "commit", "-m", "v1")

        indexed = index_local(
            path=str(repo_dir),
            use_ai_summaries=False,
            use_embeddings=False,
            storage_path=str(store_dir),
        )
        assert "repo_at_sha" in indexed

        guide.write_text("# Guide\n\nVersion two", encoding="utf-8")
        _git(repo_dir, "add", "GUIDE.md")
        _git(repo_dir, "-c", "user.name=Test", "-c", "user.email=t@example.com", "commit", "-m", "v2")
        new_sha = _git(repo_dir, "rev-parse", "HEAD")
        updated = index_file(
            file_path=str(readme),
            use_ai_summaries=False,
            storage_path=str(store_dir),
        )

        assert updated["success"] is True
        assert updated["head_sha"] == new_sha
        assert updated["source_dirty"] is True
        assert updated["sha_certified"] is False
        assert "repo_at_sha" not in updated

        repos = list_repos(storage_path=str(store_dir))
        row = next(r for r in repos["repos"] if r["repo"] == "local/repo")
        assert row["head_sha"] == new_sha
        assert row["source_dirty"] is True
        assert "repo_at_sha" not in row

    def test_index_file_preserves_existing_dirty_state_after_worktree_clean(self, tmp_path):
        repo_dir = tmp_path / "repo"
        store_dir = tmp_path / "store"
        repo_dir.mkdir()
        readme = repo_dir / "README.md"
        guide = repo_dir / "GUIDE.md"
        readme.write_text("# Readme\n\nClean", encoding="utf-8")
        guide.write_text("# Guide\n\nClean", encoding="utf-8")
        _git(repo_dir, "init")
        _git(repo_dir, "add", ".")
        _git(repo_dir, "-c", "user.name=Test", "-c", "user.email=t@example.com", "commit", "-m", "docs")

        indexed = index_local(
            path=str(repo_dir),
            use_ai_summaries=False,
            use_embeddings=False,
            storage_path=str(store_dir),
        )
        assert "repo_at_sha" in indexed

        guide.write_text("# Guide\n\nDirty", encoding="utf-8")
        dirty_update = index_file(
            file_path=str(guide),
            use_ai_summaries=False,
            storage_path=str(store_dir),
        )
        assert dirty_update["source_dirty"] is True
        _git(repo_dir, "checkout", "--", "GUIDE.md")

        clean_file_update = index_file(
            file_path=str(readme),
            use_ai_summaries=False,
            storage_path=str(store_dir),
        )

        assert clean_file_update["success"] is True
        assert clean_file_update["source_dirty"] is True
        assert clean_file_update["sha_certified"] is False
        assert "repo_at_sha" not in clean_file_update

        repos = list_repos(storage_path=str(store_dir))
        row = next(r for r in repos["repos"] if r["repo"] == "local/repo")
        assert row["source_dirty"] is True
        assert "repo_at_sha" not in row

    def test_index_file_marks_dirty_when_git_context_disappears(self, tmp_path):
        repo_dir = tmp_path / "repo"
        store_dir = tmp_path / "store"
        repo_dir.mkdir()
        readme = repo_dir / "README.md"
        readme.write_text("# Clean\n\nBody", encoding="utf-8")
        _git(repo_dir, "init")
        _git(repo_dir, "add", "README.md")
        _git(repo_dir, "-c", "user.name=Test", "-c", "user.email=t@example.com", "commit", "-m", "docs")

        indexed = index_local(
            path=str(repo_dir),
            use_ai_summaries=False,
            use_embeddings=False,
            storage_path=str(store_dir),
        )
        assert "repo_at_sha" in indexed

        _rmtree_force(repo_dir / ".git")
        readme.write_text("# Changed\n\nOutside Git", encoding="utf-8")
        updated = index_file(
            file_path=str(readme),
            use_ai_summaries=False,
            storage_path=str(store_dir),
        )

        assert updated["success"] is True
        assert updated["head_sha"] == indexed["head_sha"]
        assert updated["source_dirty"] is True
        assert updated["sha_certified"] is False
        assert "repo_at_sha" not in updated

        repos = list_repos(storage_path=str(store_dir))
        row = next(r for r in repos["repos"] if r["repo"] == "local/repo")
        assert row["source_dirty"] is True
        assert "repo_at_sha" not in row


class TestListRepos:
    def test_lists_indexed(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        result = list_repos(storage_path=storage_path)
        assert result["count"] >= 1
        assert any(r["repo"] == repo_id for r in result["repos"])

    def test_empty(self, tmp_path):
        result = list_repos(storage_path=str(tmp_path))
        assert result["count"] == 0


class TestDeleteIndex:
    def test_deletes(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        result = delete_index(repo=repo_id, storage_path=storage_path)
        assert result["success"] is True
        # Should be gone
        after = list_repos(storage_path=storage_path)
        assert all(r["repo"] != repo_id for r in after["repos"])

    def test_nonexistent(self, tmp_path):
        result = delete_index(repo="nobody/nothing", storage_path=str(tmp_path))
        assert result["success"] is False


class TestGetToc:
    def test_returns_sections(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        result = get_toc(repo=repo_id, storage_path=storage_path)
        assert "sections" in result
        assert result["section_count"] >= 1
        assert "_meta" in result

    def test_no_content_field(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        result = get_toc(repo=repo_id, storage_path=storage_path)
        for sec in result["sections"]:
            assert "content" not in sec

    def test_sorted_by_doc_and_offset(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        result = get_toc(repo=repo_id, storage_path=storage_path)
        secs = result["sections"]
        for i in range(1, len(secs)):
            prev = (secs[i - 1]["doc_path"], secs[i - 1]["byte_start"])
            curr = (secs[i]["doc_path"], secs[i]["byte_start"])
            assert prev <= curr


class TestGetTocTree:
    def test_returns_documents(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        result = get_toc_tree(repo=repo_id, storage_path=storage_path)
        assert "documents" in result
        assert result["doc_count"] >= 1

    def test_nested_structure(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        result = get_toc_tree(repo=repo_id, storage_path=storage_path)
        # Each document has sections with potential children
        for doc in result["documents"]:
            assert "doc_path" in doc
            assert "sections" in doc


class TestGetDocumentOutline:
    def test_specific_doc(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        result = get_document_outline(
            repo=repo_id,
            doc_path="sample.md",
            storage_path=storage_path,
        )
        assert "sections" in result
        assert result["section_count"] >= 1

    def test_not_found(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        result = get_document_outline(
            repo=repo_id,
            doc_path="nonexistent.md",
            storage_path=storage_path,
        )
        assert "error" in result


class TestSearchSections:
    def test_basic_search(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        result = search_sections(
            repo=repo_id,
            query="installation",
            storage_path=storage_path,
        )
        assert "results" in result
        assert "_meta" in result

    def test_no_content_in_results(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        result = search_sections(
            repo=repo_id,
            query="section",
            storage_path=storage_path,
        )
        for r in result["results"]:
            assert "content" not in r

    def test_max_results(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        result = search_sections(
            repo=repo_id,
            query="the",
            max_results=2,
            storage_path=storage_path,
        )
        assert len(result["results"]) <= 2

    def test_tokens_saved_reported(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        result = search_sections(
            repo=repo_id,
            query="install",
            storage_path=storage_path,
        )
        assert "tokens_saved" in result["_meta"]


class TestGetSection:
    def test_get_content(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        toc = get_toc(repo=repo_id, storage_path=storage_path)
        first_id = toc["sections"][0]["id"]

        result = get_section(
            repo=repo_id,
            section_id=first_id,
            storage_path=storage_path,
        )
        assert "section" in result
        assert "content" in result["section"]

    def test_verify_hash(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        toc = get_toc(repo=repo_id, storage_path=storage_path)
        first_id = toc["sections"][0]["id"]

        result = get_section(
            repo=repo_id,
            section_id=first_id,
            verify=True,
            storage_path=storage_path,
        )
        assert "section" in result
        # hash_verified may be True or None (if no hash stored)
        assert "hash_verified" in result["section"]

    def test_not_found(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        result = get_section(
            repo=repo_id,
            section_id="nobody::nowhere::nothing#1",
            storage_path=storage_path,
        )
        assert "error" in result


class TestGetSections:
    def test_batch_retrieval(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        toc = get_toc(repo=repo_id, storage_path=storage_path)
        ids = [s["id"] for s in toc["sections"][:3]]

        result = get_sections(
            repo=repo_id,
            section_ids=ids,
            storage_path=storage_path,
        )
        assert "sections" in result
        assert result["section_count"] == len(ids)

    def test_meta_complete(self, indexed_repo):
        """_meta must include tokens_saved, total_tokens_saved, and cost_avoided."""
        repo_id, storage_path = indexed_repo
        toc = get_toc(repo=repo_id, storage_path=storage_path)
        ids = [s["id"] for s in toc["sections"][:2]]

        result = get_sections(repo=repo_id, section_ids=ids, storage_path=storage_path)
        meta = result["_meta"]
        assert "tokens_saved" in meta
        assert "total_tokens_saved" in meta
        assert "cost_avoided" in meta
        assert "total_cost_avoided" in meta

    def test_invalid_section_id_returns_error(self, indexed_repo):
        """Invalid section IDs should produce per-item error dicts, not crash."""
        repo_id, storage_path = indexed_repo
        result = get_sections(
            repo=repo_id,
            section_ids=["invalid::id::nope#9"],
            storage_path=storage_path,
        )
        assert result["section_count"] == 1
        assert "error" in result["sections"][0]


class TestGetSectionContext:
    def _find_section_by_title(self, toc, title, doc_path=None):
        """Return the first section whose title matches (case-insensitive)."""
        title_lower = title.lower()
        for s in toc["sections"]:
            if s["title"].lower() == title_lower:
                if doc_path is None or s.get("doc_path", "").endswith(doc_path):
                    return s["id"]
        return None

    def test_returns_ancestors_and_section(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        toc = get_toc(repo=repo_id, storage_path=storage_path)

        # "Prerequisites" is a level-3 section under Installation → Sample Documentation
        prereq_id = self._find_section_by_title(toc, "Prerequisites", doc_path="sample.md")
        assert prereq_id, "Prerequisites section not found in fixture"

        result = get_section_context(
            repo=repo_id,
            section_id=prereq_id,
            storage_path=storage_path,
        )

        assert "error" not in result
        assert "section" in result
        assert "ancestors" in result
        assert result["section"]["id"] == prereq_id
        assert isinstance(result["section"]["content"], str)
        assert len(result["section"]["content"]) > 0
        # Should have at least one ancestor (Installation or Sample Documentation)
        assert len(result["ancestors"]) >= 1
        ancestor_titles = [a["title"] for a in result["ancestors"]]
        assert any("installation" in t.lower() or "sample" in t.lower() for t in ancestor_titles)

    def test_ancestors_ordered_root_first(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        toc = get_toc(repo=repo_id, storage_path=storage_path)

        # "Advanced Configuration" is level-4: Sample Doc > Usage > Configuration > Advanced Configuration
        adv_id = self._find_section_by_title(toc, "Advanced Configuration", doc_path="sample.md")
        assert adv_id, "Advanced Configuration section not found in fixture"

        result = get_section_context(repo=repo_id, section_id=adv_id, storage_path=storage_path)
        assert "error" not in result

        ancestors = result["ancestors"]
        assert len(ancestors) >= 2
        # Ancestors should be ordered root-first (ascending levels)
        levels = [a["level"] for a in ancestors]
        assert levels == sorted(levels)

    def test_children_included_by_default(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        toc = get_toc(repo=repo_id, storage_path=storage_path)

        # "Installation" in sample.md has children: Prerequisites, Quick Start
        install_id = self._find_section_by_title(toc, "Installation", doc_path="sample.md")
        assert install_id

        result = get_section_context(repo=repo_id, section_id=install_id, storage_path=storage_path)
        assert "error" not in result
        assert len(result["children"]) >= 2
        child_titles = [c["title"] for c in result["children"]]
        assert any("prerequisites" in t.lower() for t in child_titles)
        assert any("quick start" in t.lower() for t in child_titles)

    def test_include_children_false(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        toc = get_toc(repo=repo_id, storage_path=storage_path)

        install_id = self._find_section_by_title(toc, "Installation", doc_path="sample.md")
        assert install_id

        result = get_section_context(
            repo=repo_id, section_id=install_id,
            include_children=False, storage_path=storage_path,
        )
        assert "error" not in result
        assert result["children"] == []

    def test_max_tokens_truncates_content(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        toc = get_toc(repo=repo_id, storage_path=storage_path)

        any_id = toc["sections"][0]["id"]
        result = get_section_context(
            repo=repo_id, section_id=any_id,
            max_tokens=1,  # 4 bytes — will truncate anything non-trivial
            storage_path=storage_path,
        )
        assert "error" not in result
        # Either truncated flag is set, or content is very short
        sec = result["section"]
        assert sec.get("content_truncated") is True or len(sec["content"].encode()) <= 4

    def test_meta_fields_present(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        toc = get_toc(repo=repo_id, storage_path=storage_path)
        any_id = toc["sections"][0]["id"]

        result = get_section_context(repo=repo_id, section_id=any_id, storage_path=storage_path)
        meta = result["_meta"]
        assert "latency_ms" in meta
        assert "ancestor_count" in meta
        assert "child_count" in meta
        assert "tokens_saved" in meta

    def test_invalid_repo(self, tmp_path):
        result = get_section_context(
            repo="nonexistent/repo",
            section_id="whatever",
            storage_path=str(tmp_path),
        )
        assert "error" in result

    def test_invalid_section_id(self, indexed_repo):
        repo_id, storage_path = indexed_repo
        result = get_section_context(
            repo=repo_id,
            section_id="nobody::nowhere::nothing#99",
            storage_path=storage_path,
        )
        assert "error" in result
