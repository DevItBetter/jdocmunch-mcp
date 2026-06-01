"""Commit-SHA handling for GitHub indexing."""

import importlib

import pytest


@pytest.mark.asyncio
async def test_index_repo_fetches_tree_and_content_at_resolved_sha(tmp_path, monkeypatch):
    mod = importlib.import_module("jdocmunch_mcp.tools.index_repo")
    sha = "c" * 40
    refs = []

    async def fake_head(owner, repo, token=None, client=None, ref="HEAD"):
        assert (owner, repo, ref) == ("octo", "docs", "HEAD")
        return sha

    async def fake_tree(owner, repo, token=None, client=None, ref="HEAD"):
        refs.append(("tree", ref))
        return [{"type": "blob", "path": "README.md", "size": 64}]

    async def fake_gitignore(owner, repo, token=None, client=None, ref="HEAD"):
        refs.append(("gitignore", ref))
        return None

    async def fake_content(owner, repo, path, token=None, client=None, ref="HEAD"):
        refs.append(("content", ref, path))
        return "# README\n\nPinned content."

    monkeypatch.setattr(mod, "fetch_head_commit_sha", fake_head)
    monkeypatch.setattr(mod, "fetch_repo_tree", fake_tree)
    monkeypatch.setattr(mod, "fetch_gitignore", fake_gitignore)
    monkeypatch.setattr(mod, "fetch_file_content", fake_content)

    result = await mod.index_repo(
        "octo/docs",
        use_ai_summaries=False,
        use_embeddings=False,
        storage_path=str(tmp_path),
    )

    assert result["success"] is True
    assert result["head_sha"] == sha
    assert result["source_dirty"] is False
    assert result["repo_at_sha"] == f"octo/docs@{sha}"
    assert refs == [
        ("tree", sha),
        ("gitignore", sha),
        ("content", sha, "README.md"),
    ]
