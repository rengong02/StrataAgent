"""A run's artifacts must name the upstream state that produced them.

The failure this prevents already happened once: a rerun produced a file still
showing 30 `fmt` errors while the working tree had the fix applied, and nothing
in the file said which checkout or patch state it came from -- so the result
could not be trusted either way and had to be repeated.

The load-bearing check is `patches/` state, because a missing patch does not
make a run fail; it makes it *quietly worse*, which is the harder thing to
notice.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from strata.geobench.provenance import (
    geobenchx_provenance,
    git_revision,
    matplotlib_backend,
    patch_state,
)
from strata.geobench.tools import geobenchx_root

ROOT = Path(__file__).resolve().parents[2]


def _have_checkout() -> bool:
    return (geobenchx_root() / "geobenchx" / "tools.py").exists()


requires_geobenchx = pytest.mark.skipif(not _have_checkout(), reason="GeoBenchX checkout not present")


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "f.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@e", "-c", "user.name=t", "commit", "-qm", "x"],
        cwd=repo, check=True,
    )
    return repo


# --- the pieces ---------------------------------------------------------------


def test_git_revision_reads_a_real_repo(tmp_path):
    assert git_revision(_git_repo(tmp_path)) is not None


def test_git_revision_returns_none_outside_a_repo(tmp_path):
    """Provenance is diagnostics: it must never raise and break a run."""
    assert git_revision(tmp_path) is None


def test_a_clean_tree_has_no_digest_and_a_dirty_one_does(tmp_path):
    from strata.geobench.provenance import working_tree_digest

    repo = _git_repo(tmp_path)
    assert working_tree_digest(repo) == "", "a clean tree should read as clean"

    (repo / "f.txt").write_text("two\n", encoding="utf-8")
    digest = working_tree_digest(repo)
    assert digest, "a modified tree must produce a digest"

    (repo / "f.txt").write_text("three\n", encoding="utf-8")
    assert working_tree_digest(repo) != digest, "the digest must track the content, not just 'dirty'"


def test_the_backend_is_reported_not_raised(tmp_path):
    assert isinstance(matplotlib_backend(), str)
    assert matplotlib_backend()


# --- the patch state, which is the point --------------------------------------


def test_patch_state_detects_an_applied_patch(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "f.txt").write_text("two\n", encoding="utf-8")
    diff = subprocess.run(["git", "diff"], cwd=repo, capture_output=True, text=True, check=True).stdout
    patches = tmp_path / "patches"
    patches.mkdir()
    (patches / "0001.patch").write_text(diff, encoding="utf-8")

    assert patch_state(patches, repo) == {"0001.patch": True}


def test_patch_state_detects_a_patch_that_is_not_applied(tmp_path):
    repo = _git_repo(tmp_path)
    patches = tmp_path / "patches"
    patches.mkdir()
    (patches / "0001.patch").write_text(
        "diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-one\n+two\n",
        encoding="utf-8",
    )

    assert patch_state(patches, repo) == {"0001.patch": False}, (
        "an unapplied patch must read as False -- this is the case that silently "
        "degrades a run instead of failing it"
    )


def test_a_missing_patches_directory_is_not_an_error(tmp_path):
    assert patch_state(tmp_path / "nope", tmp_path) == {}


# --- the whole record ---------------------------------------------------------


@requires_geobenchx
def test_the_provenance_record_carries_what_a_rerun_needs():
    prov = geobenchx_provenance(ROOT)

    assert Path(prov["geobenchx_root"]).is_dir()
    assert prov["geobenchx_head"], "the checkout's revision is the first thing a reader needs"
    assert isinstance(prov["patches"], dict)
    assert prov["matplotlib_backend"]


@requires_geobenchx
def test_every_shipped_patch_is_applied_in_this_tree():
    """Guards the real workflow, not just the mechanism.

    If this fails, a run started from here produces artifacts full of a failure
    that is already fixed -- and the runner refuses to start because of it.
    """
    prov = geobenchx_provenance(ROOT)
    unapplied = sorted(name for name, ok in prov["patches"].items() if not ok)
    assert unapplied == [], f"patches/ has entries that are not applied: {unapplied}"
