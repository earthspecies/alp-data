"""Tests for provenance helpers in `alp_data.export.runner`."""

from pathlib import Path

from alp_data.export.runner import git_state


def test_git_state_reports_the_alp_data_checkout() -> None:
    commit, dirty = git_state(Path(__file__).resolve().parents[2])
    assert commit is not None and len(commit) == 40
    assert isinstance(dirty, bool)


def test_git_state_is_null_outside_an_alp_data_checkout(tmp_path: Path) -> None:
    # A directory that is inside some git repo but is not the alp_data source tree.
    assert git_state(tmp_path) == (None, None)
