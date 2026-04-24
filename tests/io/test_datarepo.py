"""Tests for the DataRepo abstraction and DatasetInfo backward compat."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from esp_data.dataset import DatasetInfo
from esp_data.io.datarepo import (
    DataRepo,
    InvalidRelativePathError,
    NoAccessibleRepoError,
    get_repo,
    join_url,
    list_repos,
    register_repo,
    resolve,
    set_access_checker,
    unregister_repo,
    validate_strict_relpath,
)


# ---------- DataRepo basics --------------------------------------------------


class TestDataRepo:
    def test_construction_and_immutability(self):
        r = DataRepo(id="test", kind="gs", base_url="gs://bucket/")
        assert r.id == "test"
        assert r.kind == "gs"
        assert r.base_url == "gs://bucket/"
        # Frozen dataclass — mutation should fail
        with pytest.raises(Exception):
            r.id = "changed"  # noqa (deliberate failure)

    def test_default_priority_and_encoding(self):
        r = DataRepo(id="x", kind="https", base_url="https://a.b/")
        assert r.priority == 100
        assert r.path_encoding == "strict"


# ---------- join_url: URL-kind dispatch --------------------------------------


class TestJoinURL:
    def test_gs_join(self):
        r = DataRepo(id="x", kind="gs", base_url="gs://bucket/")
        assert join_url(r, "folder", "file.csv") == "gs://bucket/folder/file.csv"

    def test_gs_no_trailing_slash(self):
        r = DataRepo(id="x", kind="gs", base_url="gs://bucket")  # no trailing /
        assert join_url(r, "folder", "file.csv") == "gs://bucket/folder/file.csv"

    def test_s3_join(self):
        r = DataRepo(id="x", kind="s3", base_url="s3://bkt/")
        assert join_url(r, "a", "b.wav") == "s3://bkt/a/b.wav"

    def test_https_join(self):
        r = DataRepo(id="x", kind="https", base_url="https://example.com/data/")
        assert join_url(r, "ds", "train.csv") == "https://example.com/data/ds/train.csv"

    def test_local_join(self):
        r = DataRepo(id="x", kind="local", base_url="/tmp/data")
        assert join_url(r, "ds", "train.csv") == "/tmp/data/ds/train.csv"

    def test_join_handles_extra_slashes(self):
        r = DataRepo(id="x", kind="gs", base_url="gs://bucket//")
        # Each part has leading/trailing slashes stripped
        assert join_url(r, "/folder/", "/file.csv/") == "gs://bucket/folder/file.csv"

    def test_join_no_parts_returns_base(self):
        r = DataRepo(id="x", kind="gs", base_url="gs://bucket/")
        assert join_url(r) == "gs://bucket"

    def test_join_empty_parts_dropped(self):
        r = DataRepo(id="x", kind="gs", base_url="gs://bucket/")
        assert join_url(r, "", "folder", "", "file.csv") == "gs://bucket/folder/file.csv"


# ---------- Registry + resolve with explicit registry dicts ------------------


class TestResolve:
    def _make_registry(self, *repos):
        return {r.id: r for r in repos}

    def test_single_repo(self):
        reg = self._make_registry(
            DataRepo(id="gcs", kind="gs", base_url="gs://bkt/"),
        )
        url = resolve(["gcs"], "dataset/v1", "train.csv", registry=reg)
        assert url == "gs://bkt/dataset/v1/train.csv"

    def test_picks_highest_priority(self):
        reg = self._make_registry(
            DataRepo(id="slow", kind="gs", base_url="gs://a/", priority=50),
            DataRepo(id="fast", kind="gs", base_url="gs://b/", priority=10),
        )
        url = resolve(["slow", "fast"], "ds", "train.csv", registry=reg)
        assert url.startswith("gs://b/")

    def test_skips_unregistered(self):
        reg = self._make_registry(
            DataRepo(id="only", kind="https", base_url="https://a/"),
        )
        url = resolve(["missing", "only"], "ds", "train.csv", registry=reg)
        assert url == "https://a/ds/train.csv"

    def test_raises_when_no_accessible(self):
        reg = self._make_registry(
            DataRepo(id="a", kind="gs", base_url="gs://a/"),
        )
        with pytest.raises(NoAccessibleRepoError):
            resolve(["missing"], "ds", "train.csv", registry=reg)


# ---------- Global registry (exercises the default-repo wiring) --------------


class TestGlobalRegistry:
    def test_default_repo_registered(self):
        repos = {r.id: r for r in list_repos()}
        assert "esp-internal-gcs" in repos
        assert repos["esp-internal-gcs"].kind == "gs"

    def test_register_and_retrieve(self):
        register_repo(DataRepo(id="_test_custom", kind="https", base_url="https://x/"))
        try:
            assert get_repo("_test_custom") is not None
            assert get_repo("_test_custom").kind == "https"
        finally:
            unregister_repo("_test_custom")

    def test_unregister(self):
        register_repo(DataRepo(id="_test_delete_me", kind="local", base_url="/tmp"))
        assert get_repo("_test_delete_me") is not None
        unregister_repo("_test_delete_me")
        assert get_repo("_test_delete_me") is None


# ---------- Access-checker gate ----------------------------------------------


class TestAccessChecker:
    def test_blocked_repos_excluded(self):
        reg = {
            "blocked": DataRepo(id="blocked", kind="gs", base_url="gs://a/", priority=5),
            "allowed": DataRepo(id="allowed", kind="https", base_url="https://b/", priority=10),
        }
        set_access_checker(lambda r: r.id != "blocked")
        try:
            url = resolve(["blocked", "allowed"], "ds", "x.csv", registry=reg)
            assert url.startswith("https://b/")
        finally:
            set_access_checker(None)

    def test_all_blocked_raises(self):
        reg = {"a": DataRepo(id="a", kind="gs", base_url="gs://a/")}
        set_access_checker(lambda r: False)
        try:
            with pytest.raises(NoAccessibleRepoError):
                resolve(["a"], "ds", "x.csv", registry=reg)
        finally:
            set_access_checker(None)


# ---------- Strict relative-path validator -----------------------------------


class TestStrictRelpath:
    """Validate that strict-mode repos reject paths outside the URL-safe subset.

    This is the gate that keeps xeno-canto-style mixed-encoding paths from
    sneaking into new datasets.
    """

    # -- positive cases (should NOT raise) --

    def test_simple_filename(self):
        validate_strict_relpath("file.csv")

    def test_nested_path(self):
        validate_strict_relpath("audio/cbi/wav/XC135454.wav")

    def test_allowed_punctuation(self):
        validate_strict_relpath("a/b_c/d-e.f~g.wav")

    def test_hashed_filename(self):
        # The canonical content-addressed shape we expect long-term
        validate_strict_relpath("audio/a7/a7f3b2c1d9e8f0123456789abcdef0.wav")

    # -- negative cases (should raise) --

    def test_empty_rejected(self):
        with pytest.raises(InvalidRelativePathError, match="non-empty"):
            validate_strict_relpath("")

    def test_leading_slash_rejected(self):
        with pytest.raises(InvalidRelativePathError, match="start with"):
            validate_strict_relpath("/absolute/path.csv")

    def test_trailing_slash_rejected(self):
        with pytest.raises(InvalidRelativePathError, match="end with"):
            validate_strict_relpath("folder/")

    def test_double_dot_traversal_rejected(self):
        with pytest.raises(InvalidRelativePathError, match=r"\.\."):
            validate_strict_relpath("folder/../secret.csv")

    def test_space_rejected(self):
        with pytest.raises(InvalidRelativePathError, match="disallowed"):
            validate_strict_relpath("Yellow-legged Gull/file.wav")

    def test_unicode_rejected(self):
        with pytest.raises(InvalidRelativePathError, match="disallowed"):
            validate_strict_relpath("Espiñeiro.mp3")

    def test_percent_rejected(self):
        # Percent-encoded URLs are "legacy_percent", not strict
        with pytest.raises(InvalidRelativePathError, match="disallowed"):
            validate_strict_relpath("my%20file.csv")

    def test_query_string_rejected(self):
        with pytest.raises(InvalidRelativePathError, match="disallowed"):
            validate_strict_relpath("file.csv?v=1.2")

    def test_strict_repo_rejects_at_resolve_time(self):
        """resolve() should raise when the chosen repo is strict and the path
        contains forbidden chars."""
        reg = {"strict-gs": DataRepo(id="strict-gs", kind="gs", base_url="gs://b/", path_encoding="strict")}
        with pytest.raises(InvalidRelativePathError):
            resolve(["strict-gs"], "dataset/v1", "file with space.csv", registry=reg)

    def test_legacy_repo_does_not_enforce_yet(self):
        """Legacy-encoded repos don't enforce the strict grammar (placeholder
        until conversion logic lands)."""
        reg = {
            "legacy": DataRepo(
                id="legacy", kind="gs", base_url="gs://b/", path_encoding="legacy_raw",
            )
        }
        # This would reject under strict; should pass under legacy_raw for now.
        url = resolve(["legacy"], "Yellow-legged Gull", "file.wav", registry=reg)
        assert "Yellow-legged Gull" in url


# ---------- DatasetInfo backward compatibility -------------------------------


class TestDatasetInfoBackwardCompat:
    """The most important tests: the 37 existing Dataset subclasses must
    continue to work without modification.
    """

    def test_legacy_split_paths_only(self):
        """Legacy shape: split_paths is set explicitly, new fields unset."""
        info = DatasetInfo(
            name="legacy",
            owner="test",
            split_paths={"train": "gs://esp-ml-datasets/legacy/train.csv"},
            version="0.1.0",
            description="test",
            sources=["synthetic"],
        )
        assert info.split_paths == {"train": "gs://esp-ml-datasets/legacy/train.csv"}
        assert info.repos is None
        assert info.folder is None
        assert info.splits is None

    def test_modern_shape_derives_split_paths(self):
        """Modern shape: repos+folder+splits set; split_paths is auto-derived."""
        info = DatasetInfo(
            name="modern",
            owner="test",
            repos=["esp-internal-gcs"],
            folder="modern-ds/v0.1.0/raw",
            splits={"train": "train.csv", "val": "val.csv"},
            version="0.1.0",
            description="test",
            sources=["synthetic"],
        )
        assert info.split_paths == {
            "train": "gs://esp-ml-datasets/modern-ds/v0.1.0/raw/train.csv",
            "val": "gs://esp-ml-datasets/modern-ds/v0.1.0/raw/val.csv",
        }
        # Modern fields are preserved on the instance
        assert info.repos == ["esp-internal-gcs"]
        assert info.folder == "modern-ds/v0.1.0/raw"

    def test_both_shapes_rejected(self):
        with pytest.raises(ValidationError, match="either.*legacy.*modern"):
            DatasetInfo(
                name="both",
                owner="test",
                split_paths={"train": "gs://..."},
                repos=["esp-internal-gcs"],
                folder="x",
                splits={"train": "train.csv"},
                version="0.1.0",
                description="test",
                sources=["synthetic"],
            )

    def test_neither_shape_rejected(self):
        with pytest.raises(ValidationError, match="modern shape requires"):
            DatasetInfo(
                name="neither",
                owner="test",
                version="0.1.0",
                description="test",
                sources=["synthetic"],
            )

    def test_modern_partial_fields_rejected(self):
        """Missing folder or splits when repos is set should error with clarity."""
        with pytest.raises(ValidationError, match="modern shape requires"):
            DatasetInfo(
                name="partial",
                owner="test",
                repos=["esp-internal-gcs"],
                # folder + splits missing
                version="0.1.0",
                description="test",
                sources=["synthetic"],
            )

    def test_modern_subclass_reads_split_paths_as_before(self):
        """The load path uses `info.split_paths[split]` directly — confirm that
        still works for modern-shape DatasetInfo."""
        info = DatasetInfo(
            name="subclass-compat",
            owner="test",
            repos=["esp-internal-gcs"],
            folder="compat/v0.1.0",
            splits={"train": "train.csv"},
            version="0.1.0",
            description="test",
            sources=["synthetic"],
        )
        # This is exactly what Dataset subclasses do:
        location = info.split_paths["train"]
        assert location == "gs://esp-ml-datasets/compat/v0.1.0/train.csv"
        available = list(info.split_paths.keys())
        assert available == ["train"]


# ---------- DatasetInfo resolver integration ---------------------------------


class TestDatasetInfoResolverBehavior:
    def test_runtime_repo_override(self):
        """Changing the registered repo's base_url changes the resolved URL
        for new DatasetInfo instances."""
        original = get_repo("esp-internal-gcs")
        register_repo(DataRepo(
            id="esp-internal-gcs",
            kind="gs",
            base_url="gs://staging-bucket/",
            priority=10,
        ))
        try:
            info = DatasetInfo(
                name="override",
                owner="test",
                repos=["esp-internal-gcs"],
                folder="ds/v0.1.0",
                splits={"train": "train.csv"},
                version="0.1.0",
                description="test",
                sources=["synthetic"],
            )
            assert info.split_paths["train"] == "gs://staging-bucket/ds/v0.1.0/train.csv"
        finally:
            register_repo(original)

    def test_priority_fallback_to_public_mirror(self):
        """When multiple repos are listed and the highest-priority one is
        unavailable (unregistered in this test), resolver picks the next."""
        # Save + temporarily remove ESP-internal
        original = get_repo("esp-internal-gcs")
        unregister_repo("esp-internal-gcs")
        register_repo(DataRepo(
            id="_test_public",
            kind="https",
            base_url="https://pub-test.r2.dev/esp/",
            priority=50,
        ))
        try:
            info = DatasetInfo(
                name="public-fallback",
                owner="test",
                repos=["esp-internal-gcs", "_test_public"],
                folder="ds/v0.1.0",
                splits={"train": "train.csv"},
                version="0.1.0",
                description="test",
                sources=["synthetic"],
            )
            assert info.split_paths["train"] == "https://pub-test.r2.dev/esp/ds/v0.1.0/train.csv"
        finally:
            unregister_repo("_test_public")
            register_repo(original)
