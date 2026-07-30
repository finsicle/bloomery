# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for weighted dataset blends: weights, versioning, lineage, persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from bloomery import mixture as mix
from bloomery.mixture import (
    ROLE_DOMAIN,
    ROLE_REPLAY,
    Component,
    Mixture,
    MixtureError,
)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("BLOOMERY_HOME", str(home))
    return home


def blend(**kwargs: object) -> Mixture:
    return mix.create(
        str(kwargs.get("name", "b")),
        [
            Component("new", 0.9),
            Component("old", 0.1, role=ROLE_REPLAY),
        ],
    )


class TestComponent:
    def test_rejects_nonpositive_weight(self) -> None:
        with pytest.raises(MixtureError, match="must be positive"):
            Component("a", 0.0)
        with pytest.raises(MixtureError, match="must be positive"):
            Component("a", -1.0)

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(MixtureError, match="cannot be empty"):
            Component("", 1.0)

    def test_rejects_unknown_role(self) -> None:
        with pytest.raises(MixtureError, match="unknown role"):
            Component("a", 1.0, role="somethingelse")

    def test_round_trips(self) -> None:
        original = Component("a", 0.25, role=ROLE_REPLAY)
        assert Component.from_dict(original.to_dict()) == original


class TestWeights:
    def test_normalises_to_one(self) -> None:
        m = mix.create("m", [Component("a", 60), Component("b", 15), Component("c", 25)])
        weights = m.weights()
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["a"] == pytest.approx(0.6)

    def test_raw_and_fractional_weights_are_equivalent(self) -> None:
        """60/15/25 and 0.6/0.15/0.25 must describe the same blend."""
        raw = mix.create("r", [Component("a", 60), Component("b", 15), Component("c", 25)])
        frac = mix.create("f", [Component("a", 0.6), Component("b", 0.15), Component("c", 0.25)])
        assert raw.weights() == pytest.approx(frac.weights())

    def test_replay_share(self) -> None:
        m = mix.create(
            "m",
            [Component("new", 0.75), Component("old", 0.25, role=ROLE_REPLAY)],
        )
        assert m.replay_share() == pytest.approx(0.25)

    def test_replay_share_is_zero_without_replay_components(self) -> None:
        m = mix.create("m", [Component("a", 1.0)])
        assert m.replay_share() == 0.0

    def test_rejects_duplicate_dataset(self) -> None:
        with pytest.raises(MixtureError, match="appears twice"):
            mix.create("m", [Component("a", 1.0), Component("a", 2.0)])

    def test_rejects_empty_mixture(self) -> None:
        with pytest.raises(MixtureError, match="at least one component"):
            Mixture(name="m", version=1, components=())


class TestVersioning:
    def test_with_component_bumps_version_and_records_parent(self) -> None:
        v1 = blend()
        v2 = v1.with_component("extra", 0.2)
        assert (v2.version, v2.parent_version) == (2, 1)
        assert "extra" in v2.datasets

    def test_with_component_does_not_mutate(self) -> None:
        """Versions are immutable, or a past run cannot be reproduced."""
        v1 = blend()
        before = v1.weights()
        v1.with_component("extra", 5.0)
        assert v1.weights() == before
        assert "extra" not in v1.datasets

    def test_with_component_replaces_an_existing_weight(self) -> None:
        v2 = blend().with_component("old", 0.5, role=ROLE_REPLAY)
        assert len(v2.components) == 2
        assert v2.weights()["old"] == pytest.approx(0.5 / 1.4)

    def test_with_component_can_change_a_role(self) -> None:
        v2 = blend().with_component("new", 0.9, role=ROLE_REPLAY)
        assert v2.replay_share() == pytest.approx(1.0)

    def test_without_component(self) -> None:
        v2 = blend().without_component("old")
        assert v2.datasets == ("new",)
        assert v2.parent_version == 1

    def test_without_unknown_component(self) -> None:
        with pytest.raises(MixtureError, match="no component"):
            blend().without_component("missing")

    def test_cannot_remove_the_last_component(self) -> None:
        single = mix.create("m", [Component("a", 1.0)])
        with pytest.raises(MixtureError, match="last component"):
            single.without_component("a")


class TestPersistence:
    def test_save_and_load(self) -> None:
        original = blend()
        mix.save(original)
        loaded = mix.load("b")
        assert loaded == original

    def test_save_refuses_to_overwrite(self) -> None:
        """Immutability is enforced on disk, not just in memory."""
        mix.save(blend())
        with pytest.raises(MixtureError, match="immutable"):
            mix.save(blend())

    def test_load_defaults_to_newest_version(self) -> None:
        v1 = blend()
        mix.save(v1)
        mix.save(v1.with_component("third", 0.5))
        assert mix.load("b").version == 2

    def test_load_specific_version(self) -> None:
        v1 = blend()
        mix.save(v1)
        mix.save(v1.with_component("third", 0.5))
        assert mix.load("b", 1).version == 1

    def test_load_missing_mixture(self) -> None:
        with pytest.raises(MixtureError, match="no mixture named"):
            mix.load("nope")

    def test_load_missing_version_lists_what_exists(self) -> None:
        mix.save(blend())
        with pytest.raises(MixtureError, match="v1") as caught:
            mix.load("b", 7)
        assert "no v7" in str(caught.value)

    def test_versions_are_sorted(self) -> None:
        current = blend()
        mix.save(current)
        for index in range(3):
            current = current.with_component(f"d{index}", 1.0)
            mix.save(current)
        assert mix.versions("b") == [1, 2, 3, 4]

    def test_versions_of_unknown_mixture(self) -> None:
        assert mix.versions("nope") == []

    def test_list_all_returns_newest_of_each(self) -> None:
        first = blend(name="one")
        mix.save(first)
        mix.save(first.with_component("x", 1.0))
        second = mix.create("two", [Component("a", 1.0)])
        mix.save(second)

        listed = {m.name: m.version for m in mix.list_all()}
        assert listed == {"one": 2, "two": 1}

    def test_list_all_when_empty(self) -> None:
        assert mix.list_all() == []

    def test_name_is_slugged_into_a_safe_directory(self) -> None:
        """A mixture name comes from user input and must not escape the home."""
        m = mix.create("../../escape", [Component("a", 1.0)])
        path = mix.save(m)
        assert ".." not in str(path.relative_to(mix.mixtures_root()))


class TestLineage:
    def test_walks_back_to_the_original(self) -> None:
        current = blend()
        mix.save(current)
        for index in range(3):
            current = current.with_component(f"d{index}", 1.0)
            mix.save(current)

        chain = mix.lineage(mix.load("b"))
        assert [m.version for m in chain] == [1, 2, 3, 4]

    def test_single_version_has_a_lineage_of_one(self) -> None:
        m = blend()
        mix.save(m)
        assert len(mix.lineage(m)) == 1

    def test_broken_chain_stops_instead_of_raising(self) -> None:
        """A partly deleted history should not block a training run."""
        v1 = blend()
        v2 = v1.with_component("x", 1.0)
        # v2 saved but v1 never was, so the parent is unreachable.
        mix.save(v2)
        chain = mix.lineage(v2)
        assert [m.version for m in chain] == [2]


class TestSingle:
    def test_wraps_one_dataset(self) -> None:
        m = mix.single("corpus")
        assert m.datasets == ("corpus",)
        assert m.weights() == {"corpus": 1.0}
        assert m.replay_share() == 0.0

    def test_describe_is_readable(self) -> None:
        assert "corpus 100%" in mix.single("corpus").describe()


class TestParseSpec:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("domain:0.6", ("domain", 0.6)),
            ("domain=60", ("domain", 60.0)),
            ("domain:60%", ("domain", 60.0)),
            ("  domain : 0.25 ", ("domain", 0.25)),
            ("my-data_v2:1", ("my-data_v2", 1.0)),
        ],
    )
    def test_parses(self, spec: str, expected: tuple[str, float]) -> None:
        assert mix.parse_spec(spec) == expected

    @pytest.mark.parametrize("spec", ["domain", "domain:", ":0.5", "domain:abc", ""])
    def test_rejects_malformed(self, spec: str) -> None:
        with pytest.raises(MixtureError, match="cannot parse"):
            mix.parse_spec(spec)

    def test_rejects_zero_weight(self) -> None:
        with pytest.raises(MixtureError, match="must be positive"):
            mix.parse_spec("domain:0")


class TestDescribe:
    def test_orders_by_weight_and_marks_replay(self) -> None:
        m = mix.create(
            "m",
            [
                Component("small", 0.1, role=ROLE_REPLAY),
                Component("big", 0.9, role=ROLE_DOMAIN),
            ],
        )
        described = m.describe()
        assert described.index("big") < described.index("small")
        assert "(replay)" in described
        assert "m v1" in described
