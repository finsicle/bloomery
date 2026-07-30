# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the probe's shared helpers."""

from __future__ import annotations

import pytest

from bloomery.probe.util import find_key, gfx_name, parse_int, run


class TestParseInt:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("16384", 16384),
            ("16384 MB", 16384),
            ("24 GB", 24),
            (24560, 24560),
            (24.9, 24),
            ("", None),
            (None, None),
            ("[N/A]", None),
            ("N/A", None),
            (True, None),  # bool is an int subclass; must not slip through
            ("no digits", None),
        ],
    )
    def test_parses(self, value: object, expected: int | None) -> None:
        assert parse_int(value) == expected


class TestFindKey:
    def test_ignores_separators_and_case(self) -> None:
        payload = {"Market Name": "Radeon RX 7900 XTX"}
        assert find_key(payload, ("market_name",)) == "Radeon RX 7900 XTX"
        assert find_key({"marketName": "x"}, ("market_name",)) == "x"
        assert find_key({"market-name": "y"}, ("market_name",)) == "y"

    def test_searches_nested(self) -> None:
        payload = {"asic": {"deep": {"market_name": "MI300X"}}}
        assert find_key(payload, ("market_name",)) == "MI300X"

    def test_walks_lists(self) -> None:
        payload = [{"a": 1}, {"vram": {"size": 24576}}]
        assert find_key(payload, ("size",)) == 24576

    def test_skips_placeholder_values(self) -> None:
        payload = {"outer": {"name": "N/A"}, "inner": {"nested": {"name": "real"}}}
        assert find_key(payload, ("name",)) == "real"

    def test_breadth_first_prefers_shallow(self) -> None:
        payload = {"deep": {"id": "inner"}, "id": "outer"}
        assert find_key(payload, ("id",)) == "outer"

    def test_missing_returns_none(self) -> None:
        assert find_key({"a": 1}, ("b",)) is None


class TestGfxName:
    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            # Encoding is major * 10000 + minor * 100 + step, so the digits of
            # the gfx name do not line up with the digits of the integer.
            (110000, "gfx1100"),  # RX 7900 XTX: 11, 0, 0
            (110001, "gfx1101"),  # RX 7800 XT:  11, 0, 1
            (110002, "gfx1102"),
            (100300, "gfx1030"),  # RX 6900 XT:  10, 3, 0
            (120001, "gfx1201"),  # RDNA4:       12, 0, 1
            (90402, "gfx942"),  # MI300X:        9, 4, 2
            (90010, "gfx90a"),  # MI250X:        9, 0, 10 -> hex 'a'
            (90006, "gfx906"),  # Vega 20:       9, 0, 6
            (0, None),
            (-1, None),
        ],
    )
    def test_decodes_kfd_encoding(self, version: int, expected: str | None) -> None:
        assert gfx_name(version) == expected

    def test_rejects_out_of_range_fields(self) -> None:
        # 103000 would mean minor=30, which is not a valid single hex digit and
        # therefore not a real encoding.
        assert gfx_name(103000) is None


class TestRun:
    def test_missing_binary_is_not_an_error(self) -> None:
        result = run(["definitely-not-a-real-binary-xyz"])
        assert result.ok is False
        assert "not found" in result.stderr

    def test_empty_command(self) -> None:
        assert run([]).ok is False

    def test_captures_success(self) -> None:
        result = run(["echo", "hello"])
        assert result.ok is True
        assert result.stdout.strip() == "hello"

    def test_nonzero_exit_is_not_ok(self) -> None:
        result = run(["false"])
        assert result.ok is False
        assert result.returncode != 0

    def test_timeout_does_not_raise(self) -> None:
        result = run(["sleep", "5"], timeout=0.2)
        assert result.ok is False
        assert "timed out" in result.stderr
