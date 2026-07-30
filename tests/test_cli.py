# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end tests for the CLI.

These run the real probe against whatever machine the suite is on, so they
assert on shape and invariants rather than on specific hardware.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from bloomery import __version__
from bloomery.cli import app

runner = CliRunner()


class TestVersion:
    def test_prints_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout


class TestDoctor:
    def test_runs_without_crashing(self) -> None:
        result = runner.invoke(app, ["doctor", "--no-torch"])
        # Exit 1 is legitimate: it means an error-level issue was found, such as
        # a full disk. A crash would show up as an exception instead.
        assert result.exit_code in (0, 1)
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_renders_expected_sections(self) -> None:
        result = runner.invoke(app, ["doctor", "--no-torch"])
        for heading in ("host", "gpus", "pytorch backend", "what this machine can train"):
            assert heading in result.stdout

    def test_json_is_valid_and_shaped(self) -> None:
        result = runner.invoke(app, ["doctor", "--json", "--no-torch"])
        payload = json.loads(result.stdout)

        assert payload["bloomery_version"] == __version__
        assert "home" in payload

        probe = payload["probe"]
        assert probe["host"]["platform"] in (
            "linux",
            "wsl2",
            "windows",
            "macos",
            "unknown",
        )
        assert isinstance(probe["gpus"], list)
        assert isinstance(probe["issues"], list)
        assert probe["backend"] in (
            "cu130",
            "cu128",
            "cu126",
            "cu118",
            "rocm7.2",
            "xpu",
            "mps",
            "cpu",
        )

        capability = payload["capability"]
        assert capability["budget"]["bytes"] >= 0
        assert len(capability["models"]) == 24  # 8 sizes x 3 methods
        for model in capability["models"]:
            assert model["method"] in ("full", "lora", "qlora")
            assert model["params"] > 0
            assert model["required_bytes"] > 0
            assert isinstance(model["fits"], bool)

    def test_json_has_no_enum_repr_leakage(self) -> None:
        """Enums must serialise as their values, not as ``Vendor.NVIDIA``."""
        result = runner.invoke(app, ["doctor", "--json", "--no-torch"])
        assert "Vendor." not in result.stdout
        assert "Platform." not in result.stdout
        assert "Backend." not in result.stdout

    def test_no_torch_flag_omits_torch_section(self) -> None:
        result = runner.invoke(app, ["doctor", "--json", "--no-torch"])
        payload = json.loads(result.stdout)
        assert payload["probe"]["torch"] is None

    def test_exit_code_tracks_error_issues(self) -> None:
        result = runner.invoke(app, ["doctor", "--json", "--no-torch"])
        payload = json.loads(result.stdout)
        has_error = any(i["level"] == "error" for i in payload["probe"]["issues"])
        assert result.exit_code == (1 if has_error else 0)


class TestHelp:
    def test_bare_invocation_shows_help(self) -> None:
        result = runner.invoke(app, [])
        assert "doctor" in result.stdout

    def test_doctor_help(self) -> None:
        result = runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "--json" in result.stdout
