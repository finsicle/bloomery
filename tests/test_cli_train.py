# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end CLI tests for the training commands.

Every test points ``BLOOMERY_HOME`` at a temporary directory, so nothing here
touches a real user's datasets or checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bloomery.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("BLOOMERY_HOME", str(home))
    return home


def _invoke(*args: str):  # noqa: ANN202 - typer's Result type is internal
    result = runner.invoke(app, list(args))
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise AssertionError(f"{args} raised {result.exception!r}\n{result.stdout}")
    return result


class TestPrepare:
    def test_synthetic_corpus(self, isolated_home: Path) -> None:
        result = _invoke("prepare", "--name", "t", "--synthetic", "400", "--vocab", "300")
        assert result.exit_code == 0, result.stdout
        assert (isolated_home / "datasets/t/tokenizer/tokenizer.json").is_file()
        assert (isolated_home / "datasets/t/tokens/train.bin").is_file()
        assert (isolated_home / "datasets/t/tokens/meta.json").is_file()

    def test_reads_a_source_directory(self, tmp_path: Path, isolated_home: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for index in range(60):
            (corpus / f"{index}.txt").write_text(
                f"Document number {index}. It has a few words in it so the tokenizer "
                "has something to chew on."
            )
        result = _invoke("prepare", "--name", "files", "--source", str(corpus), "--vocab", "300")
        assert result.exit_code == 0, result.stdout
        meta = json.loads((isolated_home / "datasets/files/tokens/meta.json").read_text())
        assert meta["splits"][0]["tokens"] > 0

    def test_requires_exactly_one_input(self) -> None:
        both = _invoke("prepare", "--name", "x", "--source", ".", "--synthetic", "10")
        assert both.exit_code == 1
        assert "exactly one" in both.stdout

        neither = _invoke("prepare", "--name", "x")
        assert neither.exit_code == 1

    def test_missing_source(self) -> None:
        result = _invoke("prepare", "--name", "x", "--source", "/nonexistent/path")
        assert result.exit_code == 1
        assert "does not exist" in result.stdout

    def test_source_with_no_usable_files(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "photo.png").write_bytes(b"\x89PNG")
        result = _invoke("prepare", "--name", "x", "--source", str(empty))
        assert result.exit_code == 1
        assert "no .txt" in result.stdout

    def test_name_is_sanitised(self, isolated_home: Path) -> None:
        """A dataset name must not be able to escape the bloomery home."""
        result = _invoke(
            "prepare", "--name", "../../escape", "--synthetic", "400", "--vocab", "300"
        )
        assert result.exit_code == 0
        assert not (isolated_home.parent / "escape").exists()
        assert list((isolated_home / "datasets").iterdir())


class TestTrain:
    @pytest.fixture(autouse=True)
    def prepared(self, isolated_home: Path) -> None:
        assert (
            _invoke(
                "prepare",
                "--name",
                "d",
                "--synthetic",
                "600",
                "--vocab",
                "300",
                "--val-fraction",
                "0.1",
            ).exit_code
            == 0
        )

    def test_trains_and_writes_a_checkpoint(self, isolated_home: Path) -> None:
        result = _invoke(
            "train",
            "--data",
            "d",
            "--name",
            "r",
            "--depth",
            "1",
            "--steps",
            "6",
            "--batch",
            "4",
            "--seq",
            "32",
            "--device",
            "cpu",
        )
        assert result.exit_code == 0, result.stdout
        checkpoint = isolated_home / "runs/r/latest"
        assert (checkpoint / "config.json").is_file()
        assert (checkpoint / "model.safetensors").is_file()
        assert (isolated_home / "runs/r/run.jsonl").is_file()

    def test_reports_loss_and_throughput(self) -> None:
        result = _invoke(
            "train",
            "--data",
            "d",
            "--name",
            "r",
            "--depth",
            "1",
            "--steps",
            "6",
            "--batch",
            "4",
            "--seq",
            "32",
            "--device",
            "cpu",
        )
        assert "final loss" in result.stdout
        assert "tok/s" in result.stdout

    def test_unknown_dataset(self) -> None:
        result = _invoke("train", "--data", "nope", "--steps", "2")
        assert result.exit_code == 1
        assert "prepare" in result.stdout

    def test_size_and_depth_are_mutually_exclusive(self) -> None:
        result = _invoke("train", "--data", "d", "--depth", "2", "--size", "d12", "--steps", "2")
        assert result.exit_code == 1
        assert "not both" in result.stdout

    def test_sequence_longer_than_corpus_is_a_clear_error(self) -> None:
        result = _invoke(
            "train",
            "--data",
            "d",
            "--depth",
            "1",
            "--steps",
            "2",
            "--seq",
            "500000",
            "--device",
            "cpu",
        )
        assert result.exit_code == 1
        assert "not enough" in result.stdout or "shorter --seq" in result.stdout

    def test_resume_without_a_checkpoint(self) -> None:
        result = _invoke(
            "train",
            "--data",
            "d",
            "--name",
            "fresh",
            "--depth",
            "1",
            "--steps",
            "2",
            "--resume",
            "--device",
            "cpu",
        )
        assert result.exit_code == 1
        assert "resumable" in result.stdout

    def test_resume_continues_from_the_checkpoint(self, isolated_home: Path) -> None:
        from bloomery.train.metrics import read_events

        first = _invoke(
            "train",
            "--data",
            "d",
            "--name",
            "r",
            "--depth",
            "1",
            "--steps",
            "4",
            "--batch",
            "4",
            "--seq",
            "32",
            "--device",
            "cpu",
        )
        assert first.exit_code == 0, first.stdout

        second = _invoke(
            "train",
            "--data",
            "d",
            "--name",
            "r",
            "--depth",
            "1",
            "--steps",
            "8",
            "--batch",
            "4",
            "--seq",
            "32",
            "--device",
            "cpu",
            "--resume",
        )
        assert second.exit_code == 0, second.stdout

        events = read_events(isolated_home / "runs/r/run.jsonl")
        starts = [e for e in events if e["event"] == "start"]
        assert len(starts) == 2
        assert starts[1]["resumed_from"] is not None


class TestChat:
    def test_requires_exactly_one_target(self) -> None:
        both = _invoke("chat", "--run", "r", "--checkpoint", "/tmp")
        assert both.exit_code == 1
        assert "exactly one" in both.stdout

        neither = _invoke("chat")
        assert neither.exit_code == 1

    def test_missing_checkpoint(self, tmp_path: Path) -> None:
        result = _invoke("chat", "--checkpoint", str(tmp_path), "--prompt", "hi")
        assert result.exit_code == 1

    def test_generates_from_a_trained_run(self) -> None:
        assert (
            _invoke("prepare", "--name", "d", "--synthetic", "600", "--vocab", "300").exit_code == 0
        )
        assert (
            _invoke(
                "train",
                "--data",
                "d",
                "--name",
                "r",
                "--depth",
                "1",
                "--steps",
                "4",
                "--batch",
                "4",
                "--seq",
                "32",
                "--device",
                "cpu",
            ).exit_code
            == 0
        )

        result = _invoke(
            "chat",
            "--run",
            "r",
            "--prompt",
            "Ana found",
            "--max-new-tokens",
            "8",
            "--temperature",
            "0",
            "--device",
            "cpu",
        )
        assert result.exit_code == 0, result.stdout
        assert "params on" in result.stdout


class TestBench:
    def test_reports_measured_throughput(self) -> None:
        result = _invoke(
            "bench",
            "--depth",
            "1",
            "--vocab",
            "300",
            "--batch",
            "2",
            "--seq",
            "32",
            "--steps",
            "2",
            "--device",
            "cpu",
        )
        assert result.exit_code == 0, result.stdout
        assert "tok/s" in result.stdout

    def test_json_output(self) -> None:
        result = _invoke(
            "bench",
            "--depth",
            "1",
            "--vocab",
            "300",
            "--batch",
            "2",
            "--seq",
            "32",
            "--steps",
            "2",
            "--device",
            "cpu",
            "--json",
        )
        payload = json.loads(result.stdout)
        assert payload["tokens_per_second"] > 0
        assert payload["params"] > 0
        assert payload["device"] == "cpu"
        assert payload["chinchilla_hours"] > 0

    def test_defaults_to_a_named_size(self) -> None:
        result = _invoke("bench", "--help")
        assert result.exit_code == 0
        assert "--size" in result.stdout


class TestDemo:
    def test_runs_end_to_end_and_cleans_up(self, isolated_home: Path) -> None:
        result = _invoke("demo", "--steps", "4", "--documents", "400", "--device", "cpu")
        assert result.exit_code == 0, result.stdout
        assert "samples" in result.stdout
        # Cleanup is the default, so nothing should be left behind.
        assert not (isolated_home / "datasets/demo").exists()
        assert not (isolated_home / "runs/demo").exists()

    def test_keep_retains_artifacts(self, isolated_home: Path) -> None:
        result = _invoke("demo", "--steps", "4", "--documents", "400", "--device", "cpu", "--keep")
        assert result.exit_code == 0, result.stdout
        assert (isolated_home / "runs/demo/latest/config.json").is_file()


class TestHelp:
    @pytest.mark.parametrize("command", ["prepare", "train", "chat", "bench", "demo", "doctor"])
    def test_every_command_has_help(self, command: str) -> None:
        result = _invoke(command, "--help")
        assert result.exit_code == 0
        assert command in result.stdout.lower() or "Usage" in result.stdout

    def test_root_lists_commands(self) -> None:
        result = _invoke()
        for command in ("prepare", "train", "chat", "bench", "demo", "doctor"):
            assert command in result.stdout
