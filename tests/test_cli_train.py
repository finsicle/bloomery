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


class TestMix:
    def test_create_and_show(self, isolated_home: Path) -> None:
        created = _invoke("mix", "create", "--name", "b", "--add", "new:0.9", "--replay", "old:0.1")
        assert created.exit_code == 0, created.stdout
        assert (isolated_home / "mixtures/b/v1.json").is_file()

        shown = _invoke("mix", "show", "b")
        assert shown.exit_code == 0
        assert "90.0%" in shown.stdout
        assert "replay" in shown.stdout

    def test_weights_are_normalised_not_required_to_sum_to_one(self) -> None:
        """60/15/25 must work as readily as 0.6/0.15/0.25."""
        assert (
            _invoke(
                "mix",
                "create",
                "--name",
                "raw",
                "--add",
                "a:60",
                "--add",
                "b:15",
                "--replay",
                "c:25",
            ).exit_code
            == 0
        )
        shown = _invoke("mix", "show", "raw")
        assert "60.0%" in shown.stdout
        assert "replay share  25%" in shown.stdout

    def test_create_requires_a_component(self) -> None:
        result = _invoke("mix", "create", "--name", "empty")
        assert result.exit_code == 1
        assert "at least one" in result.stdout

    def test_malformed_spec_is_rejected(self) -> None:
        result = _invoke("mix", "create", "--name", "b", "--add", "no-weight-here")
        assert result.exit_code == 1
        assert "cannot parse" in result.stdout

    def test_add_creates_a_new_version_with_lineage(self, isolated_home: Path) -> None:
        assert _invoke("mix", "create", "--name", "b", "--add", "new:0.9").exit_code == 0
        added = _invoke("mix", "add", "b", "--replay", "old:0.3", "--note", "more replay")
        assert added.exit_code == 0, added.stdout
        assert "parent     v1" in added.stdout
        assert (isolated_home / "mixtures/b/v2.json").is_file()
        # v1 must survive unchanged; that is what makes a past run reproducible.
        assert (isolated_home / "mixtures/b/v1.json").is_file()

        shown = _invoke("mix", "show", "b")
        assert "lineage" in shown.stdout
        assert "more replay" in shown.stdout

    def test_add_multiple_components_produces_one_version(self, isolated_home: Path) -> None:
        """Three additions should give v2, not v2 through v4."""
        assert _invoke("mix", "create", "--name", "b", "--add", "a:1").exit_code == 0
        assert (
            _invoke("mix", "add", "b", "--add", "c:1", "--add", "d:1", "--replay", "e:1").exit_code
            == 0
        )
        versions = sorted(p.name for p in (isolated_home / "mixtures/b").iterdir())
        assert versions == ["v1.json", "v2.json"]

    def test_add_requires_an_operation(self) -> None:
        assert _invoke("mix", "create", "--name", "b", "--add", "a:1").exit_code == 0
        result = _invoke("mix", "add", "b")
        assert result.exit_code == 1
        assert "at least one" in result.stdout

    def test_add_to_unknown_mixture(self) -> None:
        result = _invoke("mix", "add", "ghost", "--add", "a:1")
        assert result.exit_code == 1
        assert "no mixture named" in result.stdout

    def test_show_unknown_mixture(self) -> None:
        result = _invoke("mix", "show", "ghost")
        assert result.exit_code == 1

    def test_list_empty_and_populated(self) -> None:
        empty = _invoke("mix", "list")
        assert empty.exit_code == 0
        assert "No mixtures yet" in empty.stdout

        _invoke("mix", "create", "--name", "one", "--add", "a:1")
        _invoke("mix", "create", "--name", "two", "--add", "b:1")
        listed = _invoke("mix", "list")
        assert "one" in listed.stdout
        assert "two" in listed.stdout


class TestTrainOnMixture:
    @pytest.fixture(autouse=True)
    def prepared(self, isolated_home: Path) -> None:
        # Two corpora sharing one tokenizer, which is what makes them blendable.
        from bloomery import paths
        from bloomery.data import (
            build_dataset,
            eot_id,
            synthetic_documents,
            train_tokenizer,
        )

        first = synthetic_documents(500, seed=1)
        second = synthetic_documents(500, seed=2)
        shared = train_tokenizer(
            first + second, vocab_size=400, out_dir=paths.tokenizer_dir("alpha")
        )
        for name, docs in (("alpha", first), ("beta", second)):
            shared.save_pretrained(str(paths.tokenizer_dir(name)))
            build_dataset(
                docs,
                shared,
                out_dir=paths.tokens_dir(name),
                eot=eot_id(shared),
                val_fraction=0.1,
            )

    def test_trains_on_a_blend_and_reports_each_component(self) -> None:
        assert (
            _invoke(
                "mix", "create", "--name", "b", "--add", "alpha:0.8", "--replay", "beta:0.2"
            ).exit_code
            == 0
        )

        result = _invoke(
            "train",
            "--mix",
            "b",
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
            "--eval-every",
            "4",
            "--device",
            "cpu",
        )
        assert result.exit_code == 0, result.stdout
        assert "mixture" in result.stdout
        assert "replay share 20%" in result.stdout
        assert "per component" in result.stdout

    def test_records_the_mixture_in_the_run_log(self, isolated_home: Path) -> None:
        from bloomery.train.metrics import read_events

        _invoke("mix", "create", "--name", "b", "--add", "alpha:0.8", "--replay", "beta:0.2")
        assert (
            _invoke(
                "train",
                "--mix",
                "b",
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

        start = next(
            e for e in read_events(isolated_home / "runs/r/run.jsonl") if e["event"] == "start"
        )
        assert start["mixture"] == "b"
        assert start["mixture_version"] == 1
        assert start["replay_share"] == pytest.approx(0.2)
        assert set(start["components"]) == {"alpha", "beta"}

    def test_warns_when_a_blend_has_no_replay(self) -> None:
        _invoke("mix", "create", "--name", "noreplay", "--add", "alpha:0.5", "--add", "beta:0.5")
        result = _invoke(
            "train",
            "--mix",
            "noreplay",
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
        assert result.exit_code == 0, result.stdout
        assert "no component is marked as replay" in result.stdout

    def test_refuses_incompatible_tokenizers(self) -> None:
        from bloomery import paths
        from bloomery.data import (
            build_dataset,
            eot_id,
            synthetic_documents,
            train_tokenizer,
        )

        docs = synthetic_documents(300, seed=9)
        foreign = train_tokenizer(docs, vocab_size=700, out_dir=paths.tokenizer_dir("foreign"))
        build_dataset(
            docs,
            foreign,
            out_dir=paths.tokens_dir("foreign"),
            eot=eot_id(foreign),
            val_fraction=0.1,
        )
        _invoke("mix", "create", "--name", "bad", "--add", "alpha:1", "--add", "foreign:1")

        result = _invoke("train", "--mix", "bad", "--depth", "1", "--steps", "2", "--device", "cpu")
        assert result.exit_code == 1
        assert "different tokenizers" in result.stdout

    def test_data_and_mix_are_mutually_exclusive(self) -> None:
        result = _invoke("train", "--data", "alpha", "--mix", "b", "--steps", "2")
        assert result.exit_code == 1
        assert "exactly one" in result.stdout

    def test_neither_data_nor_mix(self) -> None:
        result = _invoke("train", "--steps", "2")
        assert result.exit_code == 1

    def test_pinning_a_mixture_version(self) -> None:
        _invoke("mix", "create", "--name", "b", "--add", "alpha:1")
        _invoke("mix", "add", "b", "--replay", "beta:1")
        result = _invoke(
            "train",
            "--mix",
            "b",
            "--mix-version",
            "1",
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
        assert result.exit_code == 0, result.stdout
        # v1 is a single component, so no mixture line is printed.
        assert "replay share" not in result.stdout
