# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end CLI tests for the training commands.

Every test points ``BLOOMERY_HOME`` at a temporary directory, so nothing here
touches a real user's datasets or checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from _helpers import plain
from typer.testing import CliRunner

from bloomery.cli import app

runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "200"})


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("BLOOMERY_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def _stable_memory_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop ambient RAM from deciding whether these tests pass.

    The pre-flight budget is derived from *available* system memory, and even a
    depth-1 model estimates around 0.8 GiB once the fixed context overhead is
    counted. On a machine with little free memory — or simply partway through a
    long suite run — the budget dips under that and `train` and `adapt` refuse,
    so a test about tokenizers or exports fails for reasons of its own host.

    Observed as exactly that: this file passes alone and fails inside the full
    suite, with a different test each time.

    The budget is pinned generously rather than the check disabled, so a run that
    genuinely cannot fit is still refused — TestMemoryGuard asks for shapes that
    need terabytes and is unaffected.
    """
    from bloomery import capability

    real = capability.derive_budget

    def generous(report: Any, *, device_type: str | None = None) -> Any:
        budget = real(report, device_type=device_type)
        return replace(budget, total=max(budget.total, 64 * 1024**3))

    monkeypatch.setattr(capability, "derive_budget", generous)


def _invoke(*args: str):  # noqa: ANN202 - typer's Result type is internal
    result = runner.invoke(app, list(args))
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise AssertionError(f"{args} raised {result.exception!r}\n{plain(result.stdout)}")
    return result


class TestPrepare:
    def test_synthetic_corpus(self, isolated_home: Path) -> None:
        result = _invoke("prepare", "--name", "t", "--synthetic", "400", "--vocab", "300")
        assert result.exit_code == 0, plain(result.stdout)
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
        assert result.exit_code == 0, plain(result.stdout)
        meta = json.loads((isolated_home / "datasets/files/tokens/meta.json").read_text())
        assert meta["splits"][0]["tokens"] > 0

    def test_requires_exactly_one_input(self) -> None:
        both = _invoke("prepare", "--name", "x", "--source", ".", "--synthetic", "10")
        assert both.exit_code == 1
        assert "exactly one" in plain(both.stdout)

        neither = _invoke("prepare", "--name", "x")
        assert neither.exit_code == 1

    def test_missing_source(self) -> None:
        result = _invoke("prepare", "--name", "x", "--source", "/nonexistent/path")
        assert result.exit_code == 1
        assert "does not exist" in plain(result.stdout)

    def test_source_with_no_usable_files(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "photo.png").write_bytes(b"\x89PNG")
        result = _invoke("prepare", "--name", "x", "--source", str(empty))
        assert result.exit_code == 1
        assert "no .txt" in plain(result.stdout)

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
        assert result.exit_code == 0, plain(result.stdout)
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
        assert "final loss" in plain(result.stdout)
        assert "tok/s" in plain(result.stdout)

    def test_unknown_dataset(self) -> None:
        result = _invoke("train", "--data", "nope", "--steps", "2")
        assert result.exit_code == 1
        assert "prepare" in plain(result.stdout)

    def test_size_and_depth_are_mutually_exclusive(self) -> None:
        result = _invoke("train", "--data", "d", "--depth", "2", "--size", "d12", "--steps", "2")
        assert result.exit_code == 1
        assert "not both" in plain(result.stdout)

    def test_sequence_longer_than_corpus_is_a_clear_error(self) -> None:
        """--force skips the memory guard so the corpus check is what is tested.

        Without it the pre-flight estimate rejects a 500k sequence first, which
        is correct behaviour but a different error from the one this covers.
        """
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
            "--force",
        )
        assert result.exit_code == 1
        assert "not enough" in plain(result.stdout) or "shorter --seq" in plain(result.stdout)

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
        assert "resumable" in plain(result.stdout)

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
        assert first.exit_code == 0, plain(first.stdout)

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
        assert second.exit_code == 0, plain(second.stdout)

        events = read_events(isolated_home / "runs/r/run.jsonl")
        starts = [e for e in events if e["event"] == "start"]
        assert len(starts) == 2
        assert starts[1]["resumed_from"] is not None


class TestChat:
    def test_requires_exactly_one_target(self) -> None:
        both = _invoke("chat", "--run", "r", "--checkpoint", "/tmp")
        assert both.exit_code == 1
        assert "exactly one" in plain(both.stdout)

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
        assert result.exit_code == 0, plain(result.stdout)
        assert "params on" in plain(result.stdout)


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
        assert result.exit_code == 0, plain(result.stdout)
        assert "tok/s" in plain(result.stdout)

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
        payload = json.loads(plain(result.stdout))
        assert payload["tokens_per_second"] > 0
        assert payload["params"] > 0
        assert payload["device"] == "cpu"
        assert payload["chinchilla_hours"] > 0

    def test_defaults_to_a_named_size(self) -> None:
        result = _invoke("bench", "--help")
        assert result.exit_code == 0
        assert "--size" in plain(result.stdout)


class TestDemo:
    def test_runs_end_to_end_and_cleans_up(self, isolated_home: Path) -> None:
        result = _invoke("demo", "--steps", "4", "--documents", "400", "--device", "cpu")
        assert result.exit_code == 0, plain(result.stdout)
        assert "samples" in plain(result.stdout)
        # Cleanup is the default, so nothing should be left behind.
        assert not (isolated_home / "datasets/demo").exists()
        assert not (isolated_home / "runs/demo").exists()

    def test_keep_retains_artifacts(self, isolated_home: Path) -> None:
        result = _invoke("demo", "--steps", "4", "--documents", "400", "--device", "cpu", "--keep")
        assert result.exit_code == 0, plain(result.stdout)
        assert (isolated_home / "runs/demo/latest/config.json").is_file()


class TestHelp:
    @pytest.mark.parametrize(
        "command", ["prepare", "train", "adapt", "chat", "bench", "demo", "doctor"]
    )
    def test_every_command_has_help(self, command: str) -> None:
        result = _invoke(command, "--help")
        assert result.exit_code == 0
        assert command in plain(result.stdout).lower() or "Usage" in plain(result.stdout)

    def test_root_lists_commands(self) -> None:
        result = _invoke()
        for command in ("prepare", "train", "adapt", "chat", "bench", "demo", "doctor"):
            assert command in plain(result.stdout)


class TestMix:
    def test_create_and_show(self, isolated_home: Path) -> None:
        created = _invoke("mix", "create", "--name", "b", "--add", "new:0.9", "--replay", "old:0.1")
        assert created.exit_code == 0, plain(created.stdout)
        assert (isolated_home / "mixtures/b/v1.json").is_file()

        shown = _invoke("mix", "show", "b")
        assert shown.exit_code == 0
        assert "90.0%" in plain(shown.stdout)
        assert "replay" in plain(shown.stdout)

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
        assert "60.0%" in plain(shown.stdout)
        assert "replay share  25%" in plain(shown.stdout)

    def test_create_requires_a_component(self) -> None:
        result = _invoke("mix", "create", "--name", "empty")
        assert result.exit_code == 1
        assert "at least one" in plain(result.stdout)

    def test_malformed_spec_is_rejected(self) -> None:
        result = _invoke("mix", "create", "--name", "b", "--add", "no-weight-here")
        assert result.exit_code == 1
        assert "cannot parse" in plain(result.stdout)

    def test_add_creates_a_new_version_with_lineage(self, isolated_home: Path) -> None:
        assert _invoke("mix", "create", "--name", "b", "--add", "new:0.9").exit_code == 0
        added = _invoke("mix", "add", "b", "--replay", "old:0.3", "--note", "more replay")
        assert added.exit_code == 0, plain(added.stdout)
        assert "parent     v1" in plain(added.stdout)
        assert (isolated_home / "mixtures/b/v2.json").is_file()
        # v1 must survive unchanged; that is what makes a past run reproducible.
        assert (isolated_home / "mixtures/b/v1.json").is_file()

        shown = _invoke("mix", "show", "b")
        assert "lineage" in plain(shown.stdout)
        assert "more replay" in plain(shown.stdout)

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
        assert "at least one" in plain(result.stdout)

    def test_add_to_unknown_mixture(self) -> None:
        result = _invoke("mix", "add", "ghost", "--add", "a:1")
        assert result.exit_code == 1
        assert "no mixture named" in plain(result.stdout)

    def test_show_unknown_mixture(self) -> None:
        result = _invoke("mix", "show", "ghost")
        assert result.exit_code == 1

    def test_list_empty_and_populated(self) -> None:
        empty = _invoke("mix", "list")
        assert empty.exit_code == 0
        assert "No mixtures yet" in plain(empty.stdout)

        _invoke("mix", "create", "--name", "one", "--add", "a:1")
        _invoke("mix", "create", "--name", "two", "--add", "b:1")
        listed = _invoke("mix", "list")
        assert "one" in plain(listed.stdout)
        assert "two" in plain(listed.stdout)


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
        assert result.exit_code == 0, plain(result.stdout)
        assert "mixture" in plain(result.stdout)
        assert "replay share 20%" in plain(result.stdout)
        assert "per component" in plain(result.stdout)

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
        assert result.exit_code == 0, plain(result.stdout)
        assert "no component is marked as replay" in plain(result.stdout)

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
        assert "different tokenizers" in plain(result.stdout)

    def test_data_and_mix_are_mutually_exclusive(self) -> None:
        result = _invoke("train", "--data", "alpha", "--mix", "b", "--steps", "2")
        assert result.exit_code == 1
        assert "exactly one" in plain(result.stdout)

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
        assert result.exit_code == 0, plain(result.stdout)
        # v1 is a single component, so no mixture line is printed.
        assert "replay share" not in plain(result.stdout)


class TestMemoryGuard:
    """The pre-flight check that stops a run before it OOMs partway through."""

    @pytest.fixture(autouse=True)
    def _stable_memory_budget(self) -> None:
        """Opt out of the module-wide pin: this class is what tests the budget.

        Everywhere else a pinned budget stops ambient RAM deciding unrelated
        outcomes. Here the real one is the subject.
        """

    @pytest.fixture(autouse=True)
    def prepared(self, isolated_home: Path) -> None:
        assert (
            _invoke("prepare", "--name", "d", "--synthetic", "600", "--vocab", "300").exit_code == 0
        )

    def test_reports_the_estimate_before_training(self) -> None:
        result = _invoke(
            "train",
            "--data",
            "d",
            "--name",
            "r",
            "--depth",
            "1",
            "--steps",
            "2",
            "--batch",
            "2",
            "--seq",
            "32",
            "--device",
            "cpu",
        )
        assert result.exit_code == 0, plain(result.stdout)
        assert "memory" in plain(result.stdout)
        assert "GiB needed of" in plain(result.stdout)

    def test_refuses_a_configuration_that_cannot_fit(self) -> None:
        result = _invoke(
            "train",
            "--data",
            "d",
            "--name",
            "r",
            "--depth",
            "64",
            "--steps",
            "2",
            "--batch",
            "64",
            "--seq",
            "8192",
            "--device",
            "cpu",
        )
        assert result.exit_code == 1
        out = plain(result.stdout)
        assert "needs about" in out
        assert "is available" in out

    def test_the_refusal_offers_a_concrete_fix(self) -> None:
        result = _invoke(
            "train",
            "--data",
            "d",
            "--name",
            "r",
            "--depth",
            "48",
            "--steps",
            "2",
            "--batch",
            "32",
            "--seq",
            "4096",
            "--device",
            "cpu",
        )
        assert result.exit_code == 1
        out = plain(result.stdout)
        assert "try one of" in out
        # Checkpointing is off by default, so it is the largest single saving.
        assert "--grad-checkpoint" in out

    def test_force_overrides_the_refusal(self) -> None:
        """The estimate is conservative, so there has to be a way past it."""
        result = _invoke(
            "train",
            "--data",
            "d",
            "--name",
            "r",
            "--depth",
            "10",
            "--steps",
            "1",
            "--batch",
            "8",
            "--seq",
            "256",
            "--device",
            "cpu",
            "--force",
        )
        # Either it trained, or it failed for a real reason rather than the guard.
        assert "--force given" in plain(result.stdout) or result.exit_code == 0

    def test_guard_judges_a_cpu_run_against_ram_not_vram(self) -> None:
        """A GPU on the box is irrelevant when --device cpu was asked for."""
        from bloomery.capability import check_fit
        from bloomery.probe.types import (
            CpuInfo,
            DiskInfo,
            GpuInfo,
            HostInfo,
            HostReport,
            MemoryInfo,
            Platform,
            Vendor,
        )

        gib = 1024**3
        report = HostReport(
            host=HostInfo(
                platform=Platform.LINUX,
                system="Linux",
                release="6.8",
                machine="x86_64",
                python_version="3.12",
            ),
            cpu=CpuInfo(physical_cores=8, logical_cores=16),
            memory=MemoryInfo(total=4 * gib, available=4 * gib),
            disk=DiskInfo(path="/tmp", total=100 * gib, free=90 * gib),
            gpus=[GpuInfo(index=0, vendor=Vendor.NVIDIA, name="A100", vram_total=80 * gib)],
        )
        from bloomery.capability import LADDER_BY_KEY

        on_gpu = check_fit(report, LADDER_BY_KEY["1b"], batch=4, seq=1024, device_type="cuda")
        on_cpu = check_fit(report, LADDER_BY_KEY["1b"], batch=4, seq=1024, device_type="cpu")
        assert on_gpu.fits
        assert not on_cpu.fits


class TestAdapt:
    """Continuing a model that already exists.

    The base model is a real from-scratch run rather than a download: bloomery
    checkpoints are ordinary Hugging Face directories, so one serves as "someone
    else's model" without the suite ever touching the network.
    """

    @pytest.fixture
    def base(self, isolated_home: Path) -> Path:
        """A trained checkpoint, and a corpus packed with its tokenizer."""
        assert (
            _invoke("prepare", "--name", "c", "--synthetic", "400", "--vocab", "300").exit_code == 0
        )
        assert (
            _invoke(
                "train",
                "--data",
                "c",
                "--name",
                "b",
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
        checkpoint = isolated_home / "runs/b/latest"
        assert (
            _invoke(
                "prepare",
                "--name",
                "mine",
                "--synthetic",
                "400",
                "--tokenizer",
                str(checkpoint),
            ).exit_code
            == 0
        )
        return checkpoint

    def test_lora_writes_adapters_not_a_whole_model(self, base: Path, isolated_home: Path) -> None:
        """The point of adapters: a few megabytes of diff, not a copy of the base."""
        result = _invoke(
            "adapt",
            "--from",
            str(base),
            "--data",
            "mine",
            "--name",
            "a",
            "--steps",
            "4",
            "--batch",
            "2",
            "--seq",
            "32",
            "--device",
            "cpu",
        )
        assert result.exit_code == 0, plain(result.stdout)
        out = isolated_home / "runs/a/latest"
        assert (out / "adapter_config.json").is_file()
        assert (out / "adapter_model.safetensors").is_file()
        assert not (out / "model.safetensors").is_file()

    def test_full_adaptation_writes_a_whole_model(self, base: Path, isolated_home: Path) -> None:
        result = _invoke(
            "adapt",
            "--from",
            str(base),
            "--data",
            "mine",
            "--name",
            "f",
            "--method",
            "full",
            "--steps",
            "4",
            "--batch",
            "2",
            "--seq",
            "32",
            "--device",
            "cpu",
        )
        assert result.exit_code == 0, plain(result.stdout)
        out = isolated_home / "runs/f/latest"
        assert (out / "config.json").is_file()
        assert (out / "model.safetensors").is_file()

    def test_a_corpus_the_model_cannot_read_is_refused(self, base: Path) -> None:
        """The guard the whole feature rests on.

        Token ids from another tokenizer address unrelated symbols in this
        model's embedding table. Nothing downstream raises: the run trains, the
        loss settles somewhere unremarkable, and the model comes out worse than
        it went in with no indication why.
        """
        assert (
            _invoke("prepare", "--name", "alien", "--synthetic", "400", "--vocab", "500").exit_code
            == 0
        )

        result = _invoke(
            "adapt",
            "--from",
            str(base),
            "--data",
            "alien",
            "--name",
            "x",
            "--steps",
            "2",
            "--batch",
            "2",
            "--seq",
            "32",
            "--device",
            "cpu",
        )
        assert result.exit_code == 1
        output = plain(result.stdout)
        assert "different tokenizer" in output
        # The fix has to be in the message; it is not guessable from the error.
        assert "--tokenizer" in output

    def test_the_run_reports_the_base_and_the_method(self, base: Path) -> None:
        result = _invoke(
            "adapt",
            "--from",
            str(base),
            "--data",
            "mine",
            "--name",
            "r",
            "--steps",
            "2",
            "--batch",
            "2",
            "--seq",
            "32",
            "--device",
            "cpu",
        )
        output = plain(result.stdout)
        assert "base" in output
        assert "LoRA" in output

    def test_an_unknown_method_is_refused(self, base: Path) -> None:
        result = _invoke(
            "adapt",
            "--from",
            str(base),
            "--data",
            "mine",
            "--name",
            "x",
            "--method",
            "sideways",
        )
        assert result.exit_code == 1
        assert "unknown method" in plain(result.stdout)

    def test_a_missing_model_is_reported_clearly(self, isolated_home: Path) -> None:
        assert (
            _invoke("prepare", "--name", "c", "--synthetic", "400", "--vocab", "300").exit_code == 0
        )
        result = _invoke(
            "adapt",
            "--from",
            str(isolated_home / "nope"),
            "--data",
            "c",
            "--name",
            "x",
            "--steps",
            "2",
            "--device",
            "cpu",
        )
        assert result.exit_code == 1

    def test_chat_can_open_what_a_lora_run_produced(self, base: Path) -> None:
        """Adapters are not a model, so loading one means loading its base too."""
        assert (
            _invoke(
                "adapt",
                "--from",
                str(base),
                "--data",
                "mine",
                "--name",
                "a",
                "--steps",
                "4",
                "--batch",
                "2",
                "--seq",
                "32",
                "--device",
                "cpu",
            ).exit_code
            == 0
        )
        result = _invoke(
            "chat", "--run", "a", "--prompt", "hello", "--max-new-tokens", "4", "--device", "cpu"
        )
        assert result.exit_code == 0, plain(result.stdout)

    def test_resuming_continues_rather_than_restarting(
        self, base: Path, isolated_home: Path
    ) -> None:
        for steps in ("4", "8"):
            args = [
                "adapt",
                "--from",
                str(base),
                "--data",
                "mine",
                "--name",
                "a",
                "--steps",
                steps,
                "--batch",
                "2",
                "--seq",
                "32",
                "--device",
                "cpu",
            ]
            if steps == "8":
                args.append("--resume")
            assert _invoke(*args).exit_code == 0

        events = [
            json.loads(line)
            for line in (isolated_home / "runs/a/run.jsonl").read_text().splitlines()
        ]
        finished = [e["step"] for e in events if e["event"] == "done"]
        assert finished == [4, 8], finished


class TestSupervisedFineTuning:
    """SFT has no command of its own: the dataset carries the objective.

    `prepare --chat` writes a completion mask, and `adapt` trains on it exactly
    as it trains on anything else.
    """

    TEMPLATE = (
        "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
    )

    @pytest.fixture
    def instruct(self, isolated_home: Path) -> Path:
        """A checkpoint whose tokenizer knows how a conversation is laid out."""
        from transformers import AutoTokenizer

        assert (
            _invoke("prepare", "--name", "c", "--synthetic", "500", "--vocab", "400").exit_code == 0
        )
        assert (
            _invoke(
                "train",
                "--data",
                "c",
                "--name",
                "b",
                "--depth",
                "1",
                "--steps",
                "2",
                "--batch",
                "4",
                "--seq",
                "32",
                "--device",
                "cpu",
            ).exit_code
            == 0
        )
        checkpoint = isolated_home / "runs/b/latest"
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
        tokenizer.chat_template = self.TEMPLATE
        tokenizer.save_pretrained(str(checkpoint))
        return checkpoint

    def _corpus(self, path: Path, count: int = 60) -> Path:
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"ask about {i}"},
                            {"role": "assistant", "content": f"the reply concerning {i} is here"},
                        ]
                    }
                )
                for i in range(count)
            ),
            encoding="utf-8",
        )
        return path

    def test_packs_a_masked_dataset_and_says_how_much(
        self, instruct: Path, isolated_home: Path, tmp_path: Path
    ) -> None:
        corpus = self._corpus(tmp_path / "chat.jsonl")
        result = _invoke(
            "prepare",
            "--name",
            "conv",
            "--source",
            str(corpus),
            "--chat",
            "--tokenizer",
            str(instruct),
        )
        assert result.exit_code == 0, plain(result.stdout)
        assert (isolated_home / "datasets/conv/tokens/train.mask.bin").is_file()
        # The share is the number that tells a user their data was understood.
        assert "responses" in plain(result.stdout)

    def test_adapt_trains_on_it_with_no_new_command(
        self, instruct: Path, isolated_home: Path, tmp_path: Path
    ) -> None:
        corpus = self._corpus(tmp_path / "chat.jsonl")
        assert (
            _invoke(
                "prepare",
                "--name",
                "conv",
                "--source",
                str(corpus),
                "--chat",
                "--tokenizer",
                str(instruct),
            ).exit_code
            == 0
        )
        result = _invoke(
            "adapt",
            "--from",
            str(instruct),
            "--data",
            "conv",
            "--name",
            "sft",
            "--steps",
            "4",
            "--batch",
            "2",
            "--seq",
            "64",
            "--device",
            "cpu",
        )
        assert result.exit_code == 0, plain(result.stdout)
        assert (isolated_home / "runs/sft/latest/adapter_config.json").is_file()

    def test_chat_data_sent_to_the_plain_path_says_what_it_is(
        self, isolated_home: Path, tmp_path: Path
    ) -> None:
        """It used to read as zero documents and die naming neither cause nor fix."""
        corpus = self._corpus(tmp_path / "chat.jsonl", count=5)
        result = _invoke("prepare", "--name", "oops", "--source", str(corpus))
        assert result.exit_code == 1
        output = plain(result.stdout)
        assert "conversation data" in output
        assert "--chat" in output

    def test_chat_needs_the_model_it_is_for(self, tmp_path: Path) -> None:
        corpus = self._corpus(tmp_path / "chat.jsonl", count=5)
        result = _invoke("prepare", "--name", "x", "--source", str(corpus), "--chat")
        assert result.exit_code == 1
        assert "--tokenizer" in plain(result.stdout)

    def test_a_blend_of_conversations_and_prose_trains(
        self, instruct: Path, isolated_home: Path, tmp_path: Path
    ) -> None:
        """Replay against forgetting: the prose half is trained on in full.

        This is why the mask lives in the dataset rather than in a command — a
        component with no mask simply has none.
        """
        corpus = self._corpus(tmp_path / "chat.jsonl")
        assert (
            _invoke(
                "prepare",
                "--name",
                "conv",
                "--source",
                str(corpus),
                "--chat",
                "--tokenizer",
                str(instruct),
            ).exit_code
            == 0
        )
        assert (
            _invoke(
                "prepare",
                "--name",
                "prose",
                "--synthetic",
                "400",
                "--tokenizer",
                str(instruct),
            ).exit_code
            == 0
        )
        assert (
            _invoke(
                "mix", "create", "--name", "both", "--add", "conv:0.7", "--replay", "prose:0.3"
            ).exit_code
            == 0
        )
        result = _invoke(
            "adapt",
            "--from",
            str(instruct),
            "--mix",
            "both",
            "--name",
            "blended",
            "--steps",
            "4",
            "--batch",
            "2",
            "--seq",
            "64",
            "--device",
            "cpu",
        )
        assert result.exit_code == 0, plain(result.stdout)

    def _pairs(self, path: Path, count: int = 60) -> Path:
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "prompt": f"ask about {i}",
                        "chosen": f"the careful reply concerning {i}",
                        "rejected": f"nope {i}",
                    }
                )
                for i in range(count)
            ),
            encoding="utf-8",
        )
        return path

    def _prepare_pairs(self, instruct: Path, tmp_path: Path, name: str = "pref") -> Any:
        return _invoke(
            "prepare",
            "--name",
            name,
            "--source",
            str(self._pairs(tmp_path / f"{name}.jsonl")),
            "--preference",
            "--tokenizer",
            str(instruct),
        )

    def test_preference_pairs_pack_and_report_their_count(
        self, instruct: Path, isolated_home: Path, tmp_path: Path
    ) -> None:
        result = self._prepare_pairs(instruct, tmp_path)
        assert result.exit_code == 0, plain(result.stdout)
        for part in ("prompt", "chosen", "rejected", "index"):
            assert (isolated_home / f"datasets/pref/tokens/train.{part}.bin").is_file()
        assert "pairs" in plain(result.stdout)

    def test_adapt_trains_dpo_on_them_with_no_new_command(
        self, instruct: Path, isolated_home: Path, tmp_path: Path
    ) -> None:
        """The same promise SFT keeps: the dataset carries the objective."""
        assert self._prepare_pairs(instruct, tmp_path).exit_code == 0
        result = _invoke(
            "adapt",
            "--from",
            str(instruct),
            "--data",
            "pref",
            "--name",
            "dpo",
            # Enough steps to reach the default logging interval, and evaluation
            # often enough that an eval event is actually emitted — otherwise the
            # assertions below pass over an empty list.
            "--steps",
            "10",
            "--eval-every",
            "5",
            "--batch",
            "2",
            "--seq",
            "64",
            "--device",
            "cpu",
        )
        assert result.exit_code == 0, plain(result.stdout)
        assert (isolated_home / "runs/dpo/latest/adapter_config.json").is_file()
        output = plain(result.stdout)
        assert "preference" in output, "the objective in play should be stated"
        # Two sequences go through the model per pair, so the memory pre-flight
        # is sized at twice the batch. Nominal sizing approves a run that dies on
        # step one, and the same expression is what is reported here.
        assert "4 sequences per step" in output

        events = [
            json.loads(line)
            for line in (isolated_home / "runs/dpo/run.jsonl").read_text().splitlines()
            if line.strip()
        ]
        steps = [e for e in events if e["event"] == "step"]
        assert steps, "no step was logged"
        for event in steps:
            # The two keys the progress display indexes unconditionally.
            assert "loss" in event and "tokens_per_second" in event
            for key in ("reward_margin", "reward_accuracy", "reward_chosen", "reward_rejected"):
                assert key in event, f"{key} missing from the step event"
        evals = [e for e in events if e["event"] == "eval"]
        assert evals, "no evaluation ran, so the assertion below proves nothing"
        for event in evals:
            # A preference loss is not a log-likelihood, so exp() of it is not a
            # perplexity — and 2.0 is exactly what a fresh run would report.
            assert event["perplexity"] is None
            assert "val_reward_accuracy" in event

    def test_full_weight_preference_training_is_refused_by_name(
        self, instruct: Path, tmp_path: Path
    ) -> None:
        assert self._prepare_pairs(instruct, tmp_path).exit_code == 0
        result = _invoke(
            "adapt", "--from", str(instruct), "--data", "pref", "--name", "x", "--method", "full"
        )
        assert result.exit_code == 1
        output = plain(result.stdout)
        assert "--method lora" in output
        assert "second copy" in output

    def test_preference_training_from_an_adapter_checkpoint_is_refused(
        self, instruct: Path, isolated_home: Path, tmp_path: Path
    ) -> None:
        """Its reference would be the model from before that adapter was trained.

        Not a subtle degradation: the objective would pull the model back toward
        undoing its own fine-tuning, while reporting a perfectly ordinary loss.
        """
        corpus = self._corpus(tmp_path / "chat.jsonl")
        assert (
            _invoke(
                "prepare",
                "--name",
                "conv",
                "--source",
                str(corpus),
                "--chat",
                "--tokenizer",
                str(instruct),
            ).exit_code
            == 0
        )
        assert (
            _invoke(
                "adapt",
                "--from",
                str(instruct),
                "--data",
                "conv",
                "--name",
                "sft",
                "--steps",
                "2",
                "--batch",
                "2",
                "--seq",
                "64",
                "--device",
                "cpu",
            ).exit_code
            == 0
        )
        assert self._prepare_pairs(instruct, tmp_path).exit_code == 0

        result = _invoke(
            "adapt",
            "--from",
            str(isolated_home / "runs/sft/latest"),
            "--data",
            "pref",
            "--name",
            "y",
        )
        assert result.exit_code == 1
        output = plain(result.stdout)
        assert "adapters" in output
        assert "export" in output, "the message should name the way out"

    def test_blending_preference_pairs_with_prose_is_refused(
        self, instruct: Path, tmp_path: Path
    ) -> None:
        """There is no row shape that is both, so a blend cannot be approximated."""
        assert self._prepare_pairs(instruct, tmp_path).exit_code == 0
        assert (
            _invoke(
                "prepare", "--name", "prose", "--synthetic", "400", "--tokenizer", str(instruct)
            ).exit_code
            == 0
        )
        assert (
            _invoke(
                "mix", "create", "--name", "mixed", "--add", "pref:0.5", "--add", "prose:0.5"
            ).exit_code
            == 0
        )
        result = _invoke("adapt", "--from", str(instruct), "--mix", "mixed", "--name", "z")
        assert result.exit_code == 1
        assert "preference pairs with ordinary text" in plain(result.stdout)

    def test_training_from_scratch_on_preference_pairs_is_refused(
        self, instruct: Path, tmp_path: Path
    ) -> None:
        """There is nothing to be preferred relative to yet."""
        assert self._prepare_pairs(instruct, tmp_path).exit_code == 0
        result = _invoke("train", "--data", "pref", "--name", "scratch", "--depth", "1")
        assert result.exit_code == 1
        assert "preference" in plain(result.stdout)

    def test_preference_data_sent_to_the_plain_path_says_what_it_is(self, tmp_path: Path) -> None:
        result = _invoke(
            "prepare", "--name", "oops", "--source", str(self._pairs(tmp_path / "p.jsonl", 5))
        )
        assert result.exit_code == 1
        output = plain(result.stdout)
        assert "preference data" in output
        assert "--preference" in output

    def test_chat_prompts_through_the_template_it_was_trained_with(
        self, instruct: Path, tmp_path: Path
    ) -> None:
        """Asked as plain text, a fine-tuned model is asked in a shape it never saw."""
        from bloomery.generate import as_prompted, load

        loaded = load(instruct, device="cpu")
        templated = as_prompted(loaded.tokenizer, "hello")
        assert "<|user|>" in templated and "<|assistant|>" in templated
        assert as_prompted(loaded.tokenizer, "hello", raw=True) == "hello"

    def test_the_raw_flag_reaches_generation(self, instruct: Path) -> None:
        """Exercised through the command, not through as_prompted alone.

        The unit test above proves the templating function is right; it says
        nothing about whether `chat` still passes the flag to it. Both call
        sites can be dropped in a refactor with every other test still green.
        """
        import bloomery.generate as generate_module

        seen: list[bool] = []
        real = generate_module.complete

        def record(loaded, prompt, config=None, *, raw=False):  # noqa: ANN001, ANN202
            seen.append(raw)
            return real(loaded, prompt, config, raw=raw)

        with mock.patch.object(generate_module, "complete", record):
            assert (
                _invoke(
                    "chat",
                    "--checkpoint",
                    str(instruct),
                    "--prompt",
                    "hello",
                    "--max-new-tokens",
                    "2",
                    "--device",
                    "cpu",
                    "--raw",
                ).exit_code
                == 0
            )
            assert (
                _invoke(
                    "chat",
                    "--checkpoint",
                    str(instruct),
                    "--prompt",
                    "hello",
                    "--max-new-tokens",
                    "2",
                    "--device",
                    "cpu",
                ).exit_code
                == 0
            )

        assert seen == [True, False], seen


class TestExport:
    """Writing a checkpoint out for llama.cpp and Ollama."""

    @pytest.fixture
    def trained(self, isolated_home: Path) -> Path:
        assert (
            _invoke("prepare", "--name", "c", "--synthetic", "400", "--vocab", "300").exit_code == 0
        )
        assert (
            _invoke(
                "train",
                "--data",
                "c",
                "--name",
                "r",
                "--depth",
                "1",
                "--steps",
                "2",
                "--batch",
                "4",
                "--seq",
                "32",
                "--device",
                "cpu",
            ).exit_code
            == 0
        )
        return isolated_home / "runs/r/latest"

    def test_writes_a_gguf_and_a_modelfile(self, trained: Path, isolated_home: Path) -> None:
        result = _invoke("export", "--run", "r")
        assert result.exit_code == 0, plain(result.stdout)
        out = isolated_home / "exports/r"
        assert (out / "model.gguf").is_file()
        assert (out / "Modelfile").is_file()

    @pytest.mark.parametrize("quantize", ["f16", "q8_0", "q4_0"])
    def test_each_format_writes(self, trained: Path, isolated_home: Path, quantize: str) -> None:
        result = _invoke("export", "--run", "r", "--quantize", quantize, "--name", quantize)
        assert result.exit_code == 0, plain(result.stdout)
        assert (isolated_home / f"exports/{quantize}/model.gguf").is_file()

    def test_it_reports_the_context_it_was_trained_at(self, trained: Path) -> None:
        """The number most likely to surprise: a runtime treats it as a hard limit."""
        result = _invoke("export", "--run", "r", "--json")
        assert result.exit_code == 0, plain(result.stdout)
        payload = json.loads(plain(result.stdout))
        assert payload["context_length"] == 32
        assert payload["architecture"] == "llama"

    def test_an_unknown_format_is_refused(self, trained: Path) -> None:
        result = _invoke("export", "--run", "r", "--quantize", "q4_k_m")
        assert result.exit_code == 1
        assert "unknown quantization" in plain(result.stdout)

    def test_a_missing_run_is_reported(self) -> None:
        result = _invoke("export", "--run", "nope")
        assert result.exit_code == 1

    def test_exactly_one_target(self, trained: Path) -> None:
        assert _invoke("export").exit_code == 1
        assert _invoke("export", "--run", "r", "--checkpoint", str(trained)).exit_code == 1

    def test_a_lora_checkpoint_is_merged_before_export(self, isolated_home: Path) -> None:
        """GGUF has no notion of an adapter.

        Without the merge the export would silently be of the untouched base:
        it writes, it loads, and none of the fine-tuning is in it.
        """
        pytest.importorskip("peft", reason="adapters need the adapt extra")
        from transformers import AutoTokenizer

        assert (
            _invoke("prepare", "--name", "c", "--synthetic", "400", "--vocab", "300").exit_code == 0
        )
        assert (
            _invoke(
                "train",
                "--data",
                "c",
                "--name",
                "b",
                "--depth",
                "1",
                "--steps",
                "2",
                "--batch",
                "4",
                "--seq",
                "32",
                "--device",
                "cpu",
            ).exit_code
            == 0
        )
        base = isolated_home / "runs/b/latest"
        AutoTokenizer.from_pretrained(str(base)).save_pretrained(str(base))
        assert (
            _invoke(
                "adapt",
                "--from",
                str(base),
                "--data",
                "c",
                "--name",
                "a",
                "--steps",
                "4",
                "--batch",
                "2",
                "--seq",
                "32",
                "--device",
                "cpu",
            ).exit_code
            == 0
        )

        assert _invoke("export", "--run", "a", "--name", "adapted").exit_code == 0
        assert _invoke("export", "--checkpoint", str(base), "--name", "plain").exit_code == 0

        from gguf import GGUFReader

        def tensor(name: str, export: str):  # noqa: ANN202
            reader = GGUFReader(str(isolated_home / f"exports/{export}/model.gguf"))
            return next(t.data for t in reader.tensors if t.name == name)

        merged = tensor("blk.0.attn_q.weight", "adapted")
        plain_base = tensor("blk.0.attn_q.weight", "plain")
        assert not (merged == plain_base).all(), "the adapters were not folded in"

    def test_a_half_written_export_is_not_left_behind(
        self, trained: Path, isolated_home: Path
    ) -> None:
        """A staging directory must not survive a failure looking like an export."""
        result = _invoke("export", "--run", "r", "--name", "ok")
        assert result.exit_code == 0, plain(result.stdout)
        assert not (isolated_home / "exports/ok.tmp").exists()
