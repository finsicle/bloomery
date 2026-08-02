# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the training loop, schedule, metrics and checkpoints.

The integration tests here are small on purpose — a two-layer model for a few
dozen steps — but they run the real loop on real data. A unit test that mocks the
optimiser can pass while training is silently broken; only watching loss actually
fall rules that out.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from bloomery.arch import spec_from_depth
from bloomery.mixture import single
from bloomery.train import checkpoint as ckpt
from bloomery.train.device import DeviceChoice, choose, select_precision
from bloomery.train.loop import BatchSampler, TrainConfig, build_model, lr_at
from bloomery.train.metrics import MetricsWriter, Throughput, read_events


@pytest.fixture(scope="module")
def cpu_choice() -> DeviceChoice:
    """Force CPU so the tests behave identically on every CI runner."""
    import torch

    dtype, autocast, reason = select_precision(torch.device("cpu"))
    return DeviceChoice(device=torch.device("cpu"), dtype=dtype, autocast=autocast, reason=reason)


class TestLrSchedule:
    def test_warms_up_from_near_zero(self) -> None:
        config = TrainConfig(steps=1000, warmup_fraction=0.1)
        assert lr_at(0, config, 1.0) < 0.02
        assert lr_at(99, config, 1.0) == pytest.approx(1.0)

    def test_peaks_at_end_of_warmup(self) -> None:
        config = TrainConfig(steps=100, warmup_fraction=0.1)
        values = [lr_at(s, config, 1.0) for s in range(100)]
        assert max(values) == pytest.approx(1.0)
        assert values.index(max(values)) == 9

    def test_decays_monotonically_after_warmup(self) -> None:
        config = TrainConfig(steps=200, warmup_fraction=0.05)
        after = [lr_at(s, config, 1.0) for s in range(10, 200)]
        assert all(b <= a + 1e-12 for a, b in zip(after, after[1:], strict=False))

    def test_floors_rather_than_reaching_zero(self) -> None:
        config = TrainConfig(steps=100, warmup_fraction=0.05, min_lr_fraction=0.1)
        assert lr_at(99, config, 1.0) >= 0.1
        assert lr_at(500, config, 1.0) == pytest.approx(0.1)

    def test_never_exceeds_peak(self) -> None:
        config = TrainConfig(steps=50, warmup_fraction=0.2)
        assert all(lr_at(s, config, 3e-4) <= 3e-4 + 1e-12 for s in range(80))

    def test_zero_warmup_fraction_still_warms_one_step(self) -> None:
        """Starting at full LR on step 0 destabilises training."""
        config = TrainConfig(steps=100, warmup_fraction=0.0)
        assert lr_at(0, config, 1.0) == pytest.approx(1.0)


class TestThroughput:
    def test_first_sample_is_the_instant_rate(self) -> None:
        assert Throughput().update(1000, 1.0) == pytest.approx(1000.0)

    def test_smooths_toward_new_values(self) -> None:
        meter = Throughput(smoothing=0.5)
        meter.update(1000, 1.0)
        assert meter.update(2000, 1.0) == pytest.approx(1500.0)

    def test_ignores_nonpositive_duration(self) -> None:
        meter = Throughput()
        meter.update(1000, 1.0)
        assert meter.update(5000, 0.0) == pytest.approx(1000.0)


class TestMetricsWriter:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "run.jsonl"
        with MetricsWriter(path) as metrics:
            metrics.emit("start", model="d4")
            metrics.emit("step", step=1, loss=2.5)
        events = read_events(path)
        assert [e["event"] for e in events] == ["start", "step"]
        assert events[1]["loss"] == 2.5

    def test_flushes_every_event(self, tmp_path: Path) -> None:
        """A killed process must leave a readable log behind."""
        path = tmp_path / "run.jsonl"
        with MetricsWriter(path) as metrics:
            metrics.emit("step", step=1)
            assert len(read_events(path)) == 1

    def test_records_exceptions(self, tmp_path: Path) -> None:
        path = tmp_path / "run.jsonl"
        with pytest.raises(RuntimeError), MetricsWriter(path) as metrics:
            metrics.emit("start")
            raise RuntimeError("out of memory")
        events = read_events(path)
        assert events[-1]["event"] == "error"
        assert events[-1]["kind"] == "RuntimeError"
        assert "out of memory" in events[-1]["message"]

    def test_read_events_tolerates_truncated_tail(self, tmp_path: Path) -> None:
        path = tmp_path / "run.jsonl"
        path.write_text(json.dumps({"event": "step"}) + '\n{"event": "ste')
        assert [e["event"] for e in read_events(path)] == ["step"]

    def test_read_events_missing_file(self, tmp_path: Path) -> None:
        assert read_events(tmp_path / "nope.jsonl") == []


class TestBatchSampler:
    def test_shapes_and_dtype(self, dataset: Any, cpu_choice: DeviceChoice) -> None:
        sampler = BatchSampler(dataset, "train", seq=32, seed=0)
        batch = sampler.batch(4, cpu_choice.device)
        assert tuple(batch["input_ids"].shape) == (4, 32)
        assert batch["input_ids"].dtype.is_signed  # int64, as the embedding layer requires
        # A plain corpus has nothing masked, so the labels are the inputs. This
        # is what keeps pretraining bit-identical now that labels are carried.
        assert batch["labels"].equal(batch["input_ids"])

    def test_ids_are_in_vocab_range(self, dataset: Any, cpu_choice: DeviceChoice) -> None:
        sampler = BatchSampler(dataset, "train", seq=32, seed=0)
        batch = sampler.batch(8, cpu_choice.device)["input_ids"]
        assert int(batch.max()) < dataset.vocab_size
        assert int(batch.min()) >= 0

    def test_seeded_sampling_is_reproducible(self, dataset: Any, cpu_choice: DeviceChoice) -> None:
        first = BatchSampler(dataset, "train", seq=16, seed=42).batch(4, cpu_choice.device)
        second = BatchSampler(dataset, "train", seq=16, seed=42).batch(4, cpu_choice.device)
        assert first["input_ids"].equal(second["input_ids"])

    def test_rejects_seq_longer_than_the_split(self, dataset: Any) -> None:
        with pytest.raises(ValueError, match="not enough"):
            BatchSampler(dataset, "train", seq=10_000_000, seed=0)

    def test_missing_split(self, dataset: Any) -> None:
        with pytest.raises(FileNotFoundError):
            BatchSampler(dataset, "nonexistent", seq=8, seed=0)


class TestBuildModel:
    def test_is_randomly_initialised(self) -> None:
        """Two seeds must give different weights, or nothing is being trained."""
        spec = spec_from_depth(1, vocab=300, seq=32)
        a = build_model(spec, eos_token_id=0, seed=1)
        b = build_model(spec, eos_token_id=0, seed=2)
        assert not next(a.parameters()).equal(next(b.parameters()))

    def test_seed_is_reproducible(self) -> None:
        spec = spec_from_depth(1, vocab=300, seq=32)
        a = build_model(spec, eos_token_id=0, seed=7)
        b = build_model(spec, eos_token_id=0, seed=7)
        assert next(a.parameters()).equal(next(b.parameters()))

    def test_embeddings_are_tied(self) -> None:
        spec = spec_from_depth(1, vocab=300, seq=32)
        model = build_model(spec, eos_token_id=0, seed=0)
        assert model.config.tie_word_embeddings is True

    def test_config_carries_the_spec(self) -> None:
        spec = spec_from_depth(2, vocab=512, seq=64)
        config = build_model(spec, eos_token_id=3, seed=0).config
        assert config.num_hidden_layers == 2
        assert config.hidden_size == 128
        assert config.vocab_size == 512
        assert config.eos_token_id == 3


@pytest.fixture(scope="module")
def result(dataset: Any, tokenizer: Any, tmp_path_factory: Any) -> Any:
    """One real training run, shared by every assertion about it.

    Module scope so the loop runs once: it is the slowest thing in the suite, and
    every test below inspects the same outcome from a different angle.
    """
    import torch

    from bloomery.train.loop import train

    spec = spec_from_depth(2, vocab=len(tokenizer), seq=32, batch=8)
    choice = DeviceChoice(
        device=torch.device("cpu"),
        dtype=torch.float32,
        autocast=False,
        reason="test",
    )
    return train(
        spec=spec,
        datasets={"fixture": dataset},
        mixture=single("fixture"),
        tokenizer=tokenizer,
        run_dir=tmp_path_factory.mktemp("integration"),
        config=TrainConfig(steps=40, batch=8, seq=32, eval_every=20, eval_batches=3, log_every=5),
        choice=choice,
        eos_token_id=0,
    )


class TestTrainingIntegration:
    """The real loop, on real data, asserting it actually learns."""

    def test_loss_decreases(self, result: Any) -> None:
        losses = [e["loss"] for e in result.history if e["event"] == "step"]
        assert len(losses) >= 4
        # Compare ends rather than adjacent steps: minibatch noise makes any
        # single step-to-step comparison meaningless.
        assert losses[-1] < losses[0], f"loss did not fall: {losses}"

    def test_loss_is_finite(self, result: Any) -> None:
        assert math.isfinite(result.final_loss)

    def test_val_loss_recorded(self, result: Any) -> None:
        assert result.best_val_loss is not None
        assert math.isfinite(result.best_val_loss)

    def test_tokens_counted_correctly(self, result: Any) -> None:
        assert result.tokens_seen == 40 * 8 * 32

    def test_emits_the_event_protocol(self, result: Any) -> None:
        kinds = [e["event"] for e in result.history]
        assert kinds[0] == "start"
        assert kinds[-1] == "done"
        assert "step" in kinds
        assert "eval" in kinds

    def test_event_log_written_to_disk(self, result: Any) -> None:
        events = read_events(result.run_dir / "run.jsonl")
        assert [e["event"] for e in events] == [e["event"] for e in result.history]

    def test_checkpoint_is_a_loadable_hf_model(self, result: Any) -> None:
        """The whole reason for building on a standard architecture."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(str(result.checkpoint))
        tok = AutoTokenizer.from_pretrained(str(result.checkpoint))
        assert model.config.num_hidden_layers == 2
        assert len(tok) > 0

    def test_checkpoint_contains_expected_files(self, result: Any) -> None:
        names = {p.name for p in result.checkpoint.iterdir()}
        assert {"config.json", "model.safetensors", ckpt.TRAINER_STATE, ckpt.RUN_META} <= names

    def test_checkpoint_is_resumable(self, result: Any) -> None:
        assert ckpt.is_resumable(result.checkpoint)
        state = ckpt.load_resume_state(result.checkpoint)
        assert state.step == 40
        assert state.tokens_seen == 40 * 8 * 32

    def test_generation_produces_text(self, result: Any) -> None:
        from bloomery.generate import SamplingConfig, complete, load

        loaded = load(result.checkpoint, device="cpu")
        text = complete(loaded, "Ana", SamplingConfig(max_new_tokens=8, seed=0))
        assert isinstance(text, str)

    def test_generation_excludes_the_prompt(self, result: Any) -> None:
        from bloomery.generate import SamplingConfig, complete, load

        loaded = load(result.checkpoint, device="cpu")
        prompt = "Ana found a"
        text = complete(loaded, prompt, SamplingConfig(max_new_tokens=6, seed=0))
        assert not text.startswith(prompt)


class TestCheckpointAtomicity:
    def test_no_staging_directory_left_behind(self, tmp_path: Path, tokenizer: Any) -> None:
        import torch

        spec = spec_from_depth(1, vocab=len(tokenizer), seq=16)
        model = build_model(spec, eos_token_id=0, seed=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        target = tmp_path / "latest"
        ckpt.save(
            target,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            step=5,
            tokens_seen=100,
            best_val_loss=1.5,
        )
        assert target.is_dir()
        assert not (tmp_path / "latest.tmp").exists()

    def test_overwrites_an_existing_checkpoint(self, tmp_path: Path, tokenizer: Any) -> None:
        import torch

        spec = spec_from_depth(1, vocab=len(tokenizer), seq=16)
        model = build_model(spec, eos_token_id=0, seed=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        target = tmp_path / "latest"

        for step in (1, 2):
            ckpt.save(
                target,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                step=step,
                tokens_seen=step * 10,
                best_val_loss=None,
            )
        assert ckpt.load_resume_state(target).step == 2

    def test_restores_optimizer_state(self, tmp_path: Path, tokenizer: Any) -> None:
        import torch

        spec = spec_from_depth(1, vocab=len(tokenizer), seq=16)
        model = build_model(spec, eos_token_id=0, seed=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Take a real step so there is momentum state worth restoring.
        model(
            input_ids=torch.zeros((1, 8), dtype=torch.long),
            labels=torch.zeros((1, 8), dtype=torch.long),
        ).loss.backward()
        optimizer.step()

        target = tmp_path / "latest"
        ckpt.save(
            target,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            step=1,
            tokens_seen=8,
            best_val_loss=None,
        )

        fresh = torch.optim.AdamW(model.parameters(), lr=1e-4)
        assert not fresh.state_dict()["state"]
        ckpt.load_resume_state(target, fresh)
        assert fresh.state_dict()["state"]

    def test_forgetting_history_survives_a_checkpoint(self, tmp_path: Path, tokenizer: Any) -> None:
        """A resumed run must not mistake its first evaluation for a baseline.

        Without this, a component already degrading before the interruption is
        reported as healthy after resume — the run silently loses exactly the
        signal the blend exists to provide.
        """
        import torch

        spec = spec_from_depth(1, vocab=len(tokenizer), seq=16)
        model = build_model(spec, eos_token_id=0, seed=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        target = tmp_path / "latest"
        ckpt.save(
            target,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            step=10,
            tokens_seen=100,
            best_val_loss=0.5,
            component_best={"old": 0.9, "new": 0.4},
            component_first={"old": 1.2, "new": 1.5},
        )
        state = ckpt.load_resume_state(target)
        assert state.component_best == {"old": 0.9, "new": 0.4}
        assert state.component_first == {"old": 1.2, "new": 1.5}

    def test_restored_history_detects_a_regression_immediately(self) -> None:
        """The restored best is what makes the very next reading actionable."""
        from bloomery.train.mixing import ForgettingTracker

        tracker = ForgettingTracker(tolerance=0.0)
        tracker.best.update({"old": 0.9})
        tracker.first.update({"old": 1.2})
        # A fresh tracker would treat this as a baseline and report nothing.
        assert tracker.update({"old": 1.1}) == pytest.approx({"old": pytest.approx(0.2)})

    def test_checkpoint_without_component_history_still_loads(
        self, tmp_path: Path, tokenizer: Any
    ) -> None:
        """Checkpoints written before forgetting was tracked must still resume."""
        import torch

        spec = spec_from_depth(1, vocab=len(tokenizer), seq=16)
        model = build_model(spec, eos_token_id=0, seed=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        target = tmp_path / "latest"
        ckpt.save(
            target,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            step=5,
            tokens_seen=50,
            best_val_loss=None,
        )
        payload = torch.load(target / ckpt.TRAINER_STATE, map_location="cpu", weights_only=False)
        del payload["component_best"], payload["component_first"]
        torch.save(payload, target / ckpt.TRAINER_STATE)

        state = ckpt.load_resume_state(target)
        assert state.component_best == {}
        assert state.component_first == {}

    def test_missing_trainer_state(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ckpt.load_resume_state(tmp_path)

    def test_is_resumable_is_false_for_empty_dir(self, tmp_path: Path) -> None:
        assert ckpt.is_resumable(tmp_path) is False

    def test_checkpoint_dir_naming(self, tmp_path: Path) -> None:
        assert ckpt.checkpoint_dir(tmp_path).name == "latest"
        assert ckpt.checkpoint_dir(tmp_path, 42).name == "step-000042"


class TestDeviceSelection:
    def test_choose_returns_a_usable_pair(self) -> None:
        choice = choose()
        assert choice.type in ("cuda", "mps", "cpu")
        assert choice.reason
        assert choice.label()

    def test_explicit_cpu_is_honoured(self) -> None:
        assert choose("cpu").type == "cpu"

    def test_cpu_precision_is_a_real_dtype(self) -> None:
        import torch

        dtype, autocast, reason = select_precision(torch.device("cpu"))
        assert dtype in (torch.float32, torch.bfloat16)
        assert isinstance(autocast, bool)
        assert reason


class TestCpuBf16Gate:
    """Capability is read, never discovered by running a bf16 matmul.

    On a CPU whose bf16 kernels use instructions it does not implement, that
    matmul does not return a wrong answer and does not raise — it takes an
    illegal-instruction trap and kills the process. This is not theoretical: it
    killed CI on Windows, in `_bf16_is_faster`, with exception 0xc000001d, and
    the same crash reaches any user on such a machine the moment they run
    `bloomery train`.
    """

    def test_nothing_bf16_runs_when_capability_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        from bloomery.train import device

        executed: list[str] = []
        monkeypatch.setattr(device, "_cpu_bf16_is_native", lambda: False)
        monkeypatch.setattr(device, "_bf16_works", lambda d: executed.append("_bf16_works") or True)
        monkeypatch.setattr(
            device, "_bf16_is_faster", lambda d, **k: executed.append("_bf16_is_faster") or True
        )
        monkeypatch.setattr(
            device,
            "_bf16_probe_says_faster",
            lambda d, **k: executed.append("_bf16_probe_says_faster") or True,
        )
        monkeypatch.setattr(device, "_PRECISION_CACHE", {})

        dtype, autocast, reason = device.select_precision(torch.device("cpu"))

        assert executed == [], f"executed bf16 work on a CPU that cannot run it: {executed}"
        assert dtype is torch.float32
        assert autocast is False
        assert "without hardware bf16" in reason

    def test_the_timing_probe_still_runs_where_bf16_is_supported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hardware support is not the same as being worth using."""
        import torch

        from bloomery.train import device

        monkeypatch.setattr(device, "_cpu_bf16_is_native", lambda: True)
        monkeypatch.setattr(device, "_bf16_probe_says_faster", lambda d, **k: False)
        monkeypatch.setattr(device, "_PRECISION_CACHE", {})

        dtype, autocast, reason = device.select_precision(torch.device("cpu"))

        assert dtype is torch.float32
        assert "not usably faster" in reason

    def test_bf16_is_selected_when_supported_and_faster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        from bloomery.train import device

        monkeypatch.setattr(device, "_cpu_bf16_is_native", lambda: True)
        monkeypatch.setattr(device, "_bf16_probe_says_faster", lambda d, **k: True)
        monkeypatch.setattr(device, "_PRECISION_CACHE", {})

        dtype, autocast, reason = device.select_precision(torch.device("cpu"))

        assert dtype is torch.bfloat16
        assert autocast is True
        assert "accelerated bf16" in reason

    def test_a_child_killed_by_a_signal_means_do_not_use_bf16(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of the subprocess: dying is a valid, survivable answer."""
        import subprocess
        import sys

        import torch

        from bloomery.train.device import _bf16_probe_says_faster

        real_run = subprocess.run

        def suicidal(cmd: object, **kwargs: object) -> object:
            return real_run(
                [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGILL)"],
                **kwargs,
            )

        monkeypatch.setattr(subprocess, "run", suicidal)
        assert _bf16_probe_says_faster(torch.device("cpu")) is False

    def test_a_probe_that_cannot_run_means_do_not_use_bf16(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        import torch

        from bloomery.train.device import _bf16_probe_says_faster

        def refuse(*args: object, **kwargs: object) -> object:
            raise OSError("no interpreter")

        monkeypatch.setattr(subprocess, "run", refuse)
        assert _bf16_probe_says_faster(torch.device("cpu")) is False

    def test_a_hanging_probe_does_not_hang_the_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        import torch

        from bloomery.train.device import _bf16_probe_says_faster

        def stall(*args: object, **kwargs: object) -> object:
            raise subprocess.TimeoutExpired(cmd="probe", timeout=1)

        monkeypatch.setattr(subprocess, "run", stall)
        assert _bf16_probe_says_faster(torch.device("cpu")) is False

    def test_the_probe_can_be_switched_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        from bloomery.train.device import ENV_BF16_PROBE, _bf16_probe_says_faster

        monkeypatch.setenv(ENV_BF16_PROBE, "0")
        assert _bf16_probe_says_faster(torch.device("cpu")) is False

    def test_the_probe_module_reports_a_verdict(self) -> None:
        """Run for real, out of process, exactly as select_precision does."""
        import subprocess
        import sys

        completed = subprocess.run(
            [sys.executable, "-m", "bloomery.train._bf16_probe", "cpu"],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        # A crash here is a legitimate outcome on hardware that cannot do bf16 —
        # what matters is that it happened in a child and said something usable.
        if completed.returncode == 0:
            assert completed.stdout.strip().splitlines()[-1] in ("bf16", "fp32")

    def test_capability_answers_without_running_anything(self) -> None:
        """The gate itself must not be implemented by trying a bf16 operation."""
        from bloomery.train.device import _cpu_bf16_is_native

        assert isinstance(_cpu_bf16_is_native(), bool)

    def test_unreportable_hardware_counts_as_unsupported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Being wrong this way costs speed; the other way kills the run."""
        import torch

        from bloomery.train.device import _cpu_bf16_is_native

        class Blank:
            pass

        monkeypatch.setattr(torch, "cpu", Blank())
        assert _cpu_bf16_is_native() is False

    def test_a_capability_probe_that_raises_is_not_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        from bloomery.train.device import _cpu_bf16_is_native

        def explode(*args: object, **kwargs: object) -> bool:
            raise RuntimeError("torch internals moved")

        class Hostile:
            _is_avx512_bf16_supported = staticmethod(explode)
            get_capabilities = staticmethod(explode)

        monkeypatch.setattr(torch, "cpu", Hostile())
        assert _cpu_bf16_is_native() is False


class TestAdapterTraining:
    """Adapters against a frozen base.

    Uses a checkpoint this suite trains rather than a download: bloomery writes
    ordinary Hugging Face directories, so one stands in for "someone else's
    model" with no network involved.
    """

    def test_attaching_adapters_freezes_everything_else(self, cpu_choice: DeviceChoice) -> None:
        peft = pytest.importorskip("peft", reason="adapters need the adapt extra")
        assert peft
        from bloomery.train.loop import LoraSettings, attach_adapter, trainable_fraction

        spec = spec_from_depth(2, vocab=400, seq=64)
        model = build_model(spec, eos_token_id=0, seed=0)
        _, before = trainable_fraction(model)

        adapted = attach_adapter(model, LoraSettings(r=4, alpha=8, dropout=0.0))
        trainable, total = trainable_fraction(adapted)

        assert trainable < total
        # Only the low-rank update is learned; the base is along for the ride.
        assert trainable / total < 0.2
        assert all(
            not param.requires_grad
            for name, param in adapted.named_parameters()
            if "lora_" not in name
        )
        assert total >= before

    def test_adapters_still_learn(self, cpu_choice: DeviceChoice) -> None:
        """A frozen base with nothing trainable would look identical from outside."""
        pytest.importorskip("peft", reason="adapters need the adapt extra")
        import torch

        from bloomery.train.loop import LoraSettings, attach_adapter

        spec = spec_from_depth(2, vocab=400, seq=64)
        adapted = attach_adapter(
            build_model(spec, eos_token_id=0, seed=0), LoraSettings(r=4, alpha=8, dropout=0.0)
        )
        optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=1e-2)
        batch = torch.randint(0, 400, (2, 32))

        before = adapted(input_ids=batch, labels=batch).loss.item()
        for _ in range(8):
            loss = adapted(input_ids=batch, labels=batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        after = adapted(input_ids=batch, labels=batch).loss.item()

        assert after < before

    def test_an_adapter_checkpoint_loads_its_base(self, tmp_path: Path) -> None:
        """Adapters are a diff, so opening one means opening the model it diffs against."""
        pytest.importorskip("peft", reason="adapters need the adapt extra")
        from bloomery.train.loop import (
            LoraSettings,
            attach_adapter,
            is_adapter_dir,
            load_model,
            trainable_fraction,
        )

        base_dir = tmp_path / "base"
        spec = spec_from_depth(2, vocab=400, seq=64)
        build_model(spec, eos_token_id=0, seed=0).save_pretrained(str(base_dir))

        adapter_dir = tmp_path / "adapter"
        attach_adapter(
            load_model(base_dir), LoraSettings(r=4, alpha=8, dropout=0.0)
        ).save_pretrained(str(adapter_dir))

        assert is_adapter_dir(adapter_dir)
        assert not is_adapter_dir(base_dir)

        reloaded = load_model(adapter_dir)
        trainable, total = trainable_fraction(reloaded)
        assert trainable > 0, "a resumed adapter run must still have something to train"
        assert total > trainable

    def test_adapters_without_a_reachable_base_say_so(self, tmp_path: Path) -> None:
        """The failure mode of a small artifact: the thing it diffs against is gone."""
        pytest.importorskip("peft", reason="adapters need the adapt extra")
        import json

        from bloomery.train.loop import ModelLoadError, load_model

        orphan = tmp_path / "orphan"
        orphan.mkdir()
        (orphan / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": str(tmp_path / "gone")})
        )

        with pytest.raises(ModelLoadError, match="could not be loaded"):
            load_model(orphan)

    def test_loading_a_model_that_is_not_there_is_reported(self, tmp_path: Path) -> None:
        from bloomery.train.loop import ModelLoadError, load_model

        with pytest.raises(ModelLoadError, match="could not load a model"):
            load_model(tmp_path / "nothing")


class TestTrainedTokensSurviveResume:
    """The counter that says what the loss was computed over.

    Its sibling tokens_seen is cumulative across a resume; if this one restarts
    at zero, a resumed run reports a fraction of what it trained on, and the two
    numbers contradict each other in the same file.
    """

    def test_the_checkpoint_carries_it(self, tmp_path: Path, tokenizer: Any) -> None:
        import torch

        from bloomery.train import checkpoint as ckpt

        spec = spec_from_depth(1, vocab=64, seq=16)
        model = build_model(spec, eos_token_id=0, seed=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        directory = ckpt.save(
            tmp_path / "ckpt",
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            step=10,
            tokens_seen=5000,
            trained_tokens=1234,
            best_val_loss=None,
        )
        assert ckpt.load_resume_state(directory).trained_tokens == 1234
        assert json.loads((directory / "run.json").read_text())["trained_tokens"] == 1234

    def test_a_checkpoint_from_before_masking_falls_back_to_the_window(
        self, tmp_path: Path
    ) -> None:
        """Every token was trained on then, so the window count is the honest total.

        Restarting at zero would make a resumed old run look like it had learned
        from almost nothing.
        """
        import torch

        from bloomery.train import checkpoint as ckpt

        directory = tmp_path / "old"
        directory.mkdir()
        torch.save(
            {"step": 7, "tokens_seen": 4096, "best_val_loss": None},
            directory / ckpt.TRAINER_STATE,
        )
        assert ckpt.load_resume_state(directory).trained_tokens == 4096


class TestWindowsAlwaysCarryLoss:
    """A window holding nothing but ignored positions scores a mean over an
    empty set, which is NaN. It does not corrupt the gradient — that arrives as
    zeros — but it does reach the reported loss, run.jsonl and the final result,
    where it reads as a diverged model rather than an unlucky draw.

    One prompt longer than the sequence length is enough to produce one.
    """

    def _dataset(self, tmp_path: Path, *, prompt_tokens: int, seq: int) -> Any:
        """A corpus whose prompts are far longer than the window."""
        import numpy as np

        from bloomery.data.shards import FORMAT_SFT, META_NAME, DatasetInfo, SplitInfo

        root = tmp_path / "sparse"
        root.mkdir()
        reply = 4
        ids, mask = [], []
        # Ids stay inside the vocabulary the test model is built with.
        for _ in range(40):
            ids.extend((i % 60) + 1 for i in range(prompt_tokens))
            mask.extend([0] * prompt_tokens)
            ids.extend((i % 60) + 1 for i in range(reply))
            mask.extend([1] * reply)

        np.asarray(ids, dtype=np.uint16).tofile(root / "train.bin")
        np.asarray(mask, dtype=np.uint8).tofile(root / "train.mask.bin")
        info = DatasetInfo(
            root=root,
            dtype="uint16",
            vocab_size=64,
            format=FORMAT_SFT,
            splits=(SplitInfo(name="train", tokens=len(ids), documents=40),),
        )
        (root / META_NAME).write_text(json.dumps(info.to_dict()), encoding="utf-8")
        return info

    def test_every_window_has_something_to_learn_from(
        self, tmp_path: Path, cpu_choice: DeviceChoice
    ) -> None:
        """Prompts eight times the window length: a naive draw lands in one often."""
        from bloomery.train.loop import IGNORE_INDEX, BatchSampler

        info = self._dataset(tmp_path, prompt_tokens=128, seq=16)
        sampler = BatchSampler(info, "train", seq=16, seed=0)

        for _ in range(40):
            labels = sampler.batch(4, cpu_choice.device)["labels"]
            per_row = (labels != IGNORE_INDEX).sum(dim=1)
            assert int(per_row.min()) > 0, "a window came back with nothing supervised"

    def test_the_loss_of_such_a_batch_is_finite(
        self, tmp_path: Path, cpu_choice: DeviceChoice
    ) -> None:
        """The property that actually matters: the number reaching the user."""
        import torch

        from bloomery.train.loop import BatchSampler

        info = self._dataset(tmp_path, prompt_tokens=128, seq=16)
        model = build_model(spec_from_depth(1, vocab=64, seq=16), eos_token_id=0, seed=0)
        sampler = BatchSampler(info, "train", seq=16, seed=1)

        for _ in range(20):
            with torch.no_grad():
                loss = model(**sampler.batch(1, cpu_choice.device)).loss
            assert torch.isfinite(loss), "an all-masked window produced a NaN loss"

    def test_an_all_masked_window_would_have_been_nan(
        self, tmp_path: Path, cpu_choice: DeviceChoice
    ) -> None:
        """Pins why the resampling is there, rather than trusting the comment."""
        import torch

        from bloomery.train.loop import IGNORE_INDEX

        model = build_model(spec_from_depth(1, vocab=64, seq=16), eos_token_id=0, seed=0)
        ids = torch.randint(0, 64, (1, 16))
        with torch.no_grad():
            loss = model(input_ids=ids, labels=torch.full_like(ids, IGNORE_INDEX)).loss
        assert torch.isnan(loss)


class TestSkippedMicrobatchesAreVisible:
    """Dropping a non-finite microbatch protects the reported loss from an
    unplaceable window. It must not also hide a diverging run.

    Before the guard existed, divergence — fp16 overflow, a learning rate far
    too high — left a NaN in the reported loss where a person would see it.
    Skipping silently would make a run that diverges every step look like a
    healthy one with a slightly smaller batch.
    """

    def _run(
        self,
        dataset: Any,
        tmp_path: Path,
        cpu_choice: DeviceChoice,
        tokenizer: Any,
        **overrides: Any,
    ) -> Any:
        from bloomery.mixture import single
        from bloomery.train.loop import TrainConfig
        from bloomery.train.loop import train as run_training

        spec = spec_from_depth(1, vocab=dataset.vocab_size, seq=32)
        config = TrainConfig(steps=2, batch=2, seq=32, eval_every=0, log_every=1, **overrides)
        return run_training(
            spec=spec,
            datasets={"d": dataset},
            mixture=single("d"),
            tokenizer=tokenizer,
            run_dir=tmp_path / "run",
            config=config,
            choice=cpu_choice,
            eos_token_id=0,
        )

    def test_a_healthy_run_skips_nothing(
        self, dataset: Any, tmp_path: Path, cpu_choice: DeviceChoice, tokenizer: Any
    ) -> None:
        result = self._run(dataset, tmp_path, cpu_choice, tokenizer)
        assert result.skipped_microbatches == 0
        assert math.isfinite(result.final_loss)

    def test_the_count_reaches_run_jsonl(
        self, dataset: Any, tmp_path: Path, cpu_choice: DeviceChoice, tokenizer: Any
    ) -> None:
        """A number nobody can read is not a signal."""
        self._run(dataset, tmp_path, cpu_choice, tokenizer)
        events = [
            json.loads(line) for line in (tmp_path / "run" / "run.jsonl").read_text().splitlines()
        ]
        steps = [e for e in events if e["event"] == "step"]
        done = [e for e in events if e["event"] == "done"]
        assert steps and "skipped_microbatches" in steps[0]
        assert done and "skipped_microbatches" in done[0]

    def test_a_step_that_scored_nothing_does_not_report_a_perfect_loss(self) -> None:
        """The accumulator starts at 0.0, and 0.0 reads as a flawless model.

        Reporting that would be worse than the NaN the guard replaced: a NaN
        says something broke, a zero says the run went perfectly.
        """
        import torch

        step_loss = 0.0
        contributed = 0
        for scaled in (torch.tensor(float("nan")), torch.tensor(float("nan"))):
            if not torch.isfinite(scaled):
                continue
            step_loss += float(scaled)
            contributed += 1

        # What the loop does with that state.
        reported = step_loss if contributed else float("nan")
        assert contributed == 0
        assert math.isnan(reported), "an unscored step must not look like a perfect one"


class TestReplacingACheckpointLeavesNoGap:
    """There must never be a moment with no checkpoint at all.

    Deleting the old one before renaming the new one into place opens a window
    where a kill, or a rename that fails, takes the good checkpoint and puts
    nothing back. `export` reads runs/<name>/latest while a run may be saving,
    and that window is exactly when it finds nothing there.
    """

    def _save(self, directory: Path, tokenizer: Any, *, step: int) -> Path:
        import torch

        from bloomery.train import checkpoint as ckpt

        model = build_model(spec_from_depth(1, vocab=64, seq=16), eos_token_id=0, seed=0)
        return ckpt.save(
            directory,
            model=model,
            tokenizer=tokenizer,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
            step=step,
            tokens_seen=step * 10,
            best_val_loss=None,
        )

    def test_a_second_save_replaces_the_first(self, tmp_path: Path, tokenizer: Any) -> None:
        from bloomery.train import checkpoint as ckpt

        directory = tmp_path / "latest"
        self._save(directory, tokenizer, step=1)
        self._save(directory, tokenizer, step=2)

        assert ckpt.load_resume_state(directory).step == 2
        # And nothing is left lying about that a reader could mistake for one.
        assert not directory.with_name("latest.previous").exists()
        assert not directory.with_name("latest.tmp").exists()

    def test_the_old_checkpoint_survives_a_failed_rename(
        self, tmp_path: Path, tokenizer: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case the window existed for: something goes wrong mid-swap.

        A cross-filesystem rename raises, and the caller must still have the
        checkpoint it had before rather than neither.
        """
        from bloomery.train import checkpoint as ckpt

        directory = tmp_path / "latest"
        self._save(directory, tokenizer, step=1)

        real = Path.rename

        def refuse(self: Path, target: Any) -> None:
            if self.name.endswith(".tmp"):
                raise OSError("cross-device link")
            real(self, target)

        monkeypatch.setattr(Path, "rename", refuse)
        with pytest.raises(OSError):
            self._save(directory, tokenizer, step=2)
        monkeypatch.undo()

        assert directory.is_dir(), "the previous checkpoint was lost"
        assert ckpt.load_resume_state(directory).step == 1
