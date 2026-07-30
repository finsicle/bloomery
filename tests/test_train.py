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
        assert tuple(batch.shape) == (4, 32)
        assert batch.dtype.is_signed  # int64, as the embedding layer requires

    def test_ids_are_in_vocab_range(self, dataset: Any, cpu_choice: DeviceChoice) -> None:
        sampler = BatchSampler(dataset, "train", seq=32, seed=0)
        batch = sampler.batch(8, cpu_choice.device)
        assert int(batch.max()) < dataset.vocab_size
        assert int(batch.min()) >= 0

    def test_seeded_sampling_is_reproducible(self, dataset: Any, cpu_choice: DeviceChoice) -> None:
        first = BatchSampler(dataset, "train", seq=16, seed=42).batch(4, cpu_choice.device)
        second = BatchSampler(dataset, "train", seq=16, seed=42).batch(4, cpu_choice.device)
        assert first.equal(second)

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
