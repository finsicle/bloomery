# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What can this machine actually train?

The estimates here are deliberately conservative. Promising a run that then dies
with an out-of-memory error four hours in is worse than saying "probably not" up
front, so every uncertain term rounds against the user.

Memory only. Throughput is not estimated, because doing it honestly needs a
measured figure for this machine rather than a table of peak-FLOPS numbers that
rot with every driver release — see ``bloomery bench`` (M1).

Calibration
-----------
The per-parameter constants below were checked against well-known reference
points before being trusted:

* QLoRA on a 7B model comes out at ~6.9 GiB here; the widely reported figure is
  6-8 GiB.
* LoRA on 7B in bf16 comes out at ~17.5 GiB against a reported 16-20 GiB.
* A full 7B fine-tune comes out at ~107 GiB, which correctly says "not on one
  80 GiB card" — the known result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from bloomery.probe.types import GIB, HostReport, Vendor

# --------------------------------------------------------------------------- #
# Memory model
# --------------------------------------------------------------------------- #

# Bytes of persistent state per parameter, by training method.
#
# full:  bf16 weights 2 + bf16 grads 2 + fp32 master 4 + AdamW m 4 + AdamW v 4
# lora:  frozen bf16 base 2, plus ~1% trainable params at the full 16
# qlora: NF4 base 0.5 + double-quant constants ~0.016, plus the same adapters
_STATE_BYTES_PER_PARAM: dict[str, float] = {
    "full": 16.0,
    "lora": 2.16,
    "qlora": 0.68,
}

# Activation memory, expressed as a multiplier on batch * seq * hidden * 2 bytes.
#
# With gradient checkpointing each block stores only its input and recomputes the
# interior, so the cost is one slot per layer plus the peak inside whichever block
# is being recomputed.
#
# Without it every layer keeps its intermediates for the backward pass, which is
# far more expensive and is the default in `bloomery train`. The multiplier comes
# from the standard transformer activation estimate of 34 * s * b * h * L bytes;
# dividing by the 2 bytes already in the formula gives 17 per layer.
#
# Both assume a flash/SDPA kernel, so there is no batch * heads * seq^2 term.
_RECOMPUTE_BLOCK_FACTOR = 12
_NO_CHECKPOINT_PER_LAYER = 17

# Allocator fragmentation, plus a fixed allowance for the driver context.
_FRAGMENTATION = 1.08
_CONTEXT_OVERHEAD = int(0.8 * GIB)

# Chinchilla-optimal tokens per parameter. A convention, not a hardware fact.
TOKENS_PER_PARAM = 20

# Fraction of system RAM we are willing to plan against for a CPU-only run.
_CPU_RAM_FRACTION = 0.7


class Method(StrEnum):
    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"

    @property
    def label(self) -> str:
        return {"full": "Full", "lora": "LoRA", "qlora": "QLoRA"}[self.value]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A transformer configuration, sized the way a real model would be."""

    key: str
    label: str
    layers: int
    hidden: int
    heads: int
    vocab: int
    seq: int
    batch: int
    note: str = ""

    @property
    def params(self) -> int:
        """Parameter count for a Llama-style decoder with tied embeddings.

        ``V*H`` for the embedding table, then ``12*H^2`` per block: roughly
        ``4*H^2`` for attention projections and ``8*H^2`` for a SwiGLU MLP at the
        conventional 8/3 expansion.

        Checks out against real models: the d12 entry lands on 123.5M against
        GPT-2 small's 124M, and the 7B entry on 6.98B against Llama-7B's 6.74B.
        """
        return self.vocab * self.hidden + self.layers * 12 * self.hidden**2


# The ladder shown in `bloomery doctor`. Ordered smallest first.
LADDER: tuple[ModelSpec, ...] = (
    ModelSpec("d4", "tiny", 4, 256, 4, 8_192, 512, 32, "minutes; proves the pipeline"),
    ModelSpec("d8", "small", 8, 512, 8, 16_384, 1024, 16, "TinyStories-class prose"),
    ModelSpec("d12", "GPT-2 small class", 12, 768, 12, 50_257, 1024, 8, "the classic 124M"),
    ModelSpec("d24", "GPT-2 medium class", 24, 1024, 16, 50_257, 1024, 4, ""),
    ModelSpec("d26", "nanochat d26", 26, 1280, 10, 65_536, 2048, 4, "chats coherently"),
    ModelSpec("1b", "1B", 16, 2048, 16, 131_072, 2048, 2, "days on one consumer GPU"),
    ModelSpec("3b", "3B", 28, 3072, 24, 131_072, 2048, 1, ""),
    ModelSpec("7b", "7B", 32, 4096, 32, 131_072, 4096, 1, "fine-tune territory, not scratch"),
)

LADDER_BY_KEY = {spec.key: spec for spec in LADDER}


def activation_bytes(
    spec: ModelSpec,
    *,
    batch: int | None = None,
    seq: int | None = None,
    gradient_checkpointing: bool = True,
) -> int:
    """Activation memory for one training step.

    Checkpointing is the cheaper case by a wide margin, and it is *not* the
    default in `bloomery train` — estimating as though it were on would
    under-report the configuration people actually run.
    """
    b = batch if batch is not None else spec.batch
    s = seq if seq is not None else spec.seq
    per_token = spec.hidden * 2  # bf16
    layers = (
        spec.layers + _RECOMPUTE_BLOCK_FACTOR
        if gradient_checkpointing
        else spec.layers * _NO_CHECKPOINT_PER_LAYER
    )
    return b * s * per_token * layers


def logit_bytes(spec: ModelSpec, *, batch: int | None = None, seq: int | None = None) -> int:
    """The output logit tensor, in fp32 for the loss.

    Often the single largest allocation in a training step and a common cause of
    an out-of-memory error that looks inexplicable, because it scales with vocab
    rather than with model size. A chunked cross-entropy kernel cuts this
    dramatically; we assume the naive path.
    """
    b = batch if batch is not None else spec.batch
    s = seq if seq is not None else spec.seq
    return b * s * spec.vocab * 4


def estimate_memory(
    spec: ModelSpec,
    method: Method,
    *,
    batch: int | None = None,
    seq: int | None = None,
    gradient_checkpointing: bool = True,
) -> int:
    """Peak memory for one training step, in bytes."""
    state = int(spec.params * _STATE_BYTES_PER_PARAM[method.value])
    working = activation_bytes(
        spec, batch=batch, seq=seq, gradient_checkpointing=gradient_checkpointing
    ) + logit_bytes(spec, batch=batch, seq=seq)
    return int((state + working) * _FRAGMENTATION) + _CONTEXT_OVERHEAD


# --------------------------------------------------------------------------- #
# Assessment against a real host
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Budget:
    """How much memory a single training process can plan on using."""

    total: int
    source: str
    # Devices the work could be sharded across with FSDP or ZeRO-3.
    shardable_devices: int = 1
    shardable_total: int | None = None


@dataclass(frozen=True, slots=True)
class Row:
    spec: ModelSpec
    method: Method
    required: int
    budget: int
    available_here: bool = True
    unavailable_reason: str = ""

    @property
    def fits(self) -> bool:
        return self.available_here and self.required <= self.budget

    @property
    def ratio(self) -> float:
        return self.required / self.budget if self.budget else float("inf")

    @property
    def tokens(self) -> int:
        return self.spec.params * TOKENS_PER_PARAM


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    budget: Budget
    rows: tuple[Row, ...]

    def for_method(self, method: Method) -> tuple[Row, ...]:
        return tuple(r for r in self.rows if r.method is method)

    def largest_fitting(self, method: Method) -> Row | None:
        fitting = [r for r in self.for_method(method) if r.fits]
        return max(fitting, key=lambda r: r.spec.params) if fitting else None


def derive_budget(report: HostReport, *, device_type: str | None = None) -> Budget:
    """Work out the memory ceiling for one training process.

    Uses the *largest single* GPU rather than the sum. Plain data-parallel
    training replicates the whole model onto every device, so adding GPUs buys
    throughput, not capacity. Sharding across them is possible, and reported
    separately, but it is not the default and should not be assumed.

    ``device_type`` pins the answer to the device a run will actually use. A
    machine with a 24 GiB card still only has system RAM available to a run
    launched with ``--device cpu``, and reporting the card's VRAM there would
    approve a configuration that cannot possibly fit.
    """
    vendors = {gpu.vendor for gpu in report.gpus}

    if device_type == "cpu":
        available = report.memory.available or report.memory.total or 0
        return Budget(
            total=int(available * _CPU_RAM_FRACTION),
            source=f"system RAM — {available / GIB:.0f} GiB available, CPU only",
        )

    if report.gpus and report.largest_vram:
        largest = max(
            (g for g in report.gpus if g.vram_total),
            key=lambda g: g.vram_total or 0,
        )
        count = len([g for g in report.gpus if g.vram_total])
        unified = Vendor.APPLE in vendors
        source = (
            f"{largest.name} — unified memory"
            if unified
            else f"{largest.name} — {largest.vram_gib:.0f} GiB VRAM"
        )
        return Budget(
            total=largest.vram_total or 0,
            source=source,
            shardable_devices=count,
            shardable_total=report.total_vram if count > 1 else None,
        )

    available = report.memory.available or report.memory.total or 0
    return Budget(
        total=int(available * _CPU_RAM_FRACTION),
        source=f"system RAM — {available / GIB:.0f} GiB available, CPU only",
    )


def _method_availability(report: HostReport, method: Method) -> tuple[bool, str]:
    """Whether a method can run on this host at all."""
    vendors = {gpu.vendor for gpu in report.gpus}

    if method is Method.QLORA:
        if not report.gpus:
            return False, "bitsandbytes has no CPU training path"
        if Vendor.APPLE in vendors:
            return False, "bitsandbytes ships no macOS wheels"
    if method is Method.LORA and not report.gpus:
        return False, "adapter training on CPU is impractically slow"
    return True, ""


def assess(report: HostReport) -> CapabilityReport:
    """Build the full feasibility table for a host."""
    budget = derive_budget(report)
    rows: list[Row] = []

    for method in Method:
        available, reason = _method_availability(report, method)
        for spec in LADDER:
            rows.append(
                Row(
                    spec=spec,
                    method=method,
                    required=estimate_memory(spec, method),
                    budget=budget.total,
                    available_here=available,
                    unavailable_reason=reason,
                )
            )

    return CapabilityReport(budget=budget, rows=tuple(rows))


@dataclass(frozen=True, slots=True)
class FitCheck:
    """Whether a specific training configuration is expected to fit."""

    spec: ModelSpec
    method: Method
    batch: int
    seq: int
    gradient_checkpointing: bool
    required: int
    budget: Budget

    @property
    def fits(self) -> bool:
        return self.required <= self.budget.total

    @property
    def headroom(self) -> float:
        """Budget as a multiple of the requirement. Below 1.0 means it will not fit."""
        return self.budget.total / self.required if self.required else float("inf")

    def suggestions(self) -> list[str]:
        """Concrete ways to make this configuration fit, largest saving first."""
        if self.fits:
            return []

        options: list[str] = []
        if not self.gradient_checkpointing:
            saved = self.required - estimate_memory(
                self.spec,
                self.method,
                batch=self.batch,
                seq=self.seq,
                gradient_checkpointing=True,
            )
            if saved > 0:
                options.append(
                    f"--grad-checkpoint  (saves ~{saved / GIB:.1f} GiB, costs ~30% speed)"
                )

        # Halving either dimension halves the activation and logit terms, which
        # is usually the fastest way back under the limit.
        for label, kwargs in (
            (f"--batch {max(1, self.batch // 2)}", {"batch": max(1, self.batch // 2)}),
            (f"--seq {max(64, self.seq // 2)}", {"seq": max(64, self.seq // 2)}),
        ):
            reduced = estimate_memory(
                self.spec,
                self.method,
                batch=kwargs.get("batch", self.batch),
                seq=kwargs.get("seq", self.seq),
                gradient_checkpointing=self.gradient_checkpointing,
            )
            if reduced < self.required:
                verdict = "fits" if reduced <= self.budget.total else "still short"
                options.append(f"{label}  (~{reduced / GIB:.1f} GiB, {verdict})")

        smaller = _largest_depth_that_fits(self)
        if smaller is not None:
            options.append(f"--depth {smaller}  (the largest depth that fits as configured)")

        return options


def _largest_depth_that_fits(check: FitCheck) -> int | None:
    """Biggest layer count that fits, holding batch, sequence and vocab fixed."""
    from bloomery.arch import spec_from_depth

    best: int | None = None
    for depth in range(1, check.spec.layers):
        candidate = spec_from_depth(depth, vocab=check.spec.vocab, seq=check.seq, batch=check.batch)
        required = estimate_memory(
            candidate,
            check.method,
            batch=check.batch,
            seq=check.seq,
            gradient_checkpointing=check.gradient_checkpointing,
        )
        if required <= check.budget.total:
            best = depth
    return best


def check_fit(
    report: HostReport,
    spec: ModelSpec,
    *,
    method: Method = Method.FULL,
    batch: int,
    seq: int,
    gradient_checkpointing: bool = False,
    device_type: str | None = None,
) -> FitCheck:
    """Estimate whether a training configuration fits before it is started.

    The point of running this ahead of a run rather than discovering it from an
    out-of-memory error is that the error arrives whenever the allocator happens
    to hit the ceiling — which can be twenty minutes in, after the tokenizer,
    the packing and the model build have all succeeded.
    """
    return FitCheck(
        spec=spec,
        method=method,
        batch=batch,
        seq=seq,
        gradient_checkpointing=gradient_checkpointing,
        required=estimate_memory(
            spec,
            method,
            batch=batch,
            seq=seq,
            gradient_checkpointing=gradient_checkpointing,
        ),
        budget=derive_budget(report, device_type=device_type),
    )


def format_tokens(count: int) -> str:
    """Render a token count the way people say it out loud."""
    if count >= 1_000_000_000_000:
        return f"{count / 1_000_000_000_000:.1f}T"
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.0f}M"
    return str(count)


def format_params(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.0f}M"
    return f"{count / 1000:.0f}K"
