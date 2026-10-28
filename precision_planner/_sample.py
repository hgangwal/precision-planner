"""SAMPLE DATA GENERATOR (Day-1 stub). Nothing in this module is measured.

It produces plausible, deterministic plans so the dashboard and API can be
built before the real sensitivity analysis and search exist. Every Plan made
here has ``is_sample=True``. Delete this module when the real search lands.

The fake model: each layer has a made-up accuracy drop per format, drops add
up across layers, and a greedy pass lowers the least sensitive layers first
until the budget is spent. That mirrors the real algorithm's shape so the UI
behaves realistically (bigger budget -> more low-precision layers).
"""

from __future__ import annotations

import random
import zlib

from precision_planner.formats import Format, Rounding
from precision_planner.plan import LayerAssignment, Metrics, Plan

DISTILBERT_SST2 = "distilbert-base-uncased-finetuned-sst-2-english"

# The 38 nn.Linear layers of DistilBERT-SST2 as (name, num_params incl. bias).
DISTILBERT_LINEAR_LAYERS: tuple[tuple[str, int], ...] = tuple(
    (f"distilbert.transformer.layer.{block}.{sub}", n)
    for block in range(6)
    for sub, n in (
        ("attention.q_lin", 768 * 768 + 768),
        ("attention.k_lin", 768 * 768 + 768),
        ("attention.v_lin", 768 * 768 + 768),
        ("attention.out_lin", 768 * 768 + 768),
        ("ffn.lin1", 768 * 3072 + 3072),
        ("ffn.lin2", 3072 * 768 + 768),
    )
) + (
    ("pre_classifier", 768 * 768 + 768),
    ("classifier", 768 * 2 + 2),
)
# Embeddings + LayerNorms: everything that is not an nn.Linear. Stays FP32.
DISTILBERT_OTHER_PARAMS = 23_854_080

HELD_OUT_SIZE = 436  # half of the 872 SST-2 validation sentences

# --- made-up constants (SAMPLE) ---------------------------------------------
_SAMPLE_FP32_ACCURACY = 400 / HELD_OUT_SIZE  # ~91.7%, near the checkpoint's published figure
_SAMPLE_FP32_LATENCY_MS = (28.4, 1.9)  # (mean, std)
_SAMPLE_ACTIVATION_MB = 61.0
_INT8_LATENCY_GAIN = 0.45  # fraction of linear-layer time saved by INT8 kernels

_INT8_DROP_BY_TYPE = {
    "q_lin": 0.0006, "k_lin": 0.0005, "v_lin": 0.0012, "out_lin": 0.0015,
    "lin1": 0.0009, "lin2": 0.0030, "pre_classifier": 0.0040, "classifier": 0.0080,
}
# FP8 (e4m3) has more dynamic range than INT8, so it is pretended to handle
# outlier-heavy layers better and the rest worse.
_FP8_FACTOR_BY_TYPE = {"out_lin": 0.7, "lin2": 0.7}
_FP8_FACTOR_DEFAULT = 1.4
_FP16_FACTOR_BY_TYPE = {"classifier": 0.25, "pre_classifier": 0.1}  # small head: less slack
_FP16_FACTOR_DEFAULT = 0.01
_STOCHASTIC_FACTOR = 0.85
_HELD_OUT_FACTOR = 1.05  # held-out drop is a bit worse than on the search split


def _seed(*parts: object) -> int:
    """Stable across runs and machines (unlike hash())."""
    return zlib.crc32("|".join(map(str, parts)).encode())


def sample_sensitivity(name: str, model_id: str) -> dict[Format, float]:
    """Made-up accuracy drop for lowering just this layer to each format."""
    kind = name.rsplit(".", 1)[-1]
    base = _INT8_DROP_BY_TYPE.get(kind, 0.0010)
    # Early and late blocks are more sensitive than middle ones.
    depth_factor = 1.0
    if ".layer." in name:
        block = int(name.split(".layer.")[1].split(".")[0])
        depth_factor = (1.4, 0.9, 0.7, 0.7, 0.9, 1.6)[block % 6]
    jitter = random.Random(_seed(model_id, name)).uniform(0.8, 1.2)
    int8 = base * depth_factor * jitter
    return {
        Format.FP32: 0.0,
        Format.FP16: int8 * _FP16_FACTOR_BY_TYPE.get(kind, _FP16_FACTOR_DEFAULT),
        Format.INT8: int8,
        Format.FP8: int8 * _FP8_FACTOR_BY_TYPE.get(kind, _FP8_FACTOR_DEFAULT),
    }


def _greedy(drops: dict[str, dict[Format, float]], budget: float) -> dict[str, Format]:
    """Two passes, least-sensitive layers first: FP32->FP16, then FP16->best 1-byte format.

    Taking the nearly-free FP16 step everywhere before spending budget on
    1-byte formats keeps model size monotone in the budget.
    """
    plan = {name: Format.FP32 for name in drops}
    spent = 0.0
    order = sorted(drops, key=lambda n: drops[n][Format.INT8])
    for name in order:
        if spent + drops[name][Format.FP16] <= budget:
            plan[name] = Format.FP16
            spent += drops[name][Format.FP16]
    for name in order:
        if plan[name] is not Format.FP16:
            continue
        one_byte = min((Format.INT8, Format.FP8), key=lambda f: drops[name][f])
        extra = drops[name][one_byte] - drops[name][Format.FP16]
        if spent + extra <= budget:
            plan[name] = one_byte
            spent += extra
    return plan


def _quantize_accuracy(acc: float) -> float:
    """Snap to a multiple of 1/436 so sample accuracies look like real counts."""
    return round(max(0.0, min(1.0, acc)) * HELD_OUT_SIZE) / HELD_OUT_SIZE


def _metrics(
    formats: dict[str, Format],
    drops: dict[str, dict[Format, float]],
    layer_params: dict[str, int],
    other_params: int,
) -> Metrics:
    drop = sum(drops[n][f] for n, f in formats.items()) * _HELD_OUT_FACTOR
    size_bytes = other_params * 4 + sum(
        layer_params[n] * f.bytes_per_param for n, f in formats.items()
    )
    size_mb = size_bytes / 2**20
    linear_params = sum(layer_params.values()) or 1
    int8_fraction = sum(layer_params[n] for n, f in formats.items() if f is Format.INT8) / linear_params
    # Rule: no latency for configs using formats without real CPU kernels.
    # Mixed plans run FP16/FP8 layers as FP32 on CPU, so only uniform FP16 is n/a.
    if formats and all(f is Format.FP16 for f in formats.values()):
        latency = (None, None)
    else:
        mean, std = _SAMPLE_FP32_LATENCY_MS
        speedup = 1 - _INT8_LATENCY_GAIN * int8_fraction
        latency = (round(mean * speedup, 1), round(std * speedup, 1))
    return Metrics(
        accuracy=_quantize_accuracy(_SAMPLE_FP32_ACCURACY - drop),
        size_mb=round(size_mb, 1),
        latency_ms_mean=latency[0],
        latency_ms_std=latency[1],
        peak_memory_mb=round(size_mb + _SAMPLE_ACTIVATION_MB, 1),
    )


def make_sample_plan(
    model_id: str,
    layers: tuple[tuple[str, int], ...],
    other_params: int,
    max_accuracy_drop: float,
    rounding: Rounding,
) -> Plan:
    """Build a deterministic SAMPLE plan for the given layers and budget."""
    rounding = Rounding(rounding)
    factor = _STOCHASTIC_FACTOR if rounding is Rounding.STOCHASTIC else 1.0
    drops = {
        name: {f: d * factor for f, d in sample_sensitivity(name, model_id).items()}
        for name, _ in layers
    }
    layer_params = dict(layers)
    names = [name for name, _ in layers]

    chosen = _greedy(drops, max_accuracy_drop)
    # Random mixed: same format counts as the plan, shuffled across layers.
    shuffled = list(chosen.values())
    random.Random(_seed(model_id, max_accuracy_drop, rounding.value)).shuffle(shuffled)

    configs = {
        "fp32": {n: Format.FP32 for n in names},
        "fp16": {n: Format.FP16 for n in names},
        "int8": {n: Format.INT8 for n in names},
        "random_mixed": dict(zip(names, shuffled)),
    }
    return Plan(
        model_id=model_id,
        max_accuracy_drop=max_accuracy_drop,
        rounding=rounding,
        layers=tuple(LayerAssignment(n, chosen[n], layer_params[n]) for n in names),
        metrics=_metrics(chosen, drops, layer_params, other_params),
        baselines={k: _metrics(v, drops, layer_params, other_params) for k, v in configs.items()},
        is_sample=True,
    )
