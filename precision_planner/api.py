"""Entry points: analyze() a model, or load_plan() a precomputed result."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable

from precision_planner import _sample
from precision_planner.formats import Rounding
from precision_planner.plan import Plan, SampleDataWarning

# Budgets that have precomputed results. Fractions, not percents.
BUDGETS: tuple[float, ...] = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05)

# Model id -> display name, for models with precomputed results.
SUPPORTED_MODELS: dict[str, str] = {_sample.DISTILBERT_SST2: "DistilBERT (SST-2)"}

_EPS = 1e-9


def analyze(
    model: Any,
    data: Iterable[Any],
    max_accuracy_drop: float = 0.01,
    rounding: Rounding | str = Rounding.NEAREST,
) -> Plan:
    """Choose a precision for every ``nn.Linear`` in ``model`` within an accuracy budget.

    Args:
        model: a trained ``torch.nn.Module``. Never modified.
        data: validation batches of ``(inputs, labels)``, used for the search.
        max_accuracy_drop: largest allowed absolute accuracy loss vs FP32, as a
            fraction (0.01 means one percentage point).
        rounding: ``"nearest"`` or ``"stochastic"``; recorded in the plan so
            ``plan.apply`` converts the same way the search measured.

    STUB: the layer list is read from the real model, but the format choices
    and metrics are SAMPLE data and ``data`` is not used yet.
    """
    _check_budget(max_accuracy_drop)
    rounding = Rounding(rounding)
    layers, other_params = _linear_layers(model)
    warnings.warn(
        "precision_planner is a Day-1 stub: analyze() returns SAMPLE data, not measurements.",
        SampleDataWarning,
        stacklevel=2,
    )
    return _sample.make_sample_plan(_model_id(model), layers, other_params, max_accuracy_drop, rounding)


def load_plan(
    model_id: str,
    max_accuracy_drop: float,
    rounding: Rounding | str = Rounding.NEAREST,
    results_dir: str | Path = "results",
) -> Plan:
    """Load the precomputed plan for the largest saved budget ≤ ``max_accuracy_drop``.

    Rounding the budget *down* means the returned plan never exceeds what the
    caller asked for; ``plan.max_accuracy_drop`` says which budget was used.
    """
    budget = snap_budget(max_accuracy_drop)
    path = results_path(results_dir, model_id, budget, rounding)
    if not path.exists():
        raise FileNotFoundError(
            f"No precomputed plan at {path}. Run `python scripts/generate_results.py` first."
        )
    return Plan.load(path)


def snap_budget(max_accuracy_drop: float) -> float:
    """Largest value in BUDGETS that is ≤ the requested budget."""
    _check_budget(max_accuracy_drop)
    eligible = [b for b in BUDGETS if b <= max_accuracy_drop + _EPS]
    if not eligible:
        raise ValueError(
            f"max_accuracy_drop {max_accuracy_drop} is below the smallest precomputed budget {BUDGETS[0]}"
        )
    return max(eligible)


def results_path(results_dir: str | Path, model_id: str, budget: float, rounding: Rounding | str) -> Path:
    """Where the plan for one (model, budget, rounding) lives, e.g. ``.../budget_1.00pct_nearest.json``."""
    slug = model_id.replace("/", "__")
    return Path(results_dir) / slug / f"budget_{budget * 100:.2f}pct_{Rounding(rounding).value}.json"


def _check_budget(max_accuracy_drop: float) -> None:
    if not 0.0 < max_accuracy_drop < 1.0:
        raise ValueError(
            f"max_accuracy_drop must be a fraction in (0, 1), e.g. 0.01 for 1%; got {max_accuracy_drop}"
        )


def _linear_layers(model: Any) -> tuple[tuple[tuple[str, int], ...], int]:
    """(name, num_params) for every nn.Linear, plus the count of all other params."""
    import torch.nn as nn  # lazy: loading plans must not require torch

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be a torch.nn.Module, got {type(model).__name__}")
    layers = tuple(
        (name, sum(p.numel() for p in m.parameters()))
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear)
    )
    if not layers:
        raise ValueError("model has no nn.Linear layers to plan")
    total = sum(p.numel() for p in model.parameters())
    return layers, total - sum(n for _, n in layers)


def _model_id(model: Any) -> str:
    """Hugging Face id if the model has one, else its class name."""
    config = getattr(model, "config", None)
    return getattr(config, "_name_or_path", None) or type(model).__name__
