"""The Plan: what precision each layer gets, and what that costs and saves.

A Plan is plain data. It can be saved to JSON and loaded without torch,
which is what lets the web app render precomputed results cheaply.
"""

from __future__ import annotations

import copy
import json
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from precision_planner._version import __version__
from precision_planner.formats import Format, Rounding

SCHEMA_VERSION = 1

# Keys of Plan.baselines, in display order.
BASELINE_NAMES = ("fp32", "fp16", "int8", "random_mixed")
BASELINE_LABELS = {
    "fp32": "FP32",
    "fp16": "Uniform FP16",
    "int8": "Uniform INT8",
    "random_mixed": "Random mixed",
}


class SampleDataWarning(UserWarning):
    """Raised when sample (not measured) data is used as if it were real."""


@dataclass(frozen=True)
class Metrics:
    """Measurements of one model configuration on held-out data.

    Any field that was not measured on real hardware is ``None``, never an
    estimate. For example, FP16 latency on a CPU without FP16 kernels is None.
    """

    accuracy: float
    size_mb: float | None = None
    latency_ms_mean: float | None = None
    latency_ms_std: float | None = None
    peak_memory_mb: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError(f"accuracy must be in [0, 1], got {self.accuracy}")
        for name in ("size_mb", "latency_ms_mean", "latency_ms_std", "peak_memory_mb"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if (self.latency_ms_mean is None) != (self.latency_ms_std is None):
            raise ValueError("latency_ms_mean and latency_ms_std must both be set or both be None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "size_mb": self.size_mb,
            "latency_ms_mean": self.latency_ms_mean,
            "latency_ms_std": self.latency_ms_std,
            "peak_memory_mb": self.peak_memory_mb,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metrics:
        return cls(**data)


@dataclass(frozen=True)
class LayerAssignment:
    """The format chosen for one ``nn.Linear`` layer."""

    name: str
    format: Format
    num_params: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("layer name must be non-empty")
        if self.num_params < 0:
            raise ValueError(f"num_params must be non-negative, got {self.num_params}")
        # Accept "fp16" as well as Format.FP16 so JSON loads cleanly.
        object.__setattr__(self, "format", Format(self.format))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "format": self.format.value, "num_params": self.num_params}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayerAssignment:
        return cls(name=data["name"], format=Format(data["format"]), num_params=data["num_params"])


@dataclass(frozen=True)
class Plan:
    """A per-layer precision plan plus its evaluation against baselines.

    ``metrics`` and every entry of ``baselines`` are measured on the held-out
    split, which the search never saw. ``is_sample`` is True when the numbers
    come from the Day-1 stub rather than real measurement; anything that
    displays a Plan must surface that.
    """

    model_id: str
    max_accuracy_drop: float
    rounding: Rounding
    layers: tuple[LayerAssignment, ...]
    metrics: Metrics
    baselines: dict[str, Metrics]
    is_sample: bool
    library_version: str = __version__
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.max_accuracy_drop < 1.0:
            raise ValueError(
                f"max_accuracy_drop must be a fraction in (0, 1), got {self.max_accuracy_drop}"
            )
        object.__setattr__(self, "rounding", Rounding(self.rounding))
        object.__setattr__(self, "layers", tuple(self.layers))
        missing = [name for name in BASELINE_NAMES if name not in self.baselines]
        if missing:
            raise ValueError(f"baselines missing required entries: {missing}")
        names = [layer.name for layer in self.layers]
        duplicates = sorted(name for name, n in Counter(names).items() if n > 1)
        if duplicates:
            raise ValueError(f"duplicate layer names in plan: {duplicates}")

    @property
    def accuracy_drop(self) -> float:
        """FP32 accuracy minus plan accuracy, on held-out data (absolute, not relative)."""
        return self.baselines["fp32"].accuracy - self.metrics.accuracy

    def format_counts(self) -> dict[Format, int]:
        """How many layers got each format, with every format present (zero if unused)."""
        counts = Counter(layer.format for layer in self.layers)
        return {fmt: counts.get(fmt, 0) for fmt in Format}

    # ----- conversion -------------------------------------------------------

    def apply(self, model: Any) -> Any:
        """Return a copy of ``model`` converted to this plan. The input is never mutated.

        Raises ValueError if the plan's layer names don't match the model's
        ``nn.Linear`` layers, so a plan can't silently be applied to the wrong model.
        """
        _check_layer_names(model, {layer.name for layer in self.layers})
        if self.is_sample:
            warnings.warn(
                "This plan contains SAMPLE data from the Day-1 stub; apply() returns an "
                "unconverted copy of the model. Real conversion is not implemented yet.",
                SampleDataWarning,
                stacklevel=2,
            )
            return copy.deepcopy(model)
        raise NotImplementedError("Real precision conversion is not implemented yet.")

    # ----- reporting --------------------------------------------------------

    def report(self) -> str:
        """Print a plain-text summary of the plan and its baselines, and return it."""
        text = format_report(self)
        print(text)
        return text

    # ----- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_id": self.model_id,
            "max_accuracy_drop": self.max_accuracy_drop,
            "rounding": self.rounding.value,
            "is_sample": self.is_sample,
            "library_version": self.library_version,
            "created_at": self.created_at,
            "metrics": self.metrics.to_dict(),
            "baselines": {name: m.to_dict() for name, m in self.baselines.items()},
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported plan schema_version {version!r}; expected {SCHEMA_VERSION}")
        return cls(
            model_id=data["model_id"],
            max_accuracy_drop=data["max_accuracy_drop"],
            rounding=Rounding(data["rounding"]),
            layers=tuple(LayerAssignment.from_dict(d) for d in data["layers"]),
            metrics=Metrics.from_dict(data["metrics"]),
            baselines={name: Metrics.from_dict(m) for name, m in data["baselines"].items()},
            is_sample=data["is_sample"],
            library_version=data["library_version"],
            created_at=data["created_at"],
        )

    def save(self, path: str | Path) -> Path:
        """Write the plan as JSON, creating parent directories. Returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Plan:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _check_layer_names(model: Any, plan_names: set[str]) -> None:
    import torch.nn as nn  # lazy: plan loading must not require torch

    model_names = {name for name, m in model.named_modules() if isinstance(m, nn.Linear)}
    missing = sorted(plan_names - model_names)
    unplanned = sorted(model_names - plan_names)
    if missing or unplanned:
        raise ValueError(
            "plan does not match model: "
            f"layers in plan but not model: {missing}; layers in model but not plan: {unplanned}"
        )


def _fmt(value: float | None, spec: str, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:{spec}}{suffix}"


def _fmt_latency(m: Metrics) -> str:
    if m.latency_ms_mean is None:
        return "n/a"
    return f"{m.latency_ms_mean:.1f} ± {m.latency_ms_std:.1f} ms"


def format_report(plan: Plan) -> str:
    """Render a plan as a plain-text report (no printing)."""
    lines: list[str] = []
    if plan.is_sample:
        lines += ["*** SAMPLE DATA: these numbers are placeholders, not measurements. ***", ""]
    lines += [
        f"Precision plan for {plan.model_id}",
        f"Budget: max accuracy drop {plan.max_accuracy_drop:.2%} | rounding: {plan.rounding.value}",
        f"Held-out accuracy drop vs FP32: {plan.accuracy_drop:.2%}",
        "",
    ]
    rows = [("Plan", plan.metrics)] + [
        (BASELINE_LABELS[name], plan.baselines[name]) for name in BASELINE_NAMES
    ]
    header = f"{'Config':<14}{'Accuracy':>10}{'Size':>12}{'Latency':>20}{'Peak mem':>12}"
    lines += [header, "-" * len(header)]
    for label, m in rows:
        lines.append(
            f"{label:<14}{m.accuracy:>10.2%}{_fmt(m.size_mb, '.1f', ' MB'):>12}"
            f"{_fmt_latency(m):>20}{_fmt(m.peak_memory_mb, '.1f', ' MB'):>12}"
        )
    counts = ", ".join(f"{fmt.label}: {n}" for fmt, n in plan.format_counts().items())
    lines += ["", f"Layers ({len(plan.layers)}): {counts}"]
    width = max((len(layer.name) for layer in plan.layers), default=0)
    lines += [f"  {layer.name:<{width}}  {layer.format.label}" for layer in plan.layers]
    return "\n".join(lines)
