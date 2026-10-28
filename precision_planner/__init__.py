"""Precision Planner: per-layer mixed-precision plans under an accuracy budget.

    import precision_planner as pp

    plan = pp.analyze(model, val_data, max_accuracy_drop=0.01)
    fast_model = plan.apply(model)
    plan.report()

``__all__`` is the public API. The web app may import only these names.
"""

from precision_planner._version import __version__
from precision_planner.api import BUDGETS, SUPPORTED_MODELS, analyze, load_plan
from precision_planner.formats import Format, Rounding
from precision_planner.plan import (
    BASELINE_LABELS,
    BASELINE_NAMES,
    LayerAssignment,
    Metrics,
    Plan,
    SampleDataWarning,
)

__all__ = [
    "BASELINE_LABELS",
    "BASELINE_NAMES",
    "BUDGETS",
    "SUPPORTED_MODELS",
    "Format",
    "LayerAssignment",
    "Metrics",
    "Plan",
    "Rounding",
    "SampleDataWarning",
    "__version__",
    "analyze",
    "load_plan",
]
