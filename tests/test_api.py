import types

import pytest
import torch.nn as nn

import precision_planner as pp
from precision_planner import api
from precision_planner.api import BUDGETS, _linear_layers, _model_id, results_path, snap_budget
from precision_planner.formats import Rounding
from precision_planner.plan import SampleDataWarning

MODEL_ID = "org/model-name"


# ----- analyze ---------------------------------------------------------------


def test_analyze_layers_match_linear_modules(tiny_model):
    with pytest.warns(SampleDataWarning):
        plan = pp.analyze(tiny_model, data=[], max_accuracy_drop=0.01)
    expected = [
        (n, sum(p.numel() for p in m.parameters()))
        for n, m in tiny_model.named_modules()
        if isinstance(m, nn.Linear)
    ]
    assert [(la.name, la.num_params) for la in plan.layers] == expected
    assert dict(expected) == {"fc1": 40, "fc2": 72, "fc3": 18}


def test_analyze_returns_sample_plan_with_rounding(tiny_model):
    with pytest.warns(SampleDataWarning):
        plan = pp.analyze(tiny_model, [], 0.02, rounding="stochastic")
    assert plan.is_sample is True
    assert plan.rounding is Rounding.STOCHASTIC
    assert plan.max_accuracy_drop == 0.02


def test_analyze_default_rounding_is_nearest(tiny_model):
    with pytest.warns(SampleDataWarning):
        plan = pp.analyze(tiny_model, [])
    assert plan.rounding is Rounding.NEAREST
    assert plan.model_id == "Sequential"


@pytest.mark.parametrize("budget", [0, 1, -0.1])
def test_analyze_rejects_bad_budget(tiny_model, budget):
    with pytest.raises(ValueError, match="max_accuracy_drop"):
        pp.analyze(tiny_model, [], budget)


def test_analyze_rejects_non_module():
    with pytest.raises(TypeError, match="torch.nn.Module"):
        pp.analyze(object(), [], 0.01)


def test_analyze_rejects_model_without_linear():
    with pytest.raises(ValueError, match="no nn.Linear"):
        pp.analyze(nn.Sequential(nn.LayerNorm(4), nn.ReLU()), [], 0.01)


def test_analyze_does_not_mutate_model(tiny_model):
    before = {k: v.clone() for k, v in tiny_model.state_dict().items()}
    with pytest.warns(SampleDataWarning):
        pp.analyze(tiny_model, [], 0.01)
    assert all((before[k] == v).all() for k, v in tiny_model.state_dict().items())


# ----- _model_id / _linear_layers -------------------------------------------


def test_model_id_uses_config_name_or_path():
    model = types.SimpleNamespace(config=types.SimpleNamespace(_name_or_path="org/x"))
    assert _model_id(model) == "org/x"


def test_model_id_falls_back_to_class_name(tiny_model):
    assert _model_id(tiny_model) == "Sequential"


def test_model_id_falls_back_when_name_or_path_empty():
    model = types.SimpleNamespace(config=types.SimpleNamespace(_name_or_path=""))
    assert _model_id(model) == "SimpleNamespace"


def test_linear_layers_other_params_is_total_minus_linear(tiny_model):
    layers, other = _linear_layers(tiny_model)
    total = sum(p.numel() for p in tiny_model.parameters())
    assert other == total - sum(n for _, n in layers)
    assert other == 16  # LayerNorm(8): weight + bias


def test_linear_layers_rejects_non_module():
    with pytest.raises(TypeError):
        _linear_layers("nope")


def test_linear_layers_rejects_no_linear():
    with pytest.raises(ValueError):
        _linear_layers(nn.LayerNorm(3))


# ----- _check_budget ---------------------------------------------------------


@pytest.mark.parametrize("bad", [0, 1, -0.5, 2.0])
def test_check_budget_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        api._check_budget(bad)


def test_check_budget_accepts_fraction():
    assert api._check_budget(0.5) is None


# ----- snap_budget -----------------------------------------------------------


@pytest.mark.parametrize("budget", BUDGETS)
def test_snap_budget_exact_values_map_to_themselves(budget):
    assert snap_budget(budget) == budget


@pytest.mark.parametrize(
    "requested,expected",
    [(0.015, 0.01), (0.049, 0.03), (0.06, 0.05), (0.9, 0.05), (0.0011, 0.001)],
)
def test_snap_budget_rounds_down(requested, expected):
    assert snap_budget(requested) == expected


def test_snap_budget_rejects_below_smallest():
    with pytest.raises(ValueError, match="smallest"):
        snap_budget(0.0005)


def test_snap_budget_tolerates_float_noise():
    assert snap_budget(0.1 / 100) == 0.001


def test_snap_budget_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        snap_budget(0)


# ----- results_path / load_plan ---------------------------------------------


def test_results_path_slugs_model_id(tmp_path):
    path = results_path(tmp_path, MODEL_ID, 0.01, "nearest")
    assert path.parent == tmp_path / "org__model-name"


def test_results_path_filename(tmp_path):
    assert results_path(tmp_path, MODEL_ID, 0.01, "nearest").name == "budget_1.00pct_nearest.json"
    assert results_path(tmp_path, MODEL_ID, 0.0025, Rounding.STOCHASTIC).name == (
        "budget_0.25pct_stochastic.json"
    )


def test_load_plan_snaps_budget_and_loads(sample_plan, tmp_path):
    sample_plan.save(results_path(tmp_path, sample_plan.model_id, 0.01, "nearest"))
    loaded = pp.load_plan(sample_plan.model_id, 0.015, "nearest", results_dir=tmp_path)
    assert loaded == sample_plan
    assert loaded.max_accuracy_drop == 0.01


def test_load_plan_missing_file_mentions_generator(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_results.py"):
        pp.load_plan(MODEL_ID, 0.01, results_dir=tmp_path)


def test_public_api_exports():
    assert set(pp.__all__) >= {"analyze", "load_plan", "Plan", "Format", "BUDGETS"}
    assert all(hasattr(pp, name) for name in pp.__all__)
    assert list(BUDGETS) == sorted(BUDGETS)
    assert pp.SUPPORTED_MODELS
