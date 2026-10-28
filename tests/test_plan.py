import dataclasses
import json

import pytest
import torch

import precision_planner as pp
from precision_planner import plan as plan_mod
from precision_planner.formats import Format, Rounding
from precision_planner.plan import (
    BASELINE_NAMES,
    LayerAssignment,
    Metrics,
    Plan,
    SampleDataWarning,
    format_report,
)


# ----- Metrics ---------------------------------------------------------------


@pytest.mark.parametrize("acc", [-0.01, 1.01])
def test_metrics_rejects_accuracy_out_of_range(acc):
    with pytest.raises(ValueError, match="accuracy"):
        Metrics(accuracy=acc)


@pytest.mark.parametrize("field", ["size_mb", "peak_memory_mb"])
def test_metrics_rejects_negative_fields(field):
    with pytest.raises(ValueError, match=field):
        Metrics(accuracy=0.9, **{field: -1.0})


def test_metrics_rejects_negative_latency():
    with pytest.raises(ValueError, match="latency_ms_mean"):
        Metrics(accuracy=0.9, latency_ms_mean=-1.0, latency_ms_std=0.1)


def test_metrics_rejects_mean_without_std():
    with pytest.raises(ValueError, match="both"):
        Metrics(accuracy=0.9, latency_ms_mean=1.0)


def test_metrics_rejects_std_without_mean():
    with pytest.raises(ValueError, match="both"):
        Metrics(accuracy=0.9, latency_ms_std=1.0)


def test_metrics_accepts_boundary_accuracy():
    assert Metrics(accuracy=0.0).accuracy == 0.0
    assert Metrics(accuracy=1.0).accuracy == 1.0


def test_metrics_round_trip():
    m = Metrics(0.9, 10.0, 5.0, 0.5, 70.0)
    assert Metrics.from_dict(m.to_dict()) == m


def test_metrics_round_trip_with_nones():
    m = Metrics(accuracy=0.5)
    assert Metrics.from_dict(m.to_dict()) == m


# ----- LayerAssignment -------------------------------------------------------


def test_layer_assignment_coerces_string_format():
    layer = LayerAssignment("a", "fp16", 3)
    assert layer.format is Format.FP16


def test_layer_assignment_rejects_empty_name():
    with pytest.raises(ValueError, match="name"):
        LayerAssignment("", Format.FP32, 3)


def test_layer_assignment_rejects_negative_params():
    with pytest.raises(ValueError, match="num_params"):
        LayerAssignment("a", Format.FP32, -1)


def test_layer_assignment_rejects_unknown_format():
    with pytest.raises(ValueError):
        LayerAssignment("a", "bf16", 1)


def test_layer_assignment_round_trip():
    layer = LayerAssignment("a.b", Format.INT8, 12)
    assert layer.to_dict() == {"name": "a.b", "format": "int8", "num_params": 12}
    assert LayerAssignment.from_dict(layer.to_dict()) == layer


# ----- Plan ------------------------------------------------------------------


def test_plan_rejects_budget_outside_open_interval(sample_plan):
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="max_accuracy_drop"):
            dataclasses.replace(sample_plan, max_accuracy_drop=bad)


def test_plan_rejects_missing_baseline(sample_plan):
    baselines = {k: v for k, v in sample_plan.baselines.items() if k != "int8"}
    with pytest.raises(ValueError, match="int8"):
        dataclasses.replace(sample_plan, baselines=baselines)


def test_plan_rejects_duplicate_layer_names(sample_plan):
    layers = sample_plan.layers + (sample_plan.layers[0],)
    with pytest.raises(ValueError, match="duplicate"):
        dataclasses.replace(sample_plan, layers=layers)


def test_plan_coerces_rounding_from_string(sample_plan):
    plan = dataclasses.replace(sample_plan, rounding="stochastic")
    assert plan.rounding is Rounding.STOCHASTIC


def test_plan_coerces_layers_to_tuple(sample_plan):
    plan = dataclasses.replace(sample_plan, layers=list(sample_plan.layers))
    assert isinstance(plan.layers, tuple)


def test_plan_is_frozen(sample_plan):
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample_plan.model_id = "x"


def test_accuracy_drop_is_fp32_minus_plan(sample_plan):
    expected = sample_plan.baselines["fp32"].accuracy - sample_plan.metrics.accuracy
    assert sample_plan.accuracy_drop == expected


def test_format_counts_has_all_formats_and_sums(sample_plan):
    counts = sample_plan.format_counts()
    assert set(counts) == set(Format)
    assert sum(counts.values()) == len(sample_plan.layers)


def test_format_counts_zero_for_unused_format():
    plan = _tiny_plan([("a", Format.FP32, 1)])
    assert plan.format_counts() == {
        Format.FP32: 1, Format.FP16: 0, Format.INT8: 0, Format.FP8: 0,
    }


def test_to_dict_from_dict_round_trip(sample_plan):
    assert Plan.from_dict(sample_plan.to_dict()) == sample_plan


def test_to_dict_is_json_serializable(sample_plan):
    data = json.loads(json.dumps(sample_plan.to_dict()))
    assert data["schema_version"] == plan_mod.SCHEMA_VERSION
    assert Plan.from_dict(data) == sample_plan


def test_save_load_round_trip(sample_plan, tmp_path):
    path = sample_plan.save(tmp_path / "plan.json")
    assert path == tmp_path / "plan.json"
    assert Plan.load(path) == sample_plan
    assert Plan.load(str(path)) == sample_plan


def test_save_creates_parent_dirs(sample_plan, tmp_path):
    path = sample_plan.save(tmp_path / "a" / "b" / "plan.json")
    assert path.exists()


def test_from_dict_rejects_wrong_schema_version(sample_plan):
    data = sample_plan.to_dict()
    data["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        Plan.from_dict(data)


def test_from_dict_rejects_missing_schema_version(sample_plan):
    data = sample_plan.to_dict()
    del data["schema_version"]
    with pytest.raises(ValueError, match="schema_version"):
        Plan.from_dict(data)


# ----- reporting -------------------------------------------------------------


def test_report_prints_and_returns_same_text(sample_plan, capsys):
    text = sample_plan.report()
    assert capsys.readouterr().out == text + "\n"
    assert text == format_report(sample_plan)


def test_report_marks_sample_data(sample_plan):
    assert "SAMPLE DATA" in format_report(sample_plan)


def test_report_omits_sample_marker_when_not_sample(sample_plan):
    plan = dataclasses.replace(sample_plan, is_sample=False)
    assert "SAMPLE DATA" not in format_report(plan)


def test_report_shows_na_for_none_latency(sample_plan):
    assert sample_plan.baselines["fp16"].latency_ms_mean is None
    fp16_row = next(
        line for line in format_report(sample_plan).splitlines() if line.startswith("Uniform FP16")
    )
    assert "n/a" in fp16_row


def test_format_report_has_one_line_per_layer(sample_plan):
    lines = format_report(sample_plan).splitlines()
    for layer in sample_plan.layers:
        assert sum(1 for ln in lines if ln.strip().startswith(layer.name + " ")) == 1
    assert f"Layers ({len(sample_plan.layers)})" in "\n".join(lines)


def test_format_report_lists_all_baselines(sample_plan):
    text = format_report(sample_plan)
    for label in pp.BASELINE_LABELS.values():
        assert label in text


def test_format_report_handles_no_layers(sample_plan):
    plan = dataclasses.replace(sample_plan, layers=())
    assert "Layers (0)" in format_report(plan)


def test_fmt_none_and_value():
    assert plan_mod._fmt(None, ".1f", " MB") == "n/a"
    assert plan_mod._fmt(1.25, ".1f", " MB") == "1.2 MB"


def test_fmt_latency_none_and_value():
    assert plan_mod._fmt_latency(Metrics(0.9)) == "n/a"
    assert plan_mod._fmt_latency(Metrics(0.9, latency_ms_mean=2.0, latency_ms_std=0.5)) == "2.0 ± 0.5 ms"


def test_baseline_names_match_labels():
    assert set(BASELINE_NAMES) == set(pp.BASELINE_LABELS)


# ----- apply / _check_layer_names -------------------------------------------


def _tiny_plan(rows):
    m = Metrics(0.9)
    return Plan(
        model_id="t",
        max_accuracy_drop=0.01,
        rounding="nearest",
        layers=tuple(LayerAssignment(n, f, p) for n, f, p in rows),
        metrics=m,
        baselines={k: m for k in BASELINE_NAMES},
        is_sample=True,
    )


@pytest.fixture
def tiny_plan(tiny_model):
    with pytest.warns(SampleDataWarning):
        return pp.analyze(tiny_model, data=[], max_accuracy_drop=0.01)


def test_apply_returns_copy_and_does_not_mutate(tiny_model, tiny_plan):
    before = {k: v.clone() for k, v in tiny_model.state_dict().items()}
    with pytest.warns(SampleDataWarning):
        out = tiny_plan.apply(tiny_model)
    assert out is not tiny_model
    after = tiny_model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_apply_rejects_mismatched_layer_names(sample_plan, tiny_model):
    with pytest.raises(ValueError, match="does not match"):
        sample_plan.apply(tiny_model)


def test_apply_raises_not_implemented_when_not_sample(tiny_model, tiny_plan):
    plan = dataclasses.replace(tiny_plan, is_sample=False)
    with pytest.raises(NotImplementedError):
        plan.apply(tiny_model)


def test_check_layer_names_accepts_match(tiny_model):
    plan_mod._check_layer_names(tiny_model, {"fc1", "fc2", "fc3"})


def test_check_layer_names_reports_missing_and_unplanned(tiny_model):
    with pytest.raises(ValueError) as exc:
        plan_mod._check_layer_names(tiny_model, {"fc1", "fc2", "ghost"})
    msg = str(exc.value)
    assert "ghost" in msg and "fc3" in msg
