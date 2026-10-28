import dataclasses

import pytest

from precision_planner import _sample
from precision_planner._sample import (
    DISTILBERT_LINEAR_LAYERS,
    DISTILBERT_OTHER_PARAMS,
    DISTILBERT_SST2,
    HELD_OUT_SIZE,
    _greedy,
    _metrics,
    _quantize_accuracy,
    _seed,
    make_sample_plan,
    sample_sensitivity,
)
from precision_planner.api import BUDGETS
from precision_planner.formats import Format, Rounding

LINEAR_TOTAL = 43_100_930


def _plan(budget, rounding="nearest"):
    return make_sample_plan(
        DISTILBERT_SST2, DISTILBERT_LINEAR_LAYERS, DISTILBERT_OTHER_PARAMS, budget, rounding
    )


def test_distilbert_layer_table_sizes():
    assert len(DISTILBERT_LINEAR_LAYERS) == 38
    assert sum(n for _, n in DISTILBERT_LINEAR_LAYERS) == LINEAR_TOTAL
    assert LINEAR_TOTAL + DISTILBERT_OTHER_PARAMS == 66_955_010


def test_make_sample_plan_is_deterministic():
    a, b = _plan(0.01), _plan(0.01)
    b = dataclasses.replace(b, created_at=a.created_at)
    assert a == b


def test_make_sample_plan_is_sample(sample_plan):
    assert sample_plan.is_sample is True
    assert sample_plan.model_id == DISTILBERT_SST2
    assert len(sample_plan.layers) == 38


def test_larger_budgets_shrink_size_and_lower_more_layers():
    plans = [_plan(b) for b in sorted(BUDGETS)]
    sizes = [p.metrics.size_mb for p in plans]
    lowered = [38 - p.format_counts()[Format.FP32] for p in plans]
    assert sizes == sorted(sizes, reverse=True)
    assert lowered == sorted(lowered)


def test_fp32_baseline_size(sample_plan):
    expected = round((LINEAR_TOTAL + DISTILBERT_OTHER_PARAMS) * 4 / 2**20, 1)
    assert sample_plan.baselines["fp32"].size_mb == expected


def test_uniform_fp16_baseline_has_no_latency(sample_plan):
    m = sample_plan.baselines["fp16"]
    assert m.latency_ms_mean is None and m.latency_ms_std is None


def test_fp32_baseline_has_latency(sample_plan):
    assert sample_plan.baselines["fp32"].latency_ms_mean == 28.4


@pytest.mark.parametrize("budget", BUDGETS)
def test_sample_accuracies_are_multiples_of_held_out_count(budget):
    plan = _plan(budget)
    for m in (plan.metrics, *plan.baselines.values()):
        scaled = m.accuracy * HELD_OUT_SIZE
        assert abs(scaled - round(scaled)) < 1e-9


def test_stochastic_and_nearest_plans_differ():
    assert _plan(0.01, "stochastic").layers != _plan(0.01, "nearest").layers


def test_make_sample_plan_records_budget_and_rounding():
    plan = _plan(0.01)
    assert plan.max_accuracy_drop == 0.01
    assert plan.rounding is Rounding.NEAREST


def test_make_sample_plan_accepts_string_rounding():
    assert _plan(0.01, "stochastic").rounding is Rounding.STOCHASTIC


def test_sample_sensitivity_fp32_zero_and_fp16_below_int8():
    drops = sample_sensitivity("distilbert.transformer.layer.2.attention.q_lin", DISTILBERT_SST2)
    assert drops[Format.FP32] == 0
    assert 0 < drops[Format.FP16] < drops[Format.INT8]
    assert set(drops) == set(Format)


def test_sample_sensitivity_is_deterministic():
    a = sample_sensitivity("classifier", "m")
    assert a == sample_sensitivity("classifier", "m")
    assert a != sample_sensitivity("classifier", "other")


def test_sample_sensitivity_unknown_layer_uses_default_base():
    drops = sample_sensitivity("foo.bar", "m")
    assert 0.0010 * 0.8 <= drops[Format.INT8] <= 0.0010 * 1.2


def test_seed_is_stable_int():
    assert isinstance(_seed("a", 1), int)
    assert _seed("a", 1) == _seed("a", 1)
    assert _seed("a", 1) != _seed("a", 2)


def test_quantize_accuracy_clamps():
    assert _quantize_accuracy(-0.5) == 0.0
    assert _quantize_accuracy(1.5) == 1.0


def test_quantize_accuracy_snaps_to_count():
    assert _quantize_accuracy(0.5) == 218 / 436
    assert _quantize_accuracy(0.9166) == round(0.9166 * 436) / 436


def _drops():
    return {
        "a": {Format.FP32: 0.0, Format.FP16: 0.001, Format.INT8: 0.002, Format.FP8: 0.003},
        "b": {Format.FP32: 0.0, Format.FP16: 0.002, Format.INT8: 0.004, Format.FP8: 0.003},
    }


def test_greedy_tiny_budget_keeps_fp32():
    assert _greedy(_drops(), 0.0005) == {"a": Format.FP32, "b": Format.FP32}


def test_greedy_partial_fp16_pass():
    assert _greedy(_drops(), 0.0015) == {"a": Format.FP16, "b": Format.FP32}


def test_greedy_second_pass_goes_to_least_sensitive_first():
    assert _greedy(_drops(), 0.0045) == {"a": Format.INT8, "b": Format.FP16}


def test_greedy_picks_best_one_byte_format():
    assert _greedy(_drops(), 0.1) == {"a": Format.INT8, "b": Format.FP8}


def test_metrics_hand_checked():
    drops = _drops()
    m = _metrics({"a": Format.INT8, "b": Format.FP16}, drops, {"a": 2**20, "b": 2**20}, 2**20)
    assert m.size_mb == round((4 + 1 + 2), 1)
    assert m.peak_memory_mb == round(7 + 61.0, 1)
    assert m.accuracy == _quantize_accuracy(400 / 436 - (0.002 + 0.002) * 1.05)
    assert m.latency_ms_mean is not None


def test_metrics_uniform_fp16_has_no_latency():
    m = _metrics({"a": Format.FP16}, _drops(), {"a": 10}, 0)
    assert m.latency_ms_mean is None and m.latency_ms_std is None


def test_metrics_int8_speeds_up_latency():
    fp32 = _metrics({"a": Format.FP32}, _drops(), {"a": 10}, 0)
    int8 = _metrics({"a": Format.INT8}, _drops(), {"a": 10}, 0)
    assert int8.latency_ms_mean < fp32.latency_ms_mean


def test_module_constants_consistent():
    assert _sample.HELD_OUT_SIZE == 436
