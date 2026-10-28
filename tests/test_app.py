import dataclasses

import gradio as gr
import pytest

import app
import precision_planner as pp
from precision_planner.api import results_path


@pytest.fixture
def real_plan(sample_plan):
    return dataclasses.replace(sample_plan, is_sample=False)


@pytest.fixture
def results_dir(tmp_path, sample_plan):
    for rounding in pp.Rounding:
        for budget in pp.BUDGETS:
            plan = dataclasses.replace(sample_plan, max_accuracy_drop=budget, rounding=rounding)
            plan.save(results_path(tmp_path, plan.model_id, budget, rounding))
    return tmp_path


def test_budget_from_percent():
    assert app.budget_from_percent(1.0) == 0.01
    assert app.budget_from_percent(0.1) == pytest.approx(0.001)


def test_fmt_pct_and_mb():
    assert app.fmt_pct(0.01234) == "1.23%"
    assert app.fmt_mb(12.345) == "12.3 MB"
    assert app.fmt_mb(None) == "n/a"


def test_fmt_latency():
    assert app.fmt_latency(pp.Metrics(0.9, latency_ms_mean=10.0, latency_ms_std=0.5)) == "10.0 ± 0.5 ms"
    assert app.fmt_latency(pp.Metrics(0.9)) == "n/a (accuracy-only)"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("distilbert.transformer.layer.3.ffn.lin1", ("Block 3", "ffn.lin1")),
        ("layer.0.attention.q_lin", ("Block 0", "attention.q_lin")),
        ("classifier", ("Head", "classifier")),
        ("encoder.dense", ("Head", "encoder.dense")),
    ],
)
def test_split_layer_name(name, expected):
    assert app.split_layer_name(name) == expected


def test_within_budget_tolerates_float_noise():
    assert app.within_budget(0.01 + 1e-12, 0.01)
    assert not app.within_budget(0.0101, 0.01)


def test_banner_only_for_sample_data(sample_plan, real_plan):
    assert "Demo data: real measurements coming soon" in app.banner_html(sample_plan)
    assert app.banner_html(real_plan) == ""


def test_summary_notes_snapped_budget(sample_plan):
    text = app.summary_markdown(sample_plan, 1.3)
    assert "You asked for 1.3%" in text
    assert "1.00%" in text


def test_summary_omits_snap_note_for_exact_budget(sample_plan):
    assert "You asked for" not in app.summary_markdown(sample_plan, 1.0)


def test_summary_flags_over_budget(sample_plan):
    over = dataclasses.replace(sample_plan, max_accuracy_drop=sample_plan.accuracy_drop / 2)
    assert "over budget" in app.summary_markdown(over, 1.0)


def test_layer_plan_html_has_one_chip_per_layer_with_format_color(sample_plan):
    out = app.layer_plan_html(sample_plan)
    assert out.count('class="pp-chip"') == len(sample_plan.layers)
    for layer in sample_plan.layers:
        assert app.FORMAT_COLORS[layer.format] in out
    assert "Block 5" in out and "Head" in out


def test_layer_plan_html_escapes_names(sample_plan):
    layer = pp.LayerAssignment("<script>", pp.Format.FP32, 1)
    plan = dataclasses.replace(sample_plan, layers=(layer,))
    out = app.layer_plan_html(plan)
    assert "<script>" not in out and "&lt;script&gt;" in out


def test_every_format_has_a_color():
    assert set(app.FORMAT_COLORS) == set(pp.Format)


def test_metric_tiles_has_four_tiles(sample_plan):
    out = app.metric_tiles_html(sample_plan)
    assert out.count('class="pp-tile"') == 4
    for key in ("Accuracy", "Model size", "Latency", "Peak memory"):
        assert key in out


def test_metric_tiles_handles_missing_latency(sample_plan):
    metrics = dataclasses.replace(sample_plan.metrics, latency_ms_mean=None, latency_ms_std=None)
    out = app.metric_tiles_html(dataclasses.replace(sample_plan, metrics=metrics))
    assert "n/a (accuracy-only)" in out and "not measured" in out


def test_relative():
    assert app._relative(50.0, 100.0, "smaller") == "50% smaller than FP32 (100.0)"
    assert app._relative(150.0, 100.0, "smaller") == "50% larger than FP32 (100.0)"
    assert app._relative(None, 100.0, "smaller") == "not measured"
    assert app._relative(1.0, 0.0, "smaller") == "not measured"


def test_tile():
    assert app._tile("K", "V", "D") == (
        '<div class="pp-tile"><div class="k">K</div><div class="v">V</div><div class="d">D</div></div>'
    )


def test_baseline_rows_order_and_shape(sample_plan):
    rows = app.baseline_rows(sample_plan)
    assert [r[0] for r in rows] == ["Precision plan", "FP32", "Uniform FP16", "Uniform INT8", "Random mixed"]
    assert all(len(r) == len(app.BASELINE_HEADERS) for r in rows)
    fp32 = rows[1]
    assert fp32[2] == "0.00%" and fp32[6] == "yes"


def test_render_returns_all_outputs(results_dir):
    banner, summary, tiles, layers, rows = app.render(next(iter(pp.SUPPORTED_MODELS)), 1.3, "stochastic", results_dir)
    assert "Demo data" in banner
    assert "stochastic" in summary and "1.00%" in summary
    assert "pp-tile" in tiles and "pp-chip" in layers
    assert len(rows) == 5


def test_render_against_committed_results():
    """The results/ shipped with the app must load for every slider extreme and rounding."""
    model_id = next(iter(pp.SUPPORTED_MODELS))
    for rounding in pp.Rounding:
        for percent in (0.1, 1.0, 5.0):
            assert len(app.render(model_id, percent, rounding.value)[4]) == 5


def test_build_demo_returns_blocks():
    assert isinstance(app.build_demo(), gr.Blocks)
