"""Precision Planner dashboard (Gradio).

The app renders precomputed plans from results/. It imports only the public
API (``precision_planner.__all__``), which tests/test_app_imports.py enforces,
so the real library can replace the stub without touching this file.

Helpers return plain strings/lists so they can be unit-tested without Gradio.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import gradio as gr

import precision_planner as pp

RESULTS_DIR = Path(__file__).resolve().parent / "results"

FORMAT_COLORS = {
    pp.Format.FP32: "#3b5bdb",
    pp.Format.FP16: "#2f9e44",
    pp.Format.INT8: "#e8590c",
    pp.Format.FP8: "#9c36b5",
}

_BLOCK_RE = re.compile(r"(?:^|\.)layer\.(\d+)\.(.+)$")

CSS = """
.pp-banner { background:#fff3bf; color:#5c3c00; border:1px solid #f59f00; border-radius:8px;
             padding:10px 14px; font-weight:600; }
.pp-tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
.pp-tile { border:1px solid var(--border-color-primary); border-radius:8px; padding:12px;
           background:var(--block-background-fill); }
.pp-tile .k { font-size:12px; opacity:.75; text-transform:uppercase; letter-spacing:.04em; }
.pp-tile .v { font-size:22px; font-weight:700; margin:4px 0; font-variant-numeric:tabular-nums; }
.pp-tile .d { font-size:13px; opacity:.85; }
.pp-legend { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:8px; font-size:13px; }
.pp-legend span::before { content:""; display:inline-block; width:10px; height:10px;
                          border-radius:2px; margin-right:5px; background:var(--c); }
.pp-group { display:flex; align-items:center; gap:6px; margin:4px 0; flex-wrap:wrap; }
.pp-group .g { width:64px; font-size:12px; opacity:.75; flex-shrink:0; }
.pp-table-wrap { overflow-x:auto; }
.pp-table { border-collapse:collapse; width:100%; font-size:14px; font-variant-numeric:tabular-nums; }
.pp-table th, .pp-table td { padding:7px 10px; border-bottom:1px solid var(--border-color-primary);
                             text-align:right; white-space:nowrap; }
.pp-table th:first-child, .pp-table td:first-child { text-align:left; }
.pp-table th { font-size:12px; text-transform:uppercase; letter-spacing:.04em; opacity:.75; }
.pp-table tr.plan td { font-weight:700; }
.pp-chip { color:#fff; border-radius:4px; padding:3px 7px; font-size:12px;
           font-family:var(--font-mono); white-space:nowrap; }
"""


# ----- formatting helpers ----------------------------------------------------


def budget_from_percent(percent: float) -> float:
    """Slider percent (1.0) -> API fraction (0.01)."""
    return percent / 100.0


def fmt_pct(fraction: float) -> str:
    return f"{fraction:.2%}"


def fmt_mb(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} MB"


def fmt_latency(metrics: pp.Metrics, missing: str = "n/a (accuracy-only)") -> str:
    if metrics.latency_ms_mean is None:
        return missing
    return f"{metrics.latency_ms_mean:.1f} ± {metrics.latency_ms_std:.1f} ms"


def split_layer_name(name: str) -> tuple[str, str]:
    """Group a layer for display: ``...layer.3.ffn.lin1`` -> ("Block 3", "ffn.lin1")."""
    match = _BLOCK_RE.search(name)
    if match:
        return f"Block {match.group(1)}", match.group(2)
    return "Head", name


def within_budget(drop: float, budget: float) -> bool:
    return drop <= budget + 1e-9


# ----- HTML / table builders -------------------------------------------------


def banner_html(plan: pp.Plan) -> str:
    if not plan.is_sample:
        return ""
    return (
        '<div class="pp-banner">⚠️ Demo data: real measurements coming soon. '
        "Every number on this page is a SAMPLE placeholder, not a measurement.</div>"
    )


def summary_markdown(plan: pp.Plan, requested_percent: float) -> str:
    """Which saved budget is shown, and whether the plan held its budget on held-out data."""
    used = plan.max_accuracy_drop
    lines = [f"**Budget:** {fmt_pct(used)} max accuracy drop · **Rounding:** {plan.rounding.value}"]
    if abs(budget_from_percent(requested_percent) - used) > 1e-9:
        saved = ", ".join(f"{b * 100:g}%" for b in pp.BUDGETS)
        lines.append(
            f"You asked for {requested_percent:g}%; showing the largest saved budget at or below it "
            f"(saved budgets: {saved})."
        )
    verdict = "within budget ✅" if within_budget(plan.accuracy_drop, used) else "over budget ⚠️"
    lines.append(f"**Held-out accuracy drop:** {fmt_pct(plan.accuracy_drop)} ({verdict})")
    return "\n\n".join(lines)


def layer_plan_html(plan: pp.Plan) -> str:
    """Color-coded chips, one per layer, grouped into transformer blocks."""
    legend = "".join(
        f'<span style="--c:{FORMAT_COLORS[fmt]}">{fmt.label} ({n})</span>'
        for fmt, n in plan.format_counts().items()
    )
    groups: dict[str, list[str]] = {}
    for layer in plan.layers:
        group, short = split_layer_name(layer.name)
        chip = (
            f'<span class="pp-chip" style="background:{FORMAT_COLORS[layer.format]}" '
            f'title="{html.escape(layer.name)} · {layer.num_params:,} params">'
            f"{html.escape(short)} · {layer.format.label}</span>"
        )
        groups.setdefault(group, []).append(chip)
    rows = "".join(
        f'<div class="pp-group"><span class="g">{html.escape(g)}</span>{"".join(chips)}</div>'
        for g, chips in groups.items()
    )
    return f'<div class="pp-legend">{legend}</div>{rows}'


def _tile(key: str, value: str, detail: str) -> str:
    return f'<div class="pp-tile"><div class="k">{key}</div><div class="v">{value}</div><div class="d">{detail}</div></div>'


def _relative(plan_value: float | None, fp32_value: float | None, smaller_word: str) -> str:
    if plan_value is None or fp32_value is None or fp32_value == 0:
        return "not measured"
    change = 1 - plan_value / fp32_value
    word = smaller_word if change >= 0 else "larger"
    return f"{abs(change):.0%} {word} than FP32 ({fp32_value:.1f})"


def metric_tiles_html(plan: pp.Plan) -> str:
    """Four tiles comparing the plan against FP32."""
    m, base = plan.metrics, plan.baselines["fp32"]
    drop_pp = (base.accuracy - m.accuracy) * 100
    latency_detail = (
        "not measured"
        if m.latency_ms_mean is None or base.latency_ms_mean is None
        else f"{base.latency_ms_mean / m.latency_ms_mean:.2f}× vs FP32 ({base.latency_ms_mean:.1f} ms)"
    )
    return '<div class="pp-tiles">' + "".join(
        [
            _tile("Accuracy", fmt_pct(m.accuracy), f"{-drop_pp:+.2f} pp vs FP32 ({fmt_pct(base.accuracy)})"),
            _tile("Model size", fmt_mb(m.size_mb), _relative(m.size_mb, base.size_mb, "smaller")),
            _tile("Latency", fmt_latency(m), latency_detail),
            _tile("Peak memory", fmt_mb(m.peak_memory_mb), _relative(m.peak_memory_mb, base.peak_memory_mb, "lower")),
        ]
    ) + "</div>"


BASELINE_HEADERS = ["Config", "Accuracy", "Drop", "Size", "Latency", "Peak mem", "In budget"]


def baseline_rows(plan: pp.Plan) -> list[list[str]]:
    """Plan first, then the four baselines, as display strings."""
    fp32_acc = plan.baselines["fp32"].accuracy
    configs = [("Precision plan", plan.metrics)] + [
        (pp.BASELINE_LABELS[name], plan.baselines[name]) for name in pp.BASELINE_NAMES
    ]
    rows = []
    for label, m in configs:
        drop = fp32_acc - m.accuracy
        rows.append(
            [
                label,
                fmt_pct(m.accuracy),
                fmt_pct(drop),
                fmt_mb(m.size_mb),
                fmt_latency(m, missing="accuracy-only"),
                fmt_mb(m.peak_memory_mb),
                "yes" if within_budget(drop, plan.max_accuracy_drop) else "no",
            ]
        )
    return rows


def baseline_table_html(plan: pp.Plan) -> str:
    """The baseline rows as an HTML table, plan row emphasized."""
    head = "".join(f"<th>{h}</th>" for h in BASELINE_HEADERS)
    body = "".join(
        ('<tr class="plan">' if i == 0 else "<tr>")
        + "".join(f"<td>{html.escape(c)}</td>" for c in row)
        + "</tr>"
        for i, row in enumerate(baseline_rows(plan))
    )
    return f'<div class="pp-table-wrap"><table class="pp-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render(model_id: str, percent: float, rounding: str, results_dir: Path = RESULTS_DIR):
    """Load the saved plan and build every output. Returns a tuple matching the UI outputs."""
    plan = pp.load_plan(model_id, budget_from_percent(percent), rounding, results_dir=results_dir)
    return (
        banner_html(plan),
        summary_markdown(plan, percent),
        metric_tiles_html(plan),
        layer_plan_html(plan),
        baseline_table_html(plan),
    )


# ----- UI --------------------------------------------------------------------

ABOUT = """
**How it works.** For each `nn.Linear` layer, Precision Planner measures how much accuracy drops
when only that layer is lowered to FP16, INT8 or FP8. It then lowers layers greedily, least
sensitive first, and keeps a change only if whole-model accuracy stays within your budget. All
numbers are reported on held-out SST-2 sentences that the search never saw.

**What's real today.** The layer list for DistilBERT-SST2 and the API are real. Format choices and
metrics are sample placeholders until the real conversion, search, and benchmarks land.
Latency is reported only where real CPU kernels exist (FP32, INT8). FP16 and FP8 are accuracy-only.
"""


def build_demo() -> gr.Blocks:
    model_choices = [(label, model_id) for model_id, label in pp.SUPPORTED_MODELS.items()]
    with gr.Blocks(title="Precision Planner") as demo:
        gr.Markdown("# 🎯 Precision Planner\nPer-layer mixed precision (FP32 / FP16 / INT8 / FP8) under an accuracy budget.")
        banner = gr.HTML()
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                model = gr.Dropdown(model_choices, value=model_choices[0][1], label="Model")
                budget = gr.Slider(0.1, 5.0, value=1.0, step=0.05, label="Max accuracy drop (%)")
                rounding = gr.Radio([r.value for r in pp.Rounding], value=pp.Rounding.NEAREST.value, label="Rounding")
                run = gr.Button("Run", variant="primary")
                with gr.Accordion("About", open=False):
                    gr.Markdown(ABOUT)
            with gr.Column(scale=3):
                summary = gr.Markdown()
                tiles = gr.HTML()
                gr.Markdown("### Per-layer plan")
                layers = gr.HTML()
                gr.Markdown("### Plan vs baselines")
                table = gr.HTML()
        outputs = [banner, summary, tiles, layers, table]
        run.click(render, inputs=[model, budget, rounding], outputs=outputs)
        demo.load(render, inputs=[model, budget, rounding], outputs=outputs)
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
