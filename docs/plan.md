# Precision Planner: Implementation Plan

Status: **Day 1, approved 2026-10-28**. Last updated 2026-10-28.

## Goal for Day 1

Get a deployed Gradio dashboard on Hugging Face Spaces. It runs on a stub
library whose public API already has its final signatures. Later this week
the real library replaces the stub, and `app.py` does not change.

## Key design decision: the app reads saved plans

A full sensitivity sweep plus greedy search is far too slow for the free CPU
tier, so the work happens in two places:

```
OFFLINE (laptop)                               ONLINE (HF Space)
scripts/generate_results.py                    app.py
  for budget in BUDGETS:                         plan = pp.load_plan(model_id,
    for rounding in ("nearest","stochastic"):                    budget, rounding)
      plan = pp.analyze(model, data, budget,     render(plan)
                        rounding)                if plan.is_sample: show banner
      plan.save("results/...json")  ───────────► results/*.json
```

- **Today**, `generate_results.py` writes sample plans from the stub. Every
  file has `"is_sample": true`.
- **Later**, the same script runs the real `analyze()` and overwrites the
  files with `"is_sample": false`. The banner disappears because the data
  says so, without any edit to `app.py`.
- `BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05)`. The slider is
  continuous, from 0.1% to 5%. `load_plan` picks the **largest saved budget
  that is ≤ the requested one**, so it never shows a plan that breaks the
  user's budget, and the UI says which budget it used.

## Public API (final signatures)

```python
import precision_planner as pp

plan = pp.analyze(model, data, max_accuracy_drop=0.01, rounding="nearest")
fast_model = plan.apply(model)     # returns a converted copy; never mutates input
text = plan.report()               # prints and returns a plain-text report
plan.save(path); pp.Plan.load(path)
plan = pp.load_plan(model_id, max_accuracy_drop, rounding, results_dir="results")
```

Exports: `analyze`, `load_plan`, `Plan`, `LayerAssignment`, `Metrics`,
`Format`, `Rounding`, `BUDGETS`, `SUPPORTED_MODELS`, `SampleDataWarning`.

`Plan` holds:
- the per-layer assignments, each with a name, format, and parameter count
- the plan's metrics plus four baselines: `fp32`, `fp16`, `int8`, and
  `random_mixed`
- provenance: model id, budget, rounding, `is_sample`, library version, and
  creation time

Metrics that are not real measurements are `None`, for example FP16 and FP8
latency on CPU. They are shown as "n/a (accuracy-only)" and never as a
number.

Stub behavior today:
- `analyze` lists the real `nn.Linear` layers of whatever module you pass.
  The format assignments and metrics come from a deterministic sample
  generator (`_sample.py`), marked `is_sample=True`.
- `apply` validates layer names and returns an unchanged deep copy. It
  raises `SampleDataWarning` so nobody mistakes the copy for a converted
  model.

## Repo layout

```
precision_planner/   __init__.py (public API only), api.py, plan.py,
                     formats.py, _sample.py  (later: quantize.py,
                     sensitivity.py, search.py, benchmark.py, data.py)
app.py               Gradio UI + pure helper functions (no Gradio in helpers)
results/             saved plan JSON consumed by the app
scripts/             generate_results.py
tests/               pytest, one test module per source module
docs/plan.md, agents.md, README.md (with HF Spaces metadata header)
pyproject.toml, requirements.txt (Space, pinned), requirements-dev.txt
.github/workflows/   tests.yml (pytest on push), sync-to-hf.yml (deploy)
```

**Dependency boundary:** loading plans and running the app need only
`gradio` and the standard library. `torch` is imported lazily, inside
`analyze`/`apply` only. This keeps the Space small and fast to build, and the
stub doesn't need `torch` today.

**Guardrail test:** a test parses `app.py` with `ast` and fails if the app
imports anything other than names in `precision_planner.__all__`.

## Deployment

GitHub is the source of truth. A GitHub Action pushes `main` to the HF Space
git remote using an `HF_TOKEN` repo secret, which is the sync method Hugging
Face documents. The Space runs the Gradio SDK on the free CPU tier. I'll walk
you through it step by step when we get there.

## Day 1 steps

1. Scaffold: pyproject, pinned requirements, .gitignore, pytest config,
   README stub, agents.md, plan.md. *(commit)*
2. `formats.py` + `plan.py` (dataclasses, JSON round-trip, report) + tests. *(commit)*
3. `_sample.py` + `api.py` (`analyze`, `load_plan`) + tests. *(commit)*
4. `scripts/generate_results.py`, which writes the 14 sample JSON files. *(commit)*
5. `app.py` helpers + UI + helper tests + import-guardrail test. *(commit)*
6. CI workflow, HF metadata, sync workflow. Then you create the Space, and we
   verify the public URL. *(commit)*

## Later this week (not today)

Real fake-quant conversion (with a hand-checkable test: a single `nn.Linear`
in FP16 vs FP32 within a stated tolerance), SST-2 data split, sensitivity
analysis, greedy search, benchmarking (10 warmup + ≥100 timed runs, mean ± std),
baselines, stochastic rounding, and real results.

## Open questions (decide before the real library)

- **Size for mixed plans:** report a *theoretical* size (bytes per parameter
  by format, labeled as computed) or only measured file sizes?
- **Latency for mixed plans on CPU:** FP16 and FP8 layers have no fast CPU
  kernel. Options: run them as FP32, so latency reflects only the INT8 gains,
  or report plan latency as accuracy-only.
