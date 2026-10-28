# agents.md

Guide for AI coding agents (and humans) working on this repo.

## Project

Precision Planner picks a numeric precision (FP32 / FP16 / INT8 / FP8) for
each `nn.Linear` layer of a trained model, keeping accuracy loss within a
user-set budget. It is inference only and never retrains. It returns a
per-layer plan, a converted model, and a report against four baselines (FP32,
uniform FP16, uniform INT8, random mixed).

Reference model: `distilbert-base-uncased-finetuned-sst-2-english` on the
872 SST-2 validation sentences, split 50/50 into search and held-out halves
with a fixed seed. Never evaluate on the training set.

## Architecture

- `precision_planner/__init__.py`: the **public API**. `__all__` is the
  contract.
- `api.py`: `analyze()` and `load_plan()`.
- `plan.py`: the `Plan`, `LayerAssignment`, and `Metrics` dataclasses, plus
  JSON save/load, `apply()`, and `report()`.
- `formats.py`: the `Format` and `Rounding` enums, bytes per parameter.
- `_sample.py`: **stub only**. A deterministic sample-data generator. It is
  deleted when the real search lands.
- `app.py`: the Gradio dashboard. It loads saved plans from `results/` via
  `pp.load_plan`.
- `render.yaml`: free Render web service that hosts the dashboard (see
  `docs/deploy.md`).
- `scripts/generate_results.py`: runs `analyze` offline and saves plans to
  `results/`.

The app never runs the search live. It renders precomputed plans, and a
banner appears whenever `plan.is_sample` is true.

## Rules

1. **`app.py` imports only names in `precision_planner.__all__`.** This is
   enforced by `tests/test_app_imports.py`.
2. **Sample data must be labeled.** Anything from `_sample.py` sets
   `is_sample=True`, and the UI shows the banner. Never present sample
   numbers as measured.
3. **No fake measurements.** A metric that wasn't measured on real hardware
   is `None` and is shown as "n/a". Never report energy or cost.
4. **Accuracy uses simulated quantization.** Weights are cast to the format
   and back to FP32. Size and latency use real kernels where they exist.
5. **Benchmarks:** 10 warmup runs, then ≥100 timed runs, reported as
   mean ± std.
6. **`torch` is imported lazily** inside functions that need it. Plan
   loading and the app must work without torch.
7. Every function has a pytest test. Numerical code also gets a small case
   that can be checked by hand, with a stated tolerance.
8. Small, focused commits with imperative messages, for example
   `Add Plan JSON round-trip`.

## Conventions

- Python 3.10+, type hints everywhere, dataclasses for data, enums for
  formats.
- Pure helper functions in `app.py` (no Gradio objects), so they are
  unit-testable.
- Docstrings say *why*, not just what. Keep comments sparse.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                          # run all tests
python scripts/generate_results.py   # regenerate results/*.json
python app.py                   # run the dashboard at http://127.0.0.1:7860
```
