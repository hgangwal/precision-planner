---
title: Precision Planner
emoji: 🎯
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Per-layer mixed precision under an accuracy budget
---

# Precision Planner

> **Status: Day-1 preview.** The API and dashboard are in place. The numbers
> they show are **sample data** from a stub. Real measurements are coming
> soon.

Precision Planner automatically chooses a numeric precision (FP32, FP16,
INT8, or FP8) for **each layer** of a trained neural network. You set an
accuracy budget, for example "lose at most 1% accuracy", and it keeps the
model inside that budget. It returns the per-layer plan, the converted model,
and a report against simple baselines. Inference only: the model is never
retrained.

```python
import precision_planner as pp

plan = pp.analyze(model, val_data, max_accuracy_drop=0.01)
fast_model = plan.apply(model)
plan.report()
```

## How it works

1. **Baseline:** measure FP32 accuracy, size, latency, and peak memory.
2. **Sensitivity:** lower one layer at a time to each format and record the
   accuracy drop.
3. **Greedy search:** lower the least sensitive layers first, and keep each
   change only if whole-model accuracy stays within budget.
4. **Apply** the plan and **report** results on held-out data the search
   never saw, against FP32, uniform FP16, uniform INT8, and random mixed.

The idea comes from research on mixed-precision Newton's method. In that
work, some steps tolerated FP8 and others needed FP32. Neural network layers
differ in sensitivity in the same way.

## Development

**In the browser:** on GitHub, click **Code → Codespaces → Create codespace**.
It installs Gradio and starts the dashboard, which opens on port 7860.

**Locally:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
python scripts/generate_results.py   # write results/*.json
python app.py                         # dashboard at http://127.0.0.1:7860
```

See [agents.md](agents.md) for architecture and conventions, and
[docs/plan.md](docs/plan.md) for the implementation plan.

## License

MIT
