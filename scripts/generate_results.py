"""Precompute one plan per (model, budget, rounding) and save them for the app.

The dashboard never runs the search live; it renders these files. Today they
are SAMPLE plans from the stub. When the real search lands, this script loads
the model and SST-2 search split and calls pp.analyze() instead, and the app
picks up real numbers with no code change.

Usage: python scripts/generate_results.py [--results-dir results]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import precision_planner as pp
from precision_planner import _sample
from precision_planner.api import results_path


def generate(results_dir: Path) -> list[Path]:
    """Write every precomputed plan under ``results_dir`` and return the paths."""
    written = []
    for model_id in pp.SUPPORTED_MODELS:
        for rounding in pp.Rounding:
            for budget in pp.BUDGETS:
                plan = _sample.make_sample_plan(
                    model_id,
                    _sample.DISTILBERT_LINEAR_LAYERS,
                    _sample.DISTILBERT_OTHER_PARAMS,
                    budget,
                    rounding,
                )
                written.append(plan.save(results_path(results_dir, model_id, budget, rounding)))
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    paths = generate(args.results_dir)
    print(f"Wrote {len(paths)} SAMPLE plans to {args.results_dir}/")


if __name__ == "__main__":
    main()
