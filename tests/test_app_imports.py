"""Guardrail: the app may use only the public API, so the stub can be swapped out safely."""

import ast
from pathlib import Path

import precision_planner as pp

APP = Path(__file__).resolve().parent.parent / "app.py"


def _tree():
    return ast.parse(APP.read_text(encoding="utf-8"))


def test_app_imports_no_precision_planner_submodules():
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("precision_planner"), f"app.py imports from {node.module}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name == "precision_planner" or not alias.name.startswith("precision_planner"), alias.name


def test_app_uses_only_public_names():
    used = {
        node.attr
        for node in ast.walk(_tree())
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "pp"
    }
    assert used, "expected app.py to use the pp namespace"
    assert used <= set(pp.__all__), f"non-public names used: {sorted(used - set(pp.__all__))}"
