"""Deployment config must agree on the Gradio version, or the Space, CI and Codespaces drift apart."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _readme_sdk_version():
    front_matter = (ROOT / "README.md").read_text(encoding="utf-8").split("---")[1]
    return re.search(r"^sdk_version:\s*(\S+)$", front_matter, re.MULTILINE).group(1)


def _pinned(text, package):
    return re.search(rf"\b{package}==([\w.]+)", text).group(1)


def test_readme_declares_gradio_space():
    front_matter = (ROOT / "README.md").read_text(encoding="utf-8").split("---")[1]
    assert re.search(r"^sdk:\s*gradio$", front_matter, re.MULTILINE)
    assert re.search(r"^app_file:\s*app\.py$", front_matter, re.MULTILINE)


def test_gradio_version_matches_across_configs():
    space = _readme_sdk_version()
    dev = _pinned((ROOT / "requirements-dev.txt").read_text(encoding="utf-8"), "gradio")
    devcontainer = json.loads((ROOT / ".devcontainer" / "devcontainer.json").read_text(encoding="utf-8"))
    assert dev == space
    assert _pinned(devcontainer["postCreateCommand"], "gradio") == space


def test_devcontainer_forwards_dashboard_port():
    devcontainer = json.loads((ROOT / ".devcontainer" / "devcontainer.json").read_text(encoding="utf-8"))
    assert 7860 in devcontainer["forwardPorts"]
    assert devcontainer["postAttachCommand"] == "python app.py"
