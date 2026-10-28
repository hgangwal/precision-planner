"""Deployment config must agree on the Gradio version, or Render, CI and Codespaces drift apart."""

import json
import re
from pathlib import Path

import yaml

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


def test_readme_python_version_is_supported_by_zerogpu():
    """For the optional HF Space: free accounts can only use ZeroGPU, which supports these exact versions."""
    front_matter = (ROOT / "README.md").read_text(encoding="utf-8").split("---")[1]
    version = re.search(r'^python_version:\s*"?([\d.]+)"?$', front_matter, re.MULTILINE).group(1)
    assert version in {"3.12.12", "3.10.13"}


def test_gradio_version_matches_across_configs():
    space = _readme_sdk_version()
    runtime = _pinned((ROOT / "requirements.txt").read_text(encoding="utf-8"), "gradio")
    dev = _pinned((ROOT / "requirements-dev.txt").read_text(encoding="utf-8"), "gradio")
    devcontainer = json.loads((ROOT / ".devcontainer" / "devcontainer.json").read_text(encoding="utf-8"))
    assert runtime == dev == space
    assert _pinned(devcontainer["postCreateCommand"], "gradio") == space


def test_runtime_requirements_exclude_torch():
    """The hosted app only renders saved plans; torch would add ~1 GB to every deploy."""
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    requirements = [line for line in lines if line.strip() and not line.startswith("#")]
    assert requirements == ["gradio==6.31.0"]


def test_render_blueprint_serves_app_publicly_on_free_plan():
    blueprint = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    (service,) = blueprint["services"]
    env = {var["key"]: var["value"] for var in service["envVars"]}
    assert service["plan"] == "free"
    assert service["startCommand"] == "python app.py"
    assert service["buildCommand"] == "pip install -r requirements.txt"
    assert env["GRADIO_SERVER_NAME"] == "0.0.0.0"
    assert env["GRADIO_SERVER_PORT"] == "10000"


def test_devcontainer_forwards_dashboard_port():
    devcontainer = json.loads((ROOT / ".devcontainer" / "devcontainer.json").read_text(encoding="utf-8"))
    assert 7860 in devcontainer["forwardPorts"]
    assert devcontainer["postAttachCommand"] == "python app.py"
