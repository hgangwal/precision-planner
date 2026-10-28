# Deploying the dashboard

The dashboard is hosted on **Render** as a free web service. GitHub is the
source of truth: Render watches `main` and redeploys after every push, once
CI passes.

```
git push → GitHub (main) → CI "tests" passes → Render builds → https://precision-planner-xxxx.onrender.com
```

## Why Render, not Hugging Face Spaces

As of October 2026, Hugging Face requires a paid plan (PRO) to create Gradio
Spaces. Free accounts can host Gradio only on ZeroGPU, and only once the
account is more than 30 days old. Render's free web services run any Python
app with no such limit. The trade-off is that a free instance **sleeps after
about 15 idle minutes**, and the next visitor waits about a minute while it
wakes. Open the link a minute before you demo it.

## How Render runs the app

Everything is in `render.yaml` (a Render "Blueprint"):

| Setting | Value | Why |
|---|---|---|
| `buildCommand` | `pip install -r requirements.txt` | installs only Gradio. The app never imports torch. |
| `startCommand` | `python app.py` | same as running it locally |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | listen on the public interface, not just localhost |
| `GRADIO_SERVER_PORT` | `10000` | the port Render routes traffic to |
| `PYTHON_VERSION` | `3.12.12` | match the dev environment |
| `autoDeployTrigger` | `checksPass` | never deploy a commit whose tests failed |

## One-time setup

1. **Merge to `main`.** Render reads `render.yaml` from your default branch,
   so merge the open PR first.
2. **Sign up:** go to <https://dashboard.render.com/register> and choose
   **GitHub** to sign in. When asked, let Render access the
   `precision-planner` repo. Choosing only that repo is fine.
3. **Create the service from the Blueprint:** in the Render dashboard, click
   **New → Blueprint**, pick `hgangwal/precision-planner`, and click
   **Apply**. Render reads `render.yaml` and creates a web service named
   `precision-planner` on the **Free** plan.
4. **Wait for the first deploy,** about 2–4 minutes. Watch the **Logs** tab
   until you see `Running on local URL: http://0.0.0.0:10000`.
5. **Open the URL** shown at the top of the service page. It looks like
   `https://precision-planner-xxxx.onrender.com`. That is your public link.

## Troubleshooting

- **Blueprint shows "no render.yaml found":** the PR isn't merged into
  `main` yet.
- **Deploy stuck on "Waiting for checks":** CI is still running or failed.
  See the repo's **Actions** tab.
- **"No open ports detected":** `GRADIO_SERVER_NAME` or `GRADIO_SERVER_PORT`
  is missing from the service's **Environment** settings.
- **Page takes about a minute to load:** the free instance was asleep. This
  is normal.
- **"No precomputed plan" error:** `results/` must be committed. Regenerate
  with `python scripts/generate_results.py`.

## Optional: Hugging Face Spaces

`README.md` still carries the Spaces metadata header, and
`.github/workflows/sync-to-hf.yml` mirrors `main` to a Space. The workflow is
skipped until you set the `HF_SPACE` repo variable. To enable it later, with
HF PRO or an account more than 30 days old using ZeroGPU hardware:

1. Create a Gradio Space named `precision-planner`.
2. Create a Write token at <https://huggingface.co/settings/tokens>.
3. In GitHub, go to **Settings → Secrets and variables → Actions** and add:
   - the secret `HF_TOKEN`
   - the variable `HF_SPACE` = `<hf-username>/precision-planner`

ZeroGPU only supports Python 3.12.12 or 3.10.13. That's why `README.md` pins
`python_version: "3.12.12"`.
