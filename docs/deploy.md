# Deploying the dashboard to Hugging Face Spaces

GitHub is the source of truth. Every push to `main` runs
`.github/workflows/sync-to-hf.yml`, which mirrors the repo into the Space's
own git repo. The Space then rebuilds and serves `app.py`.

```
git push → GitHub (main) → Action "sync-to-hf" → huggingface.co/spaces/<you>/precision-planner → rebuild
```

The Space reads its config from the YAML header at the top of `README.md`:
`sdk: gradio`, `sdk_version`, `python_version`, and `app_file: app.py`.
Spaces installs Gradio itself from `sdk_version`. `requirements.txt` is
intentionally empty because the app needs nothing else, and that keeps the
build fast.

## One-time setup

1. **Hugging Face account:** sign up at <https://huggingface.co/join> if you
   don't have one.
2. **Create the Space:** go to <https://huggingface.co/new-space>.
   - Owner: your username. Space name: `precision-planner`.
   - License: MIT. SDK: **Gradio**, Blank template.
   - Hardware: **CPU basic (free)**. Visibility: Public.
   - Click **Create Space**. Ignore the "clone this repo" instructions,
     because the GitHub Action pushes for us.
3. **Create a token:** go to <https://huggingface.co/settings/tokens> and
   click **Create new token**. Choose **Write**, or Fine-grained with write
   access to this Space. Name it `github-sync`. Copy it; it is shown once.
4. **Give GitHub the token:** in the GitHub repo, go to **Settings → Secrets
   and variables → Actions**.
   - **Secrets** tab, **New repository secret**: Name `HF_TOKEN`, Value =
     the token.
   - **Variables** tab, **New repository variable**: Name `HF_SPACE`,
     Value = `<your-hf-username>/precision-planner`.
5. **Deploy:** merge to `main`, or run the workflow by hand from **Actions →
   sync-to-hf → Run workflow**. Watch the Space's **Logs** tab while it
   builds, which takes about 1–3 minutes.

The app is then live at `https://huggingface.co/spaces/<your-hf-username>/precision-planner`.

## Troubleshooting

- **Action fails with "HF_TOKEN secret is not set":** step 4 is missing or
  the name has a typo.
- **403 from huggingface.co:** the token is read-only. Create a Write token.
- **Space shows "Build error":** open its Logs tab. The usual cause is an
  `sdk_version` in `README.md` that doesn't match a released Gradio version.
- **Space shows "Runtime error" about results:** `results/` must be
  committed. Regenerate with `python scripts/generate_results.py`.
