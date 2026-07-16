# Deployment Guide (Streamlit Community Cloud)

## Before you start

This app needs `imdb-sentiment/final_model.joblib` committed to the repo —
Streamlit Cloud only sees what's on GitHub, not your local machine. Confirm
it's tracked (not gitignored):

```bash
git ls-files imdb-sentiment/final_model.joblib
```

If that prints nothing, force-add it: `git add -f imdb-sentiment/final_model.joblib`.

The raw 66MB training CSV (`imdb-sentiment/data/raw/IMDB_Dataset.csv`) is
deliberately NOT committed — only the trained model is needed at runtime.

## Steps

1. Push everything to GitHub, including `imdb-sentiment/` and `final_model.joblib`.
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**.
4. Fill in:
   - **Repository:** `LacadenJulianna/nlp-text-classifier-pipeline`
   - **Branch:** `main`
   - **Main file path:** `imdb-sentiment/app/app.py`
5. Click **Deploy**.
6. **Once live, verify it against the exact documented smoke-test input
   before calling this "deployed"** — a hard-learned rule from the earlier
   Disney+/spam-harm deployments, not optional:
   - Paste: `This movie was a stunning, beautifully acted masterpiece. I loved every minute of it.` → expect **Positive**, ~99.8% confidence.
   - Paste: `Absolutely terrible. Waste of time, bad acting, worse plot.` → expect **Negative**, ~99.9% confidence.
   - If either doesn't match, do NOT assume the environment is broken —
     first confirm you used the *exact* input above, not a paraphrase or a
     different example. Only chase environment/dependency mismatches once
     the exact input is confirmed and still disagrees.

## If it fails to deploy

Check the Streamlit Cloud deploy logs (Manage app dashboard):
- `FileNotFoundError` on the model file → it wasn't actually committed (see "Before you start").
- A version-related unpickling error → the installed versions didn't match `requirements.txt`; check what the log says actually got installed.

## The public URL

Save the `https://<something>.streamlit.app` URL once live — it goes in `README.md` below.
