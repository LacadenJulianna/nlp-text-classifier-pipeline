# Deployment Guide (Streamlit Community Cloud)

## Before you start

This app needs `spam-harm/iterate/final_model.joblib` committed to the
repo — Streamlit Cloud only sees what's on GitHub, not your local
machine. Confirm it's tracked (not gitignored):

```bash
git ls-files spam-harm/iterate/final_model.joblib
```

If that prints nothing, force-add it: `git add -f spam-harm/iterate/final_model.joblib`.

## Steps

1. Push everything to GitHub, including `spam-harm/` and `final_model.joblib`.
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**.
4. Fill in:
   - **Repository:** `LacadenJulianna/nlp-text-classifier-pipeline`
   - **Branch:** `main`
   - **Main file path:** `spam-harm/app/app.py`
5. Click **Deploy**.
6. **Once live, verify it against the exact documented smoke-test input
   before calling this "deployed"** — this is a hard-learned rule from
   the Disney+ deployment, not optional:
   - Paste: `This is not a scam, click now to see the details! Limited time offer, act now! Get instant access to premium services. For more details, visit our website or contact us directly.` → expect **SPAM**.
   - Paste: `Hey, are we still on for lunch tomorrow at noon?` → expect
     **HAM**.
   - If either doesn't match, do NOT assume the environment is broken —
     first confirm you used the *exact* input above, not a paraphrase
     or a different example. Only chase environment/dependency
     mismatches once the exact input is confirmed and still disagrees.

## If it fails to deploy

Check the Streamlit Cloud deploy logs (Manage app dashboard):
- `FileNotFoundError` on the model file → it wasn't actually committed (see "Before you start").
- A version-related unpickling error → the installed versions didn't match `requirements.txt`; check what the log says actually got installed.

## The public URL

Save the `https://<something>.streamlit.app` URL once live — it goes in `README.md` below.
