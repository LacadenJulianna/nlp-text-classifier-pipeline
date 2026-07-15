# Day 19 — Deployment Guide (Streamlit Community Cloud)

I can't click through your GitHub/Streamlit accounts for you, so this is a
step-by-step guide rather than something I ran myself. Everything the guide
depends on (`requirements.txt`, `app.py`) has been tested locally first.

## Before you start

`stage04` needs the model files to actually be present in your repo for
deployment to work — Streamlit Cloud only has access to what's committed to
GitHub, not your local machine. Confirm these are committed:
- `stage03/final_model.joblib`
- `stage02/features/feature_pipeline.joblib`

If either was excluded via `.gitignore` at any point (worth checking — some
earlier commits gitignored regenerable output), you'll need to force-add it
specifically for deployment, since the app can't regenerate it on a cloud
host without your full training pipeline running there too.

## Steps

1. **Push everything to GitHub first**, including `stage04/` and the two
   model files above. Streamlit Cloud deploys directly from a GitHub repo —
   nothing to upload manually.

2. Go to **share.streamlit.io** and sign in with GitHub.

3. Click **"New app"**.

4. Fill in:
   - **Repository:** `LacadenJulianna/nlp-text-classifier-pipeline` (or wherever this ended up)
   - **Branch:** `main`
   - **Main file path:** `stage04/app.py`

5. Click **"Advanced settings"** and confirm the Python version matches what
   you tested with locally (3.11, based on your earlier terminal output) —
   mismatched Python versions can cause the same kind of subtle
   incompatibility as the sklearn version mismatch caught during Day 19
   testing here.

6. Click **Deploy**. First deploy takes a few minutes — it's installing
   everything in `requirements.txt` fresh.

7. **Once live, actually test it** — don't just confirm it loads. Run the
   same smoke-test input used throughout this build (Animation + Family
   genres, 2016, 23-minute movie) and confirm it still predicts TV-G. If it
   doesn't match what ran locally, something about the cloud environment
   differs from what was tested here, and that's worth chasing down before
   calling this "deployed."

## If it fails to deploy

The Streamlit Cloud logs (visible from your app's dashboard) will show the
actual error. Two likely candidates, given what came up during local
testing:
- **`FileNotFoundError` on the model files** — almost certainly means one of
  the two `.joblib` files from step 1 wasn't actually committed.
- **A version-related unpickling warning/error** — means the cloud
  environment didn't install the exact pinned versions from
  `requirements.txt`. Check the deploy log for which versions actually got
  installed.

## The public URL

Once deployed, Streamlit gives you a URL like
`https://<something>.streamlit.app`. That's Day 19's actual deliverable —
save it, since Day 21's README and Day 22's slide deck both need it.
