# IMDB Sentiment Classifier

Classifies movie reviews as positive or negative. TF-IDF (unigrams+bigrams) +
Logistic Regression, trained on the classic 50,000-review labeled IMDB
dataset (25,000 positive / 25,000 negative).

**Test accuracy: 90.46%** (macro F1 0.9046) — see [`metrics_report.md`](metrics_report.md).

## Structure

```
data/raw/IMDB_Dataset.csv   Raw dataset (review, sentiment columns) — not committed, see below
train.py                    Loads, cleans (strips HTML), trains, evaluates, saves the model
final_model.joblib          Trained sklearn Pipeline (TfidfVectorizer + LogisticRegression)
metrics_report.md           Accuracy, classification report, confusion matrix
app/predict.py              predict_sentiment(review) -> {"sentiment", "confidence"}
app/app.py                  Streamlit UI
app/test_predict.py         Unit tests (happy path + edge cases)
app/requirements.txt        Deployment dependencies
app/DEPLOYMENT.md           Streamlit Community Cloud deployment steps
```

## Why no baseline/candidate iteration stage

Unlike `spam-harm/`, this project trains a single tuned model directly rather
than iterating baseline → candidate → tuned. TF-IDF + Logistic Regression is
already the well-established strong classical approach for this dataset (published
benchmarks cluster around 88-90% accuracy), and 50,000 rows makes a full
grid-search sweep expensive for the marginal gain. One well-chosen
configuration reaches that ceiling directly.

## Dataset

Not committed to git (66MB, over the repo's usual data-file size — `spam-harm/`'s
raw file was 321KB by comparison). To retrain: place the classic
"IMDB Dataset of 50K Movie Reviews" CSV (`review`, `sentiment` columns) at
`data/raw/IMDB_Dataset.csv`, then run `python train.py`.

## Running locally

```
cd imdb-sentiment/app
pip install -r requirements.txt
streamlit run app.py
```

## Known limitations

- Trained only on IMDB-style long-form prose reviews; short or sarcastic
  text (e.g. "yeah, real *great* movie") may be misclassified — classical
  TF-IDF features don't capture sarcasm.
- Binary positive/negative only, no neutral class.
