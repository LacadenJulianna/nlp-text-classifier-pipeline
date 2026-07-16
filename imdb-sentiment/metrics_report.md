# IMDB Sentiment Classifier — Metrics

Model: TF-IDF (unigrams+bigrams, max_features=30000, min_df=5, sublinear_tf) +
Logistic Regression (C=10, liblinear).

Train/test split: 80/20 stratified, random_state=42.
Train size: 40000, test size: 10000.
Training time: 14.8s.

**Accuracy: 0.9046**

## Classification report

```
              precision    recall  f1-score   support

    negative     0.9102    0.8978    0.9039      5000
    positive     0.8992    0.9114    0.9052      5000

    accuracy                         0.9046     10000
   macro avg     0.9047    0.9046    0.9046     10000
weighted avg     0.9047    0.9046    0.9046     10000

```

## Confusion matrix (rows=actual, cols=predicted; order: negative, positive)

```
[[4489  511]
 [ 443 4557]]
```
