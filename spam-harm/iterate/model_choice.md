# Model Choice: Baseline vs. Candidate

| Metric | Baseline (Naive Bayes) | Candidate (tuned Logistic Regression) |
|---|---|---|
| Accuracy | 1.0000 | 1.0000 |
| Spam F1 | 1.0000 | 1.0000 |
| Spam precision | 1.0000 | 1.0000 |
| Spam recall | 1.0000 | 1.0000 |

**Selected: candidate**, based on spam F1 on the held-out test set (the metric this project treats as the headline number, not accuracy).

## Per-class breakdown, baseline

```
              precision    recall  f1-score   support

         ham       1.00      1.00      1.00       100
        spam       1.00      1.00      1.00        36

    accuracy                           1.00       136
   macro avg       1.00      1.00      1.00       136
weighted avg       1.00      1.00      1.00       136

```

## Per-class breakdown, candidate

```
              precision    recall  f1-score   support

         ham       1.00      1.00      1.00       100
        spam       1.00      1.00      1.00        36

    accuracy                           1.00       136
   macro avg       1.00      1.00      1.00       136
weighted avg       1.00      1.00      1.00       136

```
