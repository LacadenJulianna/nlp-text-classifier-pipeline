# Day 12 — Candidate Model Metrics

| | Macro F1 | Accuracy |
|---|---|---|
| Baseline (Week 2, logistic regression) | 0.5272 | 0.6172 |
| Candidate (Random Forest) | 0.5757 | 0.6000 |
| **Delta** | **+0.0485 (+9.2%)** | -0.0172 |

## Per-class breakdown

```
              precision    recall  f1-score   support

           G       0.61      0.65      0.63        51
          PG       0.65      0.66      0.65        47
       PG-13       0.80      0.62      0.70        13
       TV-14       0.31      0.25      0.28        16
        TV-G       0.62      0.56      0.59        64
       TV-PG       0.53      0.60      0.56        60
        TV-Y       0.50      0.60      0.55        10
       TV-Y7       0.73      0.73      0.73        26
    TV-Y7-FV       1.00      0.33      0.50         3

    accuracy                           0.60       290
   macro avg       0.64      0.56      0.58       290
weighted avg       0.60      0.60      0.60       290

```
