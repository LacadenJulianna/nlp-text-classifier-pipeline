# Day 14 — Tuning Log

Grid: `{'n_estimators': [200, 300, 500], 'max_depth': [None, 15, 25], 'min_samples_leaf': [1, 2]}`

Best CV macro F1 (training data, 3-fold): 0.5138

Best params: `{'max_depth': None, 'min_samples_leaf': 2, 'n_estimators': 200}`

## Final test-set result

- Macro F1: 0.5962
- Accuracy: 0.6310

```
              precision    recall  f1-score   support

           G       0.63      0.71      0.67        51
          PG       0.67      0.62      0.64        47
       PG-13       0.90      0.69      0.78        13
       TV-14       0.33      0.06      0.11        16
        TV-G       0.59      0.64      0.62        64
       TV-PG       0.57      0.63      0.60        60
        TV-Y       0.86      0.60      0.71        10
       TV-Y7       0.67      0.85      0.75        26
    TV-Y7-FV       1.00      0.33      0.50         3

    accuracy                           0.63       290
   macro avg       0.69      0.57      0.60       290
weighted avg       0.63      0.63      0.62       290

```
