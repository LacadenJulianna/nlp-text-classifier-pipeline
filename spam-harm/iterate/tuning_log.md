# Candidate Model — Tuning Log

## Grid searched

```
{'tfidf__ngram_range': [(1, 1), (1, 2)], 'tfidf__stop_words': [None, 'english'], 'clf__C': [0.01, 0.1, 1, 10]}
```

## Best result

- **Best CV spam F1:** 1.0000
- **Best params:** {'clf__C': 1, 'tfidf__ngram_range': (1, 1), 'tfidf__stop_words': None}

## Stopword removal: does it help?

Mean CV spam F1 by `tfidf__stop_words` setting (averaged over all other grid params, not just the single best row):

```
params
english    0.628579
NaN        0.702080
Name: mean_test_score, dtype: float64
```

`None` performs better on average -- kept

## Stemming: considered and discarded

Not implemented for this candidate. Stemming (e.g. via NLTK's PorterStemmer) would require an extra dependency plus a corpus download, which doesn't pay for itself on a dataset and project this size ("NLP-lite" per the project's own scope). Worth revisiting only if a future iteration shows TF-IDF token sparsity is actually a bottleneck.
