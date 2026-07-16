import pytest
from predict import predict_sentiment


def test_positive_review():
    result = predict_sentiment("A stunning, beautifully acted masterpiece. I loved it.")
    assert result["sentiment"] == "positive"
    assert result["confidence"] > 0.5


def test_negative_review():
    result = predict_sentiment("Absolutely terrible. Waste of time, bad acting, worse plot.")
    assert result["sentiment"] == "negative"
    assert result["confidence"] > 0.5


def test_empty_string_rejected():
    with pytest.raises(ValueError):
        predict_sentiment("")


def test_none_rejected():
    with pytest.raises(ValueError):
        predict_sentiment(None)


def test_html_only_rejected():
    with pytest.raises(ValueError):
        predict_sentiment("<br /><br />")


def test_html_tags_stripped_before_prediction():
    result = predict_sentiment("<br />This was great!<br />Loved it.<br />")
    assert result["sentiment"] == "positive"
