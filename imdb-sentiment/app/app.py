import streamlit as st
from predict import predict_sentiment

st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬")
st.title("🎬 IMDB Movie Review Sentiment Classifier")
st.write(
    "TF-IDF + Logistic Regression, trained on 50,000 labeled IMDB reviews "
    "(90.5% test accuracy)."
)

review = st.text_area("Paste a movie review", height=200, placeholder="Type or paste a review here...")

if st.button("Predict sentiment", type="primary"):
    try:
        result = predict_sentiment(review)
    except ValueError as e:
        st.warning(str(e))
    else:
        sentiment = result["sentiment"]
        confidence = result["confidence"]
        emoji = "😀" if sentiment == "positive" else "☹️"
        st.subheader(f"{emoji} {sentiment.capitalize()} ({confidence:.0%} confidence)")
        if confidence < 0.6:
            st.info("Low confidence — this review may be mixed or ambiguous.")
