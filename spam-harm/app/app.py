"""
spam-harm/app/app.py

Streamlit interface for the spam/harm text classifier. Single text box
in, prediction out -- try/except around the predict call so a crash
never reaches the user as a raw traceback (same pattern as the Disney+
project's app.py).

Run:
    streamlit run app.py
"""

import streamlit as st
from predict import predict_message

st.set_page_config(page_title="Spam/Harm Text Classifier", page_icon="🚫")

st.title("🚫 Spam/Harm Text Classifier")
st.caption(
    "Classifies a message as spam or ham (legitimate) using a TF-IDF + "
    "classical ML model trained on a small labeled spam/ham dataset. "
    "See README.md for accuracy and known limitations before trusting a prediction."
)

message = st.text_area("Paste a message to classify", height=120)

if st.button("Classify", type="primary"):
    if not message.strip():
        st.warning("Enter a message first.")
    else:
        try:
            result = predict_message(message)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("This shouldn't happen with valid input -- if you hit this, please note exactly what you entered.")
        else:
            label = result["label"]
            confidence = result["confidence"]
            if label == "spam":
                st.error(f"Predicted: **SPAM** (confidence: {confidence:.0%})")
            else:
                st.success(f"Predicted: **HAM** (legitimate) (confidence: {confidence:.0%})")

            st.write("Probability breakdown:")
            st.bar_chart(result["probabilities"])

            if confidence < 0.65:
                st.info(
                    "Low confidence -- the model is genuinely unsure here. "
                    "See README.md for known limitations."
                )
