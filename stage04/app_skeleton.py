"""
stage04/app_skeleton.py

Week 4, Day 17: Build your interface (part 1)

Minimal scaffold: loads the model (via predict.py) and collects user
input through Streamlit widgets. Deliberately NOT wired up to produce a
prediction yet -- that's Day 18 (app.py). Today's goal is just: does the
app start, does it load the model without error, can a user interact
with the inputs.

Run:
    streamlit run app_skeleton.py
"""

import streamlit as st
from predict import get_known_genres

st.set_page_config(page_title="Disney+ Rating Predictor (skeleton)", page_icon="🎬")

st.title("🎬 Disney+ Content Rating Predictor")
st.caption("Day 17 skeleton -- inputs only, no prediction wired up yet.")

# Confirms the model loads without error before building anything further on top of it
known_genres = get_known_genres()
st.success(f"Model loaded. Knows {len(known_genres)} genres by name (others map to 'Other').")

st.subheader("Inputs")

content_type = st.radio("Type", ["Movie", "TV Show"])

if content_type == "Movie":
    duration_value = st.number_input("Runtime (minutes)", value=90)
else:
    duration_value = st.number_input("Number of seasons", value=1)

release_year = st.number_input("Release year", value=2020)

genres = st.multiselect("Genres", options=known_genres)

st.write("---")
st.write("Selected inputs (not yet sent to the model):")
st.json({
    "type": content_type,
    "duration_value": duration_value,
    "release_year": release_year,
    "genres": genres,
})
