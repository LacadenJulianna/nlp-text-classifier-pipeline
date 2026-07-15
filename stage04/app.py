"""
stage04/app.py

Week 4, Day 18: Build your interface (part 2)

Full working demo: wires up real predictions, displays results, and adds
input validation. Validation happens two ways here:
  1. Widget-level bounds (min_value/max_value) -- prevents most bad
     input from being enterable in the first place.
  2. A try/except around the actual predict() call -- catches anything
     that slips through, so a crash never reaches the user as a raw
     traceback. Day 20 (stress-testing) exists specifically to find the
     inputs that get past validation and reach this point.

Run:
    streamlit run app.py
"""

import streamlit as st
from predict import predict_rating, get_known_genres

st.set_page_config(page_title="Disney+ Rating Predictor", page_icon="🎬")

st.title("🎬 Disney+ Content Rating Predictor")
st.caption(
    "Predicts a likely content rating (TV-G, PG, TV-14, etc.) from genre, "
    "runtime, and release year. Trained on Disney+ catalog data -- "
    "see README.md for accuracy and known limitations before trusting this."
)

known_genres = get_known_genres()

st.subheader("Tell me about the title")

content_type = st.radio("Type", ["Movie", "TV Show"])

if content_type == "Movie":
    duration_value = st.number_input("Runtime (minutes)", min_value=1, max_value=500, value=90)
else:
    duration_value = st.number_input("Number of seasons", min_value=1, max_value=50, value=1)

release_year = st.number_input("Release year", min_value=1900, max_value=2030, value=2020)

genres = st.multiselect(
    "Genres",
    options=known_genres,
    help="Any genre not in this list gets treated as 'Other' -- same as during training.",
)

if st.button("Predict rating", type="primary"):
    if not genres:
        st.warning("No genres selected -- the model can still predict, but with less to go on. Consider picking at least one.")

    try:
        result = predict_rating(
            genres=genres,
            release_year=int(release_year),
            duration_value=int(duration_value),
            is_movie=(content_type == "Movie"),
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("This shouldn't happen with valid input -- if you hit this, please note exactly what you entered.")
    else:
        st.success(f"Predicted rating: **{result['rating']}**  (confidence: {result['confidence']:.0%})")

        st.write("Full probability breakdown:")
        st.bar_chart(result["all_probabilities"])

        if result["confidence"] < 0.35:
            st.info(
                "Low confidence -- the model is genuinely unsure between several ratings for this "
                "input. See README.md: this model is weakest on TV-14 specifically."
            )
