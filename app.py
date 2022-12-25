import streamlit as st
import re
import pandas as pd
from utils.functions import display_summary
import utils.config


def app():
    try:
        summary_df = pd.read_csv(utils.config.SUMMARIZED_CSV)
        classification_df = pd.read_csv(utils.config.CLASSIFIED_CSV)
        topic_extraction_df = pd.read_csv(utils.config.TOPIC_EXTRACTED_CSV)

    except NameError:
        print("Some problem with file...")

    st.sidebar.title("Menu")
    company = st.sidebar.selectbox(
        "Choose Company", list(summary_df.loc[:, "Company"].unique())
    )
    year = st.sidebar.selectbox("Choose Year", list(summary_df.loc[:, "Year"].unique()))

    which_summary = st.sidebar.selectbox(
        "See Summary For", ["All Sections", "Section-wise"]
    )

    show_topics = st.sidebar.checkbox("Show Important Words")

    show_score = False
    if show_topics:
        show_score = st.sidebar.checkbox("Show Frequency and Co-occurence Scores")

    st.markdown(
        "<h1 style='text-align: center;'>Financial Dashboard</h1>",
        unsafe_allow_html=True,
    )

    if which_summary == "All Sections":
        display_summary(
            summary_df,
            classification_df,
            topic_extraction_df,
            summary_df.columns[2:],
            company,
            year,
            show_topics,
            show_score,
        )
    else:
        options = st.multiselect(
            "Please select Section(s) to view:", summary_df.columns[2:], "Risk Factor"
        )
        display_summary(
            summary_df,
            classification_df,
            topic_extraction_df,
            options,
            company,
            year,
            show_topics,
            show_score,
        )


if __name__ == "__main__":
    app()
