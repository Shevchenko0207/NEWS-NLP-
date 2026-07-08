"""
News Articles Analysis - NLP App
Точка входу: тільки UI (Streamlit) + диспетчеризація аналізів.
Запуск: python -m streamlit run app.py
"""

import inspect

import nltk
import streamlit as st
import textstat

from config import NEWS_SOURCES, NLP_FUNCTIONS
from rss_parser import load_rss_feed
from scraper import attach_full_text
from analysis import (
    preprocess,
    get_top_n_words,
    get_top_n_bigram,
    get_top_n_trigram,
    most_common_named_entities,
    most_common_pos_tags,
    topic_modeling,
    summarize_text,
)
from visuals import (
    show_wordcloud,
    show_ngram_table_and_chart,
    show_sentiment_barchart,
    show_entity_barchart,
    show_pos_barchart,
    show_topics,
    show_summary,
    show_text_stats,
)

nltk.download("punkt", quiet=True)

st.title("News Articles Analysis - NLP App")
st.header("This app displays the news articles appeared in the top News Publications!")

st.sidebar.header("Please select the news org from the dropdown list")
s_news = st.sidebar.selectbox("News", list(NEWS_SOURCES.keys()))

st.sidebar.header("Please select the Function")
s_nlp = st.sidebar.selectbox("Functions", NLP_FUNCTIONS)


@st.cache_data(show_spinner="Завантаження та обробка статей...")
def load_and_prepare(news_source: str):
    feed_url = NEWS_SOURCES[news_source]
    df = load_rss_feed(feed_url)
    df = attach_full_text(df)
    df["title"] = preprocess(df["title"])
    df = df.dropna()
    return df


def run_analysis(nlp_choice: str, df):
    titles = df["title"]

    if nlp_choice == "Intro":
        st.markdown(
            """
            The New York Times (NYT) is an American daily newspaper based in New York City.
            Founded in 1851, it has won 130 Pulitzer Prizes and is regarded as a
            "newspaper of record".
            """
        )

    elif nlp_choice == "Snapshot":
        st.subheader("Display the dataframe")
        st.dataframe(df)
        st.markdown(f"**Number of articles:** {df.shape[0]}")
        st.write("The URL Links:")
        for _, row in df.iterrows():
            st.write(row["link"])

    elif nlp_choice == "WordCloud":
        show_wordcloud(titles)

    elif nlp_choice == "Unigrams":
        common_words = get_top_n_words(titles, 10)
        show_ngram_table_and_chart(common_words, "Words")

    elif nlp_choice == "Bigrams":
        common_words = get_top_n_bigram(titles, 10)
        show_ngram_table_and_chart(common_words, "Bigrams")

    elif nlp_choice == "Trigrams":
        common_words = get_top_n_trigram(titles, 10)
        show_ngram_table_and_chart(common_words, "Trigrams")

    elif nlp_choice == "Sentiment Analysis TextBlob":
        show_sentiment_barchart(titles, method="TextBlob")

    elif nlp_choice == "Sentiment Analysis-Vader":
        show_sentiment_barchart(titles, method="Vader")

    elif nlp_choice == "Entity Extraction":
        st.header("Entity Extraction")
        entities = most_common_named_entities(titles, entity="PERSON")
        show_entity_barchart(entities, entity_label="PERSON")

    elif nlp_choice == "Topic Modeling":
        st.header("Topic Modeling")
        topics = topic_modeling(df["content"])
        show_topics(topics)

    elif nlp_choice == "Text Summarization":
        st.header("Text Summarization")
        for _, row in df.iterrows():
            sentences = summarize_text(row["content"], sentence_count=3)
            show_summary(row["title"], sentences)

    elif nlp_choice == "Parts of Speech":
        pos_counts = most_common_pos_tags(df["content"])
        show_pos_barchart(pos_counts)

    elif nlp_choice == "Text Stat":
        textstat.set_lang("en")
        text = df.at[df.index[0], "content"]
        method_names = [
            name
            for name, _ in inspect.getmembers(textstat, predicate=inspect.isroutine)
            if not name.startswith("_")
        ][:27]
        stat_functions = {name: getattr(textstat, name) for name in method_names}
        show_text_stats(text, stat_functions)


df = load_and_prepare(s_news)
run_analysis(s_nlp, df)
