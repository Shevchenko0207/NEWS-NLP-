"""
Візуалізація результатів NLP-аналізу через Streamlit.
"""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud, STOPWORDS

from analysis import sentiment_textblob, sentiment_vader


def show_wordcloud(titles: pd.Series):
    long_string = ",".join(list(titles.values))

    wc = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color="steelblue")
    wc.generate(long_string)
    st.image(wc.to_array(), width=700, caption="WordCloud (white)")

    wc = WordCloud(width=1500, height=1000, random_state=1, background_color="salmon",
                    colormap="Pastel1", collocations=False, stopwords=STOPWORDS).generate(long_string)
    st.image(wc.to_array(), width=700, caption="WordCloud (salmon)")

    wc = WordCloud(width=1500, height=1000, random_state=1, background_color="black",
                    colormap="Set2", collocations=False, stopwords=STOPWORDS).generate(long_string)
    st.image(wc.to_array(), width=700, caption="WordCloud (black)")


def show_ngram_table_and_chart(common_words, label: str):
    df = pd.DataFrame(common_words, columns=[label, "Count"])
    st.table(df)
    fig = px.bar(df, x=label, y="Count", color="Count", height=500)
    st.plotly_chart(fig)


def show_sentiment_barchart(text_series: pd.Series, method="TextBlob"):
    if method == "TextBlob":
        sentiment = text_series.map(sentiment_textblob)
    elif method == "Vader":
        sentiment = text_series.map(sentiment_vader)
    else:
        raise ValueError('method must be "TextBlob" or "Vader"')

    counts = sentiment.value_counts()
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values, color=["cyan", "red", "green", "black"], edgecolor="yellow")
    st.pyplot(fig)


def show_entity_barchart(entity_counts, entity_label="PERSON"):
    if not entity_counts:
        st.write("Сутностей не знайдено.")
        return
    names, counts = zip(*entity_counts)
    fig, ax = plt.subplots()
    sns.barplot(x=list(counts), y=list(names), ax=ax).set_title(entity_label)
    st.pyplot(fig)


def show_pos_barchart(pos_counts):
    if not pos_counts:
        st.write("No part-of-speech tags to display.")
        return
    tags, counts = zip(*pos_counts)
    fig, ax = plt.subplots()
    sns.barplot(x=list(counts), y=list(tags), ax=ax)
    st.pyplot(fig)


def show_topics(topics):
    st.write("Topics:")
    for topic in topics:
        st.write(topic)


def show_summary(title: str, sentences):
    st.write("Summarized Document")
    st.write(" ")
    st.write(title)
    st.write(" ")
    for sentence in sentences:
        st.write(sentence)


def show_text_stats(text: str, stat_functions: dict):
    for name, func in stat_functions.items():
        st.write(name)
        st.write(func(text))
        st.write(" ")
