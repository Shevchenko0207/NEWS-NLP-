"""
NLP-обчислення: sentiment, n-грами, NER, тематичне моделювання,
сумаризація, частини мови, текстова статистика.

Тут немає жодних st.write / st.pyplot — тільки обчислення.
Візуалізація винесена в visuals.py.
"""

import streamlit as st
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from config import SPACY_MODEL_NAME


def preprocess(review_text):
    """Прибирає html-сміття з тексту (Series pandas)."""
    review_text = review_text.str.replace("(<br/>)", "", regex=True)
    review_text = review_text.str.replace("(<a).*(>).*(</a>)", "", regex=True)
    review_text = review_text.str.replace("(&amp)", "", regex=True)
    review_text = review_text.str.replace("(&gt)", "", regex=True)
    review_text = review_text.str.replace("(&lt)", "", regex=True)
    review_text = review_text.str.replace("(\xa0)", " ", regex=True)
    return review_text


@st.cache_data
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words="english").fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]


@st.cache_data
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words="english").fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]


@st.cache_data
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words="english").fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]


@st.cache_resource
def ensure_vader_lexicon():
    nltk.download("vader_lexicon", quiet=True)
    return True


@st.cache_data
def sentiment_vader(text):
    ensure_vader_lexicon()
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    scores.pop("compound")
    return max(scores, key=scores.get)


@st.cache_data
def sentiment_textblob(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity < 0:
        return "neg"
    if polarity == 0:
        return "neu"
    return "pos"


@st.cache_resource
def load_spacy_model():
    """Завантажує модель spaCy один раз і кешує її (замість щоразового load)."""
    return spacy.load(SPACY_MODEL_NAME)


def most_common_named_entities(text_series, entity="PERSON", top_n=10):
    """Повертає список (сутність, кількість) найпоширеніших сутностей заданого типу."""
    nlp_model = load_spacy_model()

    all_entities = []
    # nlp.pipe обробляє тексти пакетно й економніше по пам'яті/CPU,
    # ніж виклик nlp(text) по одному в циклі. Вимикаємо компоненти
    # конвеєра, які не потрібні для NER (parser, lemmatizer, tagger),
    # щоб не витрачати пам'ять даремно.
    disabled = [
        name
        for name in ("parser", "lemmatizer", "tagger", "attribute_ruler")
        if name in nlp_model.pipe_names
    ]
    with nlp_model.select_pipes(disable=disabled):
        for doc in nlp_model.pipe(text_series, batch_size=32):
            all_entities.extend(ent.text for ent in doc.ents if ent.label_ == entity)

    from collections import Counter
    return Counter(all_entities).most_common(top_n)


@st.cache_resource
def ensure_nltk_pos_data():
    """
    Завантажує ресурси NLTK для POS-тегування один раз (кешовано).
    У різних версіях NLTK тегер зветься по-різному
    (averaged_perceptron_tagger або averaged_perceptron_tagger_eng),
    тому пробуємо обидві назви.
    """
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    for resource in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
    return True


def most_common_pos_tags(text_series, top_n=7):
    """Повертає список (тег, кількість) найпоширеніших частин мови."""
    ensure_nltk_pos_data()

    def _pos(text):
        if not text.strip():
            return []
        tagged = nltk.pos_tag(word_tokenize(text))
        return [tag for _, tag in tagged] if tagged else []

    tags = [t for text in text_series for t in _pos(text)]
    from collections import Counter
    return Counter(tags).most_common(top_n)


def topic_modeling(content_series, n_topics=10, n_words=6):
    """
    Повертає список рядків: топ-слова для кожної теми (LDA).
    Якщо в текстах недостатньо реальних слів (наприклад, скрапінг
    повного тексту не вдався і content порожній), повертає порожній
    список замість падіння з ValueError.
    """
    non_empty = content_series[content_series.str.strip().astype(bool)]
    if non_empty.empty:
        return []

    try:
        vectorizer = CountVectorizer(stop_words="english")
        count_data = vectorizer.fit_transform(non_empty)
    except ValueError:
        return []

    n_topics = min(n_topics, count_data.shape[0])
    if n_topics < 1:
        return []

    lda = LDA(n_components=n_topics, n_jobs=-1)
    lda.fit(count_data)

    words = vectorizer.get_feature_names_out()
    topics = []
    for topic in lda.components_:
        top_words = " ".join(words[i] for i in topic.argsort()[: -n_words - 1 : -1])
        topics.append(top_words)
    return topics


@st.cache_resource
def ensure_punkt_data():
    """
    Довантажує ресурси токенізації речень NLTK, потрібні для sumy.
    Новіші версії NLTK використовують 'punkt_tab' замість старого 'punkt',
    тому пробуємо довантажити обидва.
    """
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
    return True


def summarize_text(text, sentence_count=3):
    """Повертає список речень-резюме для одного тексту."""
    if not text.strip():
        return []
    ensure_punkt_data()
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return [str(sentence) for sentence in summary]
