"""
Константи та налаштування застосунку.
"""

NEWS_SOURCES = {
    "NY Times": "https://rss.nytimes.com/services/xml/rss/nyt/US.xml",
    # Додайте сюди інші джерела за потреби, наприклад:
    # "CNN": "http://rss.cnn.com/rss/cnn_topstories.rss",
    # "LA Times": "https://www.latimes.com/rss2.0.xml",
}

NLP_FUNCTIONS = [
    "Intro",
    "Snapshot",
    "Unigrams",
    "Bigrams",
    "Trigrams",
    "WordCloud",
    "Text Stat",
    "Topic Modeling",
    "Entity Extraction",
    "Sentiment Analysis TextBlob",
    "Sentiment Analysis-Vader",
    "Text Summarization",
    "Parts of Speech",
]

SPACY_MODEL_NAME = "en_core_web_sm"
