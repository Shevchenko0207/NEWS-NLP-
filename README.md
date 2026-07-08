# News Articles Analysis — NLP App

A Streamlit app for analyzing news articles using NLP techniques. It fetches
articles from the RSS feed of a selected publication, scrapes the full
article text, and provides several types of text analysis: n-grams,
sentiment, named entity recognition, topic modeling, summarization,
part-of-speech tagging, and text statistics.

## Project structure

```
NEWS NLP/
├── app.py            # Entry point: Streamlit UI + analysis dispatcher
├── config.py         # Constants: news sources, list of analysis functions
├── rss_parser.py      # RSS feed loading and parsing
├── scraper.py         # Full article text extraction by URL
├── analysis.py        # Pure NLP computations (no UI): sentiment, n-grams,
│                        NER, LDA, summarization, POS, textstat
├── visuals.py          # Result visualization (charts, tables, wordcloud)
└── requirements.txt    # Project dependencies
```

The whole app used to be a single ~600-line `app.py` file. It has been
split into 6 modules, each with a single responsibility, to make the
project easier to maintain and test.

## News sources

| Source         | RSS feed                                            | Status                          |
|----------------|------------------------------------------------------|----------------------------------|
| NY Times       | `rss.nytimes.com/services/xml/rss/nyt/US.xml`         | ✅ Works reliably                 |
| Forbes         | `forbes.com/business/feed/`                           | ✅ Works reliably                 |
| Investing.com  | `investing.com/rss/news_25.rss`                       | ⚠️ Frequently unavailable (see below) |

### Known limitation: Investing.com

Investing.com is protected by anti-bot infrastructure (likely something
similar to Cloudflare) that blocks automated requests at the TLS
connection level — even with a proper `User-Agent` header. This shows up
as a `ConnectionResetError` or an empty RSS parsing result.

The app handles this gracefully: if a source returns zero articles, the
user sees a clear message instead of a crash/traceback. Getting real
access to this source would require a headless browser or a specialized
library (e.g. `cloudscraper`).

## Installation

```bash
pip install -r requirements.txt
```

You'll also need the spaCy model for named entity recognition:

```bash
python -m spacy download en_core_web_sm
```

NLTK resources (`vader_lexicon`, `averaged_perceptron_tagger`, `punkt`)
are downloaded automatically the first time the relevant functions are
used — no manual setup needed.

## Running the app

```bash
python -m streamlit run app.py
```

## Analysis functions

- **Intro** — a short description of the selected news source
- **Snapshot** — a table of the fetched articles (title, link)
- **Unigrams / Bigrams / Trigrams** — most frequent n-grams in the titles
- **WordCloud** — a word cloud (3 color-scheme variants)
- **Sentiment Analysis (TextBlob / Vader)** — sentiment distribution of the titles
- **Entity Extraction** — most common named entities (spaCy, NER)
- **Topic Modeling** — topic modeling using LDA
- **Text Summarization** — automatic article summarization (LexRank)
- **Parts of Speech** — part-of-speech distribution across article texts
- **Text Stat** — text statistics (readability, complexity, etc.)

## Known behavior notes

- Sentiment analysis is computed on **titles**, not the full article text.
  Serious publications tend to write headlines in a neutral, factual tone,
  so the result may show mostly the `neu` class — this is expected
  behavior, not a bug.
- Some functions (`Topic Modeling`, `Text Summarization`, `Parts of Speech`,
  `Text Stat`) work on the full article text (`content`), so their quality
  depends on how successfully `scraper.py` was able to extract text from
  a given site.
