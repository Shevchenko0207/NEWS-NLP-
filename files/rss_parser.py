"""
Завантаження та розбір RSS-стрічок.
"""

import feedparser
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Завантаження новин з RSS...")
def load_rss_feed(feed_url: str) -> pd.DataFrame:
    """
    Завантажує RSS-стрічку та повертає DataFrame з колонками:
    title, link, description, published, content.
    """
    rows = []
    feed = feedparser.parse(feed_url)

    for entry in feed.entries:
        title = entry.get("title", "")
        link = entry.get("link", "")
        description = entry.get("description", "")
        published = entry.get("published", "")
        content = ""
        if entry.get("content"):
            content = entry.get("content")[0].get("value", "")

        rows.append(
            {
                "title": title,
                "link": link,
                "description": description,
                "published": published,
                "content": content,
            }
        )

    df = pd.DataFrame(rows, columns=["title", "link", "description", "published", "content"])
    return df
