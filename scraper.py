"""
Отримання повного тексту статей за посиланням.
"""

import time

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup


@st.cache_data(show_spinner=False)
def full_text(url: str) -> str:
    """
    Завантажує сторінку за url і повертає текст найбільшого
    змістовного блоку <p>-тегів на сторінці.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
    except Exception:
        # Ловимо не тільки requests.RequestException, а й нижчорівневі збої
        # з'єднання (напр. ConnectionResetError: WinError 10054), щоб одна
        # недоступна стаття не валила весь аналіз.
        return ""

    body = soup.find_all("body")
    if not body:
        return ""

    paragraphs = body[0].find_all("p")
    rows = []
    for p in paragraphs:
        parents = [
            f"{parent.name}id:{parent.get('id', '')}"
            for parent in p.parents
            if parent is not None
        ]
        parents.reverse()
        hierarchy = " -> ".join(parents)
        rows.append(
            {
                "parent_hierarchy": hierarchy,
                "element_text": p.text,
                "element_text_count": len(p.text),
            }
        )

    if not rows:
        return ""

    blocks_df = pd.DataFrame(rows)
    grouped = blocks_df.groupby("parent_hierarchy")["element_text_count"].sum()
    best_hierarchy = grouped.idxmax()
    merged_text = "\n".join(
        blocks_df.loc[blocks_df["parent_hierarchy"] == best_hierarchy, "element_text"]
    )
    return merged_text


def attach_full_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Для кожного рядка df дістає повний текст статті за посиланням
    і записує його в колонку 'content' (правильно, через .at,
    без старого бага df['content', ind] = text).
    """
    df = df.copy()
    for idx in df.index:
        url = df.at[idx, "link"]
        df.at[idx, "content"] = full_text(url)
        time.sleep(0.5)
    return df
