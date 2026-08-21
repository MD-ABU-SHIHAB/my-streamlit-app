"""
app.py — Islamic Mas'alah System · Global Islamic Ruling Reference
CSE 469 — Machine Learning and Pattern Recognition

Retrieval-first search over dataset.csv (exact match -> TF-IDF cosine
similarity -> fuzzy match -> honest no-match state), with a Logistic
Regression classifier used only as a secondary confirmation signal, never
as the primary displayed answer.

Multilingual support: বাংলা | English | العربية
"""

from __future__ import annotations

import base64
import difflib
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import hashlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# Import main.py functions for ML analysis
from main import run_complete_analysis

# --------------------------------------------------------------------------- #
# CONSTANTS
# --------------------------------------------------------------------------- #

DATA_PATH: str = "dataset.csv"
RANDOM_STATE: int = 42

TFIDF_SIMILARITY_THRESHOLD: float = 0.30
FUZZY_MATCH_THRESHOLD: float = 0.55
TOP_K_SUGGESTIONS: int = 5
TOP_K_RELATED: int = 4
MEANINGFUL_TOKEN_MIN_LEN: int = 3

REQUIRED_COLUMNS = [
    "id", "question_bn", "question_en", "question_banglish", "tier1_class",
    "topic", "strictness_label", "short_explanation_bn", "short_explanation_en",
    "reference_text", "reference_source", "source_type", "verification_status",
    "search_keywords",
]

NEEDS_VERIFICATION_LABEL = "NEEDS_VERIFICATION"
PLACEHOLDER_REFERENCE_TEXT = "SEE_REFERENCE_SOURCE"

# Theme colors
COLOR_PAPER = "#EDE7D9"
COLOR_CARD = "#F6F2E8"
COLOR_INK = "#2B2A26"
COLOR_INK_MUTED = "#6B6558"
COLOR_ACCENT = "#3A4F63"
COLOR_HAIRLINE = "#D8D0BC"
COLOR_FLAG_BG = "#F0E4CC"
COLOR_FLAG_INK = "#6B5033"


# --------------------------------------------------------------------------- #
# MULTILINGUAL TRANSLATIONS
# --------------------------------------------------------------------------- #

T = {
    "bn": {
        "app_title": "ইসলামী মাসআলা সিস্টেম",
        "app_subtitle": "বিশ্বব্যাপী ইসলামী রেফারেন্স ও মেশিন লার্নিং প্যাটার্ন রিকগনিশন সিস্টেম",
        "search_placeholder": "বাংলা, ইংরেজি বা বাংলিশে লিখুন...",
        "search_button": "খুঁজুন",
        "browse": "ব্রাউজ",
        "model_insights": "মডেল বিশ্লেষণ",
        "ruling": "মাসআলা",
        "topic": "বিষয়",
        "explanation": "ব্যাখ্যা",
        "reference": "তথ্যসূত্র",
        "source": "উৎস",
        "verification": "যাচাইকরণ",
        "verified": "যাচাইকৃত",
        "needs_verification": "যাচাই বাকি",
        "no_match": "কোনো মিল পাওয়া যায়নি",
        "suggestions": "পরামর্শ",
        "related": "সম্পর্কিত",
        "details": "বিবরণ",
        "disclaimer": "এই সিস্টেম তার সংগৃহীত ডেটাসেট থেকে তথ্য পুনরুদ্ধার করে এবং একজন যোগ্য ইসলামি পণ্ডিতের সাথে পরামর্শের বিকল্প নয়।",
        "about": "এই সিস্টেম সম্পর্কে",
        "viva_mode": "প্রকল্প ব্যাখ্যা",
        "search_history": "অনুসন্ধান ইতিহাস",
        "bookmarks": "বুকমার্ক",
        "font_size": "অক্ষরের আকার",
        "small": "ছোট",
        "medium": "মাঝারি",
        "large": "বড়",
        "copy_citation": "উদ্ধৃতি কপি করুন",
        "print": "প্রিন্ট",
        "examples": "উদাহরণ প্রশ্ন:",
    },
    "en": {
        "app_title": "Islamic Mas'alah System",
        "app_subtitle": "Global Islamic Ruling Reference & Machine Learning Pattern Recognition System",
        "search_placeholder": "Search in Bangla, English, or Banglish...",
        "search_button": "Search",
        "browse": "Browse",
        "model_insights": "Model Insights",
        "ruling": "Ruling",
        "topic": "Topic",
        "explanation": "Explanation",
        "reference": "Reference",
        "source": "Source",
        "verification": "Verification",
        "verified": "Verified",
        "needs_verification": "Needs Verification",
        "no_match": "No match found",
        "suggestions": "Suggestions",
        "related": "Related",
        "details": "Details",
        "disclaimer": "This system retrieves information from its curated dataset and is not a replacement for consultation with a qualified Islamic scholar.",
        "about": "About This System",
        "viva_mode": "Project Explanation",
        "search_history": "Search History",
        "bookmarks": "Bookmarks",
        "font_size": "Font Size",
        "small": "Small",
        "medium": "Medium",
        "large": "Large",
        "copy_citation": "Copy Citation",
        "print": "Print",
        "examples": "Example questions:",
    },
    "ar": {
        "app_title": "نظام المسألة الإسلامية",
        "app_subtitle": "المرجعية العالمية للأحكام الإسلامية ونظام التعرف على الأنماط وتعلم الآلة",
        "search_placeholder": "ابحث بالبنغالية أو الإنجليزية أو البنغليش...",
        "search_button": "بحث",
        "browse": "تصفح",
        "model_insights": "تحليل النماذج",
        "ruling": "الحكم",
        "topic": "الموضوع",
        "explanation": "الشرح",
        "reference": "المرجع",
        "source": "المصدر",
        "verification": "التحقق",
        "verified": "موثق",
        "needs_verification": "بحاجة إلى تحقق",
        "no_match": "لم يتم العثور على نتيجة",
        "suggestions": "اقتراحات",
        "related": "ذات صلة",
        "details": "التفاصيل",
        "disclaimer": "هذا النظام يسترجع المعلومات من مجموعة البيانات المنسقة وليس بديلاً عن استشارة عالم إسلامي مؤهل.",
        "about": "حول هذا النظام",
        "viva_mode": "شرح المشروع",
        "search_history": "سجل البحث",
        "bookmarks": "العلامات المرجعية",
        "font_size": "حجم الخط",
        "small": "صغير",
        "medium": "متوسط",
        "large": "كبير",
        "copy_citation": "نسخ الاستشهاد",
        "print": "طباعة",
        "examples": "أسئلة نموذجية:",
    }
}

LANGUAGE_NAMES = {
    "bn": "বাংলা",
    "en": "English",
    "ar": "العربية"
}

LANGUAGE_DIRECTIONS = {
    "bn": "ltr",
    "en": "ltr",
    "ar": "rtl"
}

# Arabic fallback message
ARABIC_UNAVAILABLE = {
    "bn": "আরবি অনুবাদ এখনও উপলব্ধ নয়।",
    "en": "Arabic translation is not yet available.",
    "ar": "الترجمة العربية غير متوفرة حالياً."
}


def _(key: str, lang: str = "en") -> str:
    """Get translated string for the current language."""
    if lang in T and key in T[lang]:
        return T[lang][key]
    return key


# --------------------------------------------------------------------------- #
# DATA MODELS
# --------------------------------------------------------------------------- #

@dataclass
class RetrievalResult:
    row: Optional[pd.Series]
    stage: str
    similarity: float
    suggestions: list = field(default_factory=list)


class DatasetError(Exception):
    pass


# --------------------------------------------------------------------------- #
# SESSION STATE INITIALIZATION
# --------------------------------------------------------------------------- #

def init_session_state():
    """Initialize all session state variables."""
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    if "bookmarks" not in st.session_state:
        st.session_state.bookmarks = []
    if "font_size" not in st.session_state:
        st.session_state.font_size = "medium"
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = ""
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None


# --------------------------------------------------------------------------- #
# LOADING & PREPROCESSING
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise DatasetError(f"Couldn't find '{path}'. Place dataset.csv next to app.py.")
    except Exception as exc:
        raise DatasetError(f"Couldn't read '{path}': {exc}")

    if df.empty:
        raise DatasetError(f"'{path}' was found but contains no rows.")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetError("dataset.csv is missing required column(s): " + ", ".join(missing_cols))

    df = df.dropna(subset=["question_bn", "tier1_class"]).reset_index(drop=True)
    if df.empty:
        raise DatasetError("Every row is missing question_bn and/or tier1_class.")
    return df


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s\u0980-\u09FF]", " ", text)
    tokens = text.split()
    return " ".join(tokens)


@st.cache_data(show_spinner=False)
def build_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("question_bn", "question_en", "question_banglish", "search_keywords"):
        df[col] = df[col].fillna("")
    df["combined_text_raw"] = df["question_bn"] + " " + df["question_en"] + " " + df["search_keywords"]
    df["combined_text_clean"] = df["combined_text_raw"].apply(clean_text)
    return df


@st.cache_data(show_spinner=False)
def build_banglish_map(df: pd.DataFrame) -> dict[str, set[str]]:
    banglish_map: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        group_terms = set()
        for kw in str(row.get("search_keywords", "")).split(","):
            kw = kw.strip().lower()
            if len(kw) >= MEANINGFUL_TOKEN_MIN_LEN:
                group_terms.add(kw)
        for term in group_terms:
            banglish_map.setdefault(term, set()).update(group_terms)
    return banglish_map


def normalize_query(query: str, banglish_map: dict[str, set[str]]) -> str:
    query_lower = query.lower().strip()
    tokens = re.findall(r"[\w\u0980-\u09FF]+", query_lower)
    expanded = set(tokens)
    for tok in tokens:
        if tok in banglish_map:
            expanded.update(banglish_map[tok])
    return query_lower + " " + " ".join(sorted(expanded))


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\w\u0980-\u09FF]+", text.lower()))


# --------------------------------------------------------------------------- #
# CLASSIFIER
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Indexing the reference set…")
def train_classifier(df: pd.DataFrame):
    df = build_combined_text(df)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=20000)
    X = vectorizer.fit_transform(df["combined_text_clean"])
    y = df["tier1_class"]
    classifier = None
    if y.nunique() >= 2:
        classifier = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        classifier.fit(X, y)
    return vectorizer, X, classifier


# --------------------------------------------------------------------------- #
# RETRIEVAL PIPELINE
# --------------------------------------------------------------------------- #

def _exact_match(query: str, banglish_map: dict[str, set[str]], df: pd.DataFrame) -> Optional[pd.Series]:
    query_norm = query.lower().strip()
    search_cols = ["question_bn", "question_en", "question_banglish", "search_keywords"]
    lowered = {c: df[c].fillna("").str.lower() for c in search_cols}

    if len(query_norm) >= 4:
        direct_mask = pd.Series(False, index=df.index)
        for c in search_cols:
            direct_mask = direct_mask | lowered[c].str.contains(re.escape(query_norm), na=False)
        if direct_mask.any():
            return df.loc[direct_mask.idxmax()]

    meaningful = {t for t in _tokenize(query) if len(t) >= MEANINGFUL_TOKEN_MIN_LEN}
    if not meaningful:
        return None
    expanded_str = normalize_query(query, banglish_map)
    expanded = {t for t in _tokenize(expanded_str) if len(t) >= MEANINGFUL_TOKEN_MIN_LEN}

    required_hits = len(meaningful) if len(meaningful) <= 2 else max(2, -(-len(meaningful) * 6 // 10))
    field_tokens = df[search_cols].fillna("").agg(" ".join, axis=1).apply(_tokenize)
    hit_counts = field_tokens.apply(lambda toks: len(toks & expanded))
    if hit_counts.max() >= required_hits:
        return df.loc[hit_counts.idxmax()]
    return None


def _tfidf_match(query_clean: str, df: pd.DataFrame, vectorizer, X) -> tuple[Optional[pd.Series], float]:
    if not query_clean.strip():
        return None, 0.0
    q_vec = vectorizer.transform([query_clean])
    sims = cosine_similarity(q_vec, X).flatten()
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score >= TFIDF_SIMILARITY_THRESHOLD:
        return df.iloc[best_idx], best_score
    return None, best_score


def _fuzzy_match(query_clean: str, df: pd.DataFrame) -> tuple[Optional[pd.Series], float]:
    if not query_clean.strip():
        return None, 0.0
    texts = df["combined_text_clean"].tolist()
    best_ratio, best_idx = 0.0, -1
    for i, text in enumerate(texts):
        ratio = difflib.SequenceMatcher(None, query_clean, text).ratio()
        if ratio > best_ratio:
            best_ratio, best_idx = ratio, i
    if best_idx >= 0 and best_ratio >= FUZZY_MATCH_THRESHOLD:
        return df.iloc[best_idx], best_ratio
    return None, best_ratio


def retrieve_candidates(query: str, df: pd.DataFrame, vectorizer, X, banglish_map: dict[str, set[str]]) -> RetrievalResult:
    query_clean = clean_text(query)
    row = _exact_match(query, banglish_map, df)
    if row is not None:
        return RetrievalResult(row=row, stage="exact_match", similarity=1.0)

    row, score = _tfidf_match(query_clean, df, vectorizer, X)
    if row is not None:
        return RetrievalResult(row=row, stage="tfidf_cosine", similarity=score)

    row, ratio = _fuzzy_match(query_clean, df)
    if row is not None:
        return RetrievalResult(row=row, stage="fuzzy_match", similarity=ratio)

    suggestions = []
    if query_clean.strip():
        q_vec = vectorizer.transform([query_clean])
        sims = cosine_similarity(q_vec, X).flatten()
        top_idx = np.argsort(sims)[::-1][:TOP_K_SUGGESTIONS]
        suggestions = [df.iloc[i] for i in top_idx]
    return RetrievalResult(row=None, stage="no_match", similarity=0.0, suggestions=suggestions)


def classify_query(query: str, vectorizer, classifier) -> Optional[str]:
    if classifier is None:
        return None
    q_clean = clean_text(query)
    if not q_clean.strip():
        return None
    q_vec = vectorizer.transform([q_clean])
    return classifier.predict(q_vec)[0]


# --------------------------------------------------------------------------- #
# UI HELPERS
# --------------------------------------------------------------------------- #

def get_lang() -> str:
    return st.session_state.get("language", "en")


def get_direction() -> str:
    return LANGUAGE_DIRECTIONS.get(get_lang(), "ltr")


def t(key: str) -> str:
    return _(key, get_lang())


def format_text(text: str, lang: str = None) -> str:
    """Format text for display with proper language handling."""
    if not text or pd.isna(text):
        return ""
    return str(text)


def get_field_for_lang(row: pd.Series, field_base: str) -> str:
    """Get the appropriate field value based on current language."""
    lang = get_lang()
    if lang == "bn" and f"{field_base}_bn" in row and pd.notna(row[f"{field_base}_bn"]):
        return str(row[f"{field_base}_bn"])
    elif lang == "ar" and f"{field_base}_ar" in row and pd.notna(row[f"{field_base}_ar"]):
        return str(row[f"{field_base}_ar"])
    else:
        # Fallback to English or Bangla
        if f"{field_base}_en" in row and pd.notna(row[f"{field_base}_en"]):
            return str(row[f"{field_base}_en"])
        elif f"{field_base}_bn" in row and pd.notna(row[f"{field_base}_bn"]):
            return str(row[f"{field_base}_bn"])
    return ""


# --------------------------------------------------------------------------- #
# CSS INJECTION (with RTL support)
# --------------------------------------------------------------------------- #

def inject_css() -> None:
    lang = get_lang()
    direction = get_direction()
    
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+Bengali:wght@400;600;700&family=Noto+Sans+Bengali:wght@400;500;600&family=Inter:wght@400;500;600&family=Noto+Naskh+Arabic:wght@400;600;700&family=Amiri:wght@400;700&display=swap');

        :root {{
            --paper: #EDE7D9;
            --card: #F6F2E8;
            --ink: #2B2A26;
            --ink-muted: #6B6558;
            --accent: #3A4F63;
            --accent-soft: #E4DEC9;
            --hairline: #D8D0BC;
            --flag-bg: #F0E4CC;
            --flag-ink: #6B5033;
            --font-display: 'Noto Serif Bengali', Georgia, serif;
            --font-body: 'Noto Sans Bengali', 'Inter', sans-serif;
            --font-arabic: 'Noto Naskh Arabic', 'Amiri', serif;
            --space-1: 0.4rem;
            --space-2: 0.8rem;
            --space-3: 1.4rem;
            --space-4: 2.2rem;
            --space-5: 3.2rem;
        }}

        html, body, [class*="css"] {{
            background-color: var(--paper) !important;
            color: var(--ink);
            font-family: var(--font-body);
        }}
        
        .block-container {{ max-width: 800px; padding-top: var(--space-4); padding-bottom: var(--space-5); }}
        #MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; }}

        /* RTL support for Arabic */
        .lang-ar {{
            direction: rtl !important;
            text-align: right !important;
            font-family: var(--font-arabic) !important;
        }}
        .lang-ar .entry-question {{
            font-family: var(--font-arabic) !important;
        }}
        .lang-ar .explanation-text {{
            font-family: var(--font-arabic) !important;
        }}
        .lang-ar .citation-block {{
            font-family: var(--font-arabic) !important;
        }}
        .lang-ar .category-label {{
            border-left: none !important;
            border-right: 3px solid var(--accent) !important;
            padding: 0.2rem 0.65rem 0.2rem 0 !important;
        }}

        .app-eyebrow {{
            font-family: 'Inter', sans-serif;
            font-size: 0.72rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--ink-muted);
            margin-bottom: var(--space-1);
        }}
        .app-title {{
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 2.1rem;
            color: var(--ink);
            margin-bottom: var(--space-1);
            line-height: 1.15;
        }}
        .app-subtitle {{
            font-family: var(--font-body);
            color: var(--ink-muted);
            font-size: 0.95rem;
        }}
        .app-rule {{
            border: none;
            border-top: 1px solid var(--hairline);
            margin: var(--space-3) 0 var(--space-4) 0;
        }}

        .entry {{
            background-color: var(--card);
            border: 1px solid var(--hairline);
            border-radius: 3px;
            padding: var(--space-3) var(--space-3) var(--space-2) var(--space-3);
            margin-top: var(--space-3);
        }}
        .entry-refno {{
            font-family: 'Inter', monospace;
            font-size: 0.72rem;
            color: var(--ink-muted);
            letter-spacing: 0.05em;
            margin-bottom: var(--space-2);
        }}
        .entry-question {{
            font-family: var(--font-display);
            font-size: 1.2rem;
            font-weight: 600;
            line-height: 1.4;
            margin-bottom: 0.3rem;
        }}
        .entry-question-en {{
            font-family: var(--font-body);
            font-size: 0.92rem;
            color: var(--ink-muted);
            margin-bottom: var(--space-2);
        }}
        .category-label {{
            font-family: var(--font-body);
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--ink);
            border-left: 3px solid var(--accent);
            padding: 0.2rem 0 0.2rem 0.65rem;
            margin: var(--space-2) 0 var(--space-3) 0;
        }}
        .category-mark {{ color: var(--accent); margin-right: 0.4rem; font-weight: 400; }}

        .explanation-text {{ font-size: 0.99rem; line-height: 1.6; margin-bottom: var(--space-2); }}
        .citation-block {{
            border-left: 2px solid var(--hairline);
            padding: 0.45rem 0 0.45rem 0.85rem;
            color: var(--ink-muted);
            font-size: 0.88rem;
            font-style: italic;
            margin-bottom: var(--space-2);
        }}
        .citation-source {{
            display: block;
            font-family: 'Inter', monospace;
            font-style: normal;
            font-size: 0.76rem;
            color: var(--ink-muted);
            margin-top: 0.25rem;
            letter-spacing: 0.02em;
        }}
        .verification-flag {{
            display: inline-block;
            font-family: 'Inter', monospace;
            font-size: 0.72rem;
            color: var(--flag-ink);
            background-color: var(--flag-bg);
            border: 1px solid #D8C9A0;
            border-radius: 2px;
            padding: 0.15rem 0.5rem;
            margin-bottom: var(--space-2);
            letter-spacing: 0.02em;
        }}

        .no-match-box {{
            background-color: var(--card);
            border: 1px dashed var(--hairline);
            border-radius: 3px;
            padding: var(--space-3);
            margin-top: var(--space-3);
        }}
        .no-match-heading {{
            font-family: var(--font-display);
            font-size: 1.05rem;
            margin-bottom: 0.3rem;
        }}
        .no-match-body {{ color: var(--ink-muted); font-size: 0.9rem; margin-bottom: var(--space-2); }}

        .related-heading {{
            font-family: 'Inter', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--ink-muted);
            margin: var(--space-2) 0 0.35rem 0;
            border-top: 1px solid var(--hairline);
            padding-top: var(--space-2);
        }}

        .app-footer {{
            margin-top: var(--space-5);
            padding-top: var(--space-2);
            border-top: 1px solid var(--hairline);
            color: var(--ink-muted);
            font-size: 0.8rem;
            line-height: 1.5;
        }}

        /* Font size classes */
        .font-small {{ font-size: 0.85rem; }}
        .font-medium {{ font-size: 1rem; }}
        .font-large {{ font-size: 1.15rem; }}

        @media (max-width: 480px) {{
            .block-container {{ padding-left: 1.1rem; padding-right: 1.1rem; }}
            .app-title {{ font-size: 1.6rem; }}
            .entry {{ padding: var(--space-2); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# HEADER WITH LOGO
# --------------------------------------------------------------------------- #

def render_header() -> None:
    logo_path = "assets/hstu_logo.png"
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.15rem;">
                    <img src="data:image/png;base64,{img_data}" 
                         style="height: 52px; width: auto; object-fit: contain;" />
                    <div>
                        <div style="font-family: 'Inter', sans-serif; font-size: 0.78rem; 
                                    color: #6B6558; letter-spacing: 0.04em; line-height: 1.3;">
                            Hajee Mohammad Danesh Science and Technology University
                        </div>
                        <div style="font-family: 'Inter', sans-serif; font-size: 0.72rem; 
                                    color: #6B6558; letter-spacing: 0.08em; text-transform: uppercase;">
                            CSE 469 — Machine Learning and Pattern Recognition
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# LANGUAGE SELECTOR
# --------------------------------------------------------------------------- #

def render_language_selector():
    cols = st.columns([1, 1, 1, 4])
    with cols[0]:
        if st.button("বাংলা", key="lang_bn", use_container_width=True):
            st.session_state.language = "bn"
            st.rerun()
    with cols[1]:
        if st.button("English", key="lang_en", use_container_width=True):
            st.session_state.language = "en"
            st.rerun()
    with cols[2]:
        if st.button("العربية", key="lang_ar", use_container_width=True):
            st.session_state.language = "ar"
            st.rerun()
    with cols[3]:
        current = LANGUAGE_NAMES.get(st.session_state.language, "English")
        st.markdown(f"<div style='text-align:right;color:#6B6558;font-size:0.8rem;'>{current}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# SEARCH TAB
# --------------------------------------------------------------------------- #

def render_search_tab(df: pd.DataFrame, vectorizer, X, banglish_map, classifier):
    lang = get_lang()
    
    # Example questions
    examples = {
        "bn": ["নামাজ পড়া কি ফরজ?", "সুদ খাওয়া কি হারাম?", "বিয়ে করা কি সুন্নত?"],
        "en": ["Is prayer obligatory?", "Is interest forbidden?", "Is marriage recommended?"],
        "ar": ["هل الصلاة واجبة؟", "هل الربا حرام؟", "هل الزواج سنة؟"]
    }
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Search",
            placeholder=t("search_placeholder"),
            label_visibility="collapsed",
            key="search_input"
        )
    with col2:
        if st.button(t("search_button"), use_container_width=True):
            pass
    
    # Example questions
    if not query:
        st.caption(t("examples"))
        example_cols = st.columns(3)
        for i, ex in enumerate(examples.get(lang, examples["en"])[:3]):
            with example_cols[i]:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    st.session_state.search_input = ex
                    st.rerun()
    
    if query.strip():
        # Add to search history
        if query not in st.session_state.search_history:
            st.session_state.search_history.insert(0, query)
            if len(st.session_state.search_history) > 20:
                st.session_state.search_history.pop()
        
        result = retrieve_candidates(query, df, vectorizer, X, banglish_map)
        if result.row is not None:
            render_result(result, df, vectorizer, classifier)
        else:
            render_no_match(result)


def render_result(result: RetrievalResult, df: pd.DataFrame, vectorizer, classifier):
    row = result.row
    lang = get_lang()
    direction = get_direction()
    
    lang_class = f"lang-{lang}" if lang == "ar" else ""
    
    st.markdown(f'<div class="entry {lang_class}">', unsafe_allow_html=True)
    
    # Reference number
    st.markdown(f'<div class="entry-refno">Ref. No. {int(row["id"]):04d}</div>', unsafe_allow_html=True)
    
    # Question in the right language
    question = get_field_for_lang(row, "question")
    if question:
        st.markdown(f'<div class="entry-question">{question}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="entry-question">{row["question_bn"]}</div>', unsafe_allow_html=True)
    
    # English question as subtitle (if different from displayed)
    if lang != "en" and row.get("question_en"):
        st.markdown(f'<div class="entry-question-en">{row["question_en"]}</div>', unsafe_allow_html=True)
    
    # Category label
    tier1 = str(row["tier1_class"])
    label = tier1.replace("_", " ")
    label_bn = TIER1_CLASS_BN.get(tier1, "")
    strictness = str(row.get("strictness_label", "")).strip()
    label_text = f"{label} — {strictness}" if strictness else label
    st.markdown(
        f'<div class="category-label"><span class="category-mark">۞</span>{label_text}'
        f'<span style="color:#6B6558;font-size:0.7rem;"> · {label_bn}</span></div>',
        unsafe_allow_html=True,
    )
    
    # Verification status
    if str(row.get("verification_status", "")).strip() == NEEDS_VERIFICATION_LABEL:
        st.markdown(
            f'<div class="verification-flag">⚠️ {t("needs_verification")}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="verification-flag" style="background-color:#E1E9E4;border-color:#A8C4B8;">✓ {t("verified")}</div>',
            unsafe_allow_html=True,
        )
    
    # Explanation
    explanation = get_field_for_lang(row, "short_explanation")
    if explanation:
        st.markdown(f'<div class="explanation-text">{explanation}</div>', unsafe_allow_html=True)
    else:
        # Fallback to Bangla or English
        if row.get("short_explanation_bn"):
            st.markdown(f'<div class="explanation-text">{row["short_explanation_bn"]}</div>', unsafe_allow_html=True)
        elif row.get("short_explanation_en"):
            st.markdown(f'<div class="explanation-text">{row["short_explanation_en"]}</div>', unsafe_allow_html=True)
    
    # Reference
    ref_text = str(row.get("reference_text", "")).strip()
    ref_source = str(row.get("reference_source", "")).strip()
    if ref_text and ref_text != PLACEHOLDER_REFERENCE_TEXT:
        st.markdown(
            f'<div class="citation-block">{ref_text}<span class="citation-source">{ref_source}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f'<div class="citation-block"><span class="citation-source">{ref_source}</span></div>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Buttons: Bookmark, Copy Citation
    col1, col2, col3 = st.columns(3)
    with col1:
        bookmark_key = f"bm_{row['id']}"
        if bookmark_key in st.session_state.bookmarks:
            if st.button("⭐ " + t("bookmarks"), key=f"unbm_{row['id']}"):
                st.session_state.bookmarks.remove(bookmark_key)
                st.rerun()
        else:
            if st.button("☆ " + t("bookmarks"), key=f"bm_{row['id']}"):
                st.session_state.bookmarks.append(bookmark_key)
                st.rerun()
    
    with col2:
        citation = f"Question: {row.get('question_en', row.get('question_bn', ''))}\nRuling: {row['tier1_class']}\nReference: {ref_source}"
        if st.button("📋 " + t("copy_citation"), key=f"cite_{row['id']}"):
            st.write(citation)
            st.toast("Citation copied to clipboard!", icon="✅")
    
    with col3:
        if st.button("🖨️ " + t("print"), key=f"print_{row['id']}"):
            st.write("--- Print-friendly view ---")
            st.write(citation)
    
    # Related entries
    related = df[(df["topic"] == row["topic"]) & (df["id"] != row["id"])].head(TOP_K_RELATED)
    if not related.empty:
        st.markdown(f'<div class="related-heading">{t("related")}</div>', unsafe_allow_html=True)
        for _, r in related.iterrows():
            display_text = r["question_en"] or r["question_bn"]
            if st.button(f"{display_text}", key=f"related_{row['id']}_{_}"):
                st.session_state.pending_query = str(r["question_en"] or r["question_bn"])
                st.rerun()
    
    # Details expander
    with st.expander(t("details")):
        stage_labels = {
            "exact_match": "Exact / Rule-Based Match",
            "tfidf_cosine": "TF-IDF + Cosine Similarity",
            "fuzzy_match": "Fuzzy / Instance-Based Match",
        }
        st.write(f"Matched via: {stage_labels.get(result.stage, result.stage)}")
        st.write(f"Confidence score: {result.similarity:.2f}")
        
        predicted = classify_query(str(row["question_en"] or row["question_bn"]), vectorizer, classifier)
        if predicted is not None:
            agree = "agrees" if predicted == row["tier1_class"] else "disagrees"
            st.write(f"Classifier confirmation: {predicted.replace('_', ' ')} ({agree})")
        else:
            st.write("Classifier confirmation unavailable (not enough training data)")


def render_no_match(result: RetrievalResult):
    st.markdown('<div class="no-match-box">', unsafe_allow_html=True)
    st.markdown(f'<div class="no-match-heading">{t("no_match")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="no-match-body">{t("suggestions")}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if result.suggestions:
        for i, r in enumerate(result.suggestions):
            display_text = r["question_en"] or r["question_bn"]
            if st.button(f"{display_text}", key=f"suggestion_{i}"):
                st.session_state.pending_query = str(r["question_en"] or r["question_bn"])
                st.rerun()


# --------------------------------------------------------------------------- #
# BROWSE TAB
# --------------------------------------------------------------------------- #

def render_browse_tab(df: pd.DataFrame):
    lang = get_lang()
    
    # Topic filter
    topics = ["All"] + sorted(df["topic"].unique().tolist())
    selected_topic = st.selectbox(t("topic"), topics)
    
    # Class filter
    classes = ["All"] + sorted(df["tier1_class"].unique().tolist())
    selected_class = st.selectbox(t("ruling"), classes)
    
    # Filter
    filtered = df.copy()
    if selected_topic != "All":
        filtered = filtered[filtered["topic"] == selected_topic]
    if selected_class != "All":
        filtered = filtered[filtered["tier1_class"] == selected_class]
    
    st.caption(f"{len(filtered)} entries")
    
    for _, row in filtered.iterrows():
        st.markdown('<div class="entry">', unsafe_allow_html=True)
        st.markdown(f'<div class="entry-refno">Ref. No. {int(row["id"]):04d}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="entry-question">{row["question_en"] or row["question_bn"]}</div>', unsafe_allow_html=True)
        tier1 = str(row["tier1_class"])
        label = tier1.replace("_", " ")
        st.markdown(
            f'<div class="category-label"><span class="category-mark">۞</span>{label}</div>',
            unsafe_allow_html=True,
        )
        explanation = get_field_for_lang(row, "short_explanation")
        if explanation:
            st.markdown(f'<div class="explanation-text">{explanation}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# MODEL INSIGHTS TAB
# --------------------------------------------------------------------------- #

def render_model_insights_tab(df: pd.DataFrame):
    lang = get_lang()
    
    st.markdown(f"#### {t('model_insights')}")
    
    # Load analysis results
    if st.session_state.analysis_results is None:
        with st.spinner("Loading ML analysis..."):
            st.session_state.analysis_results = run_complete_analysis(df)
    
    results = st.session_state.analysis_results
    
    # Tabs within Model Insights
    tabs = st.tabs([
        "📊 Overview",
        "📈 Classification",
        "🔄 Clustering",
        "📉 PCA",
        "📋 Viva Mode"
    ])
    
    with tabs[0]:
        render_overview_tab(results, df)
    
    with tabs[1]:
        render_classification_tab(results)
    
    with tabs[2]:
        render_clustering_tab(results)
    
    with tabs[3]:
        render_pca_tab(results)
    
    with tabs[4]:
        render_viva_mode_tab()


def render_overview_tab(results, df):
    lang = get_lang()
    
    st.markdown("### Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Classes", df["tier1_class"].nunique())
    with col3:
        st.metric("Topics", df["topic"].nunique())
    
    # Class distribution chart
    st.markdown("#### Ruling Class Distribution")
    class_counts = df["tier1_class"].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]
    
    fig = px.bar(
        class_counts,
        x="Class",
        y="Count",
        color="Class",
        title="Distribution of Rulings by Class",
        labels={"Class": "Ruling Class", "Count": "Number of Records"},
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("What does this graph mean?"):
        st.write("This chart shows how many rulings fall into each category. If some classes have fewer examples, machine learning models may have less information to learn those patterns. This is why we use macro-averaged metrics that treat every class equally.")


def render_classification_tab(results):
    lang = get_lang()
    
    st.markdown("### Classifier Comparison")
    
    # Results table
    model_data = []
    for name, r in results["results"].items():
        model_data.append({
            "Model": name.replace("_", " ").title(),
            "Macro F1": f"{r['macro_f1']:.3f}",
            "Accuracy": f"{r['accuracy']:.3f}",
            "CV F1": f"{r['cv_mean_macro_f1']:.3f} ± {r['cv_std_macro_f1']:.3f}" if not np.isnan(r['cv_mean_macro_f1']) else "N/A",
        })
    
    st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)
    
    # Bar chart
    fig = px.bar(
        pd.DataFrame(model_data),
        x="Model",
        y=[float(v) for v in [d["Macro F1"] for d in model_data]],
        title="Model Comparison by Macro F1",
        labels={"y": "Macro F1 Score"},
        color="Model",
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=300,
        yaxis_range=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Why Logistic Regression?"):
        st.write("""
        Logistic Regression runs the live Search tab. It is not necessarily the single highest score — 
        the pick is justified by cross-validation stability (low variance across folds), not the top 
        raw number from one split. Logistic Regression and Naive Bayes are the classic low-variance 
        choices on small, sparse TF-IDF data; higher-capacity models (Decision Tree, Random Forest) 
        need more data before their extra flexibility pays off.
        """)


def render_clustering_tab(results):
    lang = get_lang()
    
    st.markdown("### K-Means Clustering")
    
    kmeans = results["kmeans"]
    coords = results["pca"]["coords"]
    
    st.metric("Adjusted Rand Index", f"{kmeans['ari']:.3f}")
    st.caption("1.0 = perfect alignment with human topics, 0.0 = random")
    
    # Create two plots side by side
    fig = make_subplots(rows=1, cols=2, subplot_titles=["K-Means Clusters", "True Topics"])
    
    # K-Means clusters
    fig.add_trace(
        go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(
                color=kmeans["cluster_labels"],
                colorscale="Viridis",
                showscale=False,
                size=10,
                opacity=0.7,
            ),
            name="K-Means",
        ),
        row=1, col=1
    )
    
    # True topics (using a different color scale)
    # We'll use a simplified approach - show topics as text labels
    fig.add_trace(
        go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(
                color=results["kmeans"]["cluster_labels"],
                colorscale="Plasma",
                showscale=False,
                size=10,
                opacity=0.7,
                symbol="circle-open",
                line=dict(width=1, color="black"),
            ),
            name="Topics",
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("What does this mean?"):
        st.write("""
        The Adjusted Rand Index (ARI) measures how well the algorithm's own groupings (K-Means clusters) 
        match the human-assigned topic labels. A score close to 1.0 means the algorithm found the same 
        groupings as humans did. Perfect alignment isn't expected because K-Means groups by word similarity 
        while humans group by meaning.
        """)


def render_pca_tab(results):
    lang = get_lang()
    
    st.markdown("### PCA Visualization")
    
    coords = results["pca"]["coords"]
    y = results["y"]
    
    st.metric("Variance Explained", f"{results['pca']['explained']:.1f}%")
    st.caption("Percentage of data variance captured by the first 2 PCA components")
    
    # PCA scatter plot
    df_plot = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Class": y
    })
    
    fig = px.scatter(
        df_plot,
        x="PC1",
        y="PC2",
        color="Class",
        title="PCA Projection of TF-IDF Features",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
        color_discrete_sequence=px.colors.qualitative.Set1,
        opacity=0.8,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("What does this mean?"):
        st.write(f"""
        PCA reduces the high-dimensional TF-IDF space to just 2 dimensions for visualization. 
        These 2 dimensions capture {results['pca']['explained']:.1f}% of the variance in the data — 
        the rest is spread across dimensions we can't easily draw. Colors represent different ruling 
        classes. If classes cluster together, it suggests the text features capture meaningful differences 
        between categories.
        """)


def render_viva_mode_tab():
    lang = get_lang()
    
    st.markdown("### 📋 Viva / Project Explanation")
    
    explanations = [
        {
            "title": "Why TF-IDF?",
            "text": "TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numbers by measuring how important a word is in a document relative to its overall frequency. Words that appear often in one document but rarely elsewhere get high scores, helping the model understand what makes each document unique."
        },
        {
            "title": "Why Character N-Grams (3-5)?",
            "text": "Bangla has suffix-based morphology (e.g., নামাজ/নামাজে/নামাজের all share the same root). Character n-grams capture these shared sub-word patterns. They also handle Bangla/Banglish spelling variations (namaj/namaz) better than whole-word matching."
        },
        {
            "title": "Why Macro F1?",
            "text": "Macro F1 treats every class equally when calculating average performance. This is important because our dataset is imbalanced — some classes have many examples while others have few. Macro F1 ensures small classes aren't ignored."
        },
        {
            "title": "Why Stratified Cross-Validation?",
            "text": "Cross-validation tests the model on different subsets of data. Stratified ensures each subset has the same class distribution as the full dataset. This is crucial when classes are imbalanced to prevent some folds from missing entire classes."
        },
        {
            "title": "Why Retrieval-First, Not Classifier-First?",
            "text": "The retrieval pipeline (exact match → TF-IDF → fuzzy match) finds the closest known example. The classifier is only a secondary confirmation. This approach is safer — the system never invents a ruling; it only retrieves what's in the dataset. In Islamic context, accuracy and verifiability matter more than generating novel answers."
        },
        {
            "title": "What Happens with a Small Dataset?",
            "text": "With few examples, models can memorize the data instead of learning general patterns. This is why we: (1) use cross-validation to detect overfitting, (2) macro-average metrics, (3) warn when classes are too small, and (4) never claim high accuracy as definitive. The system is designed to be honest about its limitations."
        },
        {
            "title": "Why RANDOM_STATE = 42?",
            "text": "Random state controls the randomness in algorithms. Setting it to a fixed number (42) ensures the same results every time you run the code — essential for reproducible research and viva demonstrations."
        },
        {
            "title": "What is PCA?",
            "text": "PCA (Principal Component Analysis) reduces high-dimensional data to fewer dimensions while preserving the most important patterns. We use it here to visualize the text data in 2D, letting us see if different ruling classes naturally separate in the feature space."
        },
        {
            "title": "What is K-Means?",
            "text": "K-Means is an unsupervised clustering algorithm that groups similar data points together. We use it to see if the algorithm finds groupings similar to human-created topics. The Adjusted Rand Index measures how well the algorithm's groupings match human labels."
        }
    ]
    
    for ex in explanations:
        with st.expander(f"🔍 {ex['title']}"):
            st.write(ex['text'])


# --------------------------------------------------------------------------- #
# ABOUT TAB
# --------------------------------------------------------------------------- #

def render_about_tab():
    lang = get_lang()
    
    st.markdown(f"### {t('about')}")
    
    st.markdown("""
    ### Purpose
    
    This system is a retrieval-first Islamic ruling reference with integrated Machine Learning capabilities.
    
    ### How It Works
    
    1. **Search**: Enter a question in Bangla, English, or Banglish
    2. **Retrieval**: The system finds the closest match using:
       - Exact / Rule-Based Matching
       - TF-IDF + Cosine Similarity
       - Fuzzy / Instance-Based Matching
    3. **Result**: A verified ruling with explanation and reference
    4. **ML Confirmation**: A secondary classifier confirms the result
    
    ### Academic Context
    
    This is a CSE 469 (Machine Learning and Pattern Recognition) capstone project.
    
    ### Dataset
    
    The system uses a curated dataset of Islamic rulings with:
    - 225+ records
    - 7 ruling categories
    - 10 topics
    - Bilingual (Bangla + English) content
    
    ### Limitations
    
    - Dataset is small (ML results are illustrative)
    - Some entries need verification
    - Not a substitute for a qualified scholar
    - No Arabic data yet (coming soon)
    
    ### Future Work
    
    - Expand dataset
    - Add Arabic translations
    - Improve ML models with more data
    - Add more interactive visualizations
    """)
    
    st.markdown(f"""
    ---
    {t('disclaimer')}
    """)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

TIER1_CLASS_BN = {
    "Obligatory": "আবশ্যক",
    "Recommended": "সুপারিশকৃত",
    "Permissible": "অনুমোদিত",
    "Disliked": "অপছন্দনীয়",
    "Forbidden": "নিষিদ্ধ",
    "Religious_Innovation": "বিদআত",
    "Faith_Violation": "ঈমান পরিপন্থী",
}

STAGE_LABELS = {
    "exact_match": ("Exact match", "সরাসরি মিল"),
    "tfidf_cosine": ("TF-IDF · cosine similarity", "শব্দ-সাদৃশ্য বিশ্লেষণ"),
    "fuzzy_match": ("Approximate match", "আনুমানিক মিল"),
}


def main() -> None:
    st.set_page_config(
        page_title="Islamic Mas'alah System",
        page_icon="۞",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    
    init_session_state()
    inject_css()
    
    # Header with logo
    render_header()
    
    # Language selector
    render_language_selector()
    
    # Title
    st.markdown(f'<div class="app-eyebrow">CSE 469 — Machine Learning and Pattern Recognition</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-title">{t("app_title")} <span style="font-size:1.2rem;color:#6B6558;">۞</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-subtitle">{t("app_subtitle")}</div>', unsafe_allow_html=True)
    st.markdown('<hr class="app-rule" />', unsafe_allow_html=True)
    
    # Load data
    try:
        df = load_dataset(DATA_PATH)
        df = build_combined_text(df)
        banglish_map = build_banglish_map(df)
        vectorizer, X, classifier = train_classifier(df)
    except DatasetError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Could not start: {exc}")
        st.stop()
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### Controls")
        
        # Font size
        font_size = st.radio(
            t("font_size"),
            ["small", "medium", "large"],
            index=["small", "medium", "large"].index(st.session_state.font_size),
            horizontal=True,
        )
        st.session_state.font_size = font_size
        
        # Search history
        st.markdown("---")
        st.markdown(f"### {t('search_history')}")
        if st.session_state.search_history:
            for i, q in enumerate(st.session_state.search_history[:10]):
                if st.button(f"{i+1}. {q[:40]}", key=f"hist_{i}"):
                    st.session_state.pending_query = q
                    st.rerun()
            if st.button("Clear History"):
                st.session_state.search_history = []
                st.rerun()
        else:
            st.caption("No searches yet")
        
        # Bookmarks
        st.markdown("---")
        st.markdown(f"### {t('bookmarks')}")
        if st.session_state.bookmarks:
            for bm in st.session_state.bookmarks:
                if st.button(f"📌 {bm}", key=bm):
                    st.session_state.pending_query = bm
                    st.rerun()
        else:
            st.caption("No bookmarks yet")
    
    # Main content tabs
    tabs = st.tabs([
        f"🔍 {t('search_placeholder').split('...')[0]}",
        f"📚 {t('browse')}",
        f"🧠 {t('model_insights')}",
        f"ℹ️ {t('about')}",
    ])
    
    with tabs[0]:
        render_search_tab(df, vectorizer, X, banglish_map, classifier)
    
    with tabs[1]:
        render_browse_tab(df)
    
    with tabs[2]:
        render_model_insights_tab(df)
    
    with tabs[3]:
        render_about_tab()
    
    # Footer
    st.markdown(
        f'<div class="app-footer">{t("disclaimer")}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
