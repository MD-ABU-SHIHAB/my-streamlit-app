"""
app.py — Islamic ruling reference search (CSE 469 capstone)

Retrieval-first search over dataset.csv (exact match -> TF-IDF cosine
similarity -> fuzzy match -> honest no-match state), with a Logistic
Regression classifier used only as a secondary confirmation signal, never
as the primary displayed answer. Includes a browse/filter mode.

Run:
    streamlit run app.py

Required packages (pip):
    streamlit
    pandas
    numpy
    scikit-learn
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------- #
# CONSTANTS
# --------------------------------------------------------------------------- #

DATA_PATH: str = "dataset.csv"
RANDOM_STATE: int = 42

TFIDF_SIMILARITY_THRESHOLD: float = 0.30
FUZZY_MATCH_THRESHOLD: float = 0.55
TOP_K_SUGGESTIONS: int = 5
TOP_K_RELATED: int = 4
MIN_QUERY_LEN_FOR_EXACT: int = 2
MEANINGFUL_TOKEN_MIN_LEN: int = 3

REQUIRED_COLUMNS = [
    "id", "question_bn", "question_en", "question_banglish", "tier1_class",
    "topic", "strictness_label", "short_explanation_bn", "short_explanation_en",
    "reference_text", "reference_source", "source_type", "verification_status",
    "search_keywords",
]

BANGLA_STOPWORDS = {
    "কি", "কী", "না", "নাই", "কি না", "এবং", "ও", "তে", "এর", "এই",
    "সেই", "একটি", "একটা", "কে", "কাকে", "কোন", "কোনো", "যে", "যেটা",
    "হয়", "হয়েছে", "করা", "করে", "করতে", "কর", "কেন", "কিভাবে",
    "থেকে", "জন্য", "সাথে", "সঙ্গে", "আছে", "নেই", "হবে", "হলো",
    "তার", "তাদের", "আমার", "আমি", "আপনি", "তুমি", "সে", "তিনি",
    "এ", "ঐ", "ওই", "টা", "টি", "গুলো", "গুলি",
}

ENGLISH_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "and", "or", "it", "its",
    "this", "that", "these", "those", "do", "does", "did", "can", "will",
    "should", "would", "i", "my", "you", "your", "he", "she", "they",
    "their", "what", "which", "who", "whom", "if", "not", "no",
}

NEEDS_VERIFICATION_LABEL = "NEEDS_VERIFICATION"
PLACEHOLDER_REFERENCE_TEXT = "SEE_REFERENCE_SOURCE"
SECTION_MARK = "\u06de"  # ۞
HSTU_LOGO_URL = "https://hstu.ac.bd/img/hstu_logo_.png"

STAGE_LABELS = {
    "exact_match": ("Exact match", "সরাসরি মিল"),
    "tfidf_cosine": ("TF-IDF · cosine similarity", "শব্দ-সাদৃশ্য বিশ্লেষণ"),
    "fuzzy_match": ("Approximate match", "আনুমানিক মিল"),
}

TIER1_CLASS_BN = {
    "Obligatory": "আবশ্যক",
    "Recommended": "সুপারিশকৃত",
    "Permissible": "অনুমোদিত",
    "Disliked": "অপছন্দনীয়",
    "Forbidden": "নিষিদ্ধ",
    "Religious_Innovation": "বিদআত",
    "Faith_Violation": "ঈমান পরিপন্থী",
}

TOPIC_BN = {
    "Worship": "এবাদত",
    "Food_and_Drink": "খাদ্য ও পানীয়",
    "Family_and_Marriage": "পরিবার ও বিবাহ",
    "Business_and_Finance": "ব্যবসা ও অর্থ",
    "Purity": "পবিত্রতা",
    "Social_Conduct": "সামাজিক আচরণ",
    "Clothing_and_Adornment": "পোশাক ও সাজসজ্জা",
    "Faith_and_Aqidah": "ঈমান ও আকিদা",
    "Funeral_and_Mourning": "জানাযা ও শোক",
    "Oaths_and_Vows": "শপথ ও মানত",
}


def bilingual(en: str, bn: str) -> str:
    return f"{en} <span class='bn-inline'>· {bn}</span>"


def bilingual_plain(en: str, bn: str) -> str:
    return f"{en} · {bn}"


def bilingual_block(en: str, bn: str) -> str:
    return f'<span class="en-line">{en}</span><span class="bn-line">{bn}</span>'


# --------------------------------------------------------------------------- #
# DATA MODELS
# --------------------------------------------------------------------------- #

@dataclass
class RetrievalResult:
    row: Optional[pd.Series]
    stage: str
    similarity: float
    suggestions: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# LOADING & PREPROCESSING
# --------------------------------------------------------------------------- #

class DatasetError(Exception):
    pass


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise DatasetError(f"Couldn't find '{path}'. Place dataset.csv in the same folder as app.py.")
    except Exception as exc:
        raise DatasetError(f"Couldn't read '{path}': {exc}")

    if df.empty:
        raise DatasetError(f"'{path}' was found but contains no rows.")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetError("dataset.csv is missing required column(s): " + ", ".join(missing_cols))

    critical = ["question_bn", "tier1_class"]
    df = df.dropna(subset=critical).reset_index(drop=True)
    if df.empty:
        raise DatasetError("Every row is missing question_bn and/or tier1_class — nothing to search.")
    return df


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s\u0980-\u09FF]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in BANGLA_STOPWORDS and t not in ENGLISH_STOPWORDS]
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
            if len(kw) >= MEANINGFUL_TOKEN_MIN_LEN and kw not in BANGLA_STOPWORDS and kw not in ENGLISH_STOPWORDS:
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


# --------------------------------------------------------------------------- #
# CLASSIFIER
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Indexing the reference set…")
def train_classifier(df: pd.DataFrame):
    df = build_combined_text(df)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=20000)
    X = vectorizer.fit_transform(df["combined_text_clean"])
    y = df["tier1_class"]
    classifier: Optional[LogisticRegression] = None
    if y.nunique() >= 2 and y.value_counts().min() >= 1:
        classifier = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        classifier.fit(X, y)
    return vectorizer, X, classifier


# --------------------------------------------------------------------------- #
# RETRIEVAL PIPELINE
# --------------------------------------------------------------------------- #

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\w\u0980-\u09FF]+", text.lower()))


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
# UI — DESIGN TOKENS & STYLE
# --------------------------------------------------------------------------- #

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+Bengali:wght@400;600;700&family=Noto+Sans+Bengali:wght@400;500;600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {{
            --bg: #0F0F1A;
            --card: #1A1A2E;
            --text: #E8E8E8;
            --text-bright: #FFFFFF;
            --text-muted: #9A9AB0;
            --accent: #D4AF37;
            --hairline: #2E2E42;
            --font-display: 'Noto Serif Bengali', Georgia, serif;
            --font-body: 'Noto Sans Bengali', 'Inter', sans-serif;
            --font-mono: 'JetBrains Mono', ui-monospace, monospace;
        }}

        html, body, [class*="css"] {{
            background-color: var(--bg) !important;
            color: var(--text) !important;
            font-family: var(--font-body);
        }}

        /* HSTU Logo as fixed background watermark - BEHIND all content */
        body::before {{
            content: "";
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 400px;
            height: 400px;
            background-image: url('{HSTU_LOGO_URL}');
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            opacity: 0.04;
            pointer-events: none;
            z-index: 0;
        }}

        /* ALL Streamlit content must be ABOVE the watermark */
        .stApp {{
            position: relative;
            z-index: 1;
            background: transparent !important;
        }}

        .block-container {{
            max-width: 700px;
            padding-top: 2.2rem;
            padding-bottom: 3.2rem;
            background: transparent !important;
        }}

        #MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; }}

        /* ---------- Typography ---------- */
        .app-eyebrow {{
            font-family: var(--font-mono);
            font-size: 0.72rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--accent) !important;
            margin-bottom: 0.4rem;
        }}

        .app-title {{
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 2.1rem;
            color: var(--text-bright) !important;
            margin-bottom: 0.4rem;
            line-height: 1.15;
        }}

        .app-subtitle {{
            font-family: var(--font-body);
            color: var(--text-muted) !important;
            font-size: 0.95rem;
        }}

        .app-rule {{
            border: none;
            border-top: 1px solid var(--hairline);
            margin: 1.4rem 0 2.2rem 0;
        }}

        .bn-inline {{
            font-family: var(--font-body);
            color: var(--text-muted) !important;
        }}

        .en-line {{ display: block; color: var(--text) !important; }}
        .bn-line {{
            display: block;
            color: var(--text-muted) !important;
            margin-top: 0.2rem;
            font-size: 0.97em;
        }}

        /* ---------- Search Input ---------- */
        .field-label {{
            font-family: var(--font-body);
            font-size: 0.85rem;
            color: var(--text-muted) !important;
            margin-bottom: 0.35rem;
        }}

        div[data-testid="stTextInput"] input {{
            background-color: var(--card) !important;
            border: 1px solid var(--hairline) !important;
            border-radius: 3px !important;
            color: var(--text-bright) !important;
            font-size: 1.08rem !important;
            padding: 0.85rem 1rem !important;
        }}

        div[data-testid="stTextInput"] input:focus {{
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 1px var(--accent) !important;
        }}

        div[data-testid="stTextInput"] input::placeholder {{
            color: var(--text-muted) !important;
            opacity: 0.7;
        }}

        /* ---------- Buttons ---------- */
        .stButton > button {{
            background-color: var(--accent) !important;
            color: #0F0F1A !important;
            border: none !important;
            border-radius: 3px !important;
            padding: 0.5rem 1.2rem !important;
            font-family: var(--font-body) !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
        }}

        .stButton > button:hover {{
            background-color: #C9A030 !important;
            color: #0F0F1A !important;
        }}

        /* ---------- Radio ---------- */
        div[role="radiogroup"] {{ gap: 1.4rem; }}
        div[role="radiogroup"] label {{
            font-family: var(--font-mono) !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
            color: var(--text-muted) !important;
        }}

        /* ---------- Result Cards ---------- */
        @keyframes entryFadeIn {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .entry {{
            background-color: var(--card) !important;
            border: 1px solid var(--hairline) !important;
            border-radius: 3px !important;
            padding: 1.4rem 1.4rem 0.8rem 1.4rem !important;
            margin-top: 1.4rem !important;
            animation: entryFadeIn 220ms ease-out;
        }}

        .entry-refno {{
            font-family: var(--font-mono) !important;
            font-size: 0.72rem !important;
            color: var(--text-muted) !important;
            letter-spacing: 0.05em !important;
            margin-bottom: 0.8rem !important;
        }}

        .entry-question {{
            font-family: var(--font-display) !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            line-height: 1.4 !important;
            margin-bottom: 0.3rem !important;
            color: var(--text-bright) !important;
        }}

        .entry-question-en {{
            font-family: var(--font-body) !important;
            font-size: 0.92rem !important;
            color: var(--text-muted) !important;
            margin-bottom: 0.8rem !important;
        }}

        .category-label {{
            font-family: var(--font-body) !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
            color: var(--text-bright) !important;
            border-left: 3px solid var(--accent) !important;
            padding: 0.2rem 0 0.2rem 0.65rem !important;
            margin: 0.8rem 0 1.4rem 0 !important;
        }}

        .category-mark {{ color: var(--accent) !important; margin-right: 0.4rem !important; font-weight: 400 !important; }}

        .explanation-text {{ font-size: 0.99rem !important; line-height: 1.6 !important; margin-bottom: 0.8rem !important; color: var(--text) !important; }}

        .citation-block {{
            border-left: 2px solid var(--hairline) !important;
            padding: 0.45rem 0 0.45rem 0.85rem !important;
            color: var(--text-muted) !important;
            font-size: 0.88rem !important;
            font-style: italic !important;
            margin-bottom: 0.8rem !important;
        }}

        .citation-source {{
            display: block !important;
            font-family: var(--font-mono) !important;
            font-style: normal !important;
            font-size: 0.76rem !important;
            color: var(--accent) !important;
            margin-top: 0.25rem !important;
            letter-spacing: 0.02em !important;
        }}

        .verification-flag {{
            display: inline-block !important;
            font-family: var(--font-mono) !important;
            font-size: 0.72rem !important;
            color: #E0C060 !important;
            background-color: #3A2E10 !important;
            border: 1px solid #5A4A20 !important;
            border-radius: 2px !important;
            padding: 0.15rem 0.5rem !important;
            margin-bottom: 0.8rem !important;
            letter-spacing: 0.02em !important;
        }}

        .related-heading {{
            font-family: var(--font-mono) !important;
            font-size: 0.72rem !important;
            letter-spacing: 0.14em !important;
            text-transform: uppercase !important;
            color: var(--text-muted) !important;
            margin: 0.8rem 0 0.35rem 0 !important;
            border-top: 1px solid var(--hairline) !important;
            padding-top: 0.8rem !important;
        }}

        /* ---------- No Match ---------- */
        .no-match-box {{
            background-color: var(--card) !important;
            border: 1px dashed var(--hairline) !important;
            border-radius: 3px !important;
            padding: 1.4rem !important;
            margin-top: 1.4rem !important;
        }}

        .no-match-heading {{
            font-family: var(--font-display) !important;
            font-size: 1.05rem !important;
            margin-bottom: 0.3rem !important;
            color: var(--text-bright) !important;
        }}

        .no-match-body {{ color: var(--text-muted) !important; font-size: 0.9rem !important; margin-bottom: 0.8rem !important; }}

        /* ---------- Secondary Buttons ---------- */
        div[data-testid="stButton"] button[kind="secondary"] {{
            background-color: transparent !important;
            color: var(--text) !important;
            border: none !important;
            border-bottom: 1px solid var(--hairline) !important;
            border-radius: 0 !important;
            text-align: left !important;
            width: 100% !important;
            padding: 0.55rem 0.1rem !important;
            font-family: var(--font-body) !important;
            font-weight: 400 !important;
            font-size: 0.92rem !important;
        }}

        div[data-testid="stButton"] button[kind="secondary"]:hover {{
            background-color: #24243A !important;
            color: var(--text-bright) !important;
        }}

        /* ---------- Browse ---------- */
        .browse-count {{
            font-family: var(--font-mono) !important;
            font-size: 0.78rem !important;
            color: var(--text-muted) !important;
            letter-spacing: 0.05em !important;
            margin: 0.8rem 0 0.4rem 0 !important;
        }}

        /* ---------- Footer ---------- */
        .app-footer {{
            margin-top: 3.2rem !important;
            padding-top: 0.8rem !important;
            border-top: 1px solid var(--hairline) !important;
            color: var(--text-muted) !important;
            font-size: 0.8rem !important;
            line-height: 1.5 !important;
        }}

        @media (max-width: 480px) {{
            .block-container {{ padding-left: 1.1rem !important; padding-right: 1.1rem !important; }}
            .app-title {{ font-size: 1.6rem !important; }}
            .entry {{ padding: 0.8rem !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# UI — RENDERING
# --------------------------------------------------------------------------- #

def _explanation_for(row: pd.Series) -> str:
    parts = []
    if str(row.get("short_explanation_bn", "")).strip():
        parts.append(str(row["short_explanation_bn"]).strip())
    if str(row.get("short_explanation_en", "")).strip():
        parts.append(str(row["short_explanation_en"]).strip())
    return " ".join(parts)


def _set_pending_query(text: str) -> None:
    st.session_state["pending_query"] = text
    st.rerun()


def render_result(result: RetrievalResult, df: pd.DataFrame, vectorizer, classifier) -> None:
    row = result.row
    st.markdown('<div class="entry">', unsafe_allow_html=True)
    st.markdown(f'<div class="entry-refno">Ref. No. {int(row["id"]):04d}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="entry-question">{row["question_bn"]}</div>', unsafe_allow_html=True)
    if str(row.get("question_en", "")).strip():
        st.markdown(f'<div class="entry-question-en">{row["question_en"]}</div>', unsafe_allow_html=True)

    tier1 = str(row["tier1_class"])
    label = tier1.replace("_", " ")
    label_bn = TIER1_CLASS_BN.get(tier1, "")
    strictness = str(row.get("strictness_label", "")).strip()
    label_text = f"{label} — {strictness}" if strictness else label
    st.markdown(
        f'<div class="category-label"><span class="category-mark">{SECTION_MARK}</span>{label_text}'
        f'<span class="bn-inline"> · {label_bn}</span></div>',
        unsafe_allow_html=True,
    )

    if str(row.get("verification_status", "")).strip() == NEEDS_VERIFICATION_LABEL:
        st.markdown(
            f'<div class="verification-flag">{bilingual("Unverified reference — pending scholarly check", "যাচাই বাকি — বিশেষজ্ঞ পর্যালোচনার অপেক্ষায়")}</div>',
            unsafe_allow_html=True,
        )

    explanation = _explanation_for(row)
    if explanation:
        st.markdown(f'<div class="explanation-text">{explanation}</div>', unsafe_allow_html=True)

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

    related = df[(df["topic"] == row["topic"]) & (df["id"] != row["id"])].head(TOP_K_RELATED)
    if not related.empty:
        st.markdown(
            f'<div class="related-heading">{bilingual("Related entries", "সম্পর্কিত মাসআলা")}</div>',
            unsafe_allow_html=True,
        )
        for i, r in related.iterrows():
            label_r = str(r["tier1_class"]).replace("_", " ")
            display_text = r["question_en"] or r["question_bn"]
            if st.button(f"{label_r} — {display_text}", key=f"related_{row['id']}_{i}", type="secondary"):
                _set_pending_query(str(r["question_en"] or r["question_bn"]))

    with st.expander(bilingual_plain("Apparatus — how this was matched", "মিলের বিবরণ")):
        stage_en, stage_bn = STAGE_LABELS.get(result.stage, (result.stage, result.stage))
        st.write(f"Matched via / মিলের ধরন: **{stage_en} · {stage_bn}**")
        st.write(f"Match confidence / মিলের মাত্রা: **{result.similarity:.2f}**")
        predicted = classify_query(str(row["question_en"] or row["question_bn"]), vectorizer, classifier)
        if predicted is not None:
            agree_en = "agrees" if predicted == row["tier1_class"] else "disagrees"
            agree_bn = "মিলছে" if predicted == row["tier1_class"] else "মিলছে না"
            st.write(
                f"Classifier's independent prediction / মডেলের পৃথক পূর্বাভাস: "
                f"**{predicted.replace('_', ' ')}** ({agree_en} · {agree_bn})"
            )
        else:
            st.write(
                "Classifier confirmation unavailable / মডেলের নিশ্চিতকরণ পাওয়া যায়নি "
                "(not enough training data yet / এখনও পর্যাপ্ত প্রশিক্ষণ তথ্য নেই)."
            )


def render_no_match(result: RetrievalResult) -> None:
    st.markdown('<div class="no-match-box">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="no-match-heading">{bilingual_block("No confident match in the reference set.", "নির্ভরযোগ্য কোনো মিল পাওয়া যায়নি।")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="no-match-body">{bilingual_block("Try different wording, or open one of the closest entries below.", "অন্যভাবে লিখে চেষ্টা করুন, অথবা নিচের কাছাকাছি এন্ট্রিগুলো দেখুন।")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if result.suggestions:
        st.markdown(
            f'<div class="related-heading">{bilingual("Closest entries", "কাছাকাছি এন্ট্রি")}</div>',
            unsafe_allow_html=True,
        )
        for i, r in enumerate(result.suggestions):
            label = str(r["tier1_class"]).replace("_", " ")
            display_text = r["question_en"] or r["question_bn"]
            if st.button(f"{label} — {display_text}", key=f"suggestion_{i}", type="secondary"):
                _set_pending_query(str(r["question_en"] or r["question_bn"]))


def render_browse_mode(df: pd.DataFrame) -> None:
    topic_options = ["All · সব"] + [
        f"{t} · {TOPIC_BN.get(t, '')}" for t in sorted(df["topic"].unique().tolist())
    ]
    class_options = ["All · সব"] + [
        f"{c} · {TIER1_CLASS_BN.get(c, '')}" for c in sorted(df["tier1_class"].unique().tolist())
    ]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="field-label">{bilingual("Topic", "বিষয়")}</div>', unsafe_allow_html=True)
        topic_choice = st.selectbox("Topic", topic_options, label_visibility="collapsed")
    with col2:
        st.markdown(f'<div class="field-label">{bilingual("Category", "শ্রেণি")}</div>', unsafe_allow_html=True)
        class_choice = st.selectbox("Category", class_options, label_visibility="collapsed")

    topic_filter = topic_choice.split(" · ")[0]
    class_filter = class_choice.split(" · ")[0]

    filtered = df.copy()
    if topic_filter != "All":
        filtered = filtered[filtered["topic"] == topic_filter]
    if class_filter != "All":
        filtered = filtered[filtered["tier1_class"] == class_filter]

    st.markdown(
        f'<div class="browse-count">{bilingual(f"{len(filtered)} entries", f"{len(filtered)}টি এন্ট্রি")}</div>',
        unsafe_allow_html=True,
    )

    for _, row in filtered.iterrows():
        st.markdown('<div class="entry">', unsafe_allow_html=True)
        st.markdown(f'<div class="entry-refno">Ref. No. {int(row["id"]):04d}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="entry-question">{row["question_en"] or row["question_bn"]}</div>', unsafe_allow_html=True)
        tier1 = str(row["tier1_class"])
        label = tier1.replace("_", " ")
        label_bn = TIER1_CLASS_BN.get(tier1, "")
        strictness = str(row.get("strictness_label", "")).strip()
        label_text = f"{label} — {strictness}" if strictness else label
        st.markdown(
            f'<div class="category-label"><span class="category-mark">{SECTION_MARK}</span>{label_text}'
            f'<span class="bn-inline"> · {label_bn}</span></div>',
            unsafe_allow_html=True,
        )
        explanation = _explanation_for(row)
        if explanation:
            st.markdown(f'<div class="explanation-text">{explanation}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(page_title="Ruling Reference", layout="centered")
    inject_css()

    st.markdown(
        f'<div class="app-eyebrow">{bilingual("A reference collection of verified rulings", "যাচাইকৃত মাসআলার একটি সংকলন")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="app-title">Ruling Reference <span class="bn-inline">· মাসআলা অনুসন্ধান</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="app-subtitle">{bilingual_block("Search in Bangla, English, or Banglish — every entry traces to a cited source.", "বাংলা, ইংরেজি অথবা বাংলিশে খুঁজুন — প্রতিটি ফলাফলের সাথে যাচাইযোগ্য তথ্যসূত্র দেওয়া আছে।")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="app-rule" />', unsafe_allow_html=True)

    try:
        df = load_dataset(DATA_PATH)
        df = build_combined_text(df)
        banglish_map = build_banglish_map(df)
        vectorizer, X, classifier = train_classifier(df)
    except DatasetError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"The app couldn't start up: {exc}")
        st.stop()

    mode = st.radio(
        "Mode",
        ["Search · খুঁজুন", "Browse · সবগুলো দেখুন"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode.startswith("Search"):
        st.markdown(
            f'<div class="field-label">{bilingual("Type your question", "আপনার প্রশ্ন লিখুন")}</div>',
            unsafe_allow_html=True,
        )
        default_query = st.session_state.pop("pending_query", "")
        query = st.text_input(
            "Search",
            value=default_query,
            placeholder="namaj pora ki · is riba haram · বিয়ে করা কি সুন্নত…",
            label_visibility="collapsed",
        )
        if query.strip():
            result = retrieve_candidates(query, df, vectorizer, X, banglish_map)
            if result.row is not None:
                render_result(result, df, vectorizer, classifier)
            else:
                render_no_match(result)
    else:
        render_browse_mode(df)

    st.markdown(
        f'<div class="app-footer">{bilingual_block("Entries here are for educational reference only. For personal or complex matters, consult a qualified scholar.", "এখানে দেওয়া তথ্য শুধুমাত্র শিক্ষামূলক রেফারেন্সের জন্য। ব্যক্তিগত বা জটিল বিষয়ে অবশ্যই একজন যোগ্য আলেমের পরামর্শ নিন।")}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
