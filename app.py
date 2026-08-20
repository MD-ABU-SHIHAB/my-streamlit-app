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
RANDOM_STATE: int = 42  # matches main.py, for consistent classifier behavior

# Retrieval pipeline thresholds. Named and commented so they're easy to tune
# from one place as the dataset grows.
TFIDF_SIMILARITY_THRESHOLD: float = 0.30   # min cosine similarity to accept a TF-IDF match
FUZZY_MATCH_THRESHOLD: float = 0.55        # min difflib ratio to accept a fuzzy match
TOP_K_SUGGESTIONS: int = 5                 # suggestions shown in the no-match state
TOP_K_RELATED: int = 4                     # related rulings shown under a confident match
MIN_QUERY_LEN_FOR_EXACT: int = 2           # ignore exact-match stage for near-empty queries
MEANINGFUL_TOKEN_MIN_LEN: int = 3          # ignore short/particle-like tokens ("is", "কি") for exact matching

REQUIRED_COLUMNS = [
    "id", "question_bn", "question_en", "question_banglish", "tier1_class",
    "topic", "strictness_label", "short_explanation_bn", "short_explanation_en",
    "reference_text", "reference_source", "source_type", "verification_status",
    "search_keywords",
]

TIER1_CLASSES = [
    "Obligatory", "Recommended", "Permissible", "Disliked",
    "Forbidden", "Religious_Innovation", "Faith_Violation",
]

TOPICS = [
    "Worship", "Food_and_Drink", "Family_and_Marriage", "Business_and_Finance",
    "Purity", "Social_Conduct", "Clothing_and_Adornment", "Faith_and_Aqidah",
    "Funeral_and_Mourning", "Oaths_and_Vows",
]

# Same explicit Bangla stopword list as main.py, kept identical on purpose so
# app.py's live preprocessing matches the offline report's preprocessing exactly.
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

# Real Qur'anic section-divider glyph (rub' el hizb), used as the result
# marker. Not a decorative emoji — it is the actual mark used in printed
# Mus'haf pages to divide sections, which is why it belongs here.
SECTION_MARK = "\u06de"  # ۞

STAGE_LABELS = {
    "exact_match": "Exact match",
    "tfidf_cosine": "TF-IDF · cosine similarity",
    "fuzzy_match": "Approximate match",
}


# --------------------------------------------------------------------------- #
# DATA MODELS
# --------------------------------------------------------------------------- #

@dataclass
class RetrievalResult:
    row: Optional[pd.Series]
    stage: str                    # "exact_match" | "tfidf_cosine" | "fuzzy_match" | "no_match"
    similarity: float              # 0..1, meaning depends on stage
    suggestions: list = field(default_factory=list)  # list of pd.Series, used when stage == "no_match"


# --------------------------------------------------------------------------- #
# LOADING & PREPROCESSING (kept identical in approach to main.py)
# --------------------------------------------------------------------------- #

class DatasetError(Exception):
    """Raised for any dataset load/shape problem, so main() can show a clean message."""


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset.csv and validate it has the columns this app needs."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise DatasetError(
            f"Couldn't find '{path}'. Place dataset.csv in the same folder as app.py."
        )
    except Exception as exc:  # malformed CSV, encoding issues, etc.
        raise DatasetError(f"Couldn't read '{path}': {exc}")

    if df.empty:
        raise DatasetError(f"'{path}' was found but contains no rows.")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetError(
            "dataset.csv is missing required column(s): " + ", ".join(missing_cols)
        )

    critical = ["question_bn", "tier1_class"]
    df = df.dropna(subset=critical).reset_index(drop=True)
    if df.empty:
        raise DatasetError(
            "Every row is missing question_bn and/or tier1_class — nothing to search."
        )
    return df


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, remove Bangla+English stopwords. Matches main.py."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s\u0980-\u09FF]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in BANGLA_STOPWORDS and t not in ENGLISH_STOPWORDS]
    return " ".join(tokens)


@st.cache_data(show_spinner=False)
def build_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """Combine question_bn + question_en + search_keywords, same field construction as main.py."""
    df = df.copy()
    for col in ("question_bn", "question_en", "question_banglish", "search_keywords"):
        df[col] = df[col].fillna("")
    df["combined_text_raw"] = (
        df["question_bn"] + " " + df["question_en"] + " " + df["search_keywords"]
    )
    df["combined_text_clean"] = df["combined_text_raw"].apply(clean_text)
    return df


@st.cache_data(show_spinner=False)
def build_banglish_map(df: pd.DataFrame) -> dict[str, set[str]]:
    """Build an extendable synonym map from the dataset's own search_keywords.

    For every row, its search_keywords terms are treated as one synonym
    group — any term in the group maps to every other term in that group.
    This lets a user's query in one spelling ("namaz") pull in the dataset's
    other spellings of the same concept ("namaj", "salah", ...) at query
    time, without hardcoding any per-question logic: new rows automatically
    extend the map.

    Only search_keywords is used (not question_banglish) deliberately:
    search_keywords is a deliberately curated synonym list, whereas
    question_banglish is a full sentence containing generic connector words
    ("ki", "kora", "deya"...) that recur across almost every row. Grouping
    on those connector words would transitively merge unrelated rows'
    synonym groups into one another.
    """
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
    """Expand a raw query with known synonym terms, for use only inside retrieval (never shown to the user)."""
    query_lower = query.lower().strip()
    tokens = re.findall(r"[\w\u0980-\u09FF]+", query_lower)
    expanded = set(tokens)
    for tok in tokens:
        if tok in banglish_map:
            expanded.update(banglish_map[tok])
    return query_lower + " " + " ".join(sorted(expanded))


# --------------------------------------------------------------------------- #
# CLASSIFIER (secondary confirmation signal only — never the primary answer)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Indexing the reference set…")
def train_classifier(df: pd.DataFrame):
    """Fit the TF-IDF vectorizer + Logistic Regression once at startup.

    Character n-grams (matching main.py's choice) handle Bangla's suffix-heavy
    morphology and Bangla/Banglish spelling variation better than word-level
    tokens on this dataset. Logistic Regression is used at runtime (rather
    than the other 4 models compared in main.py) because main.py's
    bias-variance analysis identified it as the most stable generalizer
    across cross-validation folds — not necessarily the single highest
    accuracy on one split, but the most reliable choice for a live app.
    """
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
    """Word-level tokens (Bangla + Latin), not substrings — avoids "is" matching inside "wearing"."""
    return set(re.findall(r"[\w\u0980-\u09FF]+", text.lower()))


def _exact_match(query: str, banglish_map: dict[str, set[str]], df: pd.DataFrame) -> Optional[pd.Series]:
    """Stage a: exact/substring match against the question & keyword fields.

    Two sub-checks, both intentionally strict to avoid false positives from
    short common words:
      1. The full raw query appears verbatim as a substring somewhere.
      2. Every "meaningful" query token (length >= MEANINGFUL_TOKEN_MIN_LEN,
         after expanding through the synonym map) is found as a whole word
         in the same row — not just any single generic word in common.
    """
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
    """Stage b: TF-IDF + cosine similarity."""
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
    """Stage c: difflib fuzzy match, as a last resort before giving up."""
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


def retrieve_candidates(
    query: str, df: pd.DataFrame, vectorizer, X, banglish_map: dict[str, set[str]]
) -> RetrievalResult:
    """Run the full retrieval pipeline, stopping at the first confident stage."""
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
    """Secondary confirmation signal only — the predicted class, never shown as the primary answer."""
    if classifier is None:
        return None
    q_clean = clean_text(query)
    if not q_clean.strip():
        return None
    q_vec = vectorizer.transform([q_clean])
    return classifier.predict(q_vec)[0]


# --------------------------------------------------------------------------- #
# UI — DESIGN TOKENS & STYLE
#
# Direction: a hushed reading-room / critical-edition aesthetic. The subject
# is a reference library of verified rulings, so retrieval metadata (match
# stage, score, classifier agreement) is treated like a scholarly edition's
# critical apparatus — small, typographic, in the margin — rather than a
# dashboard with progress bars. Category identity is a single accent color
# plus label text, never a badge palette per category. The one deliberate
# "signature" touch is the rub' el hizb mark (۞) — the real section-divider
# glyph used in printed Qur'an pages — and a catalog-style reference number
# on each entry, both grounded in the subject rather than decorative.
# --------------------------------------------------------------------------- #

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+Bengali:wght@400;600;700&family=Noto+Sans+Bengali:wght@400;500;600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --paper: #F2ECDD;
            --card: #FBF8EF;
            --ink: #2B2A24;
            --ink-muted: #7A7360;
            --accent: #2F4858;
            --accent-soft: #E4DEC9;
            --hairline: #DDD3B8;
            --flag-bg: #F3E6C8;
            --flag-ink: #8C6B2F;
            --font-display: 'Noto Serif Bengali', Georgia, serif;
            --font-body: 'Noto Sans Bengali', 'Inter', sans-serif;
            --font-mono: 'JetBrains Mono', ui-monospace, monospace;
            --space-1: 0.4rem;
            --space-2: 0.8rem;
            --space-3: 1.4rem;
            --space-4: 2.2rem;
            --space-5: 3.2rem;
        }

        html, body, [class*="css"] {
            background-color: var(--paper) !important;
            color: var(--ink);
            font-family: var(--font-body);
        }
        .block-container { max-width: 700px; padding-top: var(--space-4); padding-bottom: var(--space-5); }
        #MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }

        /* Accessible focus states — never remove outline without replacing it */
        a:focus-visible, button:focus-visible, input:focus-visible {
            outline: 2px solid var(--accent);
            outline-offset: 2px;
        }
        @media (prefers-reduced-motion: reduce) {
            * { animation: none !important; transition: none !important; }
        }

        /* ---------- Header ---------- */
        .app-eyebrow {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--ink-muted);
            margin-bottom: var(--space-1);
        }
        .app-title {
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 2.1rem;
            color: var(--ink);
            margin-bottom: var(--space-1);
            line-height: 1.15;
        }
        .app-subtitle {
            font-family: var(--font-body);
            color: var(--ink-muted);
            font-size: 0.95rem;
        }
        .app-rule {
            border: none;
            border-top: 1px solid var(--hairline);
            margin: var(--space-3) 0 var(--space-4) 0;
        }

        /* ---------- Search ---------- */
        div[data-testid="stTextInput"] input {
            background-color: var(--card);
            border: 1px solid var(--hairline);
            border-radius: 3px;
            color: var(--ink);
            font-size: 1.08rem;
            padding: 0.85rem 1rem;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 1px var(--accent);
        }
        div[data-testid="stTextInput"] input::placeholder { color: var(--ink-muted); opacity: 0.8; }

        .stButton > button {
            background-color: var(--accent);
            color: var(--card);
            border: none;
            border-radius: 3px;
            padding: 0.5rem 1.2rem;
            font-family: var(--font-body);
            font-weight: 500;
            font-size: 0.88rem;
            letter-spacing: 0.02em;
            transition: background-color 120ms ease;
        }
        .stButton > button:hover { background-color: #24394A; color: var(--card); }

        /* Mode toggle reads like a catalog tab, not a pill switch */
        div[role="radiogroup"] { gap: var(--space-3); }
        div[role="radiogroup"] label {
            font-family: var(--font-mono);
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--ink-muted);
        }

        /* ---------- Result entry ---------- */
        @keyframes entryFadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .entry {
            background-color: var(--card);
            border: 1px solid var(--hairline);
            border-radius: 3px;
            padding: var(--space-3) var(--space-3) var(--space-2) var(--space-3);
            margin-top: var(--space-3);
            animation: entryFadeIn 220ms ease-out;
        }
        .entry-refno {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            color: var(--ink-muted);
            letter-spacing: 0.05em;
            margin-bottom: var(--space-2);
        }
        .entry-question {
            font-family: var(--font-display);
            font-size: 1.2rem;
            font-weight: 600;
            line-height: 1.4;
            margin-bottom: 0.3rem;
        }
        .entry-question-en {
            font-family: var(--font-body);
            font-size: 0.92rem;
            color: var(--ink-muted);
            margin-bottom: var(--space-2);
        }
        .category-label {
            font-family: var(--font-body);
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--ink);
            border-left: 3px solid var(--accent);
            padding: 0.2rem 0 0.2rem 0.65rem;
            margin: var(--space-2) 0 var(--space-3) 0;
        }
        .category-mark { color: var(--accent); margin-right: 0.4rem; font-weight: 400; }

        .explanation-text { font-size: 0.99rem; line-height: 1.6; margin-bottom: var(--space-2); }

        .citation-block {
            border-left: 2px solid var(--hairline);
            padding: 0.45rem 0 0.45rem 0.85rem;
            color: var(--ink-muted);
            font-size: 0.88rem;
            font-style: italic;
            margin-bottom: var(--space-2);
        }
        .citation-source {
            display: block;
            font-family: var(--font-mono);
            font-style: normal;
            font-size: 0.76rem;
            color: var(--ink-muted);
            margin-top: 0.25rem;
            letter-spacing: 0.02em;
        }

        .verification-flag {
            display: inline-block;
            font-family: var(--font-mono);
            font-size: 0.72rem;
            color: var(--flag-ink);
            background-color: var(--flag-bg);
            border: 1px solid #E0CD9C;
            border-radius: 2px;
            padding: 0.15rem 0.5rem;
            margin-bottom: var(--space-2);
            letter-spacing: 0.02em;
        }

        .related-heading {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--ink-muted);
            margin: var(--space-2) 0 0.35rem 0;
            border-top: 1px solid var(--hairline);
            padding-top: var(--space-2);
        }

        /* ---------- No-match state ---------- */
        .no-match-box {
            background-color: var(--card);
            border: 1px dashed var(--hairline);
            border-radius: 3px;
            padding: var(--space-3);
            margin-top: var(--space-3);
        }
        .no-match-heading {
            font-family: var(--font-display);
            font-size: 1.05rem;
            margin-bottom: 0.3rem;
        }
        .no-match-body { color: var(--ink-muted); font-size: 0.9rem; margin-bottom: var(--space-2); }

        /* ---------- Clickable list rows (related rulings / suggestions) ---------- */
        div[data-testid="stButton"] button[kind="secondary"] {
            background-color: transparent;
            color: var(--ink);
            border: none;
            border-bottom: 1px solid var(--hairline);
            border-radius: 0;
            text-align: left;
            width: 100%;
            padding: 0.55rem 0.1rem;
            font-family: var(--font-body);
            font-weight: 400;
            font-size: 0.92rem;
        }
        div[data-testid="stButton"] button[kind="secondary"]:hover {
            background-color: var(--accent-soft);
            color: var(--ink);
        }

        /* ---------- Browse mode ---------- */
        .browse-count {
            font-family: var(--font-mono);
            font-size: 0.78rem;
            color: var(--ink-muted);
            letter-spacing: 0.05em;
            margin: var(--space-2) 0 var(--space-1) 0;
        }

        /* ---------- Footer ---------- */
        .app-footer {
            margin-top: var(--space-5);
            padding-top: var(--space-2);
            border-top: 1px solid var(--hairline);
            color: var(--ink-muted);
            font-size: 0.8rem;
            line-height: 1.5;
        }

        @media (max-width: 480px) {
            .block-container { padding-left: 1.1rem; padding-right: 1.1rem; }
            .app-title { font-size: 1.6rem; }
            .entry { padding: var(--space-2); }
        }
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
    """Route a click on a related/suggested entry back into the search box."""
    st.session_state["pending_query"] = text
    st.rerun()


def render_result(result: RetrievalResult, df: pd.DataFrame, vectorizer, classifier) -> None:
    row = result.row
    st.markdown('<div class="entry">', unsafe_allow_html=True)

    st.markdown(f'<div class="entry-refno">Ref. No. {int(row["id"]):04d}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="entry-question">{row["question_bn"]}</div>', unsafe_allow_html=True)
    if str(row.get("question_en", "")).strip():
        st.markdown(f'<div class="entry-question-en">{row["question_en"]}</div>', unsafe_allow_html=True)

    label = str(row["tier1_class"]).replace("_", " ")
    strictness = str(row.get("strictness_label", "")).strip()
    label_text = f"{label} — {strictness}" if strictness else label
    st.markdown(
        f'<div class="category-label"><span class="category-mark">{SECTION_MARK}</span>{label_text}</div>',
        unsafe_allow_html=True,
    )

    if str(row.get("verification_status", "")).strip() == NEEDS_VERIFICATION_LABEL:
        st.markdown(
            '<div class="verification-flag">Unverified reference — pending scholarly check</div>',
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

    st.markdown("</div>", unsafe_allow_html=True)  # close .entry (buttons below can't live inside raw HTML)

    related = df[(df["topic"] == row["topic"]) & (df["id"] != row["id"])].head(TOP_K_RELATED)
    if not related.empty:
        st.markdown('<div class="related-heading">Related entries</div>', unsafe_allow_html=True)
        for i, r in related.iterrows():
            label_r = str(r["tier1_class"]).replace("_", " ")
            display_text = r["question_en"] or r["question_bn"]
            if st.button(f"{label_r} — {display_text}", key=f"related_{row['id']}_{i}", type="secondary"):
                _set_pending_query(str(r["question_en"] or r["question_bn"]))

    with st.expander("Apparatus — how this was matched"):
        st.write(f"Matched via: {STAGE_LABELS.get(result.stage, result.stage)}")
        st.write(f"Match confidence: {result.similarity:.2f}")
        predicted = classify_query(str(row["question_en"] or row["question_bn"]), vectorizer, classifier)
        if predicted is not None:
            agree = "agrees" if predicted == row["tier1_class"] else "disagrees"
            st.write(f"Classifier's independent prediction: {predicted.replace('_', ' ')} ({agree} with the retrieved entry)")
        else:
            st.write("Classifier confirmation unavailable (not enough training data yet).")


def render_no_match(result: RetrievalResult) -> None:
    st.markdown('<div class="no-match-box">', unsafe_allow_html=True)
    st.markdown('<div class="no-match-heading">No confident match in the reference set.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="no-match-body">Try different wording, or open one of the closest entries below.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if result.suggestions:
        st.markdown('<div class="related-heading">Closest entries</div>', unsafe_allow_html=True)
        for i, r in enumerate(result.suggestions):
            label = str(r["tier1_class"]).replace("_", " ")
            display_text = r["question_en"] or r["question_bn"]
            if st.button(f"{label} — {display_text}", key=f"suggestion_{i}", type="secondary"):
                _set_pending_query(str(r["question_en"] or r["question_bn"]))


def render_browse_mode(df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        topic_filter = st.selectbox("Topic", ["All"] + sorted(df["topic"].unique().tolist()))
    with col2:
        class_filter = st.selectbox("Category", ["All"] + sorted(df["tier1_class"].unique().tolist()))

    filtered = df.copy()
    if topic_filter != "All":
        filtered = filtered[filtered["topic"] == topic_filter]
    if class_filter != "All":
        filtered = filtered[filtered["tier1_class"] == class_filter]

    st.markdown(f'<div class="browse-count">{len(filtered)} entries</div>', unsafe_allow_html=True)

    for _, row in filtered.iterrows():
        st.markdown('<div class="entry">', unsafe_allow_html=True)
        st.markdown(f'<div class="entry-refno">Ref. No. {int(row["id"]):04d}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="entry-question">{row["question_en"] or row["question_bn"]}</div>', unsafe_allow_html=True)
        label = str(row["tier1_class"]).replace("_", " ")
        strictness = str(row.get("strictness_label", "")).strip()
        label_text = f"{label} — {strictness}" if strictness else label
        st.markdown(
            f'<div class="category-label"><span class="category-mark">{SECTION_MARK}</span>{label_text}</div>',
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

    st.markdown('<div class="app-eyebrow">A reference collection of verified rulings</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-title">Ruling Reference</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Search in Bangla, English, or Banglish. Every entry traces to a cited source.</div>',
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
    except Exception as exc:  # last-resort guard: never show a raw traceback
        st.error(f"The app couldn't start up: {exc}")
        st.stop()

    mode = st.radio("Mode", ["Search", "Browse"], horizontal=True, label_visibility="collapsed")

    if mode == "Search":
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
        '<div class="app-footer">Entries here are for educational reference only. '
        "For personal or complex matters, consult a qualified scholar.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
