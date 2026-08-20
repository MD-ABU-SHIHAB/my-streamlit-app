"""
app.py — Islamic ruling reference search (CSE 469 capstone)

Retrieval-first search over dataset.csv (exact match → TF-IDF cosine
similarity → fuzzy match → honest no-match state), with a Logistic
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
    """Build an extendable synonym map from the dataset's own search_keywords + question_banglish.

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
    synonym groups into one another (row A and row B both use "ki", so A's
    and B's vocabulary would incorrectly become "synonyms" of each other).
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

@st.cache_resource(show_spinner="Preparing search index...")
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
    # normalize_query() pulls in the dataset's own synonym spellings for each
    # meaningful token, so e.g. a query token "namaz" also counts a row that
    # only contains the dataset's spelling "namaj".
    expanded_str = normalize_query(query, banglish_map)
    expanded = {t for t in _tokenize(expanded_str) if len(t) >= MEANINGFUL_TOKEN_MIN_LEN}

    # Short queries (<=2 meaningful tokens) must match ALL of them to avoid a
    # single generic word ("haram", "prayer") over-confidently picking one
    # arbitrary row out of many that legitimately contain that word.
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

    # No confident match at any stage: gather the closest suggestions by
    # TF-IDF similarity anyway, so the user has somewhere useful to go.
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
# UI — STYLE
# --------------------------------------------------------------------------- #

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Tiro+Bangla&family=Noto+Serif+Bengali:wght@400;600&family=Noto+Sans+Bengali:wght@400;500&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            background-color: #EDE7D9 !important;
            color: #2B2A26;
            font-family: 'Noto Sans Bengali', 'Inter', sans-serif;
        }
        .block-container { max-width: 760px; padding-top: 2.5rem; padding-bottom: 3rem; }
        #MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }

        h1, h2, h3, .app-heading {
            font-family: 'Tiro Bangla', 'Noto Serif Bengali', Georgia, serif;
            color: #2B2A26;
            font-weight: 600;
        }
        .app-title {
            font-size: 1.9rem;
            margin-bottom: 0.15rem;
        }
        .app-subtitle {
            font-family: 'Noto Sans Bengali', 'Inter', sans-serif;
            color: #6B6558;
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }

        div[data-testid="stTextInput"] input {
            background-color: #F6F2E8;
            border: 1px solid #D8D0BC;
            border-radius: 4px;
            color: #2B2A26;
            font-size: 1.05rem;
            padding: 0.7rem 0.9rem;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #3A4F63;
            box-shadow: 0 0 0 1px #3A4F63;
        }

        .stButton > button {
            background-color: #3A4F63;
            color: #F6F2E8;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1.3rem;
            font-family: 'Noto Sans Bengali', 'Inter', sans-serif;
            font-weight: 500;
        }
        .stButton > button:hover { background-color: #2E3F50; color: #F6F2E8; }

        .result-card {
            background-color: #F6F2E8;
            border: 1px solid #D8D0BC;
            border-radius: 4px;
            padding: 1.4rem 1.6rem;
            margin-top: 1.2rem;
        }
        .result-question {
            font-family: 'Tiro Bangla', 'Noto Serif Bengali', Georgia, serif;
            font-size: 1.15rem;
            margin-bottom: 0.6rem;
        }
        .category-label {
            display: inline-block;
            font-family: 'Noto Sans Bengali', 'Inter', sans-serif;
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #2B2A26;
            border-left: 3px solid #3A4F63;
            padding: 0.15rem 0 0.15rem 0.6rem;
            margin-bottom: 0.9rem;
        }
        .explanation-text { font-size: 0.98rem; line-height: 1.55; margin-bottom: 0.9rem; }
        .citation-block {
            border-left: 3px solid #D8D0BC;
            padding: 0.5rem 0 0.5rem 0.9rem;
            color: #4A463E;
            font-size: 0.9rem;
            font-style: italic;
            margin-bottom: 0.6rem;
        }
        .verification-flag {
            display: inline-block;
            font-size: 0.78rem;
            color: #6B5033;
            background-color: #F0E4CC;
            border: 1px solid #D8C9A0;
            border-radius: 3px;
            padding: 0.15rem 0.5rem;
            margin-bottom: 0.6rem;
        }
        .related-list { margin-top: 0.9rem; font-size: 0.9rem; }
        .related-list a { color: #3A4F63; text-decoration: none; }
        .related-item { padding: 0.25rem 0; border-bottom: 1px solid #E4DFD1; }

        .no-match-box {
            background-color: #F6F2E8;
            border: 1px dashed #D8D0BC;
            border-radius: 4px;
            padding: 1.2rem 1.4rem;
            margin-top: 1.2rem;
            color: #4A463E;
        }

        .app-footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #D8D0BC;
            color: #6B6558;
            font-size: 0.82rem;
            text-align: center;
        }

        @media (max-width: 480px) {
            .block-container { padding-left: 1rem; padding-right: 1rem; }
            .app-title { font-size: 1.5rem; }
            .result-card { padding: 1.1rem 1.1rem; }
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


def render_result(result: RetrievalResult, df: pd.DataFrame, vectorizer, classifier) -> None:
    row = result.row
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="result-question">{row["question_bn"]}</div>', unsafe_allow_html=True)
    if str(row.get("question_en", "")).strip():
        st.markdown(f'<div class="result-question">{row["question_en"]}</div>', unsafe_allow_html=True)

    label = str(row["tier1_class"]).replace("_", " ")
    strictness = str(row.get("strictness_label", "")).strip()
    label_text = f"{label} — {strictness}" if strictness else label
    st.markdown(f'<div class="category-label">{label_text}</div>', unsafe_allow_html=True)

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
    citation_body = ref_source if ref_text == PLACEHOLDER_REFERENCE_TEXT or not ref_text else f"{ref_text} — {ref_source}"
    st.markdown(f'<div class="citation-block">{citation_body}</div>', unsafe_allow_html=True)

    related = df[(df["topic"] == row["topic"]) & (df["id"] != row["id"])].head(TOP_K_RELATED)
    if not related.empty:
        st.markdown('<div class="related-list">Related rulings</div>', unsafe_allow_html=True)
        for _, r in related.iterrows():
            st.markdown(f'<div class="related-item">{r["question_en"] or r["question_bn"]}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Details"):
        stage_labels = {
            "exact_match": "Exact/substring match",
            "tfidf_cosine": "TF-IDF + cosine similarity",
            "fuzzy_match": "Fuzzy text match",
        }
        st.write(f"Matched via: {stage_labels.get(result.stage, result.stage)}")
        st.write(f"Match confidence score: {result.similarity:.2f}")
        predicted = classify_query(str(row["question_en"] or row["question_bn"]), vectorizer, classifier)
        if predicted is not None:
            agree = "agrees" if predicted == row["tier1_class"] else "disagrees"
            st.write(f"Classifier's independent prediction: {predicted.replace('_', ' ')} ({agree} with the retrieved row)")
        else:
            st.write("Classifier confirmation unavailable (not enough training data yet).")


def render_no_match(result: RetrievalResult) -> None:
    st.markdown('<div class="no-match-box">', unsafe_allow_html=True)
    st.markdown("No confident match found for that question.", unsafe_allow_html=True)
    if result.suggestions:
        st.markdown("Closest entries you might mean:", unsafe_allow_html=True)
        for r in result.suggestions:
            label = str(r["tier1_class"]).replace("_", " ")
            st.markdown(
                f'<div class="related-item"><strong>{label}</strong> — {r["question_en"] or r["question_bn"]}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


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

    st.markdown(f'<div class="app-subtitle">{len(filtered)} matching entries</div>', unsafe_allow_html=True)
    for _, row in filtered.iterrows():
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="result-question">{row["question_en"] or row["question_bn"]}</div>', unsafe_allow_html=True)
        label = str(row["tier1_class"]).replace("_", " ")
        strictness = str(row.get("strictness_label", "")).strip()
        label_text = f"{label} — {strictness}" if strictness else label
        st.markdown(f'<div class="category-label">{label_text}</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="app-title app-heading">Ruling Reference</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Search Islamic rulings in Bangla, English, or Banglish.</div>',
        unsafe_allow_html=True,
    )

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
        query = st.text_input("Search", placeholder="namaj pora ki, is riba haram, বিয়ে করা কি সুন্নত...", label_visibility="collapsed")
        if query.strip():
            result = retrieve_candidates(query, df, vectorizer, X, banglish_map)
            if result.row is not None:
                render_result(result, df, vectorizer, classifier)
            else:
                render_no_match(result)
    else:
        render_browse_mode(df)

    st.markdown(
        '<div class="app-footer">Rulings shown here are for educational reference only — '
        "consult a qualified scholar for personal or complex matters.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
