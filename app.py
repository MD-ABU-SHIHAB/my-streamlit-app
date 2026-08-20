"""
app.py — Islamic Ruling Reference (CSE 469 Capstone)
Retrieval-first search over dataset.csv
(exact → TF-IDF cosine → fuzzy → honest no-match)
Logistic Regression is used only as a secondary confirmation signal.
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
@st.cache_resource(show_spinner="Preparing search index...")
def train_classifier(df: pd.DataFrame):
    df = build_combined_text(df)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=20000)
    X = vectorizer.fit_transform(df["combined_text_clean"])
    y = df["tier1_class"]

    classifier: Optional[LogisticRegression] = None
    if y.nunique() >= 2:
        classifier = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        classifier.fit(X, y)
    return vectorizer, X, classifier

# --------------------------------------------------------------------------- #
# RETRIEVAL
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

def retrieve_candidates(query: str, df: pd.DataFrame, vectorizer, X, banglish_map) -> RetrievalResult:
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
# UI — PROFESSIONAL STYLE
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600&family=Inter:wght@400;500;600&family=Noto+Sans+Bengali:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        background-color: #F7F4EF !important;
        color: #1C1C1C;
        font-family: 'Inter', 'Noto Sans Bengali', sans-serif;
    }

    .block-container {
        max-width: 780px;
        padding-top: 2.8rem;
        padding-bottom: 4rem;
    }

    #MainMenu, header[data-testid="stHeader"], footer {visibility: hidden;}

    /* Title */
    .app-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.1rem;
        font-weight: 600;
        color: #1C1C1C;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }

    .app-subtitle {
        font-size: 0.98rem;
        color: #5C5C5C;
        margin-bottom: 2.2rem;
        line-height: 1.5;
    }

    /* Search input */
    div[data-testid="stTextInput"] input {
        background-color: #FFFFFF;
        border: 1px solid #E2DDD4;
        border-radius: 8px;
        color: #1C1C1C;
        font-size: 1.05rem;
        padding: 0.85rem 1.1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #3D5A5B;
        box-shadow: 0 0 0 3px rgba(61, 90, 91, 0.12);
    }

    /* Radio buttons */
    .stRadio > div {
        gap: 1.5rem;
    }

    /* Result Card */
    .result-card {
        background: #FFFFFF;
        border: 1px solid #E8E2D9;
        border-radius: 10px;
        padding: 1.75rem 1.9rem;
        margin-top: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }

    .result-question-bn {
        font-family: 'Noto Sans Bengali', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: #1C1C1C;
        margin-bottom: 0.35rem;
        line-height: 1.45;
    }

    .result-question-en {
        font-size: 1.05rem;
        color: #444444;
        margin-bottom: 1.1rem;
        line-height: 1.45;
    }

    .category-label {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #3D5A5B;
        border-left: 3.5px solid #3D5A5B;
        padding: 0.2rem 0 0.2rem 0.7rem;
        margin-bottom: 1.1rem;
    }

    .explanation-text {
        font-size: 0.98rem;
        line-height: 1.65;
        color: #2A2A2A;
        margin-bottom: 1.3rem;
    }

    .citation-block {
        background: #F8F5F0;
        border-left: 3px solid #3D5A5B;
        padding: 0.9rem 1.1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
        color: #3A3A3A;
        margin-bottom: 0.8rem;
    }

    .verification-flag {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 500;
        color: #8B5A2B;
        background: #F8EFE0;
        border: 1px solid #E8D5B5;
        border-radius: 4px;
        padding: 0.2rem 0.55rem;
        margin-bottom: 0.9rem;
    }

    .related-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #5C5C5C;
        margin-top: 1.3rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .related-item {
        font-size: 0.92rem;
        color: #3D5A5B;
        padding: 0.35rem 0;
        border-bottom: 1px solid #F0EBE3;
    }

    /* No match */
    .no-match-box {
        background: #FFFFFF;
        border: 1px dashed #D8D0C4;
        border-radius: 10px;
        padding: 1.6rem 1.8rem;
        margin-top: 1.5rem;
        color: #4A4A4A;
    }

    /* Footer */
    .app-footer {
        margin-top: 3.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid #E2DDD4;
        color: #7A7A7A;
        font-size: 0.8rem;
        text-align: center;
        line-height: 1.5;
    }

    /* Mobile */
    @media (max-width: 480px) {
        .block-container { padding-left: 1.1rem; padding-right: 1.1rem; }
        .app-title { font-size: 1.7rem; }
        .result-card { padding: 1.3rem 1.3rem; }
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# RENDERING
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

    st.markdown(f'<div class="result-question-bn">{row["question_bn"]}</div>', unsafe_allow_html=True)
    if str(row.get("question_en", "")).strip():
        st.markdown(f'<div class="result-question-en">{row["question_en"]}</div>', unsafe_allow_html=True)

    label = str(row["tier1_class"]).replace("_", " ")
    strictness = str(row.get("strictness_label", "")).strip()
    label_text = f"{label}  ·  {strictness}" if strictness else label
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
    if ref_text == PLACEHOLDER_REFERENCE_TEXT or not ref_text:
        citation = ref_source
    else:
        citation = f"{ref_text} — {ref_source}"

    st.markdown(f'<div class="citation-block"><strong>Reference</strong><br>{citation}</div>', unsafe_allow_html=True)

    related = df[(df["topic"] == row["topic"]) & (df["id"] != row["id"])].head(TOP_K_RELATED)
    if not related.empty:
        st.markdown('<div class="related-title">Related rulings</div>', unsafe_allow_html=True)
        for _, r in related.iterrows():
            st.markdown(
                f'<div class="related-item">{r["question_en"] or r["question_bn"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Technical details"):
        stage_labels = {
            "exact_match": "Exact / substring match",
            "tfidf_cosine": "TF-IDF + cosine similarity",
            "fuzzy_match": "Fuzzy text match",
        }
        st.write(f"**Matched via:** {stage_labels.get(result.stage, result.stage)}")
        st.write(f"**Similarity score:** {result.similarity:.3f}")

        predicted = classify_query(str(row["question_en"] or row["question_bn"]), vectorizer, classifier)
        if predicted:
            agree = "agrees" if predicted == row["tier1_class"] else "disagrees"
            st.write(f"**Classifier prediction:** {predicted.replace('_', ' ')} ({agree} with retrieved row)")
        else:
            st.write("Classifier confirmation unavailable.")

def render_no_match(result: RetrievalResult) -> None:
    st.markdown('<div class="no-match-box">', unsafe_allow_html=True)
    st.markdown("**No confident match found**")
    st.markdown("The question did not closely match any verified entry. Here are the closest suggestions:")
    st.markdown("")

    if result.suggestions:
        for r in result.suggestions:
            label = str(r["tier1_class"]).replace("_", " ")
            st.markdown(
                f'<div class="related-item"><strong>{label}</strong> — {r["question_en"] or r["question_bn"]}</div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

def render_browse_mode(df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        topic_filter = st.selectbox("Topic", ["All"] + sorted(df["topic"].dropna().unique().tolist()))
    with col2:
        class_filter = st.selectbox("Category", ["All"] + sorted(df["tier1_class"].dropna().unique().tolist()))

    filtered = df.copy()
    if topic_filter != "All":
        filtered = filtered[filtered["topic"] == topic_filter]
    if class_filter != "All":
        filtered = filtered[filtered["tier1_class"] == class_filter]

    st.markdown(f"<div class='app-subtitle'>{len(filtered)} entries</div>", unsafe_allow_html=True)

    for _, row in filtered.head(40).iterrows():
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="result-question-bn">{row["question_bn"]}</div>', unsafe_allow_html=True)
        if str(row.get("question_en", "")).strip():
            st.markdown(f'<div class="result-question-en">{row["question_en"]}</div>', unsafe_allow_html=True)

        label = str(row["tier1_class"]).replace("_", " ")
        st.markdown(f'<div class="category-label">{label}</div>', unsafe_allow_html=True)

        explanation = _explanation_for(row)
        if explanation:
            st.markdown(f'<div class="explanation-text">{explanation}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="Islamic Ruling Reference",
        page_icon=None,
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    st.markdown('<div class="app-title">Islamic Ruling Reference</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Search verified rulings in Bangla, English, or Banglish</div>',
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
    except Exception as exc:
        st.error(f"The app could not start: {exc}")
        st.stop()

    mode = st.radio("Mode", ["Search", "Browse"], horizontal=True, label_visibility="collapsed")

    if mode == "Search":
        query = st.text_input(
            "Search",
            placeholder="উদাহরণ: নামাজ কি ফরজ?  /  is riba haram  /  biye kora ki sunnat",
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
        '<div class="app-footer">'
        "Educational reference only. For personal or complex matters, please consult a qualified scholar."
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
