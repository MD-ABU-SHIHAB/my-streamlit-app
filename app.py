"""
app.py — Islamic Ruling Reference (CSE 469 Capstone)
Bilingual UI (English + Bangla)
Retrieval-first search over dataset.csv
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
        raise DatasetError(f"'{path}' ফাইল পাওয়া যায়নি। app.py এর সাথে একই ফোল্ডারে dataset.csv রাখুন।")
    except Exception as exc:
        raise DatasetError(f"'{path}' পড়া যায়নি: {exc}")

    if df.empty:
        raise DatasetError(f"'{path}' ফাইলে কোনো ডেটা নেই।")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetError("dataset.csv এ প্রয়োজনীয় কলাম নেই: " + ", ".join(missing_cols))

    df = df.dropna(subset=["question_bn", "tier1_class"]).reset_index(drop=True)
    if df.empty:
        raise DatasetError("সব সারিতে question_bn অথবা tier1_class অনুপস্থিত।")
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
@st.cache_resource(show_spinner="সার্চ ইনডেক্স প্রস্তুত হচ্ছে... / Preparing search index...")
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

def _tfidf_match(query_clean: str, df: pd.DataFrame, vectorizer, X):
    if not query_clean.strip():
        return None, 0.0
    q_vec = vectorizer.transform([query_clean])
    sims = cosine_similarity(q_vec, X).flatten()
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score >= TFIDF_SIMILARITY_THRESHOLD:
        return df.iloc[best_idx], best_score
    return None, best_score

def _fuzzy_match(query_clean: str, df: pd.DataFrame):
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
# UI STYLE - High readability + Bilingual
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;500;600;700&family=Noto+Serif+Bengali:wght@500;600&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        background-color: #F7F4EF !important;
        color: #1C1C1C;
        font-family: 'Noto Sans Bengali', 'Inter', sans-serif;
    }

    .block-container {
        max-width: 820px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    #MainMenu, header[data-testid="stHeader"], footer {visibility: hidden;}

    /* Title */
    .app-title {
        font-family: 'Noto Serif Bengali', Georgia, serif;
        font-size: 1.9rem;
        font-weight: 600;
        color: #1C1C1C;
        margin-bottom: 0.3rem;
        line-height: 1.35;
    }

    .app-subtitle {
        font-size: 0.98rem;
        color: #4A4A4A;
        margin-bottom: 2rem;
        line-height: 1.55;
    }

    /* Search box */
    div[data-testid="stTextInput"] input {
        background-color: #FFFFFF !important;
        border: 1px solid #D9D2C5 !important;
        border-radius: 8px !important;
        color: #1C1C1C !important;
        font-size: 1.05rem !important;
        padding: 0.85rem 1.1rem !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #3D5A5B !important;
        box-shadow: 0 0 0 3px rgba(61, 90, 91, 0.15) !important;
    }

    /* Result Card */
    .result-card {
        background: #FFFFFF;
        border: 1px solid #E5DFD3;
        border-radius: 10px;
        padding: 1.7rem 1.8rem;
        margin-top: 1.4rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }

    .result-question-bn {
        font-family: 'Noto Sans Bengali', sans-serif;
        font-size: 1.28rem;
        font-weight: 600;
        color: #1C1C1C;
        margin-bottom: 0.35rem;
        line-height: 1.45;
    }

    .result-question-en {
        font-size: 1.05rem;
        color: #333333;
        margin-bottom: 1.1rem;
        line-height: 1.45;
    }

    .category-label {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #3D5A5B;
        border-left: 3.5px solid #3D5A5B;
        padding: 0.2rem 0 0.2rem 0.7rem;
        margin-bottom: 1rem;
    }

    .explanation-text {
        font-size: 0.98rem;
        line-height: 1.7;
        color: #222222;
        margin-bottom: 1.2rem;
    }

    .citation-block {
        background: #F3EFE7;
        border-left: 3px solid #3D5A5B;
        padding: 0.9rem 1.15rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.92rem;
        color: #333333;
        margin-bottom: 0.8rem;
        line-height: 1.55;
    }

    .verification-flag {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 500;
        color: #7A4F1D;
        background: #F8EFDF;
        border: 1px solid #E5D3B0;
        border-radius: 4px;
        padding: 0.22rem 0.6rem;
        margin-bottom: 0.9rem;
    }

    .related-title {
        font-size: 0.84rem;
        font-weight: 600;
        color: #555555;
        margin-top: 1.3rem;
        margin-bottom: 0.45rem;
        letter-spacing: 0.04em;
    }

    .related-item {
        font-size: 0.93rem;
        color: #3D5A5B;
        padding: 0.35rem 0;
        border-bottom: 1px solid #EFEBE3;
    }

    .no-match-box {
        background: #FFFFFF;
        border: 1px dashed #D0C8B8;
        border-radius: 10px;
        padding: 1.6rem 1.7rem;
        margin-top: 1.4rem;
        color: #333333;
        line-height: 1.6;
    }

    .app-footer {
        margin-top: 3.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid #E0D9CC;
        color: #666666;
        font-size: 0.82rem;
        text-align: center;
        line-height: 1.6;
    }

    @media (max-width: 480px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .app-title { font-size: 1.55rem; }
        .result-card { padding: 1.3rem; }
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
    return "<br><br>".join(parts)

def render_result(result: RetrievalResult, df: pd.DataFrame, vectorizer, classifier) -> None:
    row = result.row

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    # Question
    st.markdown(f'<div class="result-question-bn">{row["question_bn"]}</div>', unsafe_allow_html=True)
    if str(row.get("question_en", "")).strip():
        st.markdown(f'<div class="result-question-en">{row["question_en"]}</div>', unsafe_allow_html=True)

    # Category
    label = str(row["tier1_class"]).replace("_", " ")
    strictness = str(row.get("strictness_label", "")).strip()
    label_text = f"{label}  ·  {strictness}" if strictness else label
    st.markdown(f'<div class="category-label">{label_text}</div>', unsafe_allow_html=True)

    # Verification warning
    if str(row.get("verification_status", "")).strip() == NEEDS_VERIFICATION_LABEL:
        st.markdown(
            '<div class="verification-flag">'
            '⚠ Unverified reference — pending scholarly check<br>'
            '<span style="font-size:0.75rem;">অযাচাইকৃত রেফারেন্স — স্কলারলি যাচাইয়ের অপেক্ষায়</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Explanation
    explanation = _explanation_for(row)
    if explanation:
        st.markdown(f'<div class="explanation-text">{explanation}</div>', unsafe_allow_html=True)

    # Citation
    ref_text = str(row.get("reference_text", "")).strip()
    ref_source = str(row.get("reference_source", "")).strip()
    if ref_text == PLACEHOLDER_REFERENCE_TEXT or not ref_text:
        citation = ref_source
    else:
        citation = f"{ref_text}<br>{ref_source}"

    st.markdown(
        f'<div class="citation-block">'
        f'<strong>Reference / রেফারেন্স</strong><br>{citation}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Related
    related = df[(df["topic"] == row["topic"]) & (df["id"] != row["id"])].head(TOP_K_RELATED)
    if not related.empty:
        st.markdown('<div class="related-title">Related Rulings / সম্পর্কিত মাসআলা</div>', unsafe_allow_html=True)
        for _, r in related.iterrows():
            st.markdown(
                f'<div class="related-item">{r["question_bn"]}<br>'
                f'<span style="color:#555; font-size:0.88rem;">{r["question_en"]}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # Technical details
    with st.expander("Technical Details / প্রযুক্তিগত বিবরণ"):
        stage_labels = {
            "exact_match": "Exact / Substring Match",
            "tfidf_cosine": "TF-IDF + Cosine Similarity",
            "fuzzy_match": "Fuzzy Text Match",
        }
        st.write(f"**Matched via / ম্যাচ হয়েছে:** {stage_labels.get(result.stage, result.stage)}")
        st.write(f"**Similarity Score:** {result.similarity:.3f}")

        predicted = classify_query(str(row["question_en"] or row["question_bn"]), vectorizer, classifier)
        if predicted:
            agree = "agrees" if predicted == row["tier1_class"] else "disagrees"
            st.write(f"**Classifier Prediction:** {predicted.replace('_', ' ')} ({agree})")
        else:
            st.write("Classifier confirmation unavailable.")

def render_no_match(result: RetrievalResult) -> None:
    st.markdown('<div class="no-match-box">', unsafe_allow_html=True)
    st.markdown(
        "<strong>No confident match found</strong><br>"
        "<span style='color:#444;'>কোনো নিশ্চিত ম্যাচ পাওয়া যায়নি</span>"
    )
    st.markdown(
        "<br>The question did not closely match any verified entry.<br>"
        "আপনার প্রশ্নটি কোনো যাচাইকৃত এন্ট্রির সাথে ভালোভাবে মিলেনি।"
    )

    if result.suggestions:
        st.markdown("<br><strong>Closest suggestions / নিকটতম সাজেশন:</strong>")
        for r in result.suggestions:
            label = str(r["tier1_class"]).replace("_", " ")
            st.markdown(
                f'<div class="related-item">'
                f'<strong>{label}</strong><br>'
                f'{r["question_bn"]}<br>'
                f'<span style="color:#555; font-size:0.9rem;">{r["question_en"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

def render_browse_mode(df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        topic_filter = st.selectbox(
            "Topic / বিষয়",
            ["All / সব"] + sorted(df["topic"].dropna().unique().tolist())
        )
    with col2:
        class_filter = st.selectbox(
            "Category / শ্রেণী",
            ["All / সব"] + sorted(df["tier1_class"].dropna().unique().tolist())
        )

    filtered = df.copy()
    if not topic_filter.startswith("All"):
        filtered = filtered[filtered["topic"] == topic_filter]
    if not class_filter.startswith("All"):
        filtered = filtered[filtered["tier1_class"] == class_filter]

    st.markdown(
        f"<div class='app-subtitle'>{len(filtered)} entries found / {len(filtered)} টি এন্ট্রি পাওয়া গেছে</div>",
        unsafe_allow_html=True,
    )

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
        page_title="Islamic Ruling Reference | ইসলামী মাসআলা রেফারেন্স",
        page_icon=None,
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    # Header - Bilingual
    st.markdown(
        '<div class="app-title">'
        'Islamic Ruling Reference<br>'
        '<span style="font-size:1.45rem;">ইসলামী মাসআলা রেফারেন্স</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="app-subtitle">'
        'Search verified rulings in Bangla, English or Banglish<br>'
        '<span style="font-size:0.92rem;">বাংলা, ইংরেজি বা বাংলিশে যাচাইকৃত মাসআলা খুঁজুন</span>'
        '</div>',
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
        st.error(f"অ্যাপ শুরু করা যায়নি / App could not start: {exc}")
        st.stop()

    mode = st.radio(
        "Mode",
        ["Search / খুঁজুন", "Browse / ব্রাউজ করুন"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode.startswith("Search"):
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

    # Footer
    st.markdown(
        '<div class="app-footer">'
        'Educational reference only. For personal or complex matters, consult a qualified scholar.<br>'
        'শুধুমাত্র শিক্ষামূলক রেফারেন্স। ব্যক্তিগত বা জটিল বিষয়ে যোগ্য আলেমের সাথে পরামর্শ করুন।'
        '</div>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
