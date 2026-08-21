"""
Islamic Mas'alah Classification System
CSE 469 — Machine Learning and Pattern Recognition

Version: 15.0 (upgraded)
- Bangla / English / Banglish search
- Keyword + alias + TF-IDF + Fuzzy pipeline
- Dataset-backed explanations & references
- Clean professional UI
- Session search history
- Safe HTML rendering
- Academic honesty disclaimer
"""

from __future__ import annotations

import html
import re
import warnings
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "Islamic Mas'alah Classification System"
APP_VERSION = "15.0"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODEL_DIR / "best_model.pkl"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"   # matches main.py save name
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
DATASET_PATH = DATA_DIR / "dataset.csv"

# Fallbacks if files are in different locations
ALT_TFIDF = MODEL_DIR / "tfidf.pkl"
ALT_DATASET = BASE_DIR / "dataset.csv"

MIN_SEMANTIC_SCORE = 0.08
MIN_FUZZY_SCORE = 0.35
MAX_RESULTS = 5
MAX_HISTORY = 10

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CATEGORY CONFIG (quiet, typographic — no rainbow pills)
# ============================================================

CATEGORY_CONFIG: Dict[str, Dict[str, str]] = {
    "Fard": {"meaning": "ফরয — পালন করা আবশ্যক", "en": "Obligatory"},
    "Wajib": {"meaning": "ওয়াজিব — পালন করা আবশ্যক", "en": "Mandatory"},
    "Sunnah Mu'akkadah": {"meaning": "সুন্নাতে মুয়াক্কাদাহ — অত্যন্ত গুরুত্বপূর্ণ সুন্নাত", "en": "Emphasized Sunnah"},
    "Mustahabb": {"meaning": "মুস্তাহাব — করা উত্তম ও সওয়াবের কাজ", "en": "Recommended"},
    "Nafl": {"meaning": "নফল — ঐচ্ছিক ইবাদত", "en": "Voluntary"},
    "Mubah": {"meaning": "মুবাহ — শরিয়তসম্মতভাবে অনুমোদিত", "en": "Permissible"},
    "Jaiz": {"meaning": "জায়েয — বৈধ ও অনুমোদিত", "en": "Permissible"},
    "Halal": {"meaning": "হালাল — শরিয়তসম্মত ও বৈধ", "en": "Lawful"},
    "Haram": {"meaning": "হারাম — নিষিদ্ধ", "en": "Forbidden"},
    "Makruh": {"meaning": "মাকরুহ — অপছন্দনীয়", "en": "Disliked"},
    "Makruh Tahrimi": {"meaning": "মাকরুহে তাহরিমি — কঠোরভাবে বর্জনীয়", "en": "Strongly Disliked"},
    "Makruh Tanzihi": {"meaning": "মাকরুহে তানযিহি — অপছন্দনীয়, বর্জন করা উত্তম", "en": "Mildly Disliked"},
    "Bid'ah": {"meaning": "বিদআত — দ্বীনের মধ্যে ভিত্তিহীন নতুন সংযোজন", "en": "Religious Innovation"},
    "Shirk": {"meaning": "শিরক — আল্লাহর সাথে শরিক করা", "en": "Polytheism"},
    "Kufr": {"meaning": "কুফর — ইসলামের মৌলিক সত্য অস্বীকার", "en": "Disbelief"},
    "Same": {"meaning": "সাধারণ তথ্য — নির্দিষ্ট ইসলামী বিধান নয়", "en": "General Information"},
}

def get_category_config(category: str) -> Dict[str, str]:
    return CATEGORY_CONFIG.get(category, {
        "meaning": "ইসলামী বিধানের শ্রেণি",
        "en": category,
    })

# ============================================================
# CSS — Quiet professional design
# ============================================================

def load_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Noto+Sans+Bengali:wght@400;500;600&family=Noto+Serif+Bengali:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Noto Sans Bengali', sans-serif;
        background-color: #F7F4ED !important;
        color: #2B2A26;
    }

    .main-title {
        font-family: 'Noto Serif Bengali', Georgia, serif;
        font-size: 1.9rem;
        color: #2B2A26;
        margin-bottom: 0.2rem;
    }
    .main-caption {
        color: #6B6558;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }

    .result-card {
        background: #FFFdf8;
        border: 1px solid #D8D0BC;
        border-left: 4px solid #3A4F63;
        padding: 1.2rem 1.4rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .category-label {
        font-size: 1.15rem;
        font-weight: 600;
        color: #3A4F63;
        margin: 0.4rem 0;
    }
    .meaning-text {
        color: #6B6558;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }
    .section-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6B6558;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
    }
    .explanation-box {
        background: #F7F4ED;
        padding: 0.8rem 1rem;
        border-radius: 3px;
        line-height: 1.55;
        margin: 0.4rem 0;
    }
    .reference-box {
        background: #F0EBE0;
        padding: 0.8rem 1rem;
        border-radius: 3px;
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0.4rem 0;
    }
    .disclaimer {
        background: #F8F0E3;
        border: 1px solid #E0D5C0;
        padding: 0.9rem 1.1rem;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #5A5348;
        margin-top: 1.5rem;
    }
    .method-badge {
        display: inline-block;
        background: #E8E2D6;
        color: #3A4F63;
        font-size: 0.8rem;
        padding: 0.2rem 0.55rem;
        border-radius: 3px;
        margin-right: 0.4rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# LOAD MODELS & DATA
# ============================================================

@st.cache_resource(show_spinner="Loading models…")
def load_models() -> Tuple[Any, Any, Any, pd.DataFrame, Optional[str]]:
    try:
        model_path = MODEL_PATH if MODEL_PATH.exists() else (MODEL_DIR / "best_model.pkl")
        tfidf_path = TFIDF_PATH if TFIDF_PATH.exists() else ALT_TFIDF
        encoder_path = ENCODER_PATH if ENCODER_PATH.exists() else (MODEL_DIR / "label_encoder.pkl")
        dataset_path = DATASET_PATH if DATASET_PATH.exists() else ALT_DATASET

        if not model_path.exists() or not tfidf_path.exists() or not encoder_path.exists():
            return None, None, None, None, (
                "Model files not found. Run main.py first to train and save models.\n"
                f"Expected: {MODEL_DIR}"
            )

        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        encoder = joblib.load(encoder_path)

        df = pd.read_csv(dataset_path, encoding="utf-8-sig")
        required = ["question_bn", "question_en", "category", "short_explanation", "reference", "source_type", "topic"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, None, None, None, f"Dataset missing columns: {', '.join(missing)}"

        for col in required:
            df[col] = df[col].fillna("").astype(str)

        return model, tfidf, encoder, df, None

    except Exception as exc:
        return None, None, None, None, str(exc)

# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s\u0980-\u09FF]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

# ============================================================
# SEARCH PIPELINE
# ============================================================

def keyword_match(query: str, df: pd.DataFrame) -> Optional[pd.Series]:
    q = normalize_text(query)
    if len(q) < 2:
        return None
    for col in ["question_bn", "question_en"]:
        mask = df[col].str.lower().str.contains(re.escape(q), na=False)
        if mask.any():
            return df.loc[mask.idxmax()]
    return None

def semantic_match(query: str, df: pd.DataFrame, tfidf, top_k: int = 5) -> List[Tuple[pd.Series, float]]:
    try:
        q_vec = tfidf.transform([normalize_text(query)])
        # Build corpus on the fly from questions
        corpus = (df["question_bn"].fillna("") + " " + df["question_en"].fillna("")).tolist()
        doc_matrix = tfidf.transform(corpus)
        sims = cosine_similarity(q_vec, doc_matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in top_idx:
            if sims[i] >= MIN_SEMANTIC_SCORE:
                results.append((df.iloc[i], float(sims[i])))
        return results
    except Exception:
        return []

def fuzzy_match(query: str, df: pd.DataFrame, top_k: int = 3) -> List[Tuple[pd.Series, float]]:
    q = normalize_text(query)
    scores = []
    for idx, row in df.iterrows():
        text = normalize_text(row["question_bn"] + " " + row["question_en"])
        ratio = SequenceMatcher(None, q, text).ratio()
        if ratio >= MIN_FUZZY_SCORE:
            scores.append((row, ratio))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def google_style_search(
    query: str,
    df: pd.DataFrame,
    model,
    tfidf,
    encoder,
) -> Dict[str, Any]:
    """
    Retrieval-first pipeline:
    1. Exact / keyword
    2. Semantic (TF-IDF)
    3. Fuzzy
    4. ML classification as secondary signal
    """
    query = query.strip()
    if not query:
        return {"status": "empty"}

    # Stage 1 — Keyword
    exact = keyword_match(query, df)
    if exact is not None:
        return _build_result(exact, query, "keyword", 0.95, model, tfidf, encoder)

    # Stage 2 — Semantic
    semantic = semantic_match(query, df, tfidf)
    if semantic:
        best_row, score = semantic[0]
        return _build_result(best_row, query, "semantic", score, model, tfidf, encoder, suggestions=semantic[1:])

    # Stage 3 — Fuzzy
    fuzzy = fuzzy_match(query, df)
    if fuzzy:
        best_row, score = fuzzy[0]
        return _build_result(best_row, query, "fuzzy", score, model, tfidf, encoder, suggestions=fuzzy[1:])

    # No confident match
    return {
        "status": "no_match",
        "original_query": query,
        "suggestions": [r for r, _ in semantic[:3] + fuzzy[:2]],
    }

def _build_result(
    row: pd.Series,
    query: str,
    method: str,
    score: float,
    model,
    tfidf,
    encoder,
    suggestions: Optional[List] = None,
) -> Dict[str, Any]:
    category = str(row.get("category", "N/A"))
    ml_category = None
    try:
        q_vec = tfidf.transform([normalize_text(query)])
        pred = model.predict(q_vec)[0]
        ml_category = encoder.inverse_transform([pred])[0]
    except Exception:
        pass

    return {
        "status": "match",
        "original_query": query,
        "matched_question": str(row.get("question_bn", "")),
        "matched_question_en": str(row.get("question_en", "")),
        "category": category,
        "explanation": str(row.get("short_explanation", "")),
        "reference": str(row.get("reference", "")),
        "source_type": str(row.get("source_type", "")),
        "topic": str(row.get("topic", "")),
        "method": method,
        "confidence": min(100.0, max(0.0, score * 100)),
        "ml_category": ml_category,
        "suggestions": suggestions or [],
    }

# ============================================================
# UI HELPERS
# ============================================================

def method_label(method: str) -> str:
    return {
        "keyword": "Exact / Keyword Match",
        "semantic": "TF-IDF Semantic Similarity",
        "fuzzy": "Approximate (Fuzzy) Match",
    }.get(method, method)

def display_result(result: Dict[str, Any]) -> None:
    if result.get("status") == "empty":
        st.warning("দয়া করে একটি প্রশ্ন লিখুন।")
        return

    if result.get("status") == "no_match":
        st.info("নির্ভরযোগ্য কোনো মিল পাওয়া যায়নি। নিচের কাছাকাছি প্রশ্নগুলো দেখুন।")
        suggestions = result.get("suggestions", [])
        if suggestions:
            st.markdown("**কাছাকাছি প্রশ্ন**")
            for i, s in enumerate(suggestions[:5], 1):
                q = s["question_bn"] if isinstance(s, pd.Series) else str(s)
                if st.button(f"{i}. {q[:80]}", key=f"sug_{i}"):
                    st.session_state["search_query"] = q
                    st.rerun()
        return

    category = result.get("category", "N/A")
    config = get_category_config(category)
    method = method_label(result.get("method", ""))
    confidence = float(result.get("confidence", 0))

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.markdown(f'<span class="method-badge">{method}</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="category-label">{html.escape(category)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="meaning-text">{config["meaning"]} · {config["en"]}</div>', unsafe_allow_html=True)

    # Confidence
    st.progress(confidence / 100)
    st.caption(f"Match confidence: {confidence:.1f}%")

    # Matched question
    st.markdown('<div class="section-title">ডেটাসেটে মিল পাওয়া প্রশ্ন</div>', unsafe_allow_html=True)
    st.markdown(f"**{html.escape(result.get('matched_question', ''))}**")
    if result.get("matched_question_en"):
        st.caption(result["matched_question_en"])

    # Topic
    if result.get("topic"):
        st.markdown(f"**Topic:** {html.escape(result['topic'])}")

    # Explanation
    if result.get("explanation"):
        st.markdown('<div class="section-title">সংক্ষিপ্ত ব্যাখ্যা</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="explanation-box">{html.escape(result["explanation"])}</div>', unsafe_allow_html=True)

    # Reference
    if result.get("reference") and result["reference"] != "N/A":
        st.markdown('<div class="section-title">রেফারেন্স</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="reference-box">{html.escape(result["reference"])}<br>'
            f'<small>Source: {html.escape(result.get("source_type", ""))}</small></div>',
            unsafe_allow_html=True,
        )

    # ML secondary signal
    ml_cat = result.get("ml_category")
    if ml_cat:
        with st.expander("Machine Learning secondary prediction"):
            st.write(f"ML predicted category: **{ml_cat}**")
            if ml_cat != category:
                st.caption("ML prediction differs from the retrieved dataset record. "
                           "The dataset-backed result is shown as the primary answer.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown(
        '<div class="disclaimer">'
        "<strong>গুরুত্বপূর্ণ সতর্কতা</strong><br>"
        "এই সিস্টেমটি CSE 469 Machine Learning প্রজেক্টের শিক্ষামূলক উদ্দেশ্যে তৈরি। "
        "এটি কোনো ব্যক্তিগত ফতোয়া বা যোগ্য আলেমের বিকল্প নয়। "
        "জটিল বা ব্যক্তিগত মাসআলাহর ক্ষেত্রে নির্ভরযোগ্য আলেম/মুফতির পরামর্শ নিন।"
        "</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# SIDEBAR
# ============================================================

def display_sidebar(df: pd.DataFrame) -> None:
    st.sidebar.markdown("### Islamic Mas'alah")
    st.sidebar.caption(f"v{APP_VERSION} · CSE 469")

    st.sidebar.divider()
    st.sidebar.metric("Total records", len(df))
    st.sidebar.metric("Categories", df["category"].nunique())
    st.sidebar.metric("Topics", df["topic"].nunique())

    st.sidebar.divider()
    st.sidebar.markdown("**Category distribution**")
    counts = df["category"].value_counts().head(12)
    for cat, cnt in counts.items():
        pct = cnt / len(df) * 100
        st.sidebar.write(f"{cat}: {cnt} ({pct:.1f}%)")

    st.sidebar.divider()
    st.sidebar.markdown("**Search pipeline**")
    st.sidebar.caption("1. Keyword → 2. TF-IDF → 3. Fuzzy → 4. Honest no-match")

# ============================================================
# HISTORY
# ============================================================

def add_to_history(query: str, category: str) -> None:
    if "search_history" not in st.session_state:
        st.session_state["search_history"] = []
    hist = st.session_state["search_history"]
    hist.insert(0, {"query": query, "category": category})
    st.session_state["search_history"] = hist[:MAX_HISTORY]

def display_history() -> None:
    hist = st.session_state.get("search_history", [])
    if not hist:
        return
    with st.expander("সাম্প্রতিক অনুসন্ধান (Session only)"):
        for i, item in enumerate(hist):
            cols = st.columns([4, 1])
            with cols[0]:
                if st.button(f"{item['query'][:50]} → {item['category']}", key=f"hist_{i}"):
                    st.session_state["search_query"] = item["query"]
                    st.rerun()
        if st.button("Clear history"):
            st.session_state["search_history"] = []
            st.rerun()

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    load_css()

    if "search_query" not in st.session_state:
        st.session_state["search_query"] = ""

    model, tfidf, encoder, df, error = load_models()

    if error:
        st.error("System could not load.")
        st.code(error)
        st.info(
            "Project structure expected:\n\n"
            "project/\n"
            "├── app.py\n"
            "├── main.py\n"
            "├── data/\n"
            "│   └── dataset.csv\n"
            "└── models/\n"
            "    ├── best_model.pkl\n"
            "    ├── tfidf_vectorizer.pkl\n"
            "    └── label_encoder.pkl\n\n"
            "Run `python main.py` first to train and save the models."
        )
        return

    # Header
    st.markdown(f'<div class="main-title">Islamic Mas\'alah Classification System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-caption">বাংলা · English · Banglish — Dataset-backed educational search</div>',
        unsafe_allow_html=True,
    )

    display_sidebar(df)

    # Search
    st.subheader("আপনার মাসআলাহ অনুসন্ধান করুন")
    st.caption("যেকোনো শব্দ, বাক্য বা প্রশ্ন লিখুন (বাংলা / English / Banglish)")

    user_input = st.text_input(
        "Search",
        value=st.session_state["search_query"],
        placeholder="উদাহরণ: নামাজ পড়া কি? · is riba haram · namaj pora ki",
        label_visibility="collapsed",
        key="main_input",
    )

    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        search_clicked = st.button("Search", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state["search_query"] = ""
            st.rerun()

    if search_clicked and user_input.strip():
        st.session_state["search_query"] = user_input.strip()
        with st.spinner("Searching…"):
            result = google_style_search(user_input.strip(), df, model, tfidf, encoder)
        add_to_history(user_input.strip(), result.get("category", "N/A"))
        display_result(result)

    display_history()

    # Footer
    st.divider()
    st.caption(
        f"CSE 469 · Machine Learning & Pattern Recognition · v{APP_VERSION}\n\n"
        "This is an educational classification & retrieval system. "
        "It is not a substitute for a qualified scholar."
    )

if __name__ == "__main__":
    main()
