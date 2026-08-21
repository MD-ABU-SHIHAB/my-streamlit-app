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
    matplotlib
"""

from __future__ import annotations

import difflib
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

# IMPORTANT: Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
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
    "exact_match": ("Exact match", "সরাসরি মিল"),
    "tfidf_cosine": ("TF-IDF · cosine similarity", "শব্দ-সাদৃশ্য বিশ্লেষণ"),
    "fuzzy_match": ("Approximate match", "আনুমানিক মিল"),
}

# Most users of this app read Bangla first — every English UI string ships
# with a Bangla counterpart. These two maps translate the fixed category
# vocabulary (which the dataset stores in English, since it's the ML
# classification target) for display only; they never touch the underlying
# tier1_class/topic values used for matching or classification.
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

# Theme colors for matplotlib (matches CSS palette)
COLOR_PAPER = "#F2ECDD"
COLOR_CARD = "#FBF8EF"
COLOR_INK = "#2B2A24"
COLOR_INK_MUTED = "#544F42"
COLOR_ACCENT = "#2F4858"
COLOR_ACCENT_SOFT = "#E4DEC9"
COLOR_HAIRLINE = "#DDD3B8"
COLOR_FLAG_BG = "#F3E6C8"
COLOR_FLAG_INK = "#6B4E1F"


def bilingual(en: str, bn: str) -> str:
    """Compact inline pairing for short labels: 'English · বাংলা'."""
    return f"{en} <span class='bn-inline'>· {bn}</span>"


def bilingual_plain(en: str, bn: str) -> str:
    """Plain-text pairing for widget labels that don't render HTML (e.g. st.expander)."""
    return f"{en} · {bn}"


def bilingual_block(en: str, bn: str) -> str:
    """Stacked pairing for full sentences — English line, Bangla line below,
    each fully legible on its own rather than interleaved mid-sentence."""
    return f'<span class="en-line">{en}</span><span class="bn-line">{bn}</span>'


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
# ML INSIGHTS — functions for the Model Insights tab
# These are deliberately "dashboard-like" and more information-dense than the
# Search tab — this is a teaching/transparency tab, not a minimalist reference
# interface. The distinction is intentional, not a contradiction.
# --------------------------------------------------------------------------- #

def _cv_k(y: pd.Series, target: int = 5) -> int:
    return min(target, int(y.value_counts().min()))


def _style_matplotlib_fig(fig, ax):
    """Apply the app's color palette to a matplotlib figure."""
    fig.patch.set_facecolor(COLOR_PAPER)
    ax.set_facecolor(COLOR_PAPER)
    for spine in ax.spines.values():
        spine.set_color(COLOR_HAIRLINE)
    ax.tick_params(colors=COLOR_INK)
    ax.xaxis.label.set_color(COLOR_INK)
    ax.yaxis.label.set_color(COLOR_INK)
    ax.title.set_color(COLOR_INK)
    if ax.legend_ is not None:
        for text in ax.legend_.get_texts():
            text.set_color(COLOR_INK)
        ax.legend_.get_frame().set_facecolor(COLOR_PAPER)
        ax.legend_.get_frame().set_edgecolor(COLOR_HAIRLINE)
    return fig, ax


def _evaluate_classifier(model, X, y: pd.Series, cv_target: int = 5) -> dict:
    """Fit on full data + stratified CV where the smallest class allows it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    y_pred = model.predict(X)
    train_acc = accuracy_score(y, y_pred)
    train_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    train_precision = precision_score(y, y_pred, average="macro", zero_division=0)
    train_recall = recall_score(y, y_pred, average="macro", zero_division=0)

    k = _cv_k(y, cv_target)
    cv_possible = k >= 2
    cv_mean, cv_std = float("nan"), float("nan")
    cv_precision_mean, cv_recall_mean = float("nan"), float("nan")
    if cv_possible:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                f1_scores = cross_val_score(model, X, y, cv=skf, scoring="f1_macro")
                cv_mean, cv_std = float(f1_scores.mean()), float(f1_scores.std())
                precision_scores = cross_val_score(model, X, y, cv=skf, scoring="precision_macro")
                cv_precision_mean = float(precision_scores.mean())
                recall_scores = cross_val_score(model, X, y, cv=skf, scoring="recall_macro")
                cv_recall_mean = float(recall_scores.mean())
            except Exception:
                cv_possible = False

    cm = confusion_matrix(y, y_pred, labels=model.classes_)

    return {
        "model": model,
        "classes": model.classes_,
        "train_accuracy": train_acc,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_macro_f1": train_f1,
        "cv_k": k,
        "cv_possible": cv_possible,
        "cv_mean_macro_f1": cv_mean,
        "cv_std_macro_f1": cv_std,
        "cv_mean_precision": cv_precision_mean,
        "cv_mean_recall": cv_recall_mean,
        "confusion_matrix": cm,
    }


def _get_five_classifiers(knn_k: int):
    """Returns the 5 core classifiers used in Model Insights tab."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
        "KNN": KNeighborsClassifier(n_neighbors=knn_k),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }


@st.cache_resource(show_spinner="Loading model insights...")
def _get_insights_data(df: pd.DataFrame):
    """Cache all Model Insights computations."""
    df = build_combined_text(df)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=20000)
    X = vectorizer.fit_transform(df["combined_text_clean"])
    y = df["tier1_class"]
    knn_k = max(1, min(5, len(y) - 1))
    
    # Train the 5 core classifiers
    classifiers = _get_five_classifiers(knn_k)
    results = {}
    for name, model in classifiers.items():
        try:
            results[name] = _evaluate_classifier(model, X, y)
        except Exception as exc:
            results[name] = {"error": str(exc)}
    
    # K-Means
    k = df["topic"].nunique()
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    ari = adjusted_rand_score(df["topic"], cluster_labels)
    crosstab = pd.crosstab(pd.Series(cluster_labels, name="cluster"), df["topic"].reset_index(drop=True))
    
    # PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X.toarray())
    explained = float(pca.explained_variance_ratio_.sum() * 100)
    
    return {
        "X": X,
        "y": y,
        "vectorizer": vectorizer,
        "results": results,
        "kmeans": {
            "k": k,
            "ari": ari,
            "crosstab": crosstab,
            "coords": coords,
            "explained": explained,
            "cluster_labels": cluster_labels,
        },
        "knn_k": knn_k,
        "tier1_class": df["tier1_class"],
        "topic": df["topic"],
        "df": df,
    }


def render_model_insights_tab(df: pd.DataFrame, vectorizer_search, X_search, banglish_map, classifier_search) -> None:
    """Model Insights tab — demonstrates every ML/PR syllabus concept hands-on.
    
    This tab is deliberately more information-dense and "dashboard-like" than the
    Search tab, because its explicit purpose is transparency and teaching — not
    the minimalist reference interface of the Search tab.
    """
    st.markdown("#### মডেল বিশ্লেষণ · Model Insights")
    st.caption(
        "This tab exists to demonstrate every ML/PR syllabus concept hands-on, with real numbers "
        "computed from the real dataset — not static screenshots. এটি সরাসরি ডেটাসেট থেকে "
        "গণনা করা বাস্তব সংখ্যা সহ হ্যান্ডস-অন প্রদর্শনের জন্য তৈরি।"
    )
    
    insights = _get_insights_data(df)
    results = insights["results"]
    X = insights["X"]
    y = insights["y"]
    vectorizer = insights["vectorizer"]
    kmeans_data = insights["kmeans"]
    
    # ---------------------------------------------------------------------- #
    # SECTION 1: PIPELINE OVERVIEW
    # ---------------------------------------------------------------------- #
    st.markdown("---")
    st.markdown("#### ১. রিট্রিভাল পাইপলাইন · Pipeline Overview")
    st.caption(
        "How a query flows through the system: retrieval (primary) → classification (secondary confirmation). "
        "কীভাবে একটি প্রশ্ন সিস্টেমের মাধ্যমে প্রবাহিত হয়: রিট্রিভাল (প্রাথমিক) → ক্লাসিফিকেশন (গৌণ নিশ্চিতকরণ)।"
    )
    
    # Visual pipeline using columns
    pipeline_stages = [
        ("🔍", "Query\nপ্রশ্ন", COLOR_CARD),
        ("➡️", "Exact Match\nসঠিক মিল", COLOR_ACCENT_SOFT),
        ("➡️", "TF-IDF + Cosine\nটেমপ্লেট মিল", COLOR_HAIRLINE),
        ("➡️", "Fuzzy Match\nঅস্পষ্ট মিল", COLOR_FLAG_BG),
        ("➡️", "Logistic Regression\nনিশ্চিতকরণ", "#D4D4D4"),
        ("✅", "Answer\nউত্তর", COLOR_ACCENT_SOFT),
    ]
    
    cols = st.columns(len(pipeline_stages))
    for i, (icon, label, bg) in enumerate(pipeline_stages):
        with cols[i]:
            is_retrieval = i < 4
            is_classification = i == 4
            border_color = COLOR_ACCENT if is_retrieval else COLOR_FLAG_INK if is_classification else COLOR_HAIRLINE
            st.markdown(
                f"""
                <div style="
                    background-color: {bg};
                    border: 2px solid {border_color};
                    border-radius: 8px;
                    padding: 0.6rem 0.3rem;
                    text-align: center;
                    font-size: 0.75rem;
                    min-height: 85px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                    <div style="font-size: 1.4rem;">{icon}</div>
                    <div style="font-weight: 600; margin-top: 0.2rem; white-space: pre-line;">{label}</div>
                    {'' if not is_retrieval else '<span style="font-size:0.6rem;color:#2F4858;">(retrieval)</span>'}
                    {'' if not is_classification else '<span style="font-size:0.6rem;color:#6B4E1F;">(confirmation)</span>'}
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    st.caption(
        "**Retrieval** (exact → TF-IDF → fuzzy) finds the best match. **Classification** (Logistic Regression) "
        "is a secondary confirmation signal — it never overrides the retrieved answer. "
        "**রিট্রিভাল** (সঠিক → টিএফ-আইডিএফ → অস্পষ্ট) সবচেয়ে ভালো মিল খুঁজে বের করে। **ক্লাসিফিকেশন** "
        "(লজিস্টিক রিগ্রেশন) একটি গৌণ নিশ্চিতকরণ সংকেত — এটি কখনই রিট্রিভাল করা উত্তরকে ওভাররাইড করে না।"
    )
    
    # ---------------------------------------------------------------------- #
    # SECTION 2: CLASSIFIER COMPARISON
    # ---------------------------------------------------------------------- #
    st.markdown("---")
    st.markdown("#### ২. ক্লাসিফায়ার তুলনা · Classifier Comparison")
    st.caption(
        "5 core classifiers trained on the same data. একই ডেটাসেটে প্রশিক্ষিত ৫টি মূল ক্লাসিফায়ার।"
    )
    
    valid_results = {n: r for n, r in results.items() if "error" not in r}
    
    if valid_results:
        # Build comparison table
        table_rows = []
        for name, r in valid_results.items():
            cv_str = f"{r['cv_mean_macro_f1']:.3f} ± {r['cv_std_macro_f1']:.3f}" if r["cv_possible"] else "N/A"
            table_rows.append({
                "Model": name,
                "Accuracy": f"{r['train_accuracy']:.3f}",
                "Precision": f"{r['train_precision']:.3f}",
                "Recall": f"{r['train_recall']:.3f}",
                "F1 (Train)": f"{r['train_macro_f1']:.3f}",
                "F1 (CV)": cv_str,
            })
        
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        
        # Bar chart ranking by F1
        st.markdown("##### F1 স্কোর দ্বারা র্যাঙ্কিং · Ranking by F1 Score")
        
        use_cv = any(r["cv_possible"] for r in valid_results.values())
        rank_key = "cv_mean_macro_f1" if use_cv else "train_macro_f1"
        rank_label = "CV Macro-F1" if use_cv else "Train Macro-F1"
        
        names = list(valid_results.keys())
        scores = [valid_results[n][rank_key] if not np.isnan(valid_results[n][rank_key]) else 0 for n in names]
        
        sorted_pairs = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
        names_sorted, scores_sorted = zip(*sorted_pairs)
        
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(names_sorted, scores_sorted, color=COLOR_ACCENT, edgecolor=COLOR_HAIRLINE)
        ax.set_ylabel(rank_label)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{rank_label} Ranking", fontsize=12)
        plt.xticks(rotation=25, ha="right", fontsize=8)
        
        for bar, s in zip(bars, scores_sorted):
            ax.text(bar.get_x() + bar.get_width()/2, s + 0.02, f"{s:.2f}", ha="center", fontsize=8, color=COLOR_INK)
        
        fig, ax = _style_matplotlib_fig(fig, ax)
        st.pyplot(fig)
        plt.close(fig)
        
        # Confusion matrix picker
        st.markdown("##### কনফিউশন ম্যাট্রিক্স · Confusion Matrix")
        model_names = list(valid_results.keys())
        selected_model = st.selectbox(
            bilingual_plain("Select a model to view its confusion matrix", "কনফিউশন ম্যাট্রিক্স দেখতে একটি মডেল নির্বাচন করুন"),
            model_names,
            key="cm_picker"
        )
        
        if selected_model in valid_results:
            r = valid_results[selected_model]
            cm = r["confusion_matrix"]
            classes = r["classes"]
            
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels([c.replace("_", " ") for c in classes], rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels([c.replace("_", " ") for c in classes], fontsize=7)
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("True", fontsize=9)
            ax.set_title(f"{selected_model} — Confusion Matrix", fontsize=11)
            
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white", fontsize=8)
            
            fig, ax = _style_matplotlib_fig(fig, ax)
            st.pyplot(fig)
            plt.close(fig)
        
        # Why Logistic Regression
        st.markdown("##### কেন লজিস্টিক রিগ্রেশন? · Why Logistic Regression?")
        st.caption(
            "**Logistic Regression** runs the live Search tab. It is not necessarily the single highest "
            "score in the chart above — the pick is justified by cross-validation stability (low variance "
            "across folds), not the top raw number from one lucky split. Logistic Regression and Naive Bayes "
            "are the classic low-variance choices on small, sparse TF-IDF data; higher-capacity models "
            "(Decision Tree, Random Forest) tend to need substantially more training rows before their extra "
            "flexibility pays off rather than overfits.\n\n"
            "**লজিস্টিক রিগ্রেশন** লাইভ সার্চ ট্যাব চালায়। এটি উপরের চার্টে সর্বোচ্চ স্কোর নয় — বরং "
            "ক্রস-ভ্যালিডেশন স্থিতিশীলতা (কম ভ্যারিয়েন্স) এর জন্য নির্বাচিত, একক স্প্লিট থেকে প্রাপ্ত "
            "সর্বোচ্চ সংখ্যা নয়। লজিস্টিক রিগ্রেশন এবং নাইভ বেইজ হল ছোট, স্পার্স টিএফ-আইডিএফ ডেটার "
            "জন্য ক্লাসিক কম-ভ্যারিয়েন্স পছন্দ; উচ্চ-ক্ষমতার মডেলগুলির (ডিসিশন ট্রি, র্যান্ডম ফরেস্ট) "
            "জন্য যথেষ্ট বেশি প্রশিক্ষণ ডেটা প্রয়োজন হয় ওভারফিটিং এড়ানোর জন্য।"
        )
    
    # ---------------------------------------------------------------------- #
    # SECTION 3: K-MEANS CLUSTERING
    # ---------------------------------------------------------------------- #
    st.markdown("---")
    st.markdown("#### ৩. কে-মিনস ক্লাস্টারিং · K-Means Clustering")
    st.caption(
        "Unsupervised learning: the algorithm groups similar texts on its own. "
        "আনসুপারভাইসড লার্নিং: অ্যালগরিদম নিজেই অনুরূপ টেক্সট গোষ্ঠীভুক্ত করে।"
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Plot 1: Colored by K-Means cluster
    palette = plt.get_cmap("tab10")
    coords = kmeans_data["coords"]
    cluster_labels = kmeans_data["cluster_labels"]
    unique_clusters = sorted(set(cluster_labels))
    for i, c in enumerate(unique_clusters):
        mask = cluster_labels == c
        ax1.scatter(coords[mask, 0], coords[mask, 1], s=35, alpha=0.8,
                   color=palette(i % 10), label=f"Cluster {c}", edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
    ax1.set_title("K-Means Clusters", fontsize=10)
    ax1.legend(fontsize=6, loc="best")
    
    # Plot 2: Colored by true topic
    topics = insights["topic"]
    unique_topics = sorted(topics.unique())
    for i, t in enumerate(unique_topics):
        mask = (topics == t).to_numpy()
        ax2.scatter(coords[mask, 0], coords[mask, 1], s=35, alpha=0.8,
                   color=palette(i % 10), label=t.replace("_", " "), edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    ax2.set_title("True Topic Labels", fontsize=10)
    ax2.legend(fontsize=6, loc="best")
    
    fig, (ax1, ax2) = _style_matplotlib_fig(fig, ax1), _style_matplotlib_fig(fig, ax2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    ari = kmeans_data["ari"]
    ari_pct = max(0, (ari + 1) / 2 * 100)
    st.caption(
        f"**Adjusted Rand Index (ARI):** {ari:.3f} — the algorithm's own groupings matched the human topic "
        f"labels about {ari_pct:.0f}% of the way (1.0 = perfect match, 0.0 = random).\n\n"
        f"**সমন্বিত র্যান্ড ইনডেক্স (ARI):** {ari:.3f} — অ্যালগরিদমের নিজস্ব গোষ্ঠীভুক্তি মানব-নির্ধারিত "
        f"টপিক লেবেলের সাথে প্রায় {ari_pct:.0f}% মিলেছে (1.0 = নিখুঁত মিল, 0.0 = এলোমেলো)।"
    )
    
    with st.expander(bilingual_plain("Cluster vs. topic cross-tabulation", "ক্লাস্টার ও টপিকের ক্রস-ট্যাবুলেশন")):
        st.dataframe(kmeans_data["crosstab"], use_container_width=True)
    
    # ---------------------------------------------------------------------- #
    # SECTION 4: PCA VISUALIZATION
    # ---------------------------------------------------------------------- #
    st.markdown("---")
    st.markdown("#### ৪. পিসিএ ভিজুয়ালাইজেশন · PCA Visualization")
    st.caption(
        "Principal Component Analysis: reducing high-dimensional text data to 2D. "
        "প্রিন্সিপাল কম্পোনেন্ট অ্যানালাইসিস: উচ্চ-মাত্রিক টেক্সট ডেটাকে ২ডিতে রূপান্তর।"
    )
    
    fig, ax = plt.subplots(figsize=(7, 5))
    tier1_classes = insights["tier1_class"]
    unique_classes = sorted(tier1_classes.unique())
    for i, c in enumerate(unique_classes):
        mask = (tier1_classes == c).to_numpy()
        ax.scatter(coords[mask, 0], coords[mask, 1], s=40, alpha=0.85,
                   color=palette(i % 10), label=c.replace("_", " "), edgecolor="white", linewidth=0.4)
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_title(f"PCA: {kmeans_data['explained']:.1f}% variance explained", fontsize=11)
    ax.legend(fontsize=7, loc="best")
    
    fig, ax = _style_matplotlib_fig(fig, ax)
    st.pyplot(fig)
    plt.close(fig)
    
    st.caption(
        f"**{kmeans_data['explained']:.1f}%** of the total variance in the TF-IDF vectors is captured by "
        f"these 2 dimensions — the rest is spread across dimensions we can't easily draw on a flat page. "
        f"আমরা এখানে যে ২টি মাত্রা দেখছি তা টিএফ-আইডিএফ ভেক্টরগুলির মোট ভ্যারিয়েন্সের "
        f"**{kmeans_data['explained']:.1f}%** ধারণ করছে — বাকিটা এমন মাত্রায় ছড়িয়ে আছে যা আমরা "
        f"সহজে আঁকতে পারি না।"
    )
    
    # ---------------------------------------------------------------------- #
    # SECTION 5: LIVE TRY-IT
    # ---------------------------------------------------------------------- #
    st.markdown("---")
    st.markdown("#### ৫. লাইভ ট্রাই-ইট · Live Try-It")
    st.caption(
        "Type any question and see exactly how the system thinks — stage by stage. "
        "যেকোনো প্রশ্ন লিখুন এবং দেখুন সিস্টেমটি কীভাবে চিন্তা করে — ধাপে ধাপে।"
    )
    
    # Use the search-classifier's vectorizer and X for retrieval, but the insights
    # classifiers for the comparison
    live_query = st.text_input(
        bilingual_plain("Type your question to analyze", "বিশ্লেষণের জন্য আপনার প্রশ্ন লিখুন"),
        placeholder="e.g. is fasting mandatory during Ramadan? · রমজানে রোজা কি ফরজ?",
        key="live_try_input",
        label_visibility="visible"
    )
    
    if live_query.strip():
        # Run retrieval
        result = retrieve_candidates(live_query, df, vectorizer_search, X_search, banglish_map)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Retrieval Result**")
            if result.row is not None:
                stage_en, stage_bn = STAGE_LABELS.get(result.stage, (result.stage, result.stage))
                st.markdown(f"✅ Matched via: **{stage_en}** · {stage_bn}")
                st.markdown(f"Score: **{result.similarity:.3f}**")
                st.markdown(f"Ruling: **{result.row['tier1_class'].replace('_', ' ')}**")
                if result.row.get("question_en"):
                    st.markdown(f"\"{result.row['question_en']}\"")
            else:
                st.markdown("❌ No confident match found")
                if result.suggestions:
                    st.markdown("Closest suggestions:")
                    for r in result.suggestions[:3]:
                        st.markdown(f"- {r['tier1_class'].replace('_', ' ')}: {r['question_en'] or r['question_bn']}")
        
        with col2:
            st.markdown("**All 5 Classifiers' Predictions**")
            
            # Get predictions from each of the 5 classifiers
            insights = _get_insights_data(df)
            results = insights["results"]
            
            preds = {}
            for name, r in results.items():
                if "error" not in r:
                    try:
                        q_clean = clean_text(live_query)
                        if q_clean.strip():
                            q_vec = insights["vectorizer"].transform([q_clean])
                            pred = r["model"].predict(q_vec)[0]
                            preds[name] = pred
                        else:
                            preds[name] = None
                    except Exception:
                        preds[name] = None
                else:
                    preds[name] = None
            
            # Show predictions
            for name, pred in preds.items():
                if pred:
                    display_name = name.replace(" Logistic", "").replace(" Naive", "").replace(" Decision", "")
                    st.markdown(f"{display_name}: **{pred.replace('_', ' ')}**")
                else:
                    st.markdown(f"{name}: _unavailable_")
            
            # Agreement check
            if result.row is not None:
                true_label = result.row["tier1_class"]
                agreeing = [name for name, pred in preds.items() if pred == true_label]
                if agreeing:
                    st.success(f"✅ {len(agreeing)}/{len(preds)} classifiers agree with the retrieved ruling")
                else:
                    st.warning("⚠️ No classifier agrees with the retrieved ruling — the retrieval stage is doing the work here.")


# --------------------------------------------------------------------------- #
# UI — DESIGN TOKENS & STYLE
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
            --ink-muted: #544F42;
            --accent: #2F4858;
            --accent-soft: #E4DEC9;
            --hairline: #DDD3B8;
            --flag-bg: #F3E6C8;
            --flag-ink: #6B4E1F;
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

        a:focus-visible, button:focus-visible, input:focus-visible {
            outline: 2px solid var(--accent);
            outline-offset: 2px;
        }
        @media (prefers-reduced-motion: reduce) {
            * { animation: none !important; transition: none !important; }
        }

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

        .bn-inline {
            font-family: var(--font-body);
            color: var(--ink-muted);
        }
        .en-line { display: block; color: var(--ink); }
        .bn-line {
            display: block;
            color: var(--ink);
            margin-top: 0.2rem;
            font-size: 0.97em;
        }

        .field-label {
            font-family: var(--font-body);
            font-size: 0.85rem;
            color: var(--ink-muted);
            margin-bottom: 0.35rem;
        }
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

        div[role="radiogroup"] { gap: var(--space-3); }
        div[role="radiogroup"] label {
            font-family: var(--font-mono);
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--ink-muted);
        }

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

        .browse-count {
            font-family: var(--font-mono);
            font-size: 0.78rem;
            color: var(--ink-muted);
            letter-spacing: 0.05em;
            margin: var(--space-2) 0 var(--space-1) 0;
        }

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
# UI — HEADER WITH LOGO
# --------------------------------------------------------------------------- #

def render_header() -> None:
    """Render the university identity strip with logo and caption."""
    
    # Try to load the logo, fail gracefully if missing
    logo_html = ""
    logo_path = "assets/hstu_logo.png"
    if os.path.exists(logo_path):
        try:
            # Use st.image in a column for proper sizing
            logo_html = f"""
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.15rem;">
                <img src="data:image/png;base64,{_image_to_base64(logo_path)}" 
                     style="height: 52px; width: auto; object-fit: contain;" />
                <div>
                    <div style="font-family: 'Noto Sans Bengali', 'Inter', sans-serif; font-size: 0.78rem; 
                                color: #544F42; letter-spacing: 0.04em; line-height: 1.3;">
                        Hajee Mohammad Danesh Science and Technology University
                    </div>
                    <div style="font-family: 'Noto Sans Bengali', 'Inter', sans-serif; font-size: 0.72rem; 
                                color: #544F42; letter-spacing: 0.08em; text-transform: uppercase;">
                        CSE 469 — Machine Learning and Pattern Recognition
                    </div>
                </div>
            </div>
            """
        except Exception:
            # If image loading fails, fall through to text-only
            logo_html = ""
    
    if logo_html:
        st.markdown(logo_html, unsafe_allow_html=True)
    else:
        # Text-only fallback
        st.markdown(
            f"""
            <div style="margin-bottom: 0.15rem;">
                <div style="font-family: 'Noto Sans Bengali', 'Inter', sans-serif; font-size: 0.78rem; 
                            color: #544F42; letter-spacing: 0.04em; line-height: 1.3;">
                    Hajee Mohammad Danesh Science and Technology University
                </div>
                <div style="font-family: 'Noto Sans Bengali', 'Inter', sans-serif; font-size: 0.72rem; 
                            color: #544F42; letter-spacing: 0.08em; text-transform: uppercase;">
                    CSE 469 — Machine Learning and Pattern Recognition
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _image_to_base64(path: str) -> str:
    """Convert image to base64 for embedding in HTML."""
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(page_title="Ruling Reference", layout="centered")
    inject_css()

    # University identity strip (small, quiet — above the title)
    render_header()

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
    except Exception as exc:  # last-resort guard: never show a raw traceback
        st.error(f"The app couldn't start up: {exc}")
        st.stop()

    # Mode selector — now with 3 options
    mode = st.radio(
        "Mode",
        ["Search · খুঁজুন", "Browse · সবগুলো দেখুন", "Model Insights · মডেল বিশ্লেষণ"],
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
    elif mode.startswith("Model Insights"):
        render_model_insights_tab(df, vectorizer, X, banglish_map, classifier)
    else:
        render_browse_mode(df)

    st.markdown(
        f'<div class="app-footer">{bilingual_block("Entries here are for educational reference only. For personal or complex matters, consult a qualified scholar.", "এখানে দেওয়া তথ্য শুধুমাত্র শিক্ষামূলক রেফারেন্সের জন্য। ব্যক্তিগত বা জটিল বিষয়ে অবশ্যই একজন যোগ্য আলেমের পরামর্শ নিন।")}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
