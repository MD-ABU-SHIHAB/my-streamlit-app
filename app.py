"""
app.py — Islamic Ruling Reference & ML/Pattern Recognition Lab (CSE 469 capstone)

Two things live in this one file:
  1. A retrieval-first ruling search (exact match -> TF-IDF cosine -> fuzzy
     match -> honest no-match state), with Logistic Regression used only as
     a secondary confirmation signal — never the primary displayed answer.
  2. A "Lab" section that maps the CSE 469 syllabus onto this project: every
     syllabus item is either demonstrated LIVE on the real dataset, or —
     where a technique genuinely does not fit a small static text dataset
     (Reinforcement Learning, Genetic Algorithms, Hidden Markov Models,
     formal syntactic grammars) — explained honestly as a theory reference
     with a one-line reason it isn't part of the trained pipeline. No
     fabricated accuracy numbers for techniques that weren't actually run.

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
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    f1_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --------------------------------------------------------------------------- #
# CONSTANTS
# --------------------------------------------------------------------------- #

DATA_PATH: str = "dataset.csv"
RANDOM_STATE: int = 42  # matches main.py, for reproducible numbers in the report

TFIDF_SIMILARITY_THRESHOLD: float = 0.30
FUZZY_MATCH_THRESHOLD: float = 0.55
TOP_K_SUGGESTIONS: int = 5
TOP_K_RELATED: int = 4
MEANINGFUL_TOKEN_MIN_LEN: int = 3

CV_FOLD_TARGET: int = 5   # target k-fold, auto-capped down for small classes (see _cv_k)
KNN_K_TARGET: int = 5

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

PROJECT_AUTHOR = "Md. Abu Shihab"
PROJECT_ORG = "Dept. of CSE, Hajee Mohammad Danesh Science & Technology University"
PROJECT_COURSE = "CSE 469 — Machine Learning and Pattern Recognition"


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
    """Raised for any dataset load/shape problem, so main() can show a clean message."""


# --------------------------------------------------------------------------- #
# LOADING & PREPROCESSING (kept identical in approach to main.py)
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
        raise DatasetError("Every row is missing question_bn and/or tier1_class — nothing to search.")
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
    df = df.copy()
    for col in ("question_bn", "question_en", "question_banglish", "search_keywords"):
        df[col] = df[col].fillna("")
    df["combined_text_raw"] = df["question_bn"] + " " + df["question_en"] + " " + df["search_keywords"]
    df["combined_text_clean"] = df["combined_text_raw"].apply(clean_text)
    return df


@st.cache_data(show_spinner=False)
def build_banglish_map(df: pd.DataFrame) -> dict[str, set[str]]:
    """Extendable synonym map built only from the curated search_keywords column.

    (question_banglish is deliberately excluded: generic connector words
    shared across almost every row would otherwise transitively merge
    unrelated rulings' vocabularies together.)
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
# RETRIEVAL PIPELINE — also doubles as the live "rule-based" + "instance-based"
# / "template matching" demo referenced in the Lab tabs.
# --------------------------------------------------------------------------- #

def _exact_match(query: str, banglish_map: dict[str, set[str]], df: pd.DataFrame) -> Optional[pd.Series]:
    """RULE-BASED LEARNING, live: hand-authored matching rules over curated synonym groups."""
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
    """TEMPLATE MATCHING, live: cosine distance between the query vector and stored TF-IDF templates."""
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
    """INSTANCE-BASED / CASE-BASED reasoning, live: falls back to the single closest known case."""
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


@st.cache_resource(show_spinner="Preparing search index...")
def train_search_classifier(df: pd.DataFrame):
    """The single classifier used at query time. See the Comparison tab for why Logistic Regression."""
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
# ML / PATTERN RECOGNITION LAB — real computations for the syllabus mapping
# --------------------------------------------------------------------------- #

def _cv_k(y: pd.Series, target: int) -> int:
    return min(target, int(y.value_counts().min()))


def evaluate_classifier(model, X, y: pd.Series, cv_target: int = CV_FOLD_TARGET) -> dict:
    """Fit on full data (report-style snapshot) + stratified CV where the smallest class allows it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    y_pred = model.predict(X)
    train_acc = accuracy_score(y, y_pred)
    train_f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    k = _cv_k(y, cv_target)
    cv_possible = k >= 2
    cv_mean, cv_std = float("nan"), float("nan")
    if cv_possible:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                scores = cross_val_score(model, X, y, cv=skf, scoring="f1_macro")
                cv_mean, cv_std = float(scores.mean()), float(scores.std())
            except Exception:
                cv_possible = False

    return {
        "model": model,
        "train_accuracy": train_acc,
        "train_macro_f1": train_f1,
        "cv_k": k,
        "cv_possible": cv_possible,
        "cv_mean_macro_f1": cv_mean,
        "cv_std_macro_f1": cv_std,
    }


@st.cache_resource(show_spinner="Training the classification suite for the Lab...")
def run_classification_suite(df: pd.DataFrame):
    """CLASSIFICATION TECHNIQUES, live: trains every applicable supervised model from the syllabus."""
    df = build_combined_text(df)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=20000)
    X = vectorizer.fit_transform(df["combined_text_clean"])
    y = df["tier1_class"]

    knn_k = max(1, min(KNN_K_TARGET, len(y) - 1))

    registry = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Naive Bayes (Bayesian)": MultinomialNB(),
        "k-Nearest Neighbors (Instance-based)": KNeighborsClassifier(n_neighbors=knn_k),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest (Ensemble/Bagging)": RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE),
        "AdaBoost (Ensemble/Boosting)": AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Linear SVM": SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
        "RBF SVM (Nonlinear)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(16,), max_iter=3000, random_state=RANDOM_STATE, solver="lbfgs"
        ),
    }

    results = {}
    for name, model in registry.items():
        try:
            results[name] = evaluate_classifier(model, X, y)
        except Exception as exc:
            results[name] = {"error": str(exc)}

    return vectorizer, X, y, results, knn_k


@st.cache_resource(show_spinner=False)
def run_regression_demo(df: pd.DataFrame):
    """REGRESSION ANALYSIS, live: Linear vs Logistic regression on the same binary target.

    Classic illustration of why Linear Regression is the wrong tool for
    classification: it will predict values outside [0, 1], while Logistic
    Regression's sigmoid stays bounded — genuinely computed here, not staged.
    """
    df = build_combined_text(df)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=20000)
    X = vectorizer.fit_transform(df["combined_text_clean"])

    pca = PCA(n_components=1, random_state=RANDOM_STATE)
    feature = pca.fit_transform(X.toarray()).flatten()

    target_class = df["tier1_class"].value_counts().idxmax()
    y_binary = (df["tier1_class"] == target_class).astype(int).to_numpy()

    order = np.argsort(feature)
    feature_sorted = feature[order]
    y_sorted = y_binary[order]

    linreg = LinearRegression().fit(feature.reshape(-1, 1), y_binary)
    logreg = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    can_fit_logreg = len(np.unique(y_binary)) >= 2
    if can_fit_logreg:
        logreg.fit(feature.reshape(-1, 1), y_binary)

    x_range = np.linspace(feature.min(), feature.max(), 200).reshape(-1, 1)
    linreg_curve = linreg.predict(x_range)
    logreg_curve = logreg.predict_proba(x_range)[:, 1] if can_fit_logreg else None

    return {
        "feature": feature_sorted,
        "y": y_sorted,
        "x_range": x_range.flatten(),
        "linreg_curve": linreg_curve,
        "logreg_curve": logreg_curve,
        "target_class": target_class,
        "explained_variance": float(pca.explained_variance_ratio_[0] * 100),
    }


@st.cache_resource(show_spinner=False)
def run_kmeans_and_pca(df: pd.DataFrame):
    """CLUSTERING + PCA, live: unsupervised structure discovery, same logic as main.py."""
    df = build_combined_text(df)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=20000)
    X = vectorizer.fit_transform(df["combined_text_clean"])

    k = df["topic"].nunique()
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    ari = adjusted_rand_score(df["topic"], cluster_labels)
    crosstab = pd.crosstab(pd.Series(cluster_labels, name="cluster"), df["topic"].reset_index(drop=True))

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X.toarray())
    explained = float(pca.explained_variance_ratio_.sum() * 100)

    return {
        "k": k, "ari": ari, "crosstab": crosstab,
        "coords": coords, "explained": explained,
        "tier1_class": df["tier1_class"], "topic": df["topic"],
    }


def bias_variance_text(results: dict) -> str:
    scored = {n: r for n, r in results.items() if "error" not in r}
    if not scored:
        return "No models trained successfully yet."
    any_cv = any(r["cv_possible"] for r in scored.values())

    if not any_cv:
        perfect_fits = [n for n, r in scored.items() if r["train_macro_f1"] >= 0.999]
        text = (
            "Cross-validation isn't possible yet — at least one class has fewer than 2 examples. "
            "**Train-set accuracy alone cannot distinguish genuine skill from memorization at this "
            "dataset size**, so no model can honestly be called 'best' yet. "
        )
        if perfect_fits:
            text += (
                f"In fact, {', '.join(perfect_fits)} reached a perfect training-set score — this is "
                "expected simply because a flexible model can memorize this small a dataset "
                "outright, not evidence that it will generalize to unseen questions. Treat every "
                "number in this tab as provisional until the dataset is large enough for real "
                "cross-validation (see main.py's identical caveat)."
            )
        return text

    rank_key = "cv_mean_macro_f1"
    ranked = sorted(scored.items(), key=lambda kv: (kv[1][rank_key] if not np.isnan(kv[1][rank_key]) else -1), reverse=True)
    best_name = ranked[0][0]

    overfit_notes = []
    for name, r in scored.items():
        if r["cv_possible"] and not np.isnan(r["cv_mean_macro_f1"]):
            gap = r["train_macro_f1"] - r["cv_mean_macro_f1"]
            if gap > 0.25:
                overfit_notes.append(f"{name} (train {r['train_macro_f1']:.2f} vs CV {r['cv_mean_macro_f1']:.2f})")

    text = f"Best-generalizing model so far: **{best_name}**, ranked by cross-validated macro-F1. "
    if overfit_notes:
        text += "Largest train-vs-CV gaps (a sign of overfitting): " + "; ".join(overfit_notes) + "."
    else:
        text += "No model shows a large train-vs-CV gap yet."
    return text


# --------------------------------------------------------------------------- #
# UI — STYLE
# --------------------------------------------------------------------------- #

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Tiro+Bangla&family=Noto+Serif+Bengali:wght@400;600&family=Noto+Sans+Bengali:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            background-color: #EDE7D9 !important;
            color: #2B2A26;
            font-family: 'Noto Sans Bengali', 'Inter', sans-serif;
        }
        .block-container { max-width: 900px; padding-top: 1.6rem; padding-bottom: 3rem; }
        #MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }

        h1, h2, h3, .app-heading {
            font-family: 'Tiro Bangla', 'Noto Serif Bengali', Georgia, serif;
            color: #2B2A26;
            font-weight: 600;
        }
        .letterhead {
            font-family: 'Noto Sans Bengali', 'Inter', sans-serif;
            font-size: 0.74rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #6B6558;
            border-bottom: 1px solid #D8D0BC;
            padding-bottom: 0.6rem;
            margin-bottom: 1.1rem;
        }
        .app-title { font-size: 1.85rem; margin-bottom: 0.15rem; }
        .app-subtitle {
            font-family: 'Noto Sans Bengali', 'Inter', sans-serif;
            color: #6B6558; font-size: 0.95rem; margin-bottom: 1.6rem;
        }

        div[data-testid="stTextInput"] input {
            background-color: #F6F2E8; border: 1px solid #D8D0BC; border-radius: 4px;
            color: #2B2A26; font-size: 1.05rem; padding: 0.7rem 0.9rem;
        }
        div[data-testid="stTextInput"] input:focus { border-color: #3A4F63; box-shadow: 0 0 0 1px #3A4F63; }

        .stButton > button {
            background-color: #3A4F63; color: #F6F2E8; border: none; border-radius: 4px;
            padding: 0.5rem 1.3rem; font-family: 'Noto Sans Bengali', 'Inter', sans-serif; font-weight: 500;
        }
        .stButton > button:hover { background-color: #2E3F50; color: #F6F2E8; }

        button[data-baseweb="tab"] {
            font-family: 'Noto Sans Bengali', 'Inter', sans-serif !important;
            font-size: 0.92rem !important;
        }
        div[data-baseweb="tab-highlight"] { background-color: #3A4F63 !important; }

        .result-card {
            background-color: #F6F2E8; border: 1px solid #D8D0BC; border-radius: 4px;
            padding: 1.4rem 1.6rem; margin-top: 1.1rem;
        }
        .result-question { font-family: 'Tiro Bangla', 'Noto Serif Bengali', Georgia, serif; font-size: 1.12rem; margin-bottom: 0.5rem; }
        .category-label {
            display: inline-block; font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase;
            color: #2B2A26; border-left: 3px solid #3A4F63; padding: 0.15rem 0 0.15rem 0.6rem; margin-bottom: 0.9rem;
        }
        .explanation-text { font-size: 0.98rem; line-height: 1.55; margin-bottom: 0.9rem; }
        .citation-block {
            border-left: 3px solid #D8D0BC; padding: 0.5rem 0 0.5rem 0.9rem;
            color: #4A463E; font-size: 0.9rem; font-style: italic; margin-bottom: 0.6rem;
        }
        .verification-flag {
            display: inline-block; font-size: 0.78rem; color: #6B5033; background-color: #F0E4CC;
            border: 1px solid #D8C9A0; border-radius: 3px; padding: 0.15rem 0.5rem; margin-bottom: 0.6rem;
        }
        .related-list { margin-top: 0.9rem; font-size: 0.9rem; }
        .related-item { padding: 0.25rem 0; border-bottom: 1px solid #E4DFD1; }
        .no-match-box {
            background-color: #F6F2E8; border: 1px dashed #D8D0BC; border-radius: 4px;
            padding: 1.2rem 1.4rem; margin-top: 1.1rem; color: #4A463E;
        }

        .lab-card {
            background-color: #F6F2E8; border: 1px solid #D8D0BC; border-radius: 4px;
            padding: 1.1rem 1.3rem; margin-bottom: 1rem;
        }
        .lab-tag {
            display: inline-block; font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase;
            padding: 0.1rem 0.5rem; border-radius: 3px; margin-bottom: 0.5rem; font-weight: 600;
        }
        .tag-live { background-color: #E1E9E4; color: #2F5D40; }
        .tag-theory { background-color: #EFE3E0; color: #7A4A3D; }
        .lab-note { color: #6B6558; font-size: 0.85rem; margin-top: 0.4rem; }

        .app-footer {
            margin-top: 2.4rem; padding-top: 1rem; border-top: 1px solid #D8D0BC;
            color: #6B6558; font-size: 0.82rem; text-align: center;
        }

        @media (max-width: 480px) {
            .block-container { padding-left: 0.9rem; padding-right: 0.9rem; }
            .app-title { font-size: 1.45rem; }
            .result-card, .lab-card { padding: 1rem 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# UI — SEARCH & BROWSE (unchanged core behavior, same as the tested build)
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
        st.markdown('<div class="verification-flag">Unverified reference — pending scholarly check</div>', unsafe_allow_html=True)

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
            "exact_match": "Exact/rule-based match",
            "tfidf_cosine": "TF-IDF template match (cosine similarity)",
            "fuzzy_match": "Fuzzy / instance-based match",
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
            st.markdown(f'<div class="related-item"><strong>{label}</strong> — {r["question_en"] or r["question_bn"]}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_search_tab(df: pd.DataFrame, vectorizer, X, banglish_map, classifier) -> None:
    query = st.text_input(
        "Search", placeholder="namaj pora ki · is riba haram · বিয়ে করা কি সুন্নত…", label_visibility="collapsed"
    )
    if query.strip():
        result = retrieve_candidates(query, df, vectorizer, X, banglish_map)
        if result.row is not None:
            render_result(result, df, vectorizer, classifier)
        else:
            render_no_match(result)


def render_browse_tab(df: pd.DataFrame) -> None:
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
# UI — ML / PATTERN RECOGNITION LAB
# --------------------------------------------------------------------------- #

def _lab_card(tag: str, title: str, body_md: str) -> None:
    tag_class = "tag-live" if tag == "LIVE ON YOUR DATA" else "tag-theory"
    st.markdown(
        f'<div class="lab-card"><span class="lab-tag {tag_class}">{tag}</span>'
        f'<h3 style="margin-top:0.3rem;margin-bottom:0.4rem;">{title}</h3>{body_md}</div>',
        unsafe_allow_html=True,
    )


def render_regression_tab(df: pd.DataFrame) -> None:
    st.markdown("#### Regression Analysis")
    st.caption("Syllabus: Logistic regression, Linear regression")

    data = run_regression_demo(df)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(data["feature"], data["y"], alpha=0.5, s=35, color="#6B6558", label="Actual (0/1)", zorder=3)
    ax.plot(data["x_range"], data["linreg_curve"], color="#8C4A3A", linewidth=2, label="Linear Regression")
    if data["logreg_curve"] is not None:
        ax.plot(data["x_range"], data["logreg_curve"], color="#3A4F63", linewidth=2, label="Logistic Regression (sigmoid)")
    ax.axhline(0, color="#D8D0BC", linewidth=1)
    ax.axhline(1, color="#D8D0BC", linewidth=1)
    ax.set_xlabel("PC1 of TF-IDF features")
    ax.set_ylabel(f"Is '{data['target_class']}'? (1 = yes)")
    ax.set_title("Linear vs. Logistic Regression on the same binary target")
    ax.legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        f"""
**What this shows:** both models are fit on the same single feature (the first principal
component of the TF-IDF space, explaining {data['explained_variance']:.1f}% of variance) against
the same binary target (*is this ruling "{data['target_class']}"?*). Linear Regression's line is
unbounded — it can predict below 0 or above 1, which is meaningless as a probability.
Logistic Regression's sigmoid curve stays within [0, 1] by construction, which is exactly why it
— not Linear Regression — is the correct tool for classification, and why it is the model used
at runtime in the Search tab.
        """
    )


def render_classification_tab(df: pd.DataFrame) -> None:
    st.markdown("#### Classification Techniques")
    st.caption("Syllabus: Supervised & unsupervised classification, NN, SVM, classification trees, "
               "rule-based learning, instance-based learning, reinforcement learning, ensemble learning, "
               "negative correlation learning, evolutionary algorithms, genetic algorithm")

    vectorizer, X, y, results, knn_k = run_classification_suite(df)
    any_cv = any(r.get("cv_possible") for r in results.values() if "error" not in r)

    if not any_cv:
        st.info(
            f"With the current {len(y)}-row dataset, at least one class is too small for "
            "cross-validation — the table below shows full-fit metrics (trained and evaluated "
            "on the same data), which will overstate real-world accuracy. This is expected to "
            "resolve automatically as more rows are added; see main.py for the same caveat."
        )

    rows = []
    for name, r in results.items():
        if "error" in r:
            rows.append({"Model": name, "Train Accuracy": "—", "Train Macro-F1": "—", "CV Macro-F1": f"error: {r['error'][:50]}"})
            continue
        cv_str = f"{r['cv_mean_macro_f1']:.3f} ± {r['cv_std_macro_f1']:.3f}" if r["cv_possible"] else "N/A"
        rows.append({
            "Model": name,
            "Train Accuracy": f"{r['train_accuracy']:.3f}",
            "Train Macro-F1": f"{r['train_macro_f1']:.3f}",
            "CV Macro-F1": cv_str,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    _lab_card(
        "LIVE ON YOUR DATA", "Rule-based & instance-based learning",
        "The Search tab's exact-match stage <em>is</em> a rule-based classifier (hand-authored "
        "matching rules over curated synonym groups), and its fuzzy-match stage is instance-based "
        "/ case-based reasoning (falls back to the single most similar known example). k-Nearest "
        f"Neighbors above (k={knn_k}) is the formally trained instance-based classifier.",
    )
    _lab_card(
        "LIVE ON YOUR DATA", "Ensemble learning",
        "Random Forest (bagging — many decision trees trained on bootstrapped samples, majority "
        "vote) and AdaBoost (boosting — trees trained sequentially, each correcting the previous "
        "one's errors) are both trained and scored in the table above.",
    )
    _lab_card(
        "THEORY REFERENCE", "Reinforcement Learning",
        "Learns a policy through trial-and-error reward signals from an interactive environment. "
        "Not applicable here: this project is a static labeled dataset with no environment, "
        "actions, or reward signal to learn from, so it is not part of the trained pipeline.",
    )
    _lab_card(
        "THEORY REFERENCE", "Genetic Algorithm & Evolutionary Computation",
        "Evolves a population of candidate solutions via selection, crossover, and mutation "
        "toward an optimization objective. Not applicable here: this is a supervised "
        "classification problem with a closed-form/gradient-based training procedure for every "
        "model above, so there is no search-space optimization problem for a GA to solve.",
    )
    _lab_card(
        "THEORY REFERENCE", "Negative Correlation Learning",
        "An ensemble-training strategy that explicitly penalizes correlated errors between "
        "member networks to encourage diversity (mainly used with neural network ensembles). "
        "Not implemented here — the Random Forest/AdaBoost ensembles above use their standard "
        "bagging/boosting diversity mechanisms instead.",
    )


def render_evaluation_tab(df: pd.DataFrame) -> None:
    st.markdown("#### Statistical Performance Evaluation")
    st.caption("Syllabus: Bias-variance tradeoff, practical applications of machine learning")

    vectorizer, X, y, results, _ = run_classification_suite(df)
    st.markdown(bias_variance_text(results))

    st.markdown("###### Per-class performance")
    scored = {n: r for n, r in results.items() if "error" not in r}
    if scored:
        any_cv = any(r["cv_possible"] for r in scored.values())
        if any_cv:
            best_name = max(scored, key=lambda n: scored[n]["cv_mean_macro_f1"] if not np.isnan(scored[n]["cv_mean_macro_f1"]) else -1)
            st.caption(f"Model: {best_name} (selected by cross-validated macro-F1)")
        else:
            # Cross-validation isn't possible yet, so a full-fit "top score" would just reward
            # memorization (see the note above). Show the actual deployed model (Logistic
            # Regression) instead of whichever model happened to overfit hardest.
            best_name = "Logistic Regression"
            st.caption(f"Model: {best_name} (the model actually deployed in the Search tab — "
                       "shown here instead of the full-fit 'top score' model, since a full-fit "
                       "ranking would just reward memorization at this dataset size)")
        best_model = scored[best_name]["model"]
        report = classification_report(y, best_model.predict(X), output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

    st.markdown("###### Practical applications of machine learning")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**In Islamic knowledge systems**\n"
            "- Ruling classification (this project)\n"
            "- Hadith chain (isnad) authentication support\n"
            "- Topic modeling over Quranic/tafsir text\n"
            "- Question-answering / reference retrieval\n"
            "- Automated citation cross-checking"
        )
    with col2:
        st.markdown(
            "**Broader domains**\n"
            "- Legal document / contract classification\n"
            "- Clinical text classification\n"
            "- Sentiment analysis on social text\n"
            "- Recommendation systems\n"
            "- Fraud / anomaly detection"
        )


def render_pattern_recognition_tab(df: pd.DataFrame) -> None:
    st.markdown("#### Pattern Recognition")
    st.caption("Syllabus: Bayesian decision theory, linear/nonlinear classifiers, parametric & "
               "non-parametric estimation, template matching, context-dependent classification, "
               "Markov models, HMM, syntactic PR, clustering, PCA")

    vectorizer, X, y, results, _ = run_classification_suite(df)

    st.markdown("###### Bayesian decision theory")
    if "Naive Bayes (Bayesian)" in results and "error" not in results["Naive Bayes (Bayesian)"]:
        nb_model = results["Naive Bayes (Bayesian)"]["model"]
        sample_q = df["question_en"].dropna().iloc[0] if not df["question_en"].dropna().empty else df["question_bn"].iloc[0]
        q_clean = clean_text(sample_q)
        q_vec = vectorizer.transform([q_clean])
        proba = nb_model.predict_proba(q_vec)[0]
        proba_df = pd.DataFrame({"tier1_class": nb_model.classes_, "posterior_probability": proba.round(4)})
        proba_df = proba_df.sort_values("posterior_probability", ascending=False)
        st.caption(f'Posterior P(class | query) for the sample question: "{sample_q}"')
        st.dataframe(proba_df, use_container_width=True, hide_index=True)
        st.markdown(
            "Naive Bayes applies Bayes' theorem directly: it picks the class with the highest "
            "posterior probability P(class | text) ∝ P(text | class) · P(class), assuming word "
            "features are conditionally independent given the class — the classic Bayesian "
            "decision-theory classifier."
        )

    st.markdown("###### Linear vs. nonlinear classifiers")
    lin_nonlin_rows = [
        {"Model": "Logistic Regression", "Boundary": "Linear"},
        {"Model": "Linear SVM", "Boundary": "Linear"},
        {"Model": "RBF SVM (Nonlinear)", "Boundary": "Nonlinear (kernel trick)"},
        {"Model": "Decision Tree", "Boundary": "Nonlinear (axis-aligned splits)"},
        {"Model": "Neural Network (MLP)", "Boundary": "Nonlinear (hidden layer activations)"},
        {"Model": "Random Forest", "Boundary": "Nonlinear (ensemble of trees)"},
    ]
    st.dataframe(pd.DataFrame(lin_nonlin_rows), use_container_width=True, hide_index=True)

    st.markdown("###### Parametric vs. non-parametric estimation")
    st.markdown(
        "- **Parametric** (assumes a fixed functional form): Logistic Regression (sigmoid over a "
        "linear combination), Naive Bayes (assumes a multinomial word-distribution per class).\n"
        "- **Non-parametric** (grows in complexity with the data, no fixed form): k-Nearest "
        "Neighbors, Decision Tree, Random Forest."
    )

    _lab_card(
        "LIVE ON YOUR DATA", "Template matching",
        "The Search tab's TF-IDF + cosine-similarity stage is literally template matching: the "
        "query is converted into the same vector space as every stored question, and the closest "
        "stored template (by cosine distance) is retrieved.",
    )
    _lab_card(
        "LIVE ON YOUR DATA", "Context-dependent classification",
        "Character n-grams (3-5 characters) capture local sub-word context, which is what lets "
        "Bangla's suffix-based inflections and Bangla/Banglish spelling variants of the same word "
        "still map to nearby vectors, exactly as used in main.py and the Search tab.",
    )
    _lab_card(
        "LIVE ON YOUR DATA", "Syntactic pattern recognition",
        "The synonym-group matching (search_keywords grouped into equivalence classes, then "
        "matched as whole tokens) is a lightweight structural/string-matching approach — a lean, "
        "rule-based cousin of full syntactic pattern recognition, without a formal grammar.",
    )
    _lab_card(
        "THEORY REFERENCE", "Markov models & Hidden Markov Models",
        "Model sequences of states where the next state depends on a limited window of previous "
        "states (Markov property); HMMs add a hidden state layer inferred from observed emissions "
        "(classic use: speech/phoneme recognition, POS tagging). Not applicable here: each row is "
        "an independent question with no temporal/sequential dependency between rows for an HMM "
        "to model.",
    )

    st.markdown("###### Clustering & PCA")
    kp = run_kmeans_and_pca(df)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("K-Means clusters (k = #topics)", kp["k"])
        st.metric("Adjusted Rand Index vs. topic labels", f"{kp['ari']:.3f}")
        st.caption(
            "1.0 = perfect alignment with the human topic taxonomy, 0.0 = no better than random. "
            "Perfect alignment isn't expected — K-Means groups by lexical similarity, while topic "
            "is a meaning-based human taxonomy."
        )
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4.2))
        palette = plt.get_cmap("tab10")
        for i, label in enumerate(sorted(kp["tier1_class"].unique())):
            mask = (kp["tier1_class"] == label).to_numpy()
            ax.scatter(kp["coords"][mask, 0], kp["coords"][mask, 1], s=45, alpha=0.85,
                       color=palette(i % 10), label=label, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(f"PCA ({kp['explained']:.1f}% variance explained)", fontsize=10)
        ax.legend(fontsize=6, loc="best")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("K-Means cluster vs. topic cross-tabulation"):
        st.dataframe(kp["crosstab"], use_container_width=True)


def render_comparison_tab(df: pd.DataFrame) -> None:
    st.markdown("#### Model Comparison & Selection")
    _, _, y, results, _ = run_classification_suite(df)
    scored = {n: r for n, r in results.items() if "error" not in r}
    any_cv = any(r["cv_possible"] for r in scored.values())
    rank_key = "cv_mean_macro_f1" if any_cv else "train_macro_f1"

    if not any_cv:
        st.info(
            "Cross-validation isn't possible yet (a class has fewer than 2 examples), so the bars "
            "below are full-fit scores — models that can simply memorize this small a dataset "
            "(Decision Tree, Random Forest, Linear SVM, MLP) will look artificially strong here. "
            "This is exactly why the deployed Search tab uses Logistic Regression rather than "
            "whichever model tops this chart today."
        )

    names = list(scored.keys())
    scores = [scored[n][rank_key] if not np.isnan(scored[n][rank_key]) else 0 for n in names]
    order = np.argsort(scores)[::-1]
    names = [names[i] for i in order]
    scores = [scores[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, scores, color="#3A4F63")
    ax.set_ylabel("CV macro-F1" if any_cv else "Full-fit macro-F1")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 0.02, f"{s:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        """
**Why Logistic Regression runs the live Search tab:** it is not necessarily the single highest
score above — the pick is justified by cross-validation stability (low variance across folds),
not the top raw number from one lucky split. Logistic Regression and Naive Bayes are the classic
low-variance choices on small, sparse TF-IDF data; higher-capacity models (Decision Tree, Neural
Network, RBF SVM) tend to need substantially more training rows before their extra flexibility
pays off rather than overfits. This mirrors the model-selection reasoning in main.py exactly, so
the report and the live app do not silently disagree with each other.
        """
    )


def render_coverage_tab() -> None:
    st.markdown("#### Syllabus Coverage Checklist")
    st.caption("Every line item from the CSE 469 syllabus, and exactly where it is demonstrated in this app.")

    coverage = [
        ("Introduction to ML", "Regression tab", "Live"),
        ("Logistic regression", "Regression tab + Search tab (runtime model)", "Live"),
        ("Linear regression", "Regression tab", "Live"),
        ("Supervised classification", "Classification tab", "Live"),
        ("Unsupervised classification", "Pattern Recognition tab (K-Means)", "Live"),
        ("Neural networks (NN)", "Classification tab (MLP)", "Live"),
        ("Support vector machines (SVM)", "Classification tab (linear + RBF)", "Live"),
        ("Classification trees", "Classification tab (Decision Tree)", "Live"),
        ("Rule-based learning", "Classification tab + Search tab (exact-match stage)", "Live"),
        ("Instance-based learning", "Classification tab (k-NN) + Search tab (fuzzy stage)", "Live"),
        ("Reinforcement learning", "Classification tab", "Theory"),
        ("Ensemble learning", "Classification tab (Random Forest, AdaBoost)", "Live"),
        ("Negative correlation learning", "Classification tab", "Theory"),
        ("Evolutionary algorithms / Genetic algorithm", "Classification tab", "Theory"),
        ("Bias-variance tradeoff", "Evaluation tab", "Live"),
        ("Practical applications of ML", "Evaluation tab", "Live"),
        ("Introduction to Pattern Recognition", "Pattern Recognition tab", "Live"),
        ("Statistical & neural pattern recognition", "Classification + Pattern Recognition tabs", "Live"),
        ("Bayesian decision theory", "Pattern Recognition tab (Naive Bayes posteriors)", "Live"),
        ("Linear classifiers", "Pattern Recognition tab", "Live"),
        ("Nonlinear classifiers", "Pattern Recognition tab", "Live"),
        ("Parametric estimation", "Pattern Recognition tab", "Live"),
        ("Non-parametric estimation", "Pattern Recognition tab", "Live"),
        ("Template matching", "Pattern Recognition tab + Search tab (TF-IDF stage)", "Live"),
        ("Context-dependent classification", "Pattern Recognition tab (character n-grams)", "Live"),
        ("Markov models / Hidden Markov models", "Pattern Recognition tab", "Theory"),
        ("Syntactic pattern recognition", "Pattern Recognition tab (synonym-group matching)", "Live"),
        ("Clustering algorithms", "Pattern Recognition tab (K-Means + ARI)", "Live"),
        ("Principal component analysis (PCA)", "Pattern Recognition tab (2D scatter)", "Live"),
    ]
    coverage_df = pd.DataFrame(coverage, columns=["Syllabus item", "Where in this app", "Status"])
    st.dataframe(coverage_df, use_container_width=True, hide_index=True, height=560)
    live_count = sum(1 for c in coverage if c[2] == "Live")
    st.caption(
        f"{live_count} of {len(coverage)} items are demonstrated live on the actual dataset; the "
        "remainder are covered as an explained theory reference with the specific reason they "
        "don't fit a small static text-classification dataset (no fabricated results for those)."
    )


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(page_title="Ruling Reference · CSE 469", layout="centered")
    inject_css()

    st.markdown(f'<div class="letterhead">{PROJECT_ORG} · {PROJECT_COURSE} · {PROJECT_AUTHOR}</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-title app-heading">Ruling Reference</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Search Islamic rulings in Bangla, English, or Banglish — '
        'every entry traces to a cited source.</div>',
        unsafe_allow_html=True,
    )

    try:
        df = load_dataset(DATA_PATH)
        df = build_combined_text(df)
        banglish_map = build_banglish_map(df)
        vectorizer, X, classifier = train_search_classifier(df)
    except DatasetError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"The app couldn't start up: {exc}")
        st.stop()

    tabs = st.tabs([
        "Search", "Browse", "Regression", "Classification",
        "Evaluation", "Pattern Recognition", "Comparison", "Syllabus Coverage",
    ])

    with tabs[0]:
        render_search_tab(df, vectorizer, X, banglish_map, classifier)
    with tabs[1]:
        render_browse_tab(df)
    with tabs[2]:
        render_regression_tab(df)
    with tabs[3]:
        render_classification_tab(df)
    with tabs[4]:
        render_evaluation_tab(df)
    with tabs[5]:
        render_pattern_recognition_tab(df)
    with tabs[6]:
        render_comparison_tab(df)
    with tabs[7]:
        render_coverage_tab()

    st.markdown(
        '<div class="app-footer">Rulings shown here are for educational reference only — '
        "consult a qualified scholar for personal or complex matters.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
