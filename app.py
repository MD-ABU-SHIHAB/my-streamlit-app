"""
app.py — Islamic ruling reference search (CSE 469 capstone)

Retrieval-first search over dataset.csv (exact match -> TF-IDF cosine
similarity -> fuzzy match -> honest no-match state), with a Logistic
Regression classifier used only as a secondary confirmation signal, never
as the primary displayed answer. Includes a browse/filter mode and a live
ML & Pattern Recognition Lab.

Trilingual UI (English / Bangla / Arabic). Optional Arabic reference columns
are supported additively and degrade gracefully when absent.

Run:
    streamlit run app.py

Required packages (pip):
    streamlit
    pandas
    numpy
    scikit-learn
    matplotlib
    plotly
"""

from __future__ import annotations

import difflib
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
MAX_HISTORY: int = 5

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

STAGE_LABELS = {
    "exact_match": ("Exact match", "সরাসরি মিল", "تطابق مباشر"),
    "tfidf_cosine": ("TF-IDF · cosine similarity", "শব্দ-সাদৃশ্য বিশ্লেষণ", "تشابه TF-IDF"),
    "fuzzy_match": ("Approximate match", "আনুমানিক মিল", "تطابق تقريبي"),
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

TIER1_CLASS_AR = {
    "Obligatory": "واجب",
    "Recommended": "مستحب",
    "Permissible": "جائز",
    "Disliked": "مكروه",
    "Forbidden": "محرم",
    "Religious_Innovation": "بدعة",
    "Faith_Violation": "مخالفة للإيمان",
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

TOPIC_AR = {
    "Worship": "العبادة",
    "Food_and_Drink": "الطعام والشراب",
    "Family_and_Marriage": "الأسرة والزواج",
    "Business_and_Finance": "التجارة والمالية",
    "Purity": "الطهارة",
    "Social_Conduct": "السلوك الاجتماعي",
    "Clothing_and_Adornment": "اللباس والزينة",
    "Faith_and_Aqidah": "الإيمان والعقيدة",
    "Funeral_and_Mourning": "الجنازة والعزاء",
    "Oaths_and_Vows": "الأيمان والنذور",
}

# Colour palette (quiet reference-library)
COLOR_PAPER = "#EDE7D9"
COLOR_CARD = "#F6F2E8"
COLOR_INK = "#2B2A26"
COLOR_INK_MUTED = "#6B6558"
COLOR_ACCENT = "#3A4F63"
COLOR_ACCENT_SOFT = "#E4DEC9"
COLOR_HAIRLINE = "#D8D0BC"
COLOR_FLAG_BG = "#F3E6C8"
COLOR_FLAG_INK = "#6B4E1F"

PLOTLY_SAFE = px.colors.qualitative.Safe

# --------------------------------------------------------------------------- #
# UI STRINGS (trilingual)
# --------------------------------------------------------------------------- #

UI_STRINGS: dict[str, dict[str, str]] = {
    "app_title": {
        "en": "Ruling Reference",
        "bn": "মাসআলা অনুসন্ধান",
        "ar": "مرجع الأحكام الشرعية",
    },
    "app_eyebrow": {
        "en": "A reference collection of verified rulings",
        "bn": "যাচাইকৃত মাসআলার একটি সংকলন",
        "ar": "مجموعة مرجعية من الأحكام الموثقة",
    },
    "app_subtitle": {
        "en": "Search in Bangla, English, or Banglish — every entry traces to a cited source.",
        "bn": "বাংলা, ইংরেজি অথবা বাংলিশে খুঁজুন — প্রতিটি ফলাফলের সাথে যাচাইযোগ্য তথ্যসূত্র দেওয়া আছে।",
        "ar": "ابحث بالعربية أو الإنجليزية أو البنغالية — كل إدخال مرتبط بمصدر موثق.",
    },
    "search_placeholder": {
        "en": "namaj pora ki · is riba haram · বিয়ে করা কি সুন্নত…",
        "bn": "নামাজ পড়া কি · সুদ কি হারাম · বিয়ে করা কি সুন্নত…",
        "ar": "هل الصلاة واجبة · هل الربا محرم…",
    },
    "mode_search": {"en": "Search", "bn": "খুঁজুন", "ar": "بحث"},
    "mode_browse": {"en": "Browse", "bn": "সবগুলো দেখুন", "ar": "تصفح"},
    "mode_lab": {"en": "Model Insights", "bn": "মডেল বিশ্লেষণ", "ar": "تحليل النماذج"},
    "type_question": {"en": "Type your question", "bn": "আপনার প্রশ্ন লিখুন", "ar": "اكتب سؤالك"},
    "copy_citation": {"en": "Copy citation", "bn": "উদ্ধৃতি কপি", "ar": "نسخ الاقتباس"},
    "save_session": {"en": "Save this session", "bn": "এই সেশনে সংরক্ষণ", "ar": "حفظ لهذه الجلسة"},
    "unsave_session": {"en": "Remove from saved", "bn": "সংরক্ষণ থেকে সরান", "ar": "إزالة من المحفوظ"},
    "saved_panel": {
        "en": "Saved this session (cleared on refresh)",
        "bn": "এই সেশনে সংরক্ষিত (রিফ্রেশে মুছে যাবে)",
        "ar": "محفوظ لهذه الجلسة (يُمسح عند التحديث)",
    },
    "history_heading": {"en": "Recent searches", "bn": "সাম্প্রতিক অনুসন্ধান", "ar": "عمليات البحث الأخيرة"},
    "has_arabic_filter": {"en": "Has Arabic reference", "bn": "আরবি রেফারেন্স আছে", "ar": "يحتوي على مرجع عربي"},
    "font_size": {"en": "Text size", "bn": "অক্ষরের আকার", "ar": "حجم النص"},
    "font_small": {"en": "S", "bn": "ছ", "ar": "ص"},
    "font_normal": {"en": "M", "bn": "ম", "ar": "و"},
    "font_large": {"en": "L", "bn": "ব", "ar": "ك"},
    "related_heading": {"en": "Related entries", "bn": "সম্পর্কিত মাসআলা", "ar": "إدخالات ذات صلة"},
    "closest_heading": {"en": "Closest entries", "bn": "কাছাকাছি এন্ট্রি", "ar": "أقرب الإدخالات"},
    "apparatus": {"en": "Apparatus — how this was matched", "bn": "মিলের বিবরণ", "ar": "تفاصيل المطابقة"},
    "no_match_heading": {
        "en": "No confident match in the reference set.",
        "bn": "নির্ভরযোগ্য কোনো মিল পাওয়া যায়নি।",
        "ar": "لم يتم العثور على تطابق موثوق في المجموعة المرجعية.",
    },
    "no_match_body": {
        "en": "Try different wording, or open one of the closest entries below.",
        "bn": "অন্যভাবে লিখে চেষ্টা করুন, অথবা নিচের কাছাকাছি এন্ট্রিগুলো দেখুন।",
        "ar": "جرب صياغة مختلفة، أو افتح أحد أقرب الإدخالات أدناه.",
    },
    "footer": {
        "en": "Entries here are for educational reference only. For personal or complex matters, consult a qualified scholar.",
        "bn": "এখানে দেওয়া তথ্য শুধুমাত্র শিক্ষামূলক রেফারেন্সের জন্য। ব্যক্তিগত বা জটিল বিষয়ে অবশ্যই একজন যোগ্য আলেমের পরামর্শ নিন।",
        "ar": "هذه الإدخالات للمرجعية التعليمية فقط. للأمور الشخصية أو المعقدة، استشر عالماً مؤهلاً.",
    },
    "topic_label": {"en": "Topic", "bn": "বিষয়", "ar": "الموضوع"},
    "category_label": {"en": "Category", "bn": "শ্রেণি", "ar": "الفئة"},
    "all_option": {"en": "All", "bn": "সব", "ar": "الكل"},
    "browse_count": {"en": "entries", "bn": "টি এন্ট্রি", "ar": "إدخال"},
    "unverified": {
        "en": "Unverified reference — pending scholarly check",
        "bn": "যাচাই বাকি — বিশেষজ্ঞ পর্যালোচনার অপেক্ষায়",
        "ar": "مرجع غير موثق — بانتظار المراجعة العلمية",
    },
    "lab_intro": {
        "en": "This tab demonstrates every ML/PR syllabus concept hands-on, with real numbers computed from the real dataset — not static screenshots.",
        "bn": "এটি সরাসরি ডেটাসেট থেকে গণনা করা বাস্তব সংখ্যা সহ হ্যান্ডস-অন প্রদর্শনের জন্য তৈরি।",
        "ar": "تعرض هذه التبويبة كل مفهوم من مقرر التعلم الآلي والتعرف على الأنماط بأرقام حقيقية محسوبة من مجموعة البيانات.",
    },
    "why_logreg": {
        "en": "Logistic Regression runs the live Search tab. It is chosen for cross-validation stability (low variance), not the single highest raw score on one split.",
        "bn": "লজিস্টিক রিগ্রেশন লাইভ সার্চ ট্যাব চালায়। এটি ক্রস-ভ্যালিডেশন স্থিতিশীলতার জন্য নির্বাচিত।",
        "ar": "الانحدار اللوجستي يشغّل تبويب البحث المباشر. اختير لاستقرار التحقق المتقاطع (انخفاض التباين).",
    },
}


def t(key: str, lang: str) -> str:
    """Return the UI string for the given key and language; fall back to English."""
    return UI_STRINGS.get(key, {}).get(lang) or UI_STRINGS.get(key, {}).get("en", key)


def tier1_display(tier1: str, lang: str) -> str:
    if lang == "bn":
        return TIER1_CLASS_BN.get(tier1, tier1.replace("_", " "))
    if lang == "ar":
        return TIER1_CLASS_AR.get(tier1, tier1.replace("_", " "))
    return tier1.replace("_", " ")


def topic_display(topic: str, lang: str) -> str:
    if lang == "bn":
        return TOPIC_BN.get(topic, topic.replace("_", " "))
    if lang == "ar":
        return TOPIC_AR.get(topic, topic.replace("_", " "))
    return topic.replace("_", " ")


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
    """Raised for any dataset load/shape problem."""


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise DatasetError(
            f"Couldn't find '{path}'. Place dataset.csv in the same folder as app.py."
        )
    except Exception as exc:
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
    # Optional Arabic columns — never required
    for col in ("reference_text_ar", "question_ar"):
        if col not in df.columns:
            df[col] = ""
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
    df["combined_text_raw"] = (
        df["question_bn"] + " " + df["question_en"] + " " + df["search_keywords"]
    )
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
    """Fit TF-IDF + Logistic Regression once at startup.

    Character n-grams handle Bangla morphology and spelling variation.
    Logistic Regression is chosen for cross-validation stability (see main.py
    bias-variance section), not the single highest raw accuracy on one split.
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
# RETRIEVAL PIPELINE (unchanged order)
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


def retrieve_candidates(
    query: str, df: pd.DataFrame, vectorizer, X, banglish_map: dict[str, set[str]]
) -> RetrievalResult:
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
# ML INSIGHTS (Plotly)
# --------------------------------------------------------------------------- #

def _cv_k(y: pd.Series, target: int = 5) -> int:
    return min(target, int(y.value_counts().min()))


def _evaluate_classifier(model, X, y: pd.Series, cv_target: int = 5) -> dict:
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
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
        "KNN": KNeighborsClassifier(n_neighbors=knn_k),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }


@st.cache_resource(show_spinner="Loading model insights...")
def _get_insights_data(df: pd.DataFrame):
    df = build_combined_text(df)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=20000)
    X = vectorizer.fit_transform(df["combined_text_clean"])
    y = df["tier1_class"]
    knn_k = max(1, min(5, len(y) - 1))

    classifiers = _get_five_classifiers(knn_k)
    results = {}
    for name, model in classifiers.items():
        try:
            results[name] = _evaluate_classifier(model, X, y)
        except Exception as exc:
            results[name] = {"error": str(exc)}

    k = df["topic"].nunique()
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    ari = adjusted_rand_score(df["topic"], cluster_labels)
    crosstab = pd.crosstab(
        pd.Series(cluster_labels, name="cluster"),
        df["topic"].reset_index(drop=True),
    )

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


def render_model_insights_tab(
    df: pd.DataFrame, vectorizer_search, X_search, banglish_map, classifier_search, lang: str
) -> None:
    st.markdown(f"#### {t('mode_lab', lang)}")
    st.caption(t("lab_intro", lang))

    insights = _get_insights_data(df)
    results = insights["results"]
    kmeans_data = insights["kmeans"]
    coords = kmeans_data["coords"]

    # ---- Model comparison (Plotly horizontal bar, sorted) ----
    st.markdown("---")
    st.markdown("##### Which model best predicts an unseen ruling’s category?")
    model_names, f1_vals = [], []
    for name, r in results.items():
        if "error" not in r:
            model_names.append(name)
            f1_vals.append(r["train_macro_f1"])

    if model_names:
        order = np.argsort(f1_vals)[::-1]
        model_names = [model_names[i] for i in order]
        f1_vals = [f1_vals[i] for i in order]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=model_names, x=f1_vals, orientation="h",
            name="Macro-F1", marker_color=PLOTLY_SAFE[0],
            hovertemplate="%{y}<br>Macro-F1: %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(
            title="Model comparison — higher is better",
            xaxis_title="Macro-averaged F1 score",
            yaxis_title="Classifier",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor=COLOR_PAPER,
            plot_bgcolor=COLOR_PAPER,
            font=dict(color=COLOR_INK),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(t("why_logreg", lang))

        # Confusion matrix
        st.markdown("##### Where the model confuses one ruling category for another")
        selected = st.selectbox(
            "Model for confusion matrix",
            model_names,
            index=0,
            key="cm_model",
        )
        r = results.get(selected, {})
        if "confusion_matrix" in r:
            cm = r["confusion_matrix"]
            classes = [c.replace("_", " ") for c in r["classes"]]
            fig_cm = px.imshow(
                cm,
                x=classes, y=classes,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="True", color="Count"),
                text_auto=True,
                aspect="auto",
            )
            fig_cm.update_layout(
                title=f"Confusion matrix — {selected}",
                paper_bgcolor=COLOR_PAPER,
                plot_bgcolor=COLOR_PAPER,
                font=dict(color=COLOR_INK),
                height=420,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            if r.get("cv_possible"):
                st.caption(
                    f"Cross-validation ({r['cv_k']}-fold) macro-F1: "
                    f"{r['cv_mean_macro_f1']:.3f} ± {r['cv_std_macro_f1']:.3f}. "
                    "Small classes remain provisional until more examples are available."
                )
            else:
                st.caption(
                    "Cross-validation not yet reliable (at least one class has too few examples). "
                    "Treat these numbers as illustrative only."
                )

    # ---- K-Means ----
    st.markdown("---")
    st.markdown("##### Do unsupervised clusters align with human topic labels?")
    fig_k = go.Figure()
    unique_clusters = sorted(set(kmeans_data["cluster_labels"]))
    for i, c in enumerate(unique_clusters):
        mask = kmeans_data["cluster_labels"] == c
        fig_k.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers", name=f"Cluster {c}",
            marker=dict(size=8, color=PLOTLY_SAFE[i % len(PLOTLY_SAFE)]),
            hovertemplate="Cluster %{text}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
            text=[str(c)] * int(mask.sum()),
        ))
    fig_k.update_layout(
        title="K-Means clusters in PCA space",
        xaxis_title="Principal component 1",
        yaxis_title="Principal component 2",
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_PAPER,
        font=dict(color=COLOR_INK),
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_k, use_container_width=True)

    ari = kmeans_data["ari"]
    ari_pct = max(0, (ari + 1) / 2 * 100)
    st.caption(
        f"Adjusted Rand Index (ARI): {ari:.3f} — the algorithm’s own groupings matched the human "
        f"topic labels about {ari_pct:.0f}% of the way (1.0 = perfect, 0.0 = random)."
    )
    with st.expander("Cluster vs. topic cross-tabulation"):
        st.dataframe(kmeans_data["crosstab"], use_container_width=True)

    # ---- PCA ----
    st.markdown("---")
    st.markdown("##### How the rulings sit in a simplified 2-D view of the text space")
    fig_pca = go.Figure()
    unique_classes = sorted(insights["tier1_class"].unique())
    for i, c in enumerate(unique_classes):
        mask = (insights["tier1_class"] == c).to_numpy()
        fig_pca.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers", name=c.replace("_", " "),
            marker=dict(size=9, color=PLOTLY_SAFE[i % len(PLOTLY_SAFE)]),
            hovertemplate="%{text}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
            text=[c.replace("_", " ")] * int(mask.sum()),
        ))
    fig_pca.update_layout(
        title=f"PCA projection — {kmeans_data['explained']:.1f}% variance explained by these two axes",
        xaxis_title="Principal component 1",
        yaxis_title="Principal component 2",
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_PAPER,
        font=dict(color=COLOR_INK),
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    st.caption(
        f"{kmeans_data['explained']:.1f}% of the total variance in the TF-IDF vectors is captured "
        "by these two dimensions — the rest is spread across dimensions we cannot easily draw."
    )

    # ---- Live Try-It ----
    st.markdown("---")
    st.markdown("##### Live Try-It")
    live_query = st.text_input(
        t("type_question", lang),
        placeholder=t("search_placeholder", lang),
        key="live_try_input",
    )
    if live_query.strip():
        result = retrieve_candidates(live_query, df, vectorizer_search, X_search, banglish_map)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Retrieval Result**")
            if result.row is not None:
                stage_en, stage_bn, stage_ar = STAGE_LABELS.get(
                    result.stage, (result.stage, result.stage, result.stage)
                )
                stage = stage_en if lang == "en" else (stage_bn if lang == "bn" else stage_ar)
                st.markdown(f"Matched via: **{stage}**")
                st.markdown(f"Score: **{result.similarity:.3f}**")
                st.markdown(f"Ruling: **{result.row['tier1_class'].replace('_', ' ')}**")
            else:
                st.markdown(t("no_match_heading", lang))
        with col2:
            st.markdown("**All 5 Classifiers’ Predictions**")
            preds = {}
            for name, r in results.items():
                if "error" not in r:
                    try:
                        q_clean = clean_text(live_query)
                        if q_clean.strip():
                            q_vec = insights["vectorizer"].transform([q_clean])
                            preds[name] = r["model"].predict(q_vec)[0]
                    except Exception:
                        preds[name] = None
            for name, pred in preds.items():
                if pred:
                    st.markdown(f"{name}: **{pred.replace('_', ' ')}**")
                else:
                    st.markdown(f"{name}: _unavailable_")


# --------------------------------------------------------------------------- #
# UI RENDERING
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


def _citation_plain(row: pd.Series) -> str:
    parts = []
    ref_ar = str(row.get("reference_text_ar", "")).strip()
    if ref_ar:
        parts.append(ref_ar)
    ref_text = str(row.get("reference_text", "")).strip()
    if ref_text and ref_text != PLACEHOLDER_REFERENCE_TEXT:
        parts.append(ref_text)
    ref_source = str(row.get("reference_source", "")).strip()
    if ref_source:
        parts.append(f"— {ref_source}")
    return "\n".join(parts) if parts else ref_source


def render_result(
    result: RetrievalResult,
    df: pd.DataFrame,
    vectorizer,
    classifier,
    lang: str,
) -> None:
    row = result.row
    rid = int(row["id"])

    st.markdown('<div class="entry">', unsafe_allow_html=True)
    st.markdown(f'<div class="entry-refno">Ref. No. {rid:04d}</div>', unsafe_allow_html=True)

    q_bn = str(row.get("question_bn", "")).strip()
    q_en = str(row.get("question_en", "")).strip()
    q_ar = str(row.get("question_ar", "")).strip()
    if lang == "ar" and q_ar:
        st.markdown(f'<div class="entry-question ar-text" dir="rtl">{q_ar}</div>', unsafe_allow_html=True)
    elif lang == "bn" and q_bn:
        st.markdown(f'<div class="entry-question">{q_bn}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="entry-question">{q_en or q_bn}</div>', unsafe_allow_html=True)
        if q_bn and q_en and lang == "en":
            st.markdown(f'<div class="entry-question-en">{q_bn}</div>', unsafe_allow_html=True)

    tier1 = str(row["tier1_class"])
    label = tier1_display(tier1, lang)
    strictness = str(row.get("strictness_label", "")).strip()
    label_text = f"{label} — {strictness}" if strictness else label
    st.markdown(
        f'<div class="category-label"><span class="category-mark">{SECTION_MARK}</span>{label_text}</div>',
        unsafe_allow_html=True,
    )

    if str(row.get("verification_status", "")).strip() == NEEDS_VERIFICATION_LABEL:
        st.markdown(
            f'<div class="verification-flag">{t("unverified", lang)}</div>',
            unsafe_allow_html=True,
        )

    explanation = _explanation_for(row)
    if explanation:
        st.markdown(f'<div class="explanation-text">{explanation}</div>', unsafe_allow_html=True)

    ref_ar = str(row.get("reference_text_ar", "")).strip()
    ref_text = str(row.get("reference_text", "")).strip()
    ref_source = str(row.get("reference_source", "")).strip()

    citation_html = '<div class="citation-block">'
    if ref_ar:
        citation_html += f'<div class="ar-text" dir="rtl">{ref_ar}</div>'
    if ref_text and ref_text != PLACEHOLDER_REFERENCE_TEXT:
        citation_html += f"{ref_text}"
    if ref_source:
        citation_html += f'<span class="citation-source">{ref_source}</span>'
    citation_html += "</div>"
    st.markdown(citation_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        bookmarks = st.session_state.setdefault("bookmarks", set())
        is_saved = rid in bookmarks
        if st.button(
            t("unsave_session", lang) if is_saved else t("save_session", lang),
            key=f"save_{rid}",
            type="secondary",
        ):
            if is_saved:
                bookmarks.discard(rid)
            else:
                bookmarks.add(rid)
            st.session_state["bookmarks"] = bookmarks
            st.rerun()
    with col_b:
        citation = _citation_plain(row)
        if st.button(t("copy_citation", lang), key=f"copy_{rid}", type="secondary"):
            st.session_state["last_copied"] = citation
            st.toast("Citation ready — select & copy from the box below")
        if st.session_state.get("last_copied") == citation:
            st.code(citation, language=None)

    related = df[(df["topic"] == row["topic"]) & (df["id"] != row["id"])].head(TOP_K_RELATED)
    if not related.empty:
        st.markdown(f'<div class="related-heading">{t("related_heading", lang)}</div>', unsafe_allow_html=True)
        for i, r in related.iterrows():
            label_r = tier1_display(str(r["tier1_class"]), lang)
            display_text = r["question_en"] or r["question_bn"]
            if st.button(f"{label_r} — {display_text}", key=f"related_{rid}_{i}", type="secondary"):
                _set_pending_query(str(r["question_en"] or r["question_bn"]))

    with st.expander(t("apparatus", lang)):
        stage_en, stage_bn, stage_ar = STAGE_LABELS.get(
            result.stage, (result.stage, result.stage, result.stage)
        )
        stage = stage_en if lang == "en" else (stage_bn if lang == "bn" else stage_ar)
        st.write(f"**{stage}** · {result.similarity:.2f}")
        predicted = classify_query(str(row["question_en"] or row["question_bn"]), vectorizer, classifier)
        if predicted is not None:
            agree = "agrees" if predicted == row["tier1_class"] else "disagrees"
            st.write(f"Classifier prediction: **{predicted.replace('_', ' ')}** ({agree})")
        else:
            st.write("Classifier confirmation unavailable.")


def render_no_match(result: RetrievalResult, lang: str) -> None:
    st.markdown('<div class="no-match-box">', unsafe_allow_html=True)
    st.markdown(f'<div class="no-match-heading">{t("no_match_heading", lang)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="no-match-body">{t("no_match_body", lang)}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if result.suggestions:
        st.markdown(f'<div class="related-heading">{t("closest_heading", lang)}</div>', unsafe_allow_html=True)
        for i, r in enumerate(result.suggestions):
            label = tier1_display(str(r["tier1_class"]), lang)
            display_text = r["question_en"] or r["question_bn"]
            if st.button(f"{label} — {display_text}", key=f"suggestion_{i}", type="secondary"):
                _set_pending_query(str(r["question_en"] or r["question_bn"]))


def render_browse_mode(df: pd.DataFrame, lang: str) -> None:
    topic_options = [t("all_option", lang)] + [
        topic_display(tpc, lang) for tpc in sorted(df["topic"].unique().tolist())
    ]
    class_options = [t("all_option", lang)] + [
        tier1_display(c, lang) for c in sorted(df["tier1_class"].unique().tolist())
    ]
    arabic_options = [t("all_option", lang), t("has_arabic_filter", lang)]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="field-label">{t("topic_label", lang)}</div>', unsafe_allow_html=True)
        topic_choice = st.selectbox("Topic", topic_options, label_visibility="collapsed", key="browse_topic")
    with col2:
        st.markdown(f'<div class="field-label">{t("category_label", lang)}</div>', unsafe_allow_html=True)
        class_choice = st.selectbox("Category", class_options, label_visibility="collapsed", key="browse_class")
    with col3:
        st.markdown(f'<div class="field-label">{t("has_arabic_filter", lang)}</div>', unsafe_allow_html=True)
        arabic_choice = st.selectbox("Arabic", arabic_options, label_visibility="collapsed", key="browse_ar")

    filtered = df.copy()
    if topic_choice != t("all_option", lang):
        for tpc in df["topic"].unique():
            if topic_display(tpc, lang) == topic_choice:
                filtered = filtered[filtered["topic"] == tpc]
                break
    if class_choice != t("all_option", lang):
        for c in df["tier1_class"].unique():
            if tier1_display(c, lang) == class_choice:
                filtered = filtered[filtered["tier1_class"] == c]
                break
    if arabic_choice == t("has_arabic_filter", lang):
        filtered = filtered[
            filtered["reference_text_ar"].fillna("").astype(str).str.strip() != ""
        ]

    st.markdown(
        f'<div class="browse-count">{len(filtered)} {t("browse_count", lang)}</div>',
        unsafe_allow_html=True,
    )

    for _, row in filtered.iterrows():
        st.markdown('<div class="entry">', unsafe_allow_html=True)
        st.markdown(f'<div class="entry-refno">Ref. No. {int(row["id"]):04d}</div>', unsafe_allow_html=True)
        display_q = row["question_en"] or row["question_bn"]
        if lang == "bn" and str(row.get("question_bn", "")).strip():
            display_q = row["question_bn"]
        elif lang == "ar" and str(row.get("question_ar", "")).strip():
            display_q = row["question_ar"]
            st.markdown(f'<div class="entry-question ar-text" dir="rtl">{display_q}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="entry-question">{display_q}</div>', unsafe_allow_html=True)
        tier1 = str(row["tier1_class"])
        label = tier1_display(tier1, lang)
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


def render_header(lang: str) -> None:
    logo_path = "assets/hstu_logo.png"
    if os.path.exists(logo_path):
        try:
            import base64
            with open(logo_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.15rem;">
                    <img src="data:image/png;base64,{b64}" style="height:52px;width:auto;object-fit:contain;" />
                    <div>
                        <div style="font-family:'Noto Sans Bengali','Inter',sans-serif;font-size:0.78rem;color:#6B6558;letter-spacing:0.04em;line-height:1.3;">
                            Hajee Mohammad Danesh Science and Technology University
                        </div>
                        <div style="font-family:'Noto Sans Bengali','Inter',sans-serif;font-size:0.72rem;color:#6B6558;letter-spacing:0.08em;text-transform:uppercase;">
                            CSE 469 — Machine Learning and Pattern Recognition
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return
        except Exception:
            pass
    st.markdown(
        """
        <div style="margin-bottom:0.15rem;">
            <div style="font-family:'Noto Sans Bengali','Inter',sans-serif;font-size:0.78rem;color:#6B6558;letter-spacing:0.04em;line-height:1.3;">
                Hajee Mohammad Danesh Science and Technology University
            </div>
            <div style="font-family:'Noto Sans Bengali','Inter',sans-serif;font-size:0.72rem;color:#6B6558;letter-spacing:0.08em;text-transform:uppercase;">
                CSE 469 — Machine Learning and Pattern Recognition
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_css(font_size: str = "normal") -> None:
    size_map = {"small": "0.92", "normal": "1.0", "large": "1.12"}
    scale = size_map.get(font_size, "1.0")
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Inter:wght@400;500;600&family=Noto+Sans+Bengali:wght@400;500;600&family=Noto+Serif+Bengali:wght@400;600&family=Tiro+Bangla&display=swap');

        :root {{
            --paper: {COLOR_PAPER};
            --card: {COLOR_CARD};
            --ink: {COLOR_INK};
            --ink-muted: {COLOR_INK_MUTED};
            --accent: {COLOR_ACCENT};
            --hairline: {COLOR_HAIRLINE};
            --scale: {scale};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', 'Noto Sans Bengali', sans-serif;
            background-color: var(--paper) !important;
            color: var(--ink);
            font-size: calc(16px * var(--scale));
        }}

        .app-title {{
            font-family: 'Noto Serif Bengali', 'Tiro Bangla', Georgia, serif;
            font-size: calc(1.85rem * var(--scale));
            color: var(--ink);
            margin: 0.2rem 0 0.35rem 0;
            letter-spacing: -0.01em;
        }}
        .app-eyebrow {{
            font-size: calc(0.78rem * var(--scale));
            color: var(--ink-muted);
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}
        .app-subtitle {{
            font-size: calc(0.95rem * var(--scale));
            color: var(--ink-muted);
            line-height: 1.45;
            margin-bottom: 0.6rem;
        }}
        .app-rule {{
            border: none;
            border-top: 1px solid var(--hairline);
            margin: 0.6rem 0 1rem 0;
        }}
        .app-footer {{
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid var(--hairline);
            font-size: calc(0.82rem * var(--scale));
            color: var(--ink-muted);
            line-height: 1.5;
        }}

        .entry {{
            background: var(--card);
            border: 1px solid var(--hairline);
            border-left: 3px solid var(--accent);
            padding: 1rem 1.15rem;
            margin: 0.85rem 0;
            border-radius: 2px;
        }}
        .entry-refno {{
            font-size: calc(0.72rem * var(--scale));
            color: var(--ink-muted);
            letter-spacing: 0.04em;
            margin-bottom: 0.25rem;
        }}
        .entry-question {{
            font-family: 'Noto Serif Bengali', 'Tiro Bangla', Georgia, serif;
            font-size: calc(1.15rem * var(--scale));
            color: var(--ink);
            line-height: 1.4;
            margin-bottom: 0.2rem;
        }}
        .entry-question-en {{
            font-size: calc(0.92rem * var(--scale));
            color: var(--ink-muted);
            margin-bottom: 0.35rem;
        }}
        .category-label {{
            font-size: calc(0.88rem * var(--scale));
            color: var(--accent);
            letter-spacing: 0.02em;
            margin: 0.35rem 0;
        }}
        .category-mark {{
            margin-right: 0.35rem;
            opacity: 0.85;
        }}
        .explanation-text {{
            font-size: calc(0.95rem * var(--scale));
            line-height: 1.55;
            color: var(--ink);
            margin: 0.5rem 0;
        }}
        .citation-block {{
            background: var(--paper);
            border: 1px solid var(--hairline);
            padding: 0.7rem 0.85rem;
            margin-top: 0.6rem;
            font-size: calc(0.9rem * var(--scale));
            line-height: 1.5;
            color: var(--ink);
        }}
        .citation-source {{
            display: block;
            margin-top: 0.35rem;
            font-size: calc(0.8rem * var(--scale));
            color: var(--ink-muted);
            font-style: italic;
        }}
        .ar-text {{
            font-family: 'Amiri', 'Noto Naskh Arabic', serif !important;
            font-size: calc(1.2rem * var(--scale)) !important;
            line-height: 1.7 !important;
            direction: rtl;
            text-align: right;
            margin-bottom: 0.4rem;
        }}
        .verification-flag {{
            background: {COLOR_FLAG_BG};
            color: {COLOR_FLAG_INK};
            font-size: calc(0.8rem * var(--scale));
            padding: 0.3rem 0.55rem;
            margin: 0.4rem 0;
            border-radius: 2px;
        }}
        .related-heading, .field-label, .browse-count {{
            font-size: calc(0.82rem * var(--scale));
            color: var(--ink-muted);
            letter-spacing: 0.04em;
            margin: 0.7rem 0 0.35rem 0;
        }}
        .no-match-box {{
            background: var(--card);
            border: 1px solid var(--hairline);
            padding: 1.1rem;
            margin: 1rem 0;
        }}
        .no-match-heading {{
            font-family: 'Noto Serif Bengali', Georgia, serif;
            font-size: calc(1.1rem * var(--scale));
            color: var(--ink);
            margin-bottom: 0.35rem;
        }}
        .no-match-body {{
            font-size: calc(0.92rem * var(--scale));
            color: var(--ink-muted);
        }}
        .bn-inline {{ color: var(--ink-muted); font-size: 0.92em; }}

        @media print {{
            header, .stRadio, .stSelectbox, .stButton, .stTextInput,
            [data-testid="stSidebar"], .app-eyebrow, .app-subtitle,
            .app-rule, .related-heading, .field-label {{
                display: none !important;
            }}
            .entry {{
                border: none;
                border-left: 2px solid #333;
                page-break-inside: avoid;
            }}
            body {{ background: white !important; color: black !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(page_title="Ruling Reference", layout="centered")

    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
    if "font_size" not in st.session_state:
        st.session_state["font_size"] = "normal"
    if "bookmarks" not in st.session_state:
        st.session_state["bookmarks"] = set()
    if "history" not in st.session_state:
        st.session_state["history"] = []

    lang = st.session_state["lang"]
    font_size = st.session_state["font_size"]
    inject_css(font_size)

    render_header(lang)

    ctrl1, ctrl2 = st.columns([2.5, 1.5])
    with ctrl1:
        lang_choice = st.radio(
            "Language",
            options=["en", "bn", "ar"],
            format_func=lambda x: {"en": "English", "bn": "বাংলা", "ar": "العربية"}[x],
            horizontal=True,
            index=["en", "bn", "ar"].index(lang),
            label_visibility="collapsed",
            key="lang_radio",
        )
        if lang_choice != lang:
            st.session_state["lang"] = lang_choice
            st.rerun()
    with ctrl2:
        fs = st.radio(
            t("font_size", lang),
            options=["small", "normal", "large"],
            format_func=lambda x: t(f"font_{x}", lang),
            horizontal=True,
            index=["small", "normal", "large"].index(font_size),
            label_visibility="collapsed",
            key="fs_radio",
        )
        if fs != font_size:
            st.session_state["font_size"] = fs
            st.rerun()

    st.markdown(f'<div class="app-eyebrow">{t("app_eyebrow", lang)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-title">{t("app_title", lang)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-subtitle">{t("app_subtitle", lang)}</div>', unsafe_allow_html=True)
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

    mode_labels = [t("mode_search", lang), t("mode_browse", lang), t("mode_lab", lang)]
    mode = st.radio(
        "Mode",
        mode_labels,
        horizontal=True,
        label_visibility="collapsed",
        key="mode_radio",
    )

    bookmarks = st.session_state.get("bookmarks", set())
    if bookmarks:
        with st.expander(t("saved_panel", lang)):
            for bid in sorted(bookmarks):
                match = df[df["id"] == bid]
                if not match.empty:
                    r = match.iloc[0]
                    q = r["question_en"] or r["question_bn"]
                    if st.button(f"#{bid:04d} — {q[:60]}", key=f"bm_{bid}"):
                        _set_pending_query(str(q))

    history = st.session_state.get("history", [])
    if history and mode == t("mode_search", lang):
        with st.expander(t("history_heading", lang)):
            for i, h in enumerate(history[:MAX_HISTORY]):
                if st.button(h, key=f"hist_{i}"):
                    _set_pending_query(h)

    if mode == t("mode_search", lang):
        st.markdown(f'<div class="field-label">{t("type_question", lang)}</div>', unsafe_allow_html=True)
        default_query = st.session_state.pop("pending_query", "")
        query = st.text_input(
            "Search",
            value=default_query,
            placeholder=t("search_placeholder", lang),
            label_visibility="collapsed",
            key="main_search",
        )
        if query.strip():
            hist = st.session_state.get("history", [])
            if not hist or hist[0] != query.strip():
                hist = [query.strip()] + [h for h in hist if h != query.strip()]
                st.session_state["history"] = hist[:MAX_HISTORY]

            result = retrieve_candidates(query, df, vectorizer, X, banglish_map)
            if result.row is not None:
                render_result(result, df, vectorizer, classifier, lang)
            else:
                render_no_match(result, lang)

    elif mode == t("mode_lab", lang):
        render_model_insights_tab(df, vectorizer, X, banglish_map, classifier, lang)
    else:
        render_browse_mode(df, lang)

    st.markdown(f'<div class="app-footer">{t("footer", lang)}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
