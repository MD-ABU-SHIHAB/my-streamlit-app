"""HIKMA — Islamic Knowledge AI

A single-file Streamlit app: the matching engine (the HIKMA class) and the
UI live together here so the whole project only needs this file, a
requirements.txt, and dataset.csv.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
import re

import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = [
    "id", "topic", "keywords", "category", "answer_bangla",
    "answer_english", "arabic_text", "arabic_bangla", "sources", "confidence",
]


# ===========================================================================
# Matching engine
# ===========================================================================
class HIKMA:
    def __init__(self, dataset_path: str):
        """Load the dataset and build the lookup structures used at query time."""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(
                f"'{dataset_path}' was not found next to app.py. "
                "Make sure dataset.csv is committed and pushed to the repo "
                "(check it isn't excluded by .gitignore)."
            )
        if path.stat().st_size == 0:
            raise ValueError(
                f"'{dataset_path}' exists but is empty (0 bytes). "
                "This usually means the file wasn't fully pushed to GitHub — "
                "re-add and commit it, and confirm it isn't a Git LFS pointer "
                "if the real file is large."
            )

        self.df = pd.read_csv(dataset_path)
        if self.df.empty:
            raise ValueError(
                f"'{dataset_path}' loaded but contains no rows — "
                "only a header, or the wrong file was committed."
            )
        self._validate_columns()
        self._clean_data()

        # Plain Python lists — touching these is far cheaper per-row than
        # DataFrame/iterrows access, since there's no Series construction.
        self._topics_lower = self.df["topic"].astype(str).str.lower().tolist()
        self._answers_lower = self.df["answer_bangla"].astype(str).str.lower().tolist()

        self.all_keywords: set[str] = set()
        self.keyword_index: dict[str, list[int]] = defaultdict(list)
        self._build_keyword_index()

    # ---- setup ----
    def _validate_columns(self) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"dataset.csv is missing required column(s): {', '.join(missing)}"
            )

    def _clean_data(self) -> None:
        """Fill missing text fields so downstream code never has to check for NaN."""
        text_cols = ["arabic_text", "arabic_bangla", "answer_english", "sources", "category"]
        for col in text_cols:
            self.df[col] = self.df[col].fillna("")
        self.df["confidence"] = self.df["confidence"].fillna("medium")

    def _build_keyword_index(self) -> None:
        """Map every keyword variant -> the row indices it appears in.

        This is the one place we still touch every row, but it happens once
        at startup rather than once per question — important at 60k+ rows.
        """
        for idx, keywords_str in enumerate(self.df["keywords"]):
            if pd.isna(keywords_str):
                continue
            for kw in str(keywords_str).split(","):
                kw = kw.strip().lower()
                if not kw:
                    continue
                self.all_keywords.add(kw)
                self.keyword_index[kw].append(idx)

    # ---- query processing ----
    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower().strip()
        return re.sub(r"\s+", " ", text)

    def _extract_keywords(self, query: str) -> list[str]:
        """Find which known keywords appear in (or closely resemble) the query."""
        found = [kw for kw in self.all_keywords if kw in query]

        if not found:
            for word in query.split():
                if len(word) > 2:
                    found.extend(get_close_matches(word, self.all_keywords, n=3, cutoff=0.6))

        return found

    def _score_by_keywords(self, query: str, keywords: list[str]) -> Counter:
        """Tally candidate rows using only the (small) set of rows each keyword maps to."""
        scores: Counter = Counter()
        for kw in keywords:
            for idx in self.keyword_index.get(kw, ()):
                scores[idx] += 1

        for idx, topic in enumerate(self._topics_lower):
            if topic and topic in query:
                scores[idx] += 3

        return scores

    def _score_by_free_text(self, query: str) -> Counter:
        """Fallback when no keyword matched at all: raw word overlap with the
        topic name (weighted higher) and the Bangla answer text."""
        scores: Counter = Counter()
        query_words = [w for w in query.split() if len(w) > 2]
        if not query_words:
            return scores

        for idx, (topic, answer) in enumerate(zip(self._topics_lower, self._answers_lower)):
            score = 0
            for word in query_words:
                if word in topic:
                    score += 3
                if word in answer:
                    score += 1
            if score:
                scores[idx] = score

        return scores

    def _find_best_match(self, query: str) -> tuple[pd.Series | None, int]:
        query = self._normalize_text(query)
        keywords = self._extract_keywords(query)

        scores = self._score_by_keywords(query, keywords) if keywords else Counter()
        if not scores:
            scores = self._score_by_free_text(query)

        if not scores:
            return None, 0

        best_idx, best_score = scores.most_common(1)[0]
        return self.df.iloc[best_idx], best_score

    # ---- public API ----
    @lru_cache(maxsize=512)
    def _ask_cached(self, normalized_query: str) -> dict:
        match, score = self._find_best_match(normalized_query)

        if match is None:
            return {
                "bangla": "আমি এই প্রশ্নের উত্তর জানি না। অন্য প্রশ্ন জিজ্ঞাসা করুন।",
                "english": "I don't know the answer to this question. Please ask another question.",
                "arabic": "",
                "arabic_bangla": "",
                "source": "",
                "category": "",
                "topic": "",
                "confidence": 0.1,
            }

        confidence_label = str(match["confidence"]).strip().lower()
        base_confidence = {"high": 0.9, "medium": 0.7}.get(confidence_label, 0.5)
        if score <= 1:
            base_confidence = min(base_confidence, 0.5)

        return {
            "bangla": match["answer_bangla"],
            "english": match["answer_english"],
            "arabic": match["arabic_text"],
            "arabic_bangla": match["arabic_bangla"],
            "source": match["sources"],
            "category": match["category"],
            "topic": match["topic"],
            "confidence": base_confidence,
        }

    def ask(self, query: str) -> dict:
        """Process a query and return a response dict with keys: bangla,
        english, arabic, arabic_bangla, source, category, topic, confidence."""
        if not query or not query.strip():
            return {
                "bangla": "দয়া করে একটি প্রশ্ন লিখুন।",
                "english": "Please write a question.",
                "confidence": 0.0,
            }

        return self._ask_cached(self._normalize_text(query))


# ===========================================================================
# Streamlit UI
# ===========================================================================
st.set_page_config(
    page_title="HIKMA — Islamic Knowledge AI",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Tiro+Bangla&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --parchment: #F7F1E1;
    --parchment-dim: #EFE6CC;
    --ink: #1E2A28;
    --teal-deep: #0B3D3B;
    --teal: #145C56;
    --teal-light: #2F7A72;
    --gold: #C89B3C;
    --gold-light: #E4C776;
    --brick: #A64B3C;
    --muted: #6B7573;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--parchment); }

.hikma-header {
    position: relative;
    background:
        linear-gradient(45deg, rgba(200,155,60,0.10) 25%, transparent 25%, transparent 75%, rgba(200,155,60,0.10) 75%),
        linear-gradient(45deg, rgba(200,155,60,0.10) 25%, transparent 25%, transparent 75%, rgba(200,155,60,0.10) 75%),
        linear-gradient(135deg, var(--teal-deep) 0%, var(--teal) 100%);
    background-size: 26px 26px, 26px 26px, 100% 100%;
    background-position: 0 0, 13px 13px, 0 0;
    padding: 2.4rem 2rem 2rem;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 0.6rem;
    box-shadow: 0 8px 24px rgba(11,61,59,0.25);
}
.hikma-header h1 {
    font-family: 'Amiri', serif;
    font-weight: 700;
    font-size: 3rem;
    color: var(--gold-light);
    margin: 0;
    letter-spacing: 0.04em;
}
.hikma-header p {
    font-family: 'Tiro Bangla', serif;
    font-size: 1.15rem;
    color: #F2EAD3;
    opacity: 0.95;
    margin-top: 0.4rem;
}
.hikma-rule {
    height: 3px;
    margin: 0 auto 1.6rem;
    max-width: 420px;
    background: linear-gradient(90deg, transparent, var(--gold) 20%, var(--gold-light) 50%, var(--gold) 80%, transparent);
    border-radius: 2px;
}

.chip-label {
    font-family: 'Tiro Bangla', serif;
    color: var(--teal-deep);
    font-size: 1.05rem;
    margin-bottom: 0.6rem;
}
.stButton > button {
    border-radius: 999px !important;
    border: 1.5px solid var(--teal-light) !important;
    background: var(--parchment) !important;
    color: var(--teal-deep) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}
.stButton > button:hover {
    background: var(--teal-deep) !important;
    color: var(--gold-light) !important;
    border-color: var(--teal-deep) !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(11,61,59,0.22);
}
.stButton > button:focus-visible {
    outline: 3px solid var(--gold) !important;
    outline-offset: 2px;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
.msg-row { animation: fadeInUp 0.35s ease; margin: 0.7rem 0; }

.user-msg {
    background: var(--teal-deep);
    color: #F7F1E1;
    padding: 0.9rem 1.2rem;
    border-radius: 20px 20px 4px 20px;
    max-width: 78%;
    margin-left: auto;
    box-shadow: 0 3px 10px rgba(11,61,59,0.18);
}
.user-msg strong { color: var(--gold-light); }

.assistant-card {
    background: #FFFDF6;
    border: 1px solid var(--parchment-dim);
    border-top: 4px solid var(--gold);
    border-radius: 6px 22px 22px 22px;
    padding: 1.1rem 1.3rem;
    max-width: 88%;
    box-shadow: 0 4px 14px rgba(30,42,40,0.08);
}
.assistant-card .badge {
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.78rem;
    color: var(--teal-deep);
    background: var(--parchment-dim);
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    margin-bottom: 0.5rem;
    margin-right: 0.4rem;
}
.assistant-card .answer-bn {
    font-family: 'Tiro Bangla', serif;
    font-size: 1.15rem;
    line-height: 1.75;
    color: var(--ink);
}

.conf-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    margin-top: 0.7rem;
}
.conf-high  { background: rgba(11,61,59,0.10); color: var(--teal-deep); }
.conf-med   { background: rgba(200,155,60,0.18); color: #8A6A22; }
.conf-low   { background: rgba(166,75,60,0.14); color: var(--brick); }

.ayah-frame {
    direction: rtl;
    text-align: center;
    font-family: 'Amiri', serif;
    font-size: 1.5rem;
    color: var(--teal-deep);
    background: linear-gradient(180deg, #FBF6E8, #F3E9CC);
    border: 1px solid var(--gold);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    line-height: 2.1;
}
.ayah-frame .bracket { color: var(--gold); font-size: 1.7rem; }
.translation-box {
    font-family: 'Tiro Bangla', serif;
    background: var(--parchment-dim);
    border-left: 3px solid var(--teal-light);
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
    margin-top: 0.4rem;
    color: var(--ink);
}

section[data-testid="stSidebar"] {
    background: var(--teal-deep);
    color: var(--parchment);
}
section[data-testid="stSidebar"] * { color: var(--parchment) !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(247,241,225,0.2); }

.hikma-footer {
    text-align: center;
    color: var(--muted);
    padding: 1.6rem;
    font-family: 'Tiro Bangla', serif;
    font-size: 0.9rem;
    border-top: 1px solid var(--parchment-dim);
    margin-top: 1.5rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---- session state & data ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

if "hikma" not in st.session_state:
    try:
        st.session_state.hikma = HIKMA("dataset.csv")
    except Exception as e:
        st.error(f"⚠️ dataset.csv লোড করা যায়নি:\n\n**{e}**")
        st.stop()

try:
    df = st.session_state.hikma.df
    dataset_loaded = True
except Exception:
    df = None
    dataset_loaded = False

SUGGESTED_TOPICS = ["নামাজ", "রোজা", "যাকাত", "হজ্জ", "ওযু", "সুদ", "কুরবানি", "তাহাজ্জুদ"]


def handle_query(query_text: str) -> None:
    """Ask HIKMA a question and append both turns to the chat history."""
    st.session_state.messages.append({"role": "user", "content": query_text})
    response = st.session_state.hikma.ask(query_text)

    msg_data = {
        "role": "assistant",
        "content": response["bangla"],
        "confidence": response["confidence"],
    }
    for key in ("english", "arabic", "arabic_bangla", "source", "category", "topic"):
        val = response.get(key)
        if val and str(val).strip():
            msg_data[key] = val

    st.session_state.messages.append(msg_data)


# ---- sidebar ----
with st.sidebar:
    st.markdown("## 🕌 HIKMA")
    st.markdown("Islamic Knowledge AI")
    st.divider()

    st.markdown("### 🌐 যেভাবে জিজ্ঞাসা করবেন")
    st.markdown(
        "- **বাংলিশ:** namaj, roja, zakat\n"
        "- **English:** salah, fasting, zakat\n"
        "- **বাংলা:** নামাজ, রোজা, যাকাত"
    )
    st.divider()

    st.markdown("### 📊 এই সেশনে")
    q_count = sum(1 for m in st.session_state.messages if m.get("role") == "user")
    c1, c2 = st.columns(2)
    c1.metric("প্রশ্ন", q_count)
    c2.metric("বার্তা", len(st.session_state.messages))

    if dataset_loaded:
        st.divider()
        st.markdown("### 📚 ডেটাসেট")
        st.metric("মোট এন্ট্রি", f"{len(df):,}")
        if "topic" in df.columns:
            st.caption(f"স্বতন্ত্র বিষয়: {df['topic'].nunique():,}")
        if st.button("🎲 এলোমেলো একটি বিষয় জিজ্ঞাসা করুন", use_container_width=True):
            random_topic = str(df.sample(1)["topic"].iloc[0])
            handle_query(random_topic)
            st.rerun()

    st.divider()
    if st.button("🔄 কথোপকথন মুছুন", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.info(
        "✅ বাংলিশ সমর্থন\n\n"
        "✅ বাংলা ও ইংরেজি\n\n"
        "✅ আরবি রেফারেন্স\n\n"
        "✅ সূত্র-ভিত্তিক উত্তর"
    )

# ---- header ----
st.markdown(
    """
    <div class="hikma-header">
        <h1>HIKMA</h1>
        <p>বাংলিশে জিজ্ঞাসা করুন, বাংলায় উত্তর পান — সূত্রসহ</p>
    </div>
    <div class="hikma-rule"></div>
    """,
    unsafe_allow_html=True,
)

# ---- empty state: clickable topic chips ----
if not st.session_state.messages:
    st.markdown('<p class="chip-label">দ্রুত শুরু করতে একটি বিষয়ে ক্লিক করুন —</p>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, topic in enumerate(SUGGESTED_TOPICS):
        if cols[i % 4].button(topic, key=f"chip_{topic}", use_container_width=True):
            handle_query(topic)
            st.rerun()
    st.write("")

# ---- chat history ----
CONF_LABELS = [
    (0.8, "conf-high", "উচ্চ আস্থা"),
    (0.5, "conf-med", "মাঝারি আস্থা"),
    (0.0, "conf-low", "কম আস্থা"),
]

for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div class="msg-row" style="display:flex; justify-content:flex-end;">
                <div class="user-msg"><strong>আপনি</strong><br>{msg['content']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        continue

    conf = msg.get("confidence", 0.0)
    conf_class, conf_label = "conf-low", "কম আস্থা"
    for threshold, css_class, label in CONF_LABELS:
        if conf >= threshold:
            conf_class, conf_label = css_class, label
            break

    badges = '<span class="badge">HIKMA</span>'
    if msg.get("topic"):
        badges += f'<span class="badge">{msg["topic"]}</span>'
    if msg.get("category"):
        badges += f'<span class="badge">{msg["category"]}</span>'

    col_icon, col_card = st.columns([1, 11])
    with col_icon:
        st.markdown("### 🕌")
    with col_card:
        st.markdown(
            f"""
            <div class="msg-row assistant-card">
                {badges}
                <div class="answer-bn">{msg['content']}</div>
                <div class="conf-pill {conf_class}">📊 {conf_label} · {int(conf * 100)}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if msg.get("source"):
            with st.expander("📖 সূত্র দেখুন"):
                st.markdown(msg["source"])

        if msg.get("arabic"):
            with st.expander("🕋 আরবি পাঠ"):
                st.markdown(
                    f"""<div class="ayah-frame"><span class="bracket">﴿</span> {msg['arabic']} <span class="bracket">﴾</span></div>""",
                    unsafe_allow_html=True,
                )
                if msg.get("arabic_bangla"):
                    st.markdown(f'<div class="translation-box">{msg["arabic_bangla"]}</div>', unsafe_allow_html=True)

        if msg.get("english"):
            with st.expander("🌐 English"):
                st.markdown(f'<div class="translation-box">{msg["english"]}</div>', unsafe_allow_html=True)

        fb_key = f"fb_{i}"
        if st.session_state.feedback.get(fb_key):
            st.caption(f"✅ ধন্যবাদ — আপনার মতামত ({st.session_state.feedback[fb_key]}) গৃহীত হলো।")
        else:
            fb1, fb2, _ = st.columns([1, 1, 8])
            if fb1.button("👍", key=f"up_{i}"):
                st.session_state.feedback[fb_key] = "সহায়ক"
                st.rerun()
            if fb2.button("👎", key=f"down_{i}"):
                st.session_state.feedback[fb_key] = "উন্নতি প্রয়োজন"
                st.rerun()

# ---- input: Enter-to-send, auto-clears ----
st.divider()
with st.form("ask_form", clear_on_submit=True):
    col_input, col_submit = st.columns([6, 1])
    with col_input:
        user_input = st.text_input(
            "প্রশ্ন করুন",
            placeholder="বাংলিশে (namaj), ইংরেজিতে (salah), বা বাংলায় (নামাজ) লিখুন...",
            label_visibility="collapsed",
        )
    with col_submit:
        submitted = st.form_submit_button("🤖 জিজ্ঞাসা করুন", use_container_width=True)

if submitted and user_input.strip():
    with st.spinner("🕌 হিকমা উত্তর খুঁজছে..."):
        handle_query(user_input.strip())
    st.rerun()

# ---- footer ----
st.markdown(
    """
    <div class="hikma-footer">
        <strong>HIKMA</strong> — Islamic Knowledge AI<br>
        <small>বাংলিশে জিজ্ঞাসা করুন • বাংলায় উত্তর পান • সূত্র আরবিতে</small>
    </div>
    """,
    unsafe_allow_html=True,
)
