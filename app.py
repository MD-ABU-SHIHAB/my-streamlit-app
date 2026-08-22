"""HIKMA — Islamic Knowledge AI

A single-file Streamlit app: the matching engine (the HIKMA class) and the
UI live together here so the whole project only needs this file, a
requirements.txt, and dataset.csv.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
import html
import re
import time

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

    def suggest_keywords(self, query: str, n: int = 5) -> list[str]:
        """Public helper for 'did you mean' suggestions: closest known keywords
        to whatever the user typed, for use when nothing matched at all."""
        query = self._normalize_text(query)
        words = [w for w in query.split() if len(w) > 2] or [query]
        suggestions: list[str] = []
        for word in words:
            suggestions.extend(get_close_matches(word, self.all_keywords, n=n, cutoff=0.5))
        # de-duplicate while preserving order
        seen = set()
        unique = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique[:n]

    def related_topics(self, category: str, exclude_topic: str, n: int = 3) -> list[str]:
        """Other topics sharing the same category, for 'you might also ask' chips."""
        if not category:
            return []
        pool = self.df[(self.df["category"] == category) & (self.df["topic"] != exclude_topic)]
        if pool.empty:
            return []
        sample = pool["topic"].drop_duplicates()
        return sample.sample(min(n, len(sample))).tolist()

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

BASE_CSS = """
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
.stApp { background: var(--parchment); transition: background 0.3s ease; }

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

.msg-meta {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.3rem;
}

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
    min-height: 1.6rem;
}
.cursor-blink::after {
    content: "▌";
    animation: blink 0.9s steps(1) infinite;
    color: var(--gold);
}
@keyframes blink { 50% { opacity: 0; } }

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

.copy-btn, .related-btn {
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--teal-deep);
    background: var(--parchment-dim);
    border: 1px solid var(--teal-light);
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    margin: 0.5rem 0.3rem 0 0;
    cursor: pointer;
    transition: background 0.15s ease, color 0.15s ease;
}
.copy-btn:hover, .related-btn:hover { background: var(--teal-deep); color: var(--gold-light); }

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

DARK_CSS = """
<style>
.stApp { background: #0E1F1D !important; }
.assistant-card {
    background: #14302B !important;
    border-color: #23453F !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.35) !important;
}
.assistant-card .answer-bn { color: #F1EAD6 !important; }
.assistant-card .badge { background: #1E3D37 !important; color: #E4C776 !important; }
.ayah-frame { background: linear-gradient(180deg, #1B3A34, #142E29) !important; color: #E4C776 !important; }
.translation-box { background: #17332D !important; color: #F1EAD6 !important; }
.copy-btn, .related-btn { background: #17332D !important; color: #E4C776 !important; }
.hikma-footer { color: #9FB0AC !important; border-top-color: #23453F !important; }
.chip-label { color: #E4C776 !important; }
.msg-meta { color: #9FB0AC !important; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state & data
# ---------------------------------------------------------------------------
DEFAULTS = {
    "messages": [],
    "feedback": {},
    "starred": set(),
    "category_counts": Counter(),
    "dark_mode": False,
    "just_answered": None,
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.dark_mode:
    st.markdown(DARK_CSS, unsafe_allow_html=True)

if "hikma" not in st.session_state:
    try:
        st.session_state.hikma = HIKMA("dataset.csv")
    except Exception as e:
        st.error(f"⚠️ dataset.csv লোড করা যায়নি:\n\n**{e}**")
        st.stop()

hikma: HIKMA = st.session_state.hikma
df = hikma.df

SUGGESTED_TOPICS = ["নামাজ", "রোজা", "যাকাত", "হজ্জ", "ওযু", "সুদ", "কুরবানি", "তাহাজ্জুদ"]


def handle_query(query_text: str) -> None:
    """Ask HIKMA a question and append both turns to the chat history."""
    now = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": query_text, "timestamp": now})

    response = hikma.ask(query_text)
    msg_data = {
        "role": "assistant",
        "content": response["bangla"],
        "confidence": response["confidence"],
        "timestamp": datetime.now().strftime("%H:%M"),
    }
    for key in ("english", "arabic", "arabic_bangla", "source", "category", "topic"):
        val = response.get(key)
        if val and str(val).strip():
            msg_data[key] = val

    st.session_state.messages.append(msg_data)
    if msg_data.get("category"):
        st.session_state.category_counts[msg_data["category"]] += 1
    st.session_state.just_answered = len(st.session_state.messages) - 1


def stream_answer(placeholder, text: str) -> None:
    """Reveal the answer progressively, similar to a live-typed response,
    instead of dumping the whole paragraph at once."""
    steps = 40
    chunk = max(1, len(text) // steps)
    shown = ""
    for i in range(0, len(text), chunk):
        shown = text[: i + chunk]
        placeholder.markdown(f'<div class="answer-bn cursor-blink">{shown}</div>', unsafe_allow_html=True)
        time.sleep(0.012)
    placeholder.markdown(f'<div class="answer-bn">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🕌 HIKMA")
    st.markdown("Islamic Knowledge AI")
    st.divider()

    dark_toggle = st.toggle("🌙 ডার্ক মোড", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

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
    if st.session_state.category_counts:
        st.caption("বিষয় অনুযায়ী প্রশ্ন:")
        st.bar_chart(pd.Series(st.session_state.category_counts))

    st.divider()
    st.markdown("### 📚 ডেটাসেট")
    st.metric("মোট এন্ট্রি", f"{len(df):,}")
    if "topic" in df.columns:
        st.caption(f"স্বতন্ত্র বিষয়: {df['topic'].nunique():,}")

    if "category" in df.columns:
        categories = sorted(c for c in df["category"].unique() if str(c).strip())
        if categories:
            chosen_cat = st.selectbox("📂 বিষয়শ্রেণি অনুসন্ধান করুন", ["—"] + categories)
            if chosen_cat != "—":
                sample_topics = (
                    df[df["category"] == chosen_cat]["topic"].drop_duplicates().sample(
                        min(6, df[df["category"] == chosen_cat]["topic"].nunique())
                    )
                )
                for t in sample_topics:
                    if st.button(f"↳ {t}", key=f"cat_{chosen_cat}_{t}", use_container_width=True):
                        handle_query(t)
                        st.rerun()

    if st.button("🎲 এলোমেলো একটি বিষয় জিজ্ঞাসা করুন", use_container_width=True):
        random_topic = str(df.sample(1)["topic"].iloc[0])
        handle_query(random_topic)
        st.rerun()

    st.divider()
    st.markdown("### ⭐ সংরক্ষিত উত্তর")
    if st.session_state.starred:
        for idx in sorted(st.session_state.starred):
            if idx < len(st.session_state.messages):
                m = st.session_state.messages[idx]
                label = m.get("topic") or m.get("content", "")[:30]
                st.caption(f"• {label}")
    else:
        st.caption("এখনো কিছু সংরক্ষণ করা হয়নি।")

    st.divider()
    if st.session_state.messages:
        transcript_lines = []
        for m in st.session_state.messages:
            speaker = "আপনি" if m["role"] == "user" else "HIKMA"
            transcript_lines.append(f"**{speaker}** ({m.get('timestamp', '')}): {m['content']}")
        transcript = "\n\n".join(transcript_lines)
        st.download_button(
            "⬇️ কথোপকথন ডাউনলোড করুন",
            data=transcript,
            file_name="hikma_chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if st.button("🔄 কথোপকথন মুছুন", use_container_width=True):
        st.session_state.messages = []
        st.session_state.starred = set()
        st.session_state.category_counts = Counter()
        st.rerun()

    st.divider()
    st.info(
        "✅ বাংলিশ সমর্থন\n\n"
        "✅ বাংলা ও ইংরেজি\n\n"
        "✅ আরবি রেফারেন্স\n\n"
        "✅ সূত্র-ভিত্তিক উত্তর"
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Empty state — clickable topic chips
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown('<p class="chip-label">দ্রুত শুরু করতে একটি বিষয়ে ক্লিক করুন —</p>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, topic in enumerate(SUGGESTED_TOPICS):
        if cols[i % 4].button(topic, key=f"chip_{topic}", use_container_width=True):
            handle_query(topic)
            st.rerun()
    st.write("")

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
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
                <div>
                    <div class="user-msg"><strong>আপনি</strong><br>{msg['content']}</div>
                    <div class="msg-meta" style="text-align:right;">{msg.get('timestamp', '')}</div>
                </div>
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
            """,
            unsafe_allow_html=True,
        )

        answer_placeholder = st.empty()
        if st.session_state.just_answered == i:
            stream_answer(answer_placeholder, msg["content"])
            st.session_state.just_answered = None
        else:
            answer_placeholder.markdown(f'<div class="answer-bn">{msg["content"]}</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
                <div class="conf-pill {conf_class}">📊 {conf_label} · {int(conf * 100)}%</div>
                <div class="msg-meta">{msg.get('timestamp', '')}</div>
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

        # Copy-to-clipboard
        escaped_text = html.escape(msg["content"])
        copy_id = f"copy_text_{i}"
        st.markdown(
            f"""
            <span id="{copy_id}" style="display:none;">{escaped_text}</span>
            <button class="copy-btn" onclick="navigator.clipboard.writeText(document.getElementById('{copy_id}').innerText)">📋 কপি করুন</button>
            """,
            unsafe_allow_html=True,
        )

        # Did-you-mean suggestions for low-confidence / no-match answers
        if conf <= 0.15:
            suggestions = hikma.suggest_keywords(
                st.session_state.messages[i - 1]["content"] if i > 0 else ""
            )
            if suggestions:
                st.caption("এই বিষয়গুলো বুঝিয়ে থাকতে পারেন —")
                s_cols = st.columns(len(suggestions))
                for j, s in enumerate(suggestions):
                    if s_cols[j].button(s, key=f"suggest_{i}_{j}"):
                        handle_query(s)
                        st.rerun()

        # Related topics ("you might also ask")
        if msg.get("category") and msg.get("topic"):
            related = hikma.related_topics(msg["category"], msg["topic"])
            if related:
                st.caption("🔎 আরও জানতে পারেন —")
                r_cols = st.columns(len(related))
                for j, t in enumerate(related):
                    if r_cols[j].button(t, key=f"related_{i}_{j}"):
                        handle_query(t)
                        st.rerun()

        # Feedback + bookmark row
        fb_key = f"fb_{i}"
        fb1, fb2, fb3, _ = st.columns([1, 1, 1, 7])
        if st.session_state.feedback.get(fb_key):
            st.caption(f"✅ ধন্যবাদ — আপনার মতামত ({st.session_state.feedback[fb_key]}) গৃহীত হলো।")
        else:
            if fb1.button("👍", key=f"up_{i}"):
                st.session_state.feedback[fb_key] = "সহায়ক"
                st.rerun()
            if fb2.button("👎", key=f"down_{i}"):
                st.session_state.feedback[fb_key] = "উন্নতি প্রয়োজন"
                st.rerun()
        starred = i in st.session_state.starred
        if fb3.button("⭐" if not starred else "✅", key=f"star_{i}"):
            if starred:
                st.session_state.starred.discard(i)
            else:
                st.session_state.starred.add(i)
            st.rerun()

# ---------------------------------------------------------------------------
# Input — Enter-to-send, auto-clears
# ---------------------------------------------------------------------------
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
    handle_query(user_input.strip())
    st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hikma-footer">
        <strong>HIKMA</strong> — Islamic Knowledge AI<br>
        <small>স্পষ্ট প্রশ্ন • নির্ভরযোগ্য উত্তর • যাচাইকৃত সূত্র</small>
    </div>
    """,
    unsafe_allow_html=True,
)
