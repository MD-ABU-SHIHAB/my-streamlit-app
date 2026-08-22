"""HIKMA — Islamic Knowledge AI

A single-file Streamlit app: the matching engine (the HIKMA class) and the
UI live together here so the whole project only needs this file, a
requirements.txt, and dataset.csv.

Visual language deliberately mirrors Claude.ai's own product UI: warm
cream background, a single terracotta accent, serif headings over a plain
sans body, no gradients or heavy cards — flat, quiet, generous whitespace.
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

        self._topics_lower = self.df["topic"].astype(str).str.lower().tolist()
        self._answers_lower = self.df["answer_bangla"].astype(str).str.lower().tolist()

        self.all_keywords: set[str] = set()
        self.keyword_index: dict[str, list[int]] = defaultdict(list)
        self._build_keyword_index()

    def _validate_columns(self) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"dataset.csv is missing required column(s): {', '.join(missing)}"
            )

    def _clean_data(self) -> None:
        text_cols = ["arabic_text", "arabic_bangla", "answer_english", "sources", "category"]
        for col in text_cols:
            self.df[col] = self.df[col].fillna("")
        self.df["confidence"] = self.df["confidence"].fillna("medium")

    def _build_keyword_index(self) -> None:
        for idx, keywords_str in enumerate(self.df["keywords"]):
            if pd.isna(keywords_str):
                continue
            for kw in str(keywords_str).split(","):
                kw = kw.strip().lower()
                if not kw:
                    continue
                self.all_keywords.add(kw)
                self.keyword_index[kw].append(idx)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower().strip()
        return re.sub(r"\s+", " ", text)

    def _extract_keywords(self, query: str) -> list[str]:
        found = [kw for kw in self.all_keywords if kw in query]
        if not found:
            for word in query.split():
                if len(word) > 2:
                    found.extend(get_close_matches(word, self.all_keywords, n=3, cutoff=0.6))
        return found

    def _score_by_keywords(self, query: str, keywords: list[str]) -> Counter:
        scores: Counter = Counter()
        for kw in keywords:
            for idx in self.keyword_index.get(kw, ()):
                scores[idx] += 1
        for idx, topic in enumerate(self._topics_lower):
            if topic and topic in query:
                scores[idx] += 3
        return scores

    def _score_by_free_text(self, query: str) -> Counter:
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
        query = self._normalize_text(query)
        words = [w for w in query.split() if len(w) > 2] or [query]
        suggestions: list[str] = []
        for word in words:
            suggestions.extend(get_close_matches(word, self.all_keywords, n=n, cutoff=0.5))
        seen, unique = set(), []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique[:n]

    def related_topics(self, category: str, exclude_topic: str, n: int = 3) -> list[str]:
        if not category:
            return []
        pool = self.df[(self.df["category"] == category) & (self.df["topic"] != exclude_topic)]
        if pool.empty:
            return []
        sample = pool["topic"].drop_duplicates()
        return sample.sample(min(n, len(sample))).tolist()

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
        if not query or not query.strip():
            return {
                "bangla": "দয়া করে একটি প্রশ্ন লিখুন।",
                "english": "Please write a question.",
                "confidence": 0.0,
            }
        return self._ask_cached(self._normalize_text(query))


# ===========================================================================
# Streamlit UI — Claude.ai visual language
# ===========================================================================
st.set_page_config(
    page_title="HIKMA — Islamic Knowledge AI",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Tiro+Bangla&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #FAF9F5;
    --bg-alt: #F0EEE6;
    --bg-card: #FFFFFF;
    --text: #1F1E1C;
    --text-muted: #83827D;
    --accent: #D97757;
    --accent-hover: #C15F3C;
    --border: #E5E2D9;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }
.stApp { background: var(--bg); }
section[data-testid="stSidebar"] > div { background: var(--bg-alt); }
section[data-testid="stSidebar"] { border-right: 1px solid var(--border); }

/* ---------- Header ---------- */
.hikma-header {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    padding: 1.4rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}
.hikma-header .mark {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 1.9rem;
    color: var(--text);
}
.hikma-header .tagline {
    font-family: 'Tiro Bangla', serif;
    font-size: 0.98rem;
    color: var(--text-muted);
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    box-shadow: none !important;
    transition: background 0.12s ease, border-color 0.12s ease;
}
.stButton > button:hover {
    background: var(--bg-alt) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
.stButton > button:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 1px; }

div[data-testid="stFormSubmitButton"] > button {
    background: var(--accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 600 !important;
}
div[data-testid="stFormSubmitButton"] > button:hover { background: var(--accent-hover) !important; color: #fff !important; }

/* ---------- Chat turns ---------- */
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
.turn { animation: fadeIn 0.25s ease; padding: 1.1rem 0; border-bottom: 1px solid var(--border); }
.turn:last-child { border-bottom: none; }

.role-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.35rem;
}
.role-label.user { text-align: right; }

.user-bubble {
    background: var(--bg-alt);
    color: var(--text);
    padding: 0.75rem 1rem;
    border-radius: 14px;
    max-width: 70%;
    margin-left: auto;
    font-family: 'Tiro Bangla', serif;
    font-size: 1.02rem;
}

.answer-bn {
    font-family: 'Tiro Bangla', serif;
    font-size: 1.08rem;
    line-height: 1.85;
    color: var(--text);
    min-height: 1.6rem;
}
.cursor-blink::after { content: "▌"; animation: blink 0.9s steps(1) infinite; color: var(--accent); }
@keyframes blink { 50% { opacity: 0; } }

.tag-row { margin-bottom: 0.5rem; }
.tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--text-muted);
    background: var(--bg-alt);
    border: 1px solid var(--border);
    padding: 0.1rem 0.55rem;
    border-radius: 6px;
    margin-right: 0.35rem;
}

.conf-tag {
    display: inline-block;
    font-size: 0.76rem;
    font-weight: 500;
    padding: 0.15rem 0.6rem;
    border-radius: 6px;
    margin-top: 0.6rem;
    border: 1px solid var(--border);
}
.conf-high { color: var(--accent); border-color: var(--accent); }
.conf-med  { color: var(--text-muted); }
.conf-low  { color: var(--text-muted); border-style: dashed; }

.msg-meta { font-size: 0.72rem; color: var(--text-muted); margin-top: 0.3rem; }

.ayah-frame {
    direction: rtl;
    text-align: center;
    font-family: 'Source Serif 4', serif;
    font-size: 1.35rem;
    color: var(--text);
    background: var(--bg-alt);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    line-height: 2;
}
.ayah-frame .bracket { color: var(--accent); }
.translation-box {
    font-family: 'Tiro Bangla', serif;
    background: var(--bg-alt);
    border-left: 3px solid var(--accent);
    padding: 0.7rem 1rem;
    border-radius: 0 8px 8px 0;
    margin-top: 0.4rem;
    color: var(--text);
}

.icon-btn, .related-btn {
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-size: 0.76rem;
    color: var(--text-muted);
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    margin: 0.5rem 0.35rem 0 0;
    cursor: pointer;
}
.icon-btn:hover, .related-btn:hover { border-color: var(--accent); color: var(--accent); }

.hikma-footer {
    text-align: center;
    color: var(--text-muted);
    padding: 1.6rem;
    font-family: 'Tiro Bangla', serif;
    font-size: 0.88rem;
    margin-top: 1rem;
}
</style>
"""

DARK_CSS = """
<style>
:root {
    --bg: #262624 !important;
    --bg-alt: #30302E !important;
    --bg-card: #2B2B29 !important;
    --text: #F5F4EF !important;
    --text-muted: #9A9990 !important;
    --border: #3D3D3A !important;
}
.stApp { background: #262624 !important; }
section[data-testid="stSidebar"] > div { background: #21211F !important; }
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
    st.markdown("#### 🕌 HIKMA")
    st.caption("Islamic Knowledge AI")
    st.divider()

    dark_toggle = st.toggle("Dark mode", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

    st.divider()
    st.markdown("**যেভাবে জিজ্ঞাসা করবেন**")
    st.caption("বাংলিশ: namaj, roja, zakat")
    st.caption("English: salah, fasting, zakat")
    st.caption("বাংলা: নামাজ, রোজা, যাকাত")

    st.divider()
    st.markdown("**এই সেশনে**")
    q_count = sum(1 for m in st.session_state.messages if m.get("role") == "user")
    c1, c2 = st.columns(2)
    c1.metric("প্রশ্ন", q_count)
    c2.metric("বার্তা", len(st.session_state.messages))
    if st.session_state.category_counts:
        st.caption("বিষয় অনুযায়ী প্রশ্ন")
        st.bar_chart(pd.Series(st.session_state.category_counts))

    st.divider()
    st.markdown("**ডেটাসেট**")
    st.metric("মোট এন্ট্রি", f"{len(df):,}")
    if "topic" in df.columns:
        st.caption(f"স্বতন্ত্র বিষয়: {df['topic'].nunique():,}")

    if "category" in df.columns:
        categories = sorted(c for c in df["category"].unique() if str(c).strip())
        if categories:
            chosen_cat = st.selectbox("বিষয়শ্রেণি অনুসন্ধান করুন", ["—"] + categories)
            if chosen_cat != "—":
                pool = df[df["category"] == chosen_cat]["topic"].drop_duplicates()
                sample_topics = pool.sample(min(6, len(pool)))
                for t in sample_topics:
                    if st.button(f"↳ {t}", key=f"cat_{chosen_cat}_{t}", use_container_width=True):
                        handle_query(t)
                        st.rerun()

    if st.button("🎲 এলোমেলো একটি বিষয়", use_container_width=True):
        random_topic = str(df.sample(1)["topic"].iloc[0])
        handle_query(random_topic)
        st.rerun()

    st.divider()
    st.markdown("**⭐ সংরক্ষিত উত্তর**")
    if st.session_state.starred:
        for idx in sorted(st.session_state.starred):
            if idx < len(st.session_state.messages):
                m = st.session_state.messages[idx]
                st.caption(f"· {m.get('topic') or m.get('content', '')[:30]}")
    else:
        st.caption("এখনো কিছু সংরক্ষণ করা হয়নি।")

    st.divider()
    if st.session_state.messages:
        transcript_lines = [
            f"**{'আপনি' if m['role'] == 'user' else 'HIKMA'}** ({m.get('timestamp', '')}): {m['content']}"
            for m in st.session_state.messages
        ]
        st.download_button(
            "⬇️ কথোপকথন ডাউনলোড",
            data="\n\n".join(transcript_lines),
            file_name="hikma_chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if st.button("কথোপকথন মুছুন", use_container_width=True):
        st.session_state.messages = []
        st.session_state.starred = set()
        st.session_state.category_counts = Counter()
        st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hikma-header">
        <span class="mark">🕌 HIKMA</span>
        <span class="tagline">বাংলিশে জিজ্ঞাসা করুন, বাংলায় উত্তর পান</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.caption("দ্রুত শুরু করতে একটি বিষয়ে ক্লিক করুন")
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
            <div class="turn">
                <div class="role-label user">আপনি</div>
                <div class="user-bubble">{msg['content']}</div>
                <div class="msg-meta" style="text-align:right;">{msg.get('timestamp', '')}</div>
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

    tags = '<span class="tag">HIKMA</span>'
    if msg.get("topic"):
        tags += f'<span class="tag">{msg["topic"]}</span>'
    if msg.get("category"):
        tags += f'<span class="tag">{msg["category"]}</span>'

    st.markdown(f'<div class="turn"><div class="tag-row">{tags}</div>', unsafe_allow_html=True)

    answer_placeholder = st.empty()
    if st.session_state.just_answered == i:
        stream_answer(answer_placeholder, msg["content"])
        st.session_state.just_answered = None
    else:
        answer_placeholder.markdown(f'<div class="answer-bn">{msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="conf-tag {conf_class}">{conf_label} · {int(conf * 100)}%</div>'
        f'<div class="msg-meta">{msg.get("timestamp", "")}</div>',
        unsafe_allow_html=True,
    )

    if msg.get("source"):
        with st.expander("সূত্র দেখুন"):
            st.markdown(msg["source"])

    if msg.get("arabic"):
        with st.expander("আরবি পাঠ"):
            st.markdown(
                f'<div class="ayah-frame"><span class="bracket">﴿</span> {msg["arabic"]} <span class="bracket">﴾</span></div>',
                unsafe_allow_html=True,
            )
            if msg.get("arabic_bangla"):
                st.markdown(f'<div class="translation-box">{msg["arabic_bangla"]}</div>', unsafe_allow_html=True)

    if msg.get("english"):
        with st.expander("English"):
            st.markdown(f'<div class="translation-box">{msg["english"]}</div>', unsafe_allow_html=True)

    escaped_text = html.escape(msg["content"])
    copy_id = f"copy_text_{i}"
    st.markdown(
        f"""
        <span id="{copy_id}" style="display:none;">{escaped_text}</span>
        <button class="icon-btn" onclick="navigator.clipboard.writeText(document.getElementById('{copy_id}').innerText)">কপি করুন</button>
        """,
        unsafe_allow_html=True,
    )

    if conf <= 0.15:
        suggestions = hikma.suggest_keywords(st.session_state.messages[i - 1]["content"] if i > 0 else "")
        if suggestions:
            st.caption("এই বিষয়গুলো বুঝিয়ে থাকতে পারেন")
            s_cols = st.columns(len(suggestions))
            for j, s in enumerate(suggestions):
                if s_cols[j].button(s, key=f"suggest_{i}_{j}"):
                    handle_query(s)
                    st.rerun()

    if msg.get("category") and msg.get("topic"):
        related = hikma.related_topics(msg["category"], msg["topic"])
        if related:
            st.caption("আরও জানতে পারেন")
            r_cols = st.columns(len(related))
            for j, t in enumerate(related):
                if r_cols[j].button(t, key=f"related_{i}_{j}"):
                    handle_query(t)
                    st.rerun()

    fb_key = f"fb_{i}"
    fb1, fb2, fb3, _ = st.columns([1, 1, 1, 7])
    if st.session_state.feedback.get(fb_key):
        st.caption(f"ধন্যবাদ — মতামত ({st.session_state.feedback[fb_key]}) গৃহীত হলো।")
    else:
        if fb1.button("👍", key=f"up_{i}"):
            st.session_state.feedback[fb_key] = "সহায়ক"
            st.rerun()
        if fb2.button("👎", key=f"down_{i}"):
            st.session_state.feedback[fb_key] = "উন্নতি প্রয়োজন"
            st.rerun()
    starred = i in st.session_state.starred
    if fb3.button("⭐" if not starred else "✓", key=f"star_{i}"):
        if starred:
            st.session_state.starred.discard(i)
        else:
            st.session_state.starred.add(i)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input
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
        submitted = st.form_submit_button("জিজ্ঞাসা করুন", use_container_width=True)

if submitted and user_input.strip():
    handle_query(user_input.strip())
    st.rerun()

st.markdown(
    """
    <div class="hikma-footer">
        HIKMA — স্পষ্ট প্রশ্ন · নির্ভরযোগ্য উত্তর · যাচাইকৃত সূত্র
    </div>
    """,
    unsafe_allow_html=True,
)
