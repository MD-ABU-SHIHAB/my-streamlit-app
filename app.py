"""HIKMA — Islamic Knowledge AI
Streamlit UI: manuscript / mosque-tile visual identity, Bangla-first, source-grounded.
"""
import random

import pandas as pd
import streamlit as st

from main import HIKMA

st.set_page_config(
    page_title="HIKMA — Islamic Knowledge AI",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens
#   Parchment + deep teal + manuscript gold, instead of the generic
#   cream/terracotta or near-black/neon palettes — grounded in illuminated
#   Qur'an manuscripts and İznik mosque tilework rather than picked at random.
# ---------------------------------------------------------------------------
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

/* ---------- Header ---------- */
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

/* ---------- Suggestion chips (empty state) ---------- */
.chip-label {
    font-family: 'Tiro Bangla', serif;
    color: var(--teal-deep);
    font-size: 1.05rem;
    margin-bottom: 0.6rem;
}
div[data-testid="stButton"] > button.chip-btn,
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

/* ---------- Chat bubbles ---------- */
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
}
.assistant-card .answer-bn {
    font-family: 'Tiro Bangla', serif;
    font-size: 1.15rem;
    line-height: 1.75;
    color: var(--ink);
}

/* ---------- Confidence pill ---------- */
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

/* ---------- Ayah frame ---------- */
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

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--teal-deep);
    color: var(--parchment);
}
section[data-testid="stSidebar"] * { color: var(--parchment) !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(247,241,225,0.2); }

/* ---------- Footer ---------- */
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

# ---------------------------------------------------------------------------
# Session state & data
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "hikma" not in st.session_state:
    st.session_state.hikma = HIKMA("dataset.csv")
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

try:
    df = pd.read_csv("dataset.csv")
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
    for key in ("english", "arabic", "arabic_bangla", "source"):
        val = response.get(key)
        if val and str(val).strip():
            msg_data[key] = val

    st.session_state.messages.append(msg_data)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
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

    col_icon, col_card = st.columns([1, 11])
    with col_icon:
        st.markdown("### 🕌")
    with col_card:
        st.markdown(
            f"""
            <div class="msg-row assistant-card">
                <span class="badge">HIKMA</span>
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
    with st.spinner("🕌 হিকমা উত্তর খুঁজছে..."):
        handle_query(user_input.strip())
    st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hikma-footer">
        <strong>HIKMA</strong> — Islamic Knowledge AI<br>
        <small>বাংলিশে জিজ্ঞাসা করুন • বাংলায় উত্তর পান • সূত্র আরবিতে</small>
    </div>
    """,
    unsafe_allow_html=True,
)
