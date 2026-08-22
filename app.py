"""HIKMA — Islamic Knowledge AI

A single-file Streamlit app. Sections, in order:
  1. Matching engine (the HIKMA class)
  2. Pure-math/utility helpers (Qibla bearing, prayer times, Hijri date, Zakat math)
  3. Streamlit UI: sidebar + seven tabs

Only two files are needed to run this: this file and requirements.txt,
alongside dataset.csv.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, date, timedelta
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
import csv
import html
import io
import math
import random
import re
import time

import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = [
    "id", "topic", "keywords", "category", "answer_bangla",
    "answer_english", "arabic_text", "arabic_bangla", "sources", "confidence",
]


# ===========================================================================
# 1. Matching engine
# ===========================================================================
class HIKMA:
    """Turns a Banglish / Bangla / English query into the best matching row
    of dataset.csv, using an inverted keyword index so lookups stay fast
    even at tens of thousands of rows."""

    def __init__(self, dataset_path: str):
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

    # ---- setup ----
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

    # ---- query processing ----
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
        if not query or not query.strip():
            return {
                "bangla": "দয়া করে একটি প্রশ্ন লিখুন।",
                "english": "Please write a question.",
                "confidence": 0.0,
            }
        return self._ask_cached(self._normalize_text(query))

    def get_by_topic(self, topic: str) -> pd.Series | None:
        rows = self.df[self.df["topic"] == topic]
        return rows.iloc[0] if not rows.empty else None


# ===========================================================================
# 2. Pure utility helpers (no Streamlit dependency — independently testable)
# ===========================================================================

KAABA_LAT = 21.4225
KAABA_LON = 39.8262

BD_CITY_PRESETS = {
    "ঢাকা": (23.8103, 90.4125),
    "দিনাজপুর": (25.6279, 88.6332),
    "চট্টগ্রাম": (22.3569, 91.7832),
    "সিলেট": (24.8949, 91.8687),
    "রাজশাহী": (24.3745, 88.6042),
    "খুলনা": (22.8456, 89.5403),
    "বরিশাল": (22.7010, 90.3535),
    "রংপুর": (25.7439, 89.2752),
}

NISAB_GOLD_GRAMS = 87.48
NISAB_SILVER_GRAMS = 612.36
ZAKAT_RATE = 0.025

HIJRI_MONTHS_BN = [
    "মুহাররম", "সফর", "রবিউল আউয়াল", "রবিউস সানি", "জমাদিউল আউয়াল", "জমাদিউস সানি",
    "রজব", "শাবান", "রমজান", "শাওয়াল", "জিলক্বদ", "জিলহজ্জ",
]


def compute_qibla_bearing(lat: float, lon: float) -> float:
    phi1, phi2 = math.radians(lat), math.radians(KAABA_LAT)
    delta_lambda = math.radians(KAABA_LON - lon)
    x = math.sin(delta_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def bearing_to_compass_bn(bearing: float) -> str:
    directions = [
        "উত্তর", "উত্তর-পূর্ব", "পূর্ব", "দক্ষিণ-পূর্ব",
        "দক্ষিণ", "দক্ষিণ-পশ্চিম", "পশ্চিম", "উত্তর-পশ্চিম",
    ]
    return directions[round(bearing / 45) % 8]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def compute_zakat(cash, gold_grams, gold_price_per_gram, silver_grams, silver_price_per_gram,
                   business_value, other_investments, liabilities, nisab_basis) -> dict:
    gold_value = gold_grams * gold_price_per_gram
    silver_value = silver_grams * silver_price_per_gram
    total_assets = cash + gold_value + silver_value + business_value + other_investments
    net_wealth = max(0.0, total_assets - liabilities)

    if nisab_basis == "silver":
        nisab_value = NISAB_SILVER_GRAMS * silver_price_per_gram if silver_price_per_gram > 0 else 0.0
    else:
        nisab_value = NISAB_GOLD_GRAMS * gold_price_per_gram if gold_price_per_gram > 0 else 0.0

    obligatory = nisab_value > 0 and net_wealth >= nisab_value
    zakat_due = round(net_wealth * ZAKAT_RATE, 2) if obligatory else 0.0

    return {
        "gold_value": gold_value, "silver_value": silver_value, "total_assets": total_assets,
        "net_wealth": net_wealth, "nisab_value": nisab_value,
        "obligatory": obligatory, "zakat_due": zakat_due,
    }


def _decimal_hours_to_str(h: float) -> str:
    h = h % 24
    hh = int(h)
    mm = int(round((h - hh) * 60))
    if mm == 60:
        hh, mm = hh + 1, 0
    return f"{hh:02d}:{mm:02d}"


def compute_prayer_times(lat: float, lon: float, tz_offset: float, d: date,
                          fajr_angle: float = 18.0, isha_angle: float = 18.0,
                          asr_hanafi: bool = True) -> dict:
    """Standard solar-position based prayer time estimation:
    - Solar declination via Cooper's equation.
    - Equation of time via the common short approximation.
    - Sunrise/sunset at -0.833° (accounts for refraction + solar radius).
    - Fajr/Isha at the given twilight angle below the horizon.
    - Asr via the shadow-length method (Hanafi: factor 2, Shafi'i/others: factor 1).

    This is a well-established computational method, not a fabricated
    ruling — but local, moon-sighting-adjusted, authority-published
    timetables should always take precedence for actual worship.
    """
    N = d.timetuple().tm_yday
    B = math.radians(360 / 365 * (N - 81))
    eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    decl = math.radians(23.45 * math.sin(math.radians(360 / 365 * (284 + N))))
    lat_r = math.radians(lat)

    dhuhr = 12 + tz_offset - lon / 15 - eot / 60

    def hour_angle(angle_deg: float) -> float:
        angle = math.radians(angle_deg)
        cos_h = (math.sin(-angle) - math.sin(lat_r) * math.sin(decl)) / (math.cos(lat_r) * math.cos(decl))
        cos_h = max(-1.0, min(1.0, cos_h))
        return math.degrees(math.acos(cos_h))

    h_sun = hour_angle(0.833)
    sunrise, sunset = dhuhr - h_sun / 15, dhuhr + h_sun / 15

    fajr = dhuhr - hour_angle(fajr_angle) / 15
    isha = dhuhr + hour_angle(isha_angle) / 15

    shadow_factor = 2 if asr_hanafi else 1
    t = math.atan(1 / (shadow_factor + math.tan(abs(lat_r - decl))))
    cos_h_asr = (math.sin(t) - math.sin(lat_r) * math.sin(decl)) / (math.cos(lat_r) * math.cos(decl))
    cos_h_asr = max(-1.0, min(1.0, cos_h_asr))
    asr = dhuhr + math.degrees(math.acos(cos_h_asr)) / 15

    return {
        "ফজর": _decimal_hours_to_str(fajr),
        "সূর্যোদয়": _decimal_hours_to_str(sunrise),
        "যোহর": _decimal_hours_to_str(dhuhr),
        "আসর": _decimal_hours_to_str(asr),
        "মাগরিব": _decimal_hours_to_str(sunset),
        "এশা": _decimal_hours_to_str(isha),
    }


def gregorian_to_hijri(d: date) -> tuple[int, int, int]:
    """Tabular (arithmetic) Hijri conversion — the standard 30-year-cycle
    civil calendar used by most calendar-conversion libraries. This can
    differ by 1-2 days from a moon-sighting-announced date; treat it as an
    estimate, not an authoritative religious date."""
    jdn = d.toordinal() + 1721425

    l = jdn - 1948440 + 10632
    n = (l - 1) // 10631
    l = l - 10631 * n + 354
    j = ((10985 - l) // 5316) * ((50 * l) // 17719) + (l // 5670) * ((43 * l) // 15238)
    l = l - ((30 - j) // 15) * ((17719 * j) // 50) - (j // 16) * ((15238 * j) // 43) + 29
    month = (24 * l) // 709
    day = l - (709 * month) // 24
    year = 30 * n + j - 30
    return year, month, day


def build_quiz_questions(df: pd.DataFrame, n: int = 5, seed: int | None = None) -> list[dict]:
    rng = random.Random(seed)
    pool = df.drop_duplicates(subset=["topic"])
    n = min(n, len(pool))
    chosen = pool.sample(n, random_state=rng.randint(0, 1_000_000))

    questions = []
    all_answers = df["answer_bangla"].tolist()
    for _, row in chosen.iterrows():
        correct = row["answer_bangla"]
        distractors = rng.sample(
            [a for a in all_answers if a != correct], k=min(3, max(0, len(all_answers) - 1))
        )
        options = distractors + [correct]
        rng.shuffle(options)
        questions.append({
            "topic": row["topic"], "category": row.get("category", ""),
            "correct": correct, "options": options, "source": row.get("sources", ""),
        })
    return questions


def build_flashcard_deck(df: pd.DataFrame, category: str | None, n: int, seed: int | None = None) -> list[dict]:
    rng = random.Random(seed)
    pool = df.drop_duplicates(subset=["topic"])
    if category and category != "সব বিষয়শ্রেণি":
        pool = pool[pool["category"] == category]
    n = min(n, len(pool))
    if n == 0:
        return []
    chosen = pool.sample(n, random_state=rng.randint(0, 1_000_000))
    return [
        {"topic": row["topic"], "answer": row["answer_bangla"], "source": row.get("sources", "")}
        for _, row in chosen.iterrows()
    ]


def messages_to_csv(messages: list[dict]) -> str:
    """Serialize the Q&A history to CSV text for download."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["timestamp", "role", "content", "topic", "category", "confidence"])
    for m in messages:
        writer.writerow([
            m.get("timestamp", ""), m["role"], m["content"],
            m.get("topic", ""), m.get("category", ""), m.get("confidence", ""),
        ])
    return buf.getvalue()


# ===========================================================================
# 3. Streamlit UI — Claude.ai-style visual language
# ===========================================================================
st.set_page_config(
    page_title="HIKMA — Islamic Knowledge AI",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded",
)

FONT_SCALE = {"ছোট": "0.92", "মাঝারি": "1.0", "বড়": "1.14"}

BASE_CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Tiro+Bangla&family=Inter:wght@400;500;600&display=swap');

:root {{
    --bg: #FAF9F5;
    --bg-alt: #F0EEE6;
    --bg-card: #FFFFFF;
    --text: #1F1E1C;
    --text-muted: #83827D;
    --accent: #D97757;
    --accent-hover: #C15F3C;
    --border: #E5E2D9;
    --good: #3D7A5C;
    --bad: #B3542E;
    --font-scale: {font_scale};
}}

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: var(--text); }}
.stApp {{ background: var(--bg); }}
section[data-testid="stSidebar"] > div {{ background: var(--bg-alt); }}
section[data-testid="stSidebar"] {{ border-right: 1px solid var(--border); }}

.hikma-header {{
    display: flex; align-items: baseline; gap: 0.6rem;
    padding: 1.4rem 0 1rem; border-bottom: 1px solid var(--border); margin-bottom: 1.2rem;
}}
.hikma-header .mark {{
    font-family: 'Source Serif 4', serif; font-weight: 600;
    font-size: calc(1.9rem * var(--font-scale)); color: var(--text);
}}
.hikma-header .tagline {{
    font-family: 'Tiro Bangla', serif; font-size: calc(0.98rem * var(--font-scale)); color: var(--text-muted);
}}

.stButton > button {{
    border-radius: 8px !important; border: 1px solid var(--border) !important;
    background: var(--bg-card) !important; color: var(--text) !important;
    font-family: 'Inter', sans-serif !important; font-weight: 500 !important;
    font-size: 0.88rem !important; box-shadow: none !important;
    transition: background 0.12s ease, border-color 0.12s ease;
}}
.stButton > button:hover {{ background: var(--bg-alt) !important; border-color: var(--accent) !important; color: var(--accent) !important; }}
.stButton > button:focus-visible {{ outline: 2px solid var(--accent) !important; outline-offset: 1px; }}

div[data-testid="stFormSubmitButton"] > button {{
    background: var(--accent) !important; color: #FFFFFF !important; border: none !important; font-weight: 600 !important;
}}
div[data-testid="stFormSubmitButton"] > button:hover {{ background: var(--accent-hover) !important; color: #fff !important; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 0.3rem; border-bottom: 1px solid var(--border); flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'Inter', sans-serif; font-weight: 500; color: var(--text-muted); padding: 0.5rem 0.8rem; }}
.stTabs [aria-selected="true"] {{ color: var(--accent) !important; }}

@keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
.turn {{ animation: fadeIn 0.25s ease; padding: 1.1rem 0; border-bottom: 1px solid var(--border); }}
.turn:last-child {{ border-bottom: none; }}

.role-label {{
    font-family: 'Inter', sans-serif; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.35rem;
}}
.role-label.user {{ text-align: right; }}

.user-bubble {{
    background: var(--bg-alt); color: var(--text); padding: 0.75rem 1rem; border-radius: 14px;
    max-width: 70%; margin-left: auto; font-family: 'Tiro Bangla', serif; font-size: calc(1.02rem * var(--font-scale));
}}

.answer-bn {{
    font-family: 'Tiro Bangla', serif; font-size: calc(1.08rem * var(--font-scale));
    line-height: 1.85; color: var(--text); min-height: 1.6rem;
}}
.cursor-blink::after {{ content: "▌"; animation: blink 0.9s steps(1) infinite; color: var(--accent); }}
@keyframes blink {{ 50% {{ opacity: 0; }} }}

.tag-row {{ margin-bottom: 0.5rem; }}
.tag {{
    display: inline-block; font-size: 0.72rem; font-weight: 500; color: var(--text-muted);
    background: var(--bg-alt); border: 1px solid var(--border); padding: 0.1rem 0.55rem;
    border-radius: 6px; margin-right: 0.35rem;
}}

.conf-tag {{
    display: inline-block; font-size: 0.76rem; font-weight: 500; padding: 0.15rem 0.6rem;
    border-radius: 6px; margin-top: 0.6rem; border: 1px solid var(--border);
}}
.conf-high {{ color: var(--accent); border-color: var(--accent); }}
.conf-med  {{ color: var(--text-muted); }}
.conf-low  {{ color: var(--text-muted); border-style: dashed; }}

.msg-meta {{ font-size: 0.72rem; color: var(--text-muted); margin-top: 0.3rem; }}

.ayah-frame {{
    direction: rtl; text-align: center; font-family: 'Source Serif 4', serif;
    font-size: calc(1.35rem * var(--font-scale)); color: var(--text); background: var(--bg-alt);
    border: 1px solid var(--border); border-left: 3px solid var(--accent);
    border-radius: 8px; padding: 0.9rem 1.2rem; line-height: 2;
}}
.ayah-frame .bracket {{ color: var(--accent); }}
.translation-box {{
    font-family: 'Tiro Bangla', serif; background: var(--bg-alt); border-left: 3px solid var(--accent);
    padding: 0.7rem 1rem; border-radius: 0 8px 8px 0; margin-top: 0.4rem; color: var(--text);
}}

.icon-btn, .related-btn {{
    display: inline-block; font-family: 'Inter', sans-serif; font-size: 0.76rem; color: var(--text-muted);
    background: transparent; border: 1px solid var(--border); border-radius: 6px;
    padding: 0.15rem 0.6rem; margin: 0.5rem 0.35rem 0 0; cursor: pointer;
}}
.icon-btn:hover, .related-btn:hover {{ border-color: var(--accent); color: var(--accent); }}

.tool-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 1.1rem 1.3rem; margin-bottom: 1rem; }}
.tool-card h4 {{ font-family: 'Source Serif 4', serif; margin: 0 0 0.6rem; font-size: calc(1.15rem * var(--font-scale)); }}
.result-line {{ display: flex; justify-content: space-between; padding: 0.35rem 0; border-bottom: 1px dashed var(--border); font-size: 0.95rem; }}
.result-line:last-child {{ border-bottom: none; }}
.result-line .label {{ color: var(--text-muted); }}
.result-line .value {{ font-weight: 600; }}
.result-final {{ margin-top: 0.6rem; padding: 0.8rem 1rem; border-radius: 8px; background: var(--bg-alt); border-left: 3px solid var(--accent); font-weight: 600; }}

.compass-wrap {{ text-align: center; padding: 1.2rem 0; }}
.compass-circle {{ width: 160px; height: 160px; border-radius: 50%; border: 2px solid var(--border); margin: 0 auto; position: relative; background: var(--bg-alt); }}
.compass-arrow {{ position: absolute; left: 50%; top: 50%; width: 3px; height: 68px; background: var(--accent); transform-origin: bottom center; margin-left: -1.5px; margin-top: -68px; }}

.quiz-option {{ padding: 0.6rem 0.9rem; border: 1px solid var(--border); border-radius: 8px; margin-bottom: 0.4rem; font-family: 'Tiro Bangla', serif; }}
.quiz-correct {{ border-color: var(--good); background: rgba(61,122,92,0.08); }}
.quiz-wrong {{ border-color: var(--bad); background: rgba(179,84,46,0.08); }}

.flashcard {{
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px;
    padding: 2.4rem 1.6rem; text-align: center; min-height: 160px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Tiro Bangla', serif; font-size: calc(1.2rem * var(--font-scale)); line-height: 1.7;
}}
.flashcard.front {{ font-family: 'Source Serif 4', serif; font-weight: 600; font-size: calc(1.5rem * var(--font-scale)); }}

.prayer-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.7rem; }}
.prayer-cell {{ background: var(--bg-alt); border-radius: 8px; padding: 0.8rem; text-align: center; }}
.prayer-cell .name {{ font-family: 'Tiro Bangla', serif; color: var(--text-muted); font-size: 0.85rem; }}
.prayer-cell .time {{ font-weight: 700; font-size: 1.15rem; margin-top: 0.2rem; }}

.disclaimer {{ font-size: 0.82rem; color: var(--text-muted); background: var(--bg-alt); border-radius: 8px; padding: 0.7rem 0.9rem; margin-top: 0.8rem; }}
.hikma-footer {{ text-align: center; color: var(--text-muted); padding: 1.6rem; font-family: 'Tiro Bangla', serif; font-size: 0.88rem; margin-top: 1rem; }}
</style>
"""

DARK_CSS = """
<style>
:root {
    --bg: #262624 !important; --bg-alt: #30302E !important; --bg-card: #2B2B29 !important;
    --text: #F5F4EF !important; --text-muted: #9A9990 !important; --border: #3D3D3A !important;
}
.stApp { background: #262624 !important; }
section[data-testid="stSidebar"] > div { background: #21211F !important; }
</style>
"""

# ---------------------------------------------------------------------------
# Session state & data
# ---------------------------------------------------------------------------
DEFAULTS = {
    "messages": [], "feedback": {}, "starred": set(), "category_counts": Counter(),
    "dark_mode": False, "just_answered": None, "font_size": "মাঝারি",
    "recent_searches": [],
    "quiz_questions": [], "quiz_index": 0, "quiz_score": 0, "quiz_answered": False, "quiz_selected": None,
    "flash_deck": [], "flash_index": 0, "flash_flipped": False, "flash_known": 0, "flash_review": 0,
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown(BASE_CSS_TEMPLATE.format(font_scale=FONT_SCALE[st.session_state.font_size]), unsafe_allow_html=True)
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
        "role": "assistant", "content": response["bangla"], "confidence": response["confidence"],
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

    recents = st.session_state.recent_searches
    if query_text in recents:
        recents.remove(query_text)
    recents.insert(0, query_text)
    st.session_state.recent_searches = recents[:8]


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

    font_choice = st.select_slider("লেখার আকার", options=list(FONT_SCALE.keys()), value=st.session_state.font_size)
    if font_choice != st.session_state.font_size:
        st.session_state.font_size = font_choice
        st.rerun()

    st.divider()
    st.markdown("**যেভাবে জিজ্ঞাসা করবেন**")
    st.caption("বাংলিশ: namaj, roja, zakat")
    st.caption("English: salah, fasting, zakat")
    st.caption("বাংলা: নামাজ, রোজা, যাকাত")

    if st.session_state.recent_searches:
        st.divider()
        st.markdown("**🕘 সাম্প্রতিক অনুসন্ধান**")
        for q in st.session_state.recent_searches:
            if st.button(q, key=f"recent_{q}", use_container_width=True):
                handle_query(q)
                st.rerun()

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
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "⬇️ Markdown", data="\n\n".join(transcript_lines),
            file_name="hikma_chat.md", mime="text/markdown", use_container_width=True,
        )
        dl2.download_button(
            "⬇️ CSV", data=messages_to_csv(st.session_state.messages),
            file_name="hikma_chat.csv", mime="text/csv", use_container_width=True,
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

tab_chat, tab_zakat, tab_qibla, tab_prayer, tab_browse, tab_quiz, tab_flash = st.tabs([
    "💬 প্রশ্ন-উত্তর", "🧮 যাকাত", "🕋 কিবলা", "🕰️ নামাজের সময়",
    "📚 বিষয় ব্রাউজ", "🎯 কুইজ", "🗂️ ফ্ল্যাশকার্ড",
])

# ===========================================================================
# TAB 1 — Chat Q&A
# ===========================================================================
with tab_chat:
    if not st.session_state.messages:
        st.caption("দ্রুত শুরু করতে একটি বিষয়ে ক্লিক করুন")
        cols = st.columns(4)
        for i, topic in enumerate(SUGGESTED_TOPICS):
            if cols[i % 4].button(topic, key=f"chip_{topic}", use_container_width=True):
                handle_query(topic)
                st.rerun()
        st.write("")

    CONF_LABELS = [
        (0.8, "conf-high", "উচ্চ আস্থা"), (0.5, "conf-med", "মাঝারি আস্থা"), (0.0, "conf-low", "কম আস্থা"),
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

# ===========================================================================
# TAB 2 — Zakat calculator
# ===========================================================================
with tab_zakat:
    st.markdown('<div class="tool-card"><h4>যাকাত হিসাব করুন</h4>', unsafe_allow_html=True)
    st.caption("নগদ অর্থ, স্বর্ণ-রূপার বর্তমান মূল্য ও অন্যান্য সম্পদের তথ্য দিন — নিসাব ও প্রদেয় যাকাত হিসাব করা হবে।")

    zc1, zc2 = st.columns(2)
    with zc1:
        cash = st.number_input("নগদ ও ব্যাংক ব্যালেন্স (টাকা)", min_value=0.0, value=0.0, step=1000.0)
        gold_grams = st.number_input("স্বর্ণের পরিমাণ (গ্রাম)", min_value=0.0, value=0.0, step=1.0)
        gold_price = st.number_input("প্রতি গ্রাম স্বর্ণের বর্তমান মূল্য (টাকা)", min_value=0.0, value=0.0, step=100.0)
        business_value = st.number_input("ব্যবসায়িক পণ্যের মূল্য (টাকা)", min_value=0.0, value=0.0, step=1000.0)
    with zc2:
        silver_grams = st.number_input("রূপার পরিমাণ (গ্রাম)", min_value=0.0, value=0.0, step=1.0)
        silver_price = st.number_input("প্রতি গ্রাম রূপার বর্তমান মূল্য (টাকা)", min_value=0.0, value=0.0, step=10.0)
        other_investments = st.number_input("অন্যান্য বিনিয়োগ/সঞ্চয় (টাকা)", min_value=0.0, value=0.0, step=1000.0)
        liabilities = st.number_input("তাৎক্ষণিক পরিশোধযোগ্য ঋণ (টাকা)", min_value=0.0, value=0.0, step=1000.0)

    nisab_basis = st.radio(
        "নিসাবের ভিত্তি", options=["silver", "gold"],
        format_func=lambda x: "রূপা (৬১২.৩৬ গ্রাম) — বেশি ক্ষেত্রে যাকাত ফরজ করে" if x == "silver" else "স্বর্ণ (৮৭.৪৮ গ্রাম)",
        horizontal=True,
    )

    if st.button("যাকাত হিসাব করুন", type="primary"):
        result = compute_zakat(cash, gold_grams, gold_price, silver_grams, silver_price,
                                business_value, other_investments, liabilities, nisab_basis)
        st.markdown(
            f"""
            <div class="result-line"><span class="label">স্বর্ণের মূল্য</span><span class="value">৳ {result['gold_value']:,.2f}</span></div>
            <div class="result-line"><span class="label">রূপার মূল্য</span><span class="value">৳ {result['silver_value']:,.2f}</span></div>
            <div class="result-line"><span class="label">মোট সম্পদ (ঋণ বাদ দেওয়ার আগে)</span><span class="value">৳ {result['total_assets']:,.2f}</span></div>
            <div class="result-line"><span class="label">নিট যাকাতযোগ্য সম্পদ</span><span class="value">৳ {result['net_wealth']:,.2f}</span></div>
            <div class="result-line"><span class="label">নিসাবের বর্তমান মূল্য</span><span class="value">৳ {result['nisab_value']:,.2f}</span></div>
            """,
            unsafe_allow_html=True,
        )
        if result["obligatory"]:
            st.markdown(
                f'<div class="result-final">✅ আপনার সম্পদ নিসাবের সমান বা বেশি — প্রদেয় যাকাত: ৳ {result["zakat_due"]:,.2f} (নিট সম্পদের ২.৫%)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="result-final">আপনার নিট সম্পদ নিসাবের কম — এই মুহূর্তে যাকাত ফরজ নয়।</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="disclaimer">
        ⚠️ এই ক্যালকুলেটরটি সাধারণ হানাফি নিসাব ওজন (স্বর্ণ ৮৭.৪৮ গ্রাম, রূপা ৬১২.৩৬ গ্রাম) ও ২.৫% হার ব্যবহার করে
        একটি সরল হিসাব দেয়। এটি কৃষি যাকাত, গবাদি পশুর যাকাত, বা মাযহাবভেদে ভিন্ন মতামত বিবেচনা করে না।
        সঠিক ও ব্যক্তিগত পরিস্থিতি অনুযায়ী হিসাবের জন্য একজন নির্ভরযোগ্য আলেমের পরামর্শ নিন।
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# TAB 3 — Qibla direction finder
# ===========================================================================
with tab_qibla:
    st.markdown('<div class="tool-card"><h4>কিবলার দিক নির্ণয় করুন</h4>', unsafe_allow_html=True)
    st.caption("আপনার অবস্থানের অক্ষাংশ ও দ্রাঘিমাংশ দিন, অথবা নিচের শহরগুলো থেকে বেছে নিন।")

    preset_cols = st.columns(4)
    preset_choice = None
    for i, city in enumerate(BD_CITY_PRESETS):
        if preset_cols[i % 4].button(city, key=f"city_{city}", use_container_width=True):
            preset_choice = city
    if preset_choice:
        st.session_state["loc_lat"] = BD_CITY_PRESETS[preset_choice][0]
        st.session_state["loc_lon"] = BD_CITY_PRESETS[preset_choice][1]

    qc1, qc2 = st.columns(2)
    with qc1:
        lat = st.number_input("অক্ষাংশ (Latitude)", min_value=-90.0, max_value=90.0,
                               value=st.session_state.get("loc_lat", 23.8103), format="%.4f")
    with qc2:
        lon = st.number_input("দ্রাঘিমাংশ (Longitude)", min_value=-180.0, max_value=180.0,
                               value=st.session_state.get("loc_lon", 90.4125), format="%.4f")

    if st.button("কিবলার দিক নির্ণয় করুন", type="primary"):
        bearing = compute_qibla_bearing(lat, lon)
        distance = haversine_km(lat, lon, KAABA_LAT, KAABA_LON)
        compass = bearing_to_compass_bn(bearing)
        st.markdown(
            f"""
            <div class="compass-wrap">
                <div class="compass-circle"><div class="compass-arrow" style="transform: rotate({bearing}deg);"></div></div>
            </div>
            <div class="result-final" style="text-align:center;">
                কিবলা উত্তর দিক থেকে {bearing:.1f}° ({compass}) কোণে অবস্থিত — কাবা শরীফ থেকে দূরত্ব প্রায় {distance:,.0f} কিমি
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="disclaimer">
        ℹ️ এই কম্পাসটি একটি গাণিতিক গ্রেট-সার্কেল হিসাব দেখায়, বাস্তব কম্পাস নয় — সঠিক দিকনির্দেশনার জন্য
        একটি ফোনের কম্পাস অ্যাপ বা ভৌত কম্পাসের সাথে এই কোণ মিলিয়ে দিক নির্ধারণ করুন।
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# TAB 4 — Prayer times + Hijri date
# ===========================================================================
with tab_prayer:
    st.markdown('<div class="tool-card"><h4>নামাজের আনুমানিক সময়সূচি ও হিজরি তারিখ</h4>', unsafe_allow_html=True)

    preset_cols2 = st.columns(4)
    preset_choice2 = None
    for i, city in enumerate(BD_CITY_PRESETS):
        if preset_cols2[i % 4].button(city, key=f"pcity_{city}", use_container_width=True):
            preset_choice2 = city
    if preset_choice2:
        st.session_state["loc_lat"] = BD_CITY_PRESETS[preset_choice2][0]
        st.session_state["loc_lon"] = BD_CITY_PRESETS[preset_choice2][1]

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        p_lat = st.number_input("অক্ষাংশ", min_value=-90.0, max_value=90.0,
                                 value=st.session_state.get("loc_lat", 23.8103), format="%.4f", key="p_lat")
    with pc2:
        p_lon = st.number_input("দ্রাঘিমাংশ", min_value=-180.0, max_value=180.0,
                                 value=st.session_state.get("loc_lon", 90.4125), format="%.4f", key="p_lon")
    with pc3:
        p_date = st.date_input("তারিখ", value=date.today())

    mc1, mc2 = st.columns(2)
    with mc1:
        angle_method = st.selectbox(
            "গণনা পদ্ধতি (ফজর/এশার কোণ)",
            ["করাচি (১৮°/১৮°)", "মুসলিম ওয়ার্ল্ড লীগ (১৮°/১৭°)", "উম্মুল কুরা (১৯°/৯০ মিনিট)"],
        )
    with mc2:
        asr_method = st.radio("আসরের হিসাব", ["হানাফি (ছায়া×২)", "শাফি/মালিকি/হাম্বলি (ছায়া×১)"], horizontal=True)

    angle_map = {
        "করাচি (১৮°/১৮°)": (18.0, 18.0),
        "মুসলিম ওয়ার্ল্ড লীগ (১৮°/১৭°)": (18.0, 17.0),
        "উম্মুল কুরা (১৯°/৯০ মিনিট)": (19.0, 18.0),
    }
    fajr_angle, isha_angle = angle_map[angle_method]
    asr_hanafi = asr_method.startswith("হানাফি")

    if st.button("সময়সূচি দেখান", type="primary"):
        times = compute_prayer_times(p_lat, p_lon, 6.0, p_date, fajr_angle, isha_angle, asr_hanafi)
        cells = "".join(
            f'<div class="prayer-cell"><div class="name">{name}</div><div class="time">{t}</div></div>'
            for name, t in times.items()
        )
        st.markdown(f'<div class="prayer-grid">{cells}</div>', unsafe_allow_html=True)

        hy, hm, hd = gregorian_to_hijri(p_date)
        st.markdown(
            f'<div class="result-final" style="margin-top:1rem;">হিজরি তারিখ (আনুমানিক): {hd} {HIJRI_MONTHS_BN[hm - 1]} {hy} হিজরি</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="disclaimer">
        ⚠️ সময়সূচি সূর্যের অবস্থানের উপর ভিত্তি করে গাণিতিকভাবে হিসাব করা — এটি একটি প্রতিষ্ঠিত পদ্ধতি,
        কিন্তু স্থানীয় চাঁদ দেখা কমিটি বা মসজিদ কর্তৃক প্রকাশিত অফিসিয়াল সময়সূচির সাথে ১-২ মিনিট পার্থক্য হতে পারে।
        হিজরি তারিখও একটি হিসাবভিত্তিক (তাবুলার) রূপান্তর — বাস্তব চাঁদ দেখা অনুযায়ী ঘোষিত তারিখের চেয়ে ১-২ দিন
        ভিন্ন হতে পারে। নামাজ ও রোজার নির্ভুল সময়ের জন্য সর্বদা স্থানীয় মসজিদ বা ইসলামিক ফাউন্ডেশনের সময়সূচি অনুসরণ করুন।
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# TAB 5 — Dataset browser
# ===========================================================================
with tab_browse:
    st.markdown('<div class="tool-card"><h4>বিষয়সমূহ ব্রাউজ করুন</h4>', unsafe_allow_html=True)

    bc1, bc2 = st.columns([3, 1])
    with bc1:
        search_term = st.text_input("খুঁজুন", placeholder="বিষয়, কিওয়ার্ড, বা উত্তরে খুঁজুন...", label_visibility="collapsed")
    with bc2:
        cat_options = ["সব বিষয়শ্রেণি"] + sorted(c for c in df["category"].unique() if str(c).strip())
        browse_cat = st.selectbox("বিষয়শ্রেণি", cat_options, label_visibility="collapsed")

    filtered = df
    if browse_cat != "সব বিষয়শ্রেণি":
        filtered = filtered[filtered["category"] == browse_cat]
    if search_term.strip():
        term = search_term.strip().lower()
        mask = (
            filtered["topic"].astype(str).str.lower().str.contains(term, na=False)
            | filtered["keywords"].astype(str).str.lower().str.contains(term, na=False)
            | filtered["answer_bangla"].astype(str).str.lower().str.contains(term, na=False)
        )
        filtered = filtered[mask]

    st.caption(f"{len(filtered):,}টি ফলাফল পাওয়া গেছে" + (" — প্রথম ২০০টি দেখানো হচ্ছে" if len(filtered) > 200 else ""))

    if not filtered.empty:
        conf_counts = filtered["confidence"].value_counts()
        st.caption("আস্থার মাত্রা অনুযায়ী বণ্টন")
        st.bar_chart(conf_counts)

    display_df = filtered[["topic", "category", "answer_bangla"]].head(200).copy()
    display_df["answer_bangla"] = display_df["answer_bangla"].str.slice(0, 120) + "…"
    display_df.columns = ["বিষয়", "বিষয়শ্রেণি", "সংক্ষিপ্ত উত্তর"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="tool-card"><h4>দুটি বিষয় পাশাপাশি তুলনা করুন</h4>', unsafe_allow_html=True)
    all_topics = sorted(df["topic"].drop_duplicates().tolist())
    cmp1, cmp2 = st.columns(2)
    with cmp1:
        topic_a = st.selectbox("প্রথম বিষয়", all_topics, index=0 if all_topics else None, key="cmp_a")
    with cmp2:
        topic_b = st.selectbox("দ্বিতীয় বিষয়", all_topics, index=min(1, len(all_topics) - 1) if all_topics else None, key="cmp_b")

    if topic_a and topic_b:
        row_a, row_b = hikma.get_by_topic(topic_a), hikma.get_by_topic(topic_b)
        cola, colb = st.columns(2)
        for col, row, name in [(cola, row_a, topic_a), (colb, row_b, topic_b)]:
            with col:
                st.markdown(f"**{name}**")
                if row is not None:
                    st.markdown(f'<div class="answer-bn">{row["answer_bangla"]}</div>', unsafe_allow_html=True)
                    if str(row.get("sources", "")).strip():
                        st.caption(f"সূত্র: {row['sources']}")

    st.markdown("</div>", unsafe_allow_html=True)

# ===========================================================================
# TAB 6 — Quiz / practice mode
# ===========================================================================
with tab_quiz:
    st.markdown('<div class="tool-card"><h4>নিজেকে যাচাই করুন</h4>', unsafe_allow_html=True)
    st.caption("একটি বিষয়ের নাম দেখানো হবে — চারটি উত্তরের মধ্যে থেকে সঠিকটি বেছে নিন।")

    qcol1, qcol2 = st.columns([1, 3])
    with qcol1:
        if st.button("🔄 নতুন কুইজ শুরু করুন", use_container_width=True):
            st.session_state.quiz_questions = build_quiz_questions(df, n=5)
            st.session_state.quiz_index = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_answered = False
            st.session_state.quiz_selected = None
            st.rerun()
    with qcol2:
        if st.session_state.quiz_questions:
            st.caption(f"স্কোর: {st.session_state.quiz_score} / {len(st.session_state.quiz_questions)}")

    if not st.session_state.quiz_questions:
        st.info("শুরু করতে 'নতুন কুইজ শুরু করুন' বাটনে ক্লিক করুন।")
    elif st.session_state.quiz_index >= len(st.session_state.quiz_questions):
        total, score = len(st.session_state.quiz_questions), st.session_state.quiz_score
        st.markdown(
            f'<div class="result-final">কুইজ শেষ! আপনার স্কোর: {score} / {total} ({int(100 * score / total)}%)</div>',
            unsafe_allow_html=True,
        )
    else:
        q = st.session_state.quiz_questions[st.session_state.quiz_index]
        st.markdown(f"**প্রশ্ন {st.session_state.quiz_index + 1}: \"{q['topic']}\" সম্পর্কে সঠিক উত্তরটি কোনটি?**")

        if not st.session_state.quiz_answered:
            choice = st.radio("উত্তর বেছে নিন", q["options"], key=f"quiz_radio_{st.session_state.quiz_index}", label_visibility="collapsed")
            if st.button("উত্তর জমা দিন"):
                st.session_state.quiz_selected = choice
                st.session_state.quiz_answered = True
                if choice == q["correct"]:
                    st.session_state.quiz_score += 1
                st.rerun()
        else:
            for opt in q["options"]:
                css_class = "quiz-option"
                if opt == q["correct"]:
                    css_class += " quiz-correct"
                elif opt == st.session_state.quiz_selected:
                    css_class += " quiz-wrong"
                st.markdown(f'<div class="{css_class}">{opt}</div>', unsafe_allow_html=True)

            if st.session_state.quiz_selected == q["correct"]:
                st.success("সঠিক উত্তর! ✅")
            else:
                st.error("সঠিক উত্তর নয় — সঠিকটি সবুজে চিহ্নিত করা হয়েছে।")

            if q.get("source"):
                st.caption(f"সূত্র: {q['source']}")

            if st.button("পরবর্তী প্রশ্ন ➜"):
                st.session_state.quiz_index += 1
                st.session_state.quiz_answered = False
                st.session_state.quiz_selected = None
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ===========================================================================
# TAB 7 — Flashcard study mode
# ===========================================================================
with tab_flash:
    st.markdown('<div class="tool-card"><h4>ফ্ল্যাশকার্ড দিয়ে অনুশীলন করুন</h4>', unsafe_allow_html=True)
    st.caption("বিষয়ের নাম দেখে উত্তর মনে করার চেষ্টা করুন, তারপর কার্ড উল্টে মিলিয়ে নিন।")

    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        flash_cat_options = ["সব বিষয়শ্রেণি"] + sorted(c for c in df["category"].unique() if str(c).strip())
        flash_cat = st.selectbox("বিষয়শ্রেণি বেছে নিন", flash_cat_options, key="flash_cat_select")
    with fc2:
        flash_n = st.number_input("কার্ড সংখ্যা", min_value=3, max_value=20, value=10, step=1)
    with fc3:
        st.write("")
        st.write("")
        if st.button("🔄 নতুন ডেক", use_container_width=True):
            st.session_state.flash_deck = build_flashcard_deck(df, flash_cat, int(flash_n))
            st.session_state.flash_index = 0
            st.session_state.flash_flipped = False
            st.session_state.flash_known = 0
            st.session_state.flash_review = 0
            st.rerun()

    deck = st.session_state.flash_deck
    if not deck:
        st.info("শুরু করতে 'নতুন ডেক' বাটনে ক্লিক করুন।")
    elif st.session_state.flash_index >= len(deck):
        seen = st.session_state.flash_known + st.session_state.flash_review
        st.markdown(
            f'<div class="result-final">ডেক শেষ! জানা: {st.session_state.flash_known} · '
            f'আরও অনুশীলন প্রয়োজন: {st.session_state.flash_review} (মোট {seen})</div>',
            unsafe_allow_html=True,
        )
    else:
        card = deck[st.session_state.flash_index]
        st.caption(f"কার্ড {st.session_state.flash_index + 1} / {len(deck)}")

        if not st.session_state.flash_flipped:
            st.markdown(f'<div class="flashcard front">{card["topic"]}</div>', unsafe_allow_html=True)
            if st.button("🔄 কার্ড উল্টান", use_container_width=True):
                st.session_state.flash_flipped = True
                st.rerun()
        else:
            st.markdown(f'<div class="flashcard">{card["answer"]}</div>', unsafe_allow_html=True)
            if card.get("source"):
                st.caption(f"সূত্র: {card['source']}")

            gcol1, gcol2 = st.columns(2)
            if gcol1.button("✅ জানতাম", use_container_width=True):
                st.session_state.flash_known += 1
                st.session_state.flash_index += 1
                st.session_state.flash_flipped = False
                st.rerun()
            if gcol2.button("🔁 আবার অনুশীলন দরকার", use_container_width=True):
                st.session_state.flash_review += 1
                st.session_state.flash_index += 1
                st.session_state.flash_flipped = False
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hikma-footer">HIKMA — স্পষ্ট প্রশ্ন · নির্ভরযোগ্য উত্তর · যাচাইকৃত সূত্র</div>
    """,
    unsafe_allow_html=True,
)
