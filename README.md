# 🕌 HIKMA — Islamic Knowledge AI

**Ask in Banglish, get answers in Bangla — with sources.**

HIKMA is a Bangla-first Q&A tool for everyday Islamic knowledge. Ask a
question the way you'd actually type it — Banglish (`namaj`, `roja`,
`zakat`), English, or Bangla script — and get a sourced answer in Bangla,
with the original Arabic text, its translation, and an honesty-first
confidence score.

---

## ✨ Features

- **Multilingual input** — Banglish, English, and Bangla script all resolve to the same answer.
- **Source-grounded answers** — every response is tied back to a reference (Qur'an ayah / hadith) in the dataset, never generated freely.
- **Arabic + translation** — the original Arabic text (when available) is shown framed with traditional ﴾ ﴿ ayah brackets, alongside its Bangla translation.
- **Confidence, honestly reported** — a weak keyword match is never dressed up as a strong one.
- **Fast at scale** — an inverted keyword index means lookups stay quick even across tens of thousands of entries, instead of scanning the whole dataset per question.
- **Interactive UI** — clickable topic chips to start fast, a "random topic" explorer, Enter-to-send, and quick 👍/👎 feedback on each answer.

## 🖼️ Preview

*(Add a screenshot or screen recording of the app here once deployed.)*

## 🧱 Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [pandas](https://pandas.pydata.org/) — dataset loading and lookups
- Python standard library (`difflib`, `functools.lru_cache`) — fuzzy matching and query caching

## 📂 Project Structure

```
.
├── app.py             # matching engine + Streamlit UI, single file
├── dataset.csv         # the knowledge base (see below)
└── requirements.txt
```

## 🚀 Running Locally

```bash
git clone https://github.com/MD-ABU-SHIHAB/my-streamlit-app.git
cd my-streamlit-app
pip install -r requirements.txt
streamlit run app.py
```

The app expects `dataset.csv` in the same folder as `app.py`.

## 📊 Dataset

`dataset.csv` holds the knowledge base, one row per topic/question:

| Column | Description |
|---|---|
| `id` | Row identifier |
| `topic` | Topic name |
| `keywords` | Comma-separated search variants (Banglish, English, Bangla, common typos) |
| `category` | Topic category (e.g. fiqh, aqidah) |
| `answer_bangla` | The main answer, in Bangla |
| `answer_english` | English summary of the same answer |
| `arabic_text` | Arabic source text, where confidently verified |
| `arabic_bangla` | Bangla translation of `arabic_text` |
| `sources` | Citation (e.g. Qur'an — Surah, ayah / hadith reference) |
| `confidence` | `high` / `medium` — how well-verified the entry is |

New entries are added carefully: unverified Arabic text and references are
left blank rather than guessed, and fiqh rulings with known scholarly
disagreement are presented as such rather than as settled.

## ⚠️ Disclaimer

HIKMA is an educational tool built from a curated dataset, not a substitute
for a qualified scholar. Answers — especially on fiqh rulings involving
personal circumstances (marriage, divorce, inheritance, zakat calculation,
etc.) — should be verified with a trusted local scholar before acting on them.

## 📄 License

*(Add a license, e.g. MIT, if you want others to be able to reuse this.)*

## 👤 Author

**Md. Abu Shihab**
[GitHub](https://github.com/MD-ABU-SHIHAB) · [LinkedIn](https://linkedin.com/in/mashihab)
