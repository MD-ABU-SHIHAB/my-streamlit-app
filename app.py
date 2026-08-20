""")

# Show actual model performance
try:
y_true = le.inverse_transform(classifier.classes_)
y_pred = le.inverse_transform(classifier.predict(X))
acc = accuracy_score(y_true, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{acc:.1%}")
col2.metric("Classes", len(TIER1_CLASSES))
col3.metric("Samples", len(df))
col4.metric("Features", X.shape[1])

st.markdown("### Per-Class Performance")
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose().round(3)
st.dataframe(report_df, use_container_width=True)

# Confusion matrix
fig = create_confusion_matrix(y_true, y_pred, le.classes_)
st.plotly_chart(fig, use_container_width=True)

# Cross-validation
cv_scores = cross_val_score(classifier, X, le.transform(y_true), cv=5)
st.markdown(f"**5-Fold Cross-Validation**: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

except Exception as e:
st.warning(f"Performance metrics not available: {str(e)}")

st.markdown("## 📋 5. Algorithm Comparison")
comparison_data = {
"Algorithm": ["Logistic Regression (Used)", "Linear Regression", "Neural Networks", "SVM", "Decision Trees", "k-NN", "Random Forest", "K-Means", "PCA"],
"Type": ["Classification", "Regression", "Both", "Classification", "Classification", "Classification", "Ensemble", "Clustering", "Dim. Reduction"],
"Pros": ["Interpretable", "Simple", "Powerful", "Great margins", "Interpretable", "No training", "Robust", "Simple", "Visualization"],
"Cons": ["Linear boundaries", "Linear only", "Black box", "Kernel selection", "Overfitting", "Slow", "Less interpretable", "K selection", "Linear only"]
}
st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)


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


# --------------------------------------------------------------------------- #
# MAIN APPLICATION
# --------------------------------------------------------------------------- #

def main() -> None:
st.set_page_config(
page_title="Islamic Ruling Reference",
page_icon="🕌",
layout="wide",
initial_sidebar_state="collapsed"
)

inject_css()

# Watermark
st.markdown("""
<div class="watermark">
<img src="https://hstu.ac.bd/img/hstu_logo_.png" alt="HSTU Logo" />
</div>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
<div class="app-title">Ruling Reference</div>
<div class="app-subtitle">Mas'alah Search · CSE 469 Capstone · HSTU</div>
</div>
<hr class="header-divider">
""", unsafe_allow_html=True)

try:
df = load_dataset(DATA_PATH)
df = build_combined_text(df)
banglish_map = build_banglish_map(df)
vectorizer, X, classifier, le = train_classifier(df)
except DatasetError as exc:
st.error(str(exc))
st.stop()
except Exception as exc:
st.error(f"Startup error: {exc}")
st.stop()

# Mode toggle
mode = st.radio(
"Mode",
["🔍 Search", "📚 Browse"],
horizontal=True,
label_visibility="collapsed"
)

if mode == "🔍 Search":
with st.container():
st.markdown('<div class="search-container">', unsafe_allow_html=True)
st.markdown('<div class="search-label">🔎 Type your question</div>', unsafe_allow_html=True)

col_search, col_button = st.columns([5, 1])
with col_search:
    default_query = st.session_state.pop("pending_query", "")
    query = st.text_input(
        "Search",
        value=default_query,
        placeholder="namaj pora ki · is riba haram · বিয়ে করা কি সুন্নত…",
        label_visibility="collapsed",
        key="search_input"
    )
with col_button:
    search_clicked = st.button("Search", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if query.strip() or search_clicked:
if query.strip():
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    
    result = retrieve_candidates(query, df, vectorizer, X, banglish_map)
    
    st.session_state.search_history.append(
        SearchHistory(
            query=query,
            timestamp=datetime.now(),
            result_id=int(result.row["id"]) if result.row is not None else None,
            stage=result.stage if result.row is not None else None
        )
    )
    
    with st.container():
        col_left, col_center, col_right = st.columns([1, 2.5, 1.5], gap="large")
        
        with col_left:
            render_stats(df)
            render_search_history()
        
        with col_center:
            if result.row is not None:
                render_result_card(result, df, vectorizer, classifier, le)
            else:
                render_no_match(result)
        
        with col_right:
            if result.row is not None:
                render_apparatus_panel(result, vectorizer, classifier, le)
else:
    st.info("Please type a question to search")
else:
with st.container():
    col_left, col_center, col_right = st.columns([1, 2.5, 1.5], gap="large")
    with col_left:
        render_stats(df)
        render_search_history()
    with col_center:
        st.markdown("""
        <div style="text-align:center; padding:4rem 0;">
            <div style="font-size:4rem; opacity:0.3;">🕌</div>
            <div style="color:var(--text-muted); margin-top:1rem;">
                Search for Islamic rulings in Bangla, English, or Banglish
            </div>
            <div style="color:var(--text-muted); font-size:0.85rem; margin-top:0.5rem;">
                Try: "namaz", "riba", "বিয়ে", "interest"
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_right:
        st.markdown("""
        <div class="glass-card">
            <div class="glass-card-title">💡 Quick Tips</div>
            <div style="font-size:0.85rem; color:var(--text-secondary); line-height:1.8;">
                • Use Bangla, English, or Banglish<br>
                • Be specific for better results<br>
                • Browse all rulings via "Browse" mode<br>
                • Check Apparatus panel for ML details
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
render_browse_mode(df)

# ML Details
render_ml_details(df, vectorizer, classifier, le, X)

# Footer
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; border-top:1px solid rgba(255,255,255,0.05); margin-top:2rem;">
<div style="font-family:'Noto Serif Bengali',serif; font-size:1.2rem; color:var(--gold); opacity:0.6;">
রব্বি জিদনি ইলমা
</div>
<div style="color:var(--text-muted); font-size:0.75rem; margin-top:0.5rem;">
CSE 469 · Machine Learning · Hajee Mohammad Danesh Science and Technology University
</div>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
main()
