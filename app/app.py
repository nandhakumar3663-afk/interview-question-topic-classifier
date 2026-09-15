"""
EMP-26 Interview Question Topic Classifier
Interactive Streamlit Web Application
Features:
- Single-question inference with confidence gauge and probability distribution
- Configurable confidence threshold with visual low-confidence human review flags
- Bonus Feature: Semantic similar-question retrieval
- Batch CSV upload and automated classification export
- Model comparison and evaluation metrics dashboard
"""

import sys
import io
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import (
    TOPIC_CATEGORIES,
    TOPIC_COLORS,
    CONFUSION_MATRIX_PATH,
    MODEL_COMPARISON_PATH,
    EVALUATION_METRICS_PATH,
    TFIDF_MODEL_PATH,
    EMBEDDING_CLASSIFIER_PATH,
)
from src.predict import predict_topic, find_similar_questions

# Page Configuration
st.set_page_config(
    page_title="EMP-26 Interview Topic Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .topic-badge {
        display: inline-block;
        padding: 0.4rem 1.1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        margin-top: 0.5rem;
    }
    .review-flag-danger {
        background-color: #FEF2F2;
        border: 1px solid #F87171;
        border-left: 5px solid #DC2626;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        color: #991B1B;
        font-weight: 600;
        margin-top: 0.8rem;
    }
    .review-flag-success {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        border-left: 5px solid #16A34A;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        color: #166534;
        font-weight: 600;
        margin-top: 0.8rem;
    }
    .similar-box {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=70)
st.sidebar.title("Configuration")
st.sidebar.markdown("EMP-26 Lightweight NLP Classifier")

available_models = []
if TFIDF_MODEL_PATH.exists():
    available_models.append("TF-IDF Baseline")
if EMBEDDING_CLASSIFIER_PATH.exists():
    available_models.append("Sentence Transformer")

# Default fallback if models are being built
if not available_models:
    available_models = ["TF-IDF Baseline"]

selected_model_name = st.sidebar.selectbox(
    "Select Classifier Model",
    options=available_models,
    index=len(available_models) - 1,
    help="Switch between the lightweight TF-IDF baseline and the semantic SentenceTransformer candidate.",
)
model_key = "embedding" if "Sentence" in selected_model_name else "tfidf"

confidence_threshold = st.sidebar.slider(
    "Review Confidence Threshold",
    min_value=0.40,
    max_value=0.95,
    value=0.65,
    step=0.05,
    help="Predictions with confidence below this threshold will be flagged for human review.",
)

show_similar = st.sidebar.checkbox("Enable Similar-Question Retrieval", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Required Categories")
for cat in TOPIC_CATEGORIES:
    color = TOPIC_COLORS.get(cat, "#3B82F6")
    st.sidebar.markdown(
        f"<span style='color:{color}; font-weight:bold;'>•</span> {cat}",
        unsafe_allow_html=True,
    )


# Header
st.markdown('<div class="main-header">🎯 EMP-26 Interview Question Topic Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated classification across 5 core competencies with confidence scoring, human review flagging, and semantic similarity search.</div>', unsafe_allow_html=True)

tab_single, tab_batch, tab_evaluation = st.tabs([
    "🔍 Single Question Inference",
    "📁 Batch CSV Processing",
    "📊 Evaluation & Model Performance",
])

# =========================================================
# TAB 1: Single Question Inference
# =========================================================
with tab_single:
    col_input, col_preset = st.columns([3, 1])

    with col_preset:
        st.markdown("**Sample Presets**")
        presets = [
            "How do you handle race conditions and deadlocks in multithreaded services?",
            "How do you explain technical debt to non-technical business executives?",
            "Tell me about a time when you diagnosed a critical memory leak in production.",
            "Describe a time when you mentored a junior engineer struggling with code quality.",
            "How do you configure Kubernetes Horizontal Pod Autoscalers with Prometheus metrics?",
            "What is your general approach to working on a team project?",  # Ambiguous edge case
        ]
        selected_preset = st.selectbox("Select Preset Question:", [""] + presets, index=0)

    with col_input:
        default_val = selected_preset if selected_preset else "How does database indexing work under the hood in PostgreSQL?"
        input_question = st.text_area(
            "Enter Interview Question to Classify:",
            value=default_val,
            height=95,
            placeholder="Type or paste any software engineering interview question...",
        )
        classify_btn = st.button("🚀 Classify Question", type="primary", use_container_width=True)

    if input_question and (classify_btn or selected_preset):
        with st.spinner("Analyzing question..."):
            result = predict_topic(
                input_question,
                model_type=model_key,
                threshold=confidence_threshold,
            )

        st.markdown("---")
        res_col1, res_col2 = st.columns([1.1, 1.9])

        with res_col1:
            st.subheader("Prediction Result")
            topic = result["predicted_topic"]
            confidence = result["confidence"]
            conf_pct = result["confidence_pct"]
            badge_color = TOPIC_COLORS.get(topic, "#3B82F6")

            st.markdown(
                f'<div class="topic-badge" style="background-color: {badge_color};">{topic}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Confidence Score:** `{conf_pct}`")
            st.markdown(f"**Model Engine:** `{result['model_used']}`")

            # Human review flag
            if result["needs_review"]:
                st.markdown(
                    f"""
                    <div class="review-flag-danger">
                        ⚠️ <strong>FLAGGED FOR HUMAN REVIEW</strong><br>
                        Confidence ({conf_pct}) is below the {int(confidence_threshold * 100)}% threshold.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="review-flag-success">
                        ✅ <strong>HIGH CONFIDENCE PREDICTION</strong><br>
                        Confidence exceeds the {int(confidence_threshold * 100)}% quality threshold.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with res_col2:
            st.subheader("Topic Probability Distribution")
            prob_df = pd.DataFrame([
                {"Topic": cat, "Probability": prob * 100}
                for cat, prob in result["probabilities"].items()
            ]).sort_values(by="Probability", ascending=True)

            st.bar_chart(prob_df.set_index("Topic"), color="#2563EB", horizontal=True)

        # Bonus: Similar Questions
        if show_similar:
            st.markdown("---")
            st.subheader("💡 Semantically Similar Historical Questions (Bonus Feature)")
            with st.spinner("Searching historical knowledge base..."):
                similar_items = find_similar_questions(input_question, top_k=4)

            if not similar_items:
                st.info("No close historical matches found in training corpus.")
            else:
                sim_cols = st.columns(len(similar_items))
                for i, item in enumerate(similar_items):
                    with sim_cols[i]:
                        sim_color = TOPIC_COLORS.get(item["topic"], "#3B82F6")
                        st.markdown(
                            f"""
                            <div class="similar-box">
                                <span style="background:{sim_color}; color:white; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:bold;">
                                    {item['topic']}
                                </span>
                                <div style="font-size:0.85rem; color:#475569; margin-top:6px; font-weight:600;">
                                    Match: {item['similarity_pct']}
                                </div>
                                <div style="font-size:0.85rem; color:#1E293B; margin-top:4px;">
                                    {item['question']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

# =========================================================
# TAB 2: Batch CSV Processing
# =========================================================
with tab_batch:
    st.subheader("📁 Bulk Classification via CSV Upload")
    st.markdown("Upload a CSV file containing an interview question column (named `question` or `text`). The classifier will process all rows, append the predicted topic, confidence score, and review flag, and provide a downloadable CSV.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            col_options = list(batch_df.columns)
            
            # Auto-detect question column
            default_col = "question" if "question" in col_options else col_options[0]
            selected_col = st.selectbox("Select the Question column:", col_options, index=col_options.index(default_col))

            if st.button("⚡ Run Batch Classification", type="primary"):
                with st.spinner(f"Classifying {len(batch_df)} questions..."):
                    results = []
                    for q in batch_df[selected_col]:
                        res = predict_topic(str(q), model_type=model_key, threshold=confidence_threshold)
                        results.append({
                            "predicted_topic": res["predicted_topic"],
                            "confidence": res["confidence"],
                            "needs_review": res["needs_review"],
                        })
                    res_df = pd.DataFrame(results)
                    output_df = pd.concat([batch_df.reset_index(drop=True), res_df], axis=1)

                st.success(f"Successfully processed {len(output_df)} questions!")

                # Distribution stats
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Questions", len(output_df))
                m2.metric("Flagged for Review", int(output_df["needs_review"].sum()))
                m3.metric("Average Confidence", f"{output_df['confidence'].mean() * 100:.1f}%")

                st.dataframe(output_df.head(10), use_container_width=True)

                # Download CSV
                csv_buffer = io.StringIO()
                output_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Labeled CSV Results",
                    data=csv_buffer.getvalue(),
                    file_name="classified_interview_questions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# =========================================================
# TAB 3: Evaluation & Model Performance
# =========================================================
with tab_evaluation:
    st.subheader("📊 Empirical Model Performance & Benchmarks")
    st.markdown("Evaluated on the held-out test split (15% stratified test partition) across all 5 topic classes.")

    eval_col1, eval_col2 = st.columns(2)

    with eval_col1:
        st.markdown("**Confusion Matrix**")
        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate the test set confusion matrix heatmap.")

    with eval_col2:
        st.markdown("**Model Metric Comparison**")
        if MODEL_COMPARISON_PATH.exists():
            st.image(str(MODEL_COMPARISON_PATH), use_container_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate the comparative model benchmark chart.")

    if EVALUATION_METRICS_PATH.exists():
        import json
        with open(EVALUATION_METRICS_PATH, "r") as f:
            metrics_data = json.load(f)

        st.markdown("---")
        st.subheader("Detailed Test Metrics")
        for model_title, data in metrics_data.items():
            with st.expander(f"📌 {model_title} Classification Report", expanded=True):
                m = data.get("metrics", {})
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Accuracy", f"{m.get('accuracy', 0)*100:.2f}%")
                k2.metric("Macro Precision", f"{m.get('macro_precision', 0)*100:.2f}%")
                k3.metric("Macro Recall", f"{m.get('macro_recall', 0)*100:.2f}%")
                k4.metric("Macro F1-Score", f"{m.get('macro_f1', 0):.4f}")

                if "classification_report_str" in data:
                    st.text(data["classification_report_str"])
