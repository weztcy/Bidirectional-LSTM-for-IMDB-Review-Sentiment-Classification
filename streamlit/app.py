"""
Streamlit Web Application for IMDb Sentiment Classification.
Interactive UI for real-time review sentiment prediction using the trained BiLSTM model.
"""
import os
import json
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="IMDb Sentiment Classifier | BiLSTM",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .sentiment-card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .pos-card {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
    }
    .neg-card {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Lazy import prediction wrapper
@st.cache_resource
def load_predictor():
    from src.predict import SentimentPredictor
    return SentimentPredictor()

@st.cache_data
def load_metrics():
    bilstm_metrics_path = os.path.join("metrics", "bilstm_test_metrics.json")
    if not os.path.exists(bilstm_metrics_path):
        bilstm_metrics_path = os.path.join("outputs", "metrics", "bilstm_test_metrics.json")
        
    baseline_metrics_path = os.path.join("metrics", "baseline_metrics.json")
    if not os.path.exists(baseline_metrics_path):
        baseline_metrics_path = os.path.join("outputs", "metrics", "baseline_metrics.json")
    
    bilstm_data, baseline_data = None, None
    if os.path.exists(bilstm_metrics_path):
        with open(bilstm_metrics_path, "r", encoding="utf-8") as f:
            bilstm_data = json.load(f)
    if os.path.exists(baseline_metrics_path):
        with open(baseline_metrics_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
    return bilstm_data, baseline_data

def find_figure(name):
    p1 = os.path.join("figures", name)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join("outputs", "figures", name)
    if os.path.exists(p2):
        return p2
    return None

# Sidebar
st.sidebar.title("🎬 About Project")
st.sidebar.markdown("""
**Model:** Bidirectional LSTM (BiLSTM)  
**Dataset:** IMDb Large Movie Review Dataset (Maas et al., 2011)  
**Task:** Binary Sentiment Classification  
**Framework:** PyTorch & Scikit-Learn  
""")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Architecture Highlights")
st.sidebar.markdown("""
- **Embedding Dim:** 128
- **Hidden Dim:** 128 (x2 directions = 256)
- **Pooling:** Global Max Pooling over time
- **Regularization:** Dropout (0.30)
- **Sequence Length:** 256 tokens
- **Vocabulary:** 30,000 tokens
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Model Limitation:** Trained exclusively on movie reviews. It may misinterpret extreme sarcasm, highly ambivalent reviews, or reviews containing genre-specific horror terminology.")

# Main Header
st.markdown('<div class="main-header">🎬 IMDb Review Sentiment Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Classify movie reviews into Positive or Negative polarity using a deep Bidirectional LSTM network.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Live Prediction", "📊 Model Evaluation & Metrics", "📖 Methodology & Architecture"])

# TAB 1: Live Prediction
with tab1:
    col1, col2 = st.columns([1.2, 0.8])
    
    sample_reviews = {
        "Custom Review": "",
        "Positive Masterpiece": "This film was an absolute masterpiece! The character development was phenomenal, the cinematography was breathtaking, and the climax left the entire theater in tears. A truly remarkable cinematic experience.",
        "Terrible Disaster": "What an awful excuse for a movie. The acting was stiff and completely devoid of emotion, the plot holes were big enough to drive a truck through, and the pacing dragged relentlessly. Don't waste your money.",
        "Nuanced / Complex Review": "While the movie had visually stunning special effects and a stellar performance by the lead actor, the disjointed screenplay and confusing third act unfortunately kept it from reaching its true potential.",
        "Subtle Negation": "I was told this would be a boring flop, but I was not disappointed in the slightest; on the contrary, it surprised me with genuine wit and heart."
    }
    
    with col1:
        st.subheader("Enter Movie Review")
        sample_keys = list(sample_reviews.keys())
        default_sample_idx = 1
        if "sample" in st.query_params:
            try:
                default_sample_idx = int(st.query_params["sample"])
                default_sample_idx = max(0, min(len(sample_keys) - 1, default_sample_idx))
            except (ValueError, TypeError):
                pass
        selected_sample = st.selectbox("Or choose a pre-filled sample:", sample_keys, index=default_sample_idx)
        
        default_text = sample_reviews[selected_sample] if selected_sample != "Custom Review" else ""
        user_input = st.text_area(
            "Write or paste your review below:",
            value=default_text,
            height=180,
            placeholder="Type your movie review here..."
        )
        
        analyze_btn = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)
        
    with col2:
        st.subheader("Classification Result")
        
        if analyze_btn or (user_input and selected_sample != "Custom Review"):
            if not user_input.strip():
                st.warning("⚠️ Please enter review text before submitting.")
            else:
                with st.spinner("Analyzing text sequence..."):
                    try:
                        predictor = load_predictor()
                        result = predictor.predict(user_input)
                        
                        is_pos = result["sentiment"] == "positive"
                        card_class = "pos-card" if is_pos else "neg-card"
                        emoji = "🌟" if is_pos else "👎"
                        badge_color = "#10B981" if is_pos else "#EF4444"
                        
                        st.markdown(f"""
                        <div class="sentiment-card {card_class}">
                            <h2 style="color: {badge_color}; margin: 0;">{emoji} {result['sentiment'].upper()}</h2>
                            <p style="margin: 0.5rem 0 0 0; color: #374151;">Confidence: <strong>{result['confidence']}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Probability progress bars
                        st.write("**Probability Distribution:**")
                        col_p1, col_p2 = st.columns(2)
                        with col_p1:
                            st.metric("Positive Probability", f"{result['probability_positive'] * 100:.1f}%")
                            st.progress(float(result["probability_positive"]))
                        with col_p2:
                            st.metric("Negative Probability", f"{result['probability_negative'] * 100:.1f}%")
                            st.progress(float(result["probability_negative"]))
                            
                        # Input stats
                        st.markdown("---")
                        st.write(f"📏 **Word / Token Count:** `{result['tokens_count']}` tokens")
                        st.write(f"🔡 **Normalized Text:** `{result['cleaned_sample']}`")
                        
                    except Exception as e:
                        st.error(f"Inference error: {e}")
        else:
            st.info("👈 Enter a review on the left and click **Analyze Sentiment** to see the prediction.")

# TAB 2: Model Evaluation & Metrics
with tab2:
    st.subheader("Holdout Test Set Performance (25,000 Reviews)")
    bilstm_data, baseline_data = load_metrics()
    
    if bilstm_data:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Test Accuracy", f"{bilstm_data['accuracy']*100:.2f}%", 
                  f"{(bilstm_data['accuracy'] - (baseline_data['accuracy'] if baseline_data else 0))*100:+.2f}% vs Baseline" if baseline_data else None)
        m2.metric("Precision", f"{bilstm_data['precision']*100:.2f}%")
        m3.metric("Recall", f"{bilstm_data['recall']*100:.2f}%")
        m4.metric("F1-Score", f"{bilstm_data['f1_score']*100:.2f}%")
        m5.metric("ROC-AUC", f"{bilstm_data['roc_auc']:.4f}")
        
        st.markdown("---")
        
        c_left, c_right = st.columns(2)
        with c_left:
            st.write("#### Confusion Matrix (BiLSTM)")
            cm = bilstm_data["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                index=["True Negative", "True Positive"],
                columns=["Pred Negative", "Pred Positive"]
            )
            st.dataframe(cm_df, use_container_width=True)
            
            fig_cm = find_figure("confusion_matrix.png")
            if fig_cm and os.path.exists(fig_cm):
                st.image(fig_cm, caption="Confusion Matrix Heatmap", use_container_width=True)
                
        with c_right:
            st.write("#### Comparison with Baseline (TF-IDF + Logistic Regression)")
            if baseline_data:
                comp_df = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                    "BiLSTM (Ours)": [
                        f"{bilstm_data['accuracy']:.4f}",
                        f"{bilstm_data['precision']:.4f}",
                        f"{bilstm_data['recall']:.4f}",
                        f"{bilstm_data['f1_score']:.4f}",
                        f"{bilstm_data['roc_auc']:.4f}"
                    ],
                    "TF-IDF + LR": [
                        f"{baseline_data['accuracy']:.4f}",
                        f"{baseline_data['precision']:.4f}",
                        f"{baseline_data['recall']:.4f}",
                        f"{baseline_data['f1_score']:.4f}",
                        f"{baseline_data['roc_auc']:.4f}"
                    ]
                })
                st.table(comp_df)
                
            fig_comp = find_figure("model_comparison.png")
            if fig_comp and os.path.exists(fig_comp):
                st.image(fig_comp, caption="Multi-Metric Comparison", use_container_width=True)
    else:
        st.info("Metrics not loaded yet. Run evaluation to view test metrics.")

# TAB 3: Methodology & Architecture
with tab3:
    st.subheader("Architectural Design & Pipeline")
    st.markdown(r"""
    The system follows a strict, end-to-end Deep Learning sequence pipeline:
    
    1. **Text Preprocessing:**
       - Decodes HTML tags (`<br />`) and HTML entities.
       - Expands common English contractions, ensuring that negation terms (`not`, `never`, `isn't`, `didn't`) are strictly preserved.
       - Tokenizes text into word-level tokens using a robust regular expression.
       - Post-truncation and post-padding to a uniform maximum sequence length of `256` tokens.
       
    2. **Bidirectional LSTM Architecture:**
       - **Embedding Layer:** Projects discrete token indices into a dense 128-dimensional continuous vector space.
       - **Bidirectional LSTM Layer:** Processes sequences forward and backward simultaneously, accumulating contextual cues from past and future tokens (output size: 256).
       - **Global Max Pooling:** Operates across the sequence dimension to preserve the strongest sentiment activations regardless of token position.
       - **Dense Classifier:** Linear layer (256 → 64), ReLU activation, Dropout (0.30), and final projection to single logit.
       - **Output:** Sigmoid activation mapping logits to calibrated sentiment probability $P(pos \mid x)$.
       
    3. **Training & Regularization:**
       - Binary Cross-Entropy with Logits loss (`BCEWithLogitsLoss`).
       - Adam Optimizer with learning rate 0.001 and weight decay $1 \times 10^{-5}$.
       - Validation loss monitoring with Early Stopping (patience=3) and ReduceLROnPlateau.
    """)
    
    fig_hist = find_figure("loss_history.png")
    if fig_hist and os.path.exists(fig_hist):
        st.image(fig_hist, caption="Training & Validation Loss History", use_container_width=True)
