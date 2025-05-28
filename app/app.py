"""
Live Fake News Checker - Streamlit Application
Combines ML prediction with live news verification and fact-checking
"""

import streamlit as st
import os
import sys
from datetime import datetime
import logging

# Add the app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import FakeNewsChecker, NewsAPIClient, FactCheckAPI, get_combined_verdict

# Add debug logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Live Fake News Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .fake-prediction {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        color: #c62828;
    }
    
    .real-prediction {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        color: #2e7d32;
    }
    
    .uncertain-prediction {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        color: #ef6c00;
    }
    
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        background-color: #e0e0e0;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        transition: width 0.3s ease;
    }
    
    .article-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #17a2b8;
    }
    
    .fact-check-card {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #6c757d;
    }
    
    .sidebar-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'checker' not in st.session_state:
        st.session_state.checker = None
    if 'news_client' not in st.session_state:
        st.session_state.news_client = None
    if 'fact_check_client' not in st.session_state:
        st.session_state.fact_check_client = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

def setup_api_clients():
    """Setup API clients with user-provided keys"""
    st.sidebar.markdown("### 🔑 API Configuration")
    
    # NewsAPI Key
    news_api_key = st.sidebar.text_input(
        "NewsAPI Key",
        type="password",
        help="Get your free API key from https://newsapi.org"
    )
    
    # Google Fact Check API Key
    fact_check_api_key = st.sidebar.text_input(
        "Google Fact Check API Key",
        type="password",
        help="Get your API key from Google Cloud Console"
    )
    
    # Initialize clients when keys are provided
    if news_api_key:
        st.session_state.news_client = NewsAPIClient(news_api_key)
    
    if fact_check_api_key:
        st.session_state.fact_check_client = FactCheckAPI(fact_check_api_key)
    
    return news_api_key, fact_check_api_key

def display_model_info():
    """Display information about the loaded model"""
    if st.session_state.checker and st.session_state.checker.model_info:
        info = st.session_state.checker.model_info
        st.sidebar.markdown("### 🤖 Model Information")
        st.sidebar.markdown(f"""
        <div class="sidebar-info">
        <strong>Algorithm:</strong> {info.get('model_name', 'Unknown')}<br>
        <strong>Accuracy:</strong> {info.get('accuracy', 0):.2%}<br>
        <strong>Features:</strong> {info.get('features_count', 0):,}<br>
        <strong>Training Samples:</strong> {info.get('training_samples', 0):,}
        </div>
        """, unsafe_allow_html=True)

def display_ml_prediction(prediction_result):
    """Display ML model prediction results"""
    st.subheader("🤖 ML Model Prediction")
    
    if 'error' in prediction_result:
        st.error(f"Error: {prediction_result['error']}")
        return
    
    prediction = prediction_result['prediction']
    confidence = prediction_result['confidence']
    
    # Determine styling based on prediction
    if prediction == 'FAKE':
        box_class = "fake-prediction"
        color = "#f44336"
    elif prediction == 'REAL':
        box_class = "real-prediction"  
        color = "#4caf50"
    else:
        box_class = "uncertain-prediction"
        color = "#ff9800"
    
    # Display prediction box
    st.markdown(f"""
    <div class="prediction-box {box_class}">
        Prediction: {prediction} ({confidence}% confidence)
    </div>
    """, unsafe_allow_html=True)
    
    # Display confidence bar
    st.markdown("**Confidence Breakdown:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Real Probability", f"{prediction_result.get('real_probability', 0)}%")
    with col2:
        st.metric("Fake Probability", f"{prediction_result.get('fake_probability', 0)}%")
    
    # Show processed text (optional)
    with st.expander("View Processed Text"):
        st.text(prediction_result.get('processed_text', 'No processed text available'))

def display_news_articles(news_result):
    """Display related news articles"""
    st.subheader("📰 Related News Articles")
    
    if 'error' in news_result:
        st.warning(f"News search error: {news_result['error']}")
        return []
    
    articles = news_result.get('articles', [])
    
    if not articles:
        st.info("No related news articles found.")
        return []
    
    st.success(f"Found {len(articles)} related articles (Total available: {news_result.get('total_results', 0)})")
    
    for i, article in enumerate(articles, 1):
        st.markdown(f"""
        <div class="article-card">
            <h4>{i}. {article['title']}</h4>
            <p><strong>Source:</strong> {article['source']}</p>
            <p>{article['description']}</p>
            <p><strong>Published:</strong> {article['published_at']}</p>
            <a href="{article['url']}" target="_blank">Read Full Article →</a>
        </div>
        """, unsafe_allow_html=True)
    
    return articles

def display_fact_checks(fact_check_result):
    """Display fact-check results"""
    st.subheader("✅ Fact-Check Results")
    
    if 'error' in fact_check_result:
        st.warning(f"Fact-check search error: {fact_check_result['error']}")
        return []
    
    fact_checks = fact_check_result.get('fact_checks', [])
    
    if not fact_checks:
        st.info("No related fact-checks found.")
        return []
    
    st.success(f"Found {len(fact_checks)} related fact-checks")
    
    for i, fc in enumerate(fact_checks, 1):
        # Determine rating color
        rating = fc['rating'].lower()
        if any(word in rating for word in ['false', 'fake', 'pants on fire']):
            rating_color = "#f44336"
        elif any(word in rating for word in ['true', 'correct']):
            rating_color = "#4caf50"
        else:
            rating_color = "#ff9800"
        
        st.markdown(f"""
        <div class="fact-check-card">
            <h4>{i}. {fc['title']}</h4>
            <p><strong>Claim:</strong> {fc['claim']}</p>
            <p><strong>Rating:</strong> <span style="color: {rating_color}; font-weight: bold;">{fc['rating']}</span></p>
            <p><strong>Reviewer:</strong> {fc['reviewer']}</p>
            <p><strong>Review Date:</strong> {fc['review_date']}</p>
            <a href="{fc['url']}" target="_blank">Read Full Fact-Check →</a>
        </div>
        """, unsafe_allow_html=True)
    
    return fact_checks

def display_combined_verdict(verdict):
    """Display the combined verdict"""
    st.subheader("⚖️ Combined Verdict")
    
    assessment = verdict['overall_assessment']
    confidence = verdict['confidence_level']
    score = verdict['credibility_score']
    
    # Determine styling
    if 'FAKE' in assessment:
        verdict_class = "fake-prediction"
    elif 'REAL' in assessment:
        verdict_class = "real-prediction"
    else:
        verdict_class = "uncertain-prediction"
    
    # Display verdict
    st.markdown(f"""
    <div class="prediction-box {verdict_class}">
        Overall Assessment: {assessment}<br>
        Confidence Level: {confidence}<br>
        Credibility Score: {score}/100
    </div>
    """, unsafe_allow_html=True)
    
    # Display reasoning
    st.markdown("**Analysis Reasoning:**")
    for reason in verdict['reasoning']:
        st.markdown(f"• {reason}")

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Live Fake News Checker</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p>Enter a news headline to check its authenticity using AI prediction, live news verification, and fact-checking.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Setup API clients in sidebar
    news_api_key, fact_check_api_key = setup_api_clients()
    
    # Initialize ML model
    if not st.session_state.model_loaded:
        with st.spinner("Loading ML model..."):
            try:
                logger.info("Attempting to initialize FakeNewsChecker...")
                st.session_state.checker = FakeNewsChecker()
                if st.session_state.checker.model is None:
                    st.error("Failed to load ML model. Please check the console for details.")
                    logger.error("Model initialization failed - model is None")
                else:
                    st.session_state.model_loaded = True
                    logger.info("Model loaded successfully")
            except Exception as e:
                st.error(f"Error loading ML model: {str(e)}")
                logger.error(f"Model initialization error: {str(e)}", exc_info=True)
    
    # Display model information
    display_model_info()
    
    # Sidebar instructions
    st.sidebar.markdown("### 📋 Instructions")
    st.sidebar.markdown("""
    1. **Enter API Keys** above (optional but recommended)
    2. **Input a news headline** in the main area
    3. **Click 'Check Headline'** to analyze
    4. **Review results** from multiple sources
    
    **Note:** The ML model works without API keys, but live verification requires them.
    """)
    
    # Main input area
    st.markdown("### Enter News Headline")
    headline = st.text_area(
        "Paste or type the news headline you want to check:",
        height=100,
        placeholder="Example: Scientists discover cure for cancer in breakthrough study"
    )
    
    # Analysis settings
    col1, col2 = st.columns(2)
    with col1:
        max_articles = st.slider("Max news articles to fetch", 1, 10, 5)
    with col2:
        max_fact_checks = st.slider("Max fact-checks to fetch", 1, 10, 5)
    
    # Check button
    if st.button("🔍 Check Headline", type="primary", use_container_width=True):
        if not headline.strip():
            st.warning("Please enter a headline to check.")
            return
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: ML Prediction
        status_text.text("🤖 Running ML prediction...")
        progress_bar.progress(25)
        
        ml_result = st.session_state.checker.predict_headline(headline)
        
        # Step 2: News API Search
        status_text.text("📰 Searching related news...")
        progress_bar.progress(50)
        
        news_result = {'articles': [], 'error': 'NewsAPI key not provided'}
        if st.session_state.news_client:
            news_result = st.session_state.news_client.search_news(headline, max_articles)
        
        # Step 3: Fact Check Search
        status_text.text("✅ Searching fact-checks...")
        progress_bar.progress(75)
        
        fact_check_result = {'fact_checks': [], 'error': 'Fact Check API key not provided'}
        if st.session_state.fact_check_client:
            fact_check_result = st.session_state.fact_check_client.search_fact_checks(headline, max_fact_checks)
        
        # Step 4: Generate Combined Verdict
        status_text.text("⚖️ Generating combined verdict...")
        progress_bar.progress(100)
        
        combined_verdict = get_combined_verdict(
            ml_result,
            news_result.get('articles', []),
            fact_check_result.get('fact_checks', [])
        )
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Create tabs for organized display
        tab1, tab2, tab3, tab4 = st.tabs(["🤖 ML Prediction", "📰 News Articles", "✅ Fact Checks", "⚖️ Final Verdict"])
        
        with tab1:
            display_ml_prediction(ml_result)
        
        with tab2:
            articles = display_news_articles(news_result)
        
        with tab3:
            fact_checks = display_fact_checks(fact_check_result)
        
        with tab4:
            display_combined_verdict(combined_verdict)
        
        # Summary section
        st.markdown("---")
        st.markdown("### 📋 Quick Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ML Prediction", ml_result.get('prediction', 'Unknown'))
        with col2:
            st.metric("ML Confidence", f"{ml_result.get('confidence', 0)}%")
        with col3:
            st.metric("News Articles Found", len(news_result.get('articles', [])))
        with col4:
            st.metric("Fact Checks Found", len(fact_check_result.get('fact_checks', [])))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🔬 This tool combines machine learning with live data sources for comprehensive fact-checking.</p>
        <p>Always verify important information through multiple trusted sources.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()