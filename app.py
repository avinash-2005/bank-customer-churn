"""
Digital Banking Churn Prediction - Streamlit Application
Behavior + Emotion Based Customer Churn Prediction for Digital Banking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import time

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import DataPreprocessor
from src.sentiment import SentimentAnalyzer
from src.model import ChurnPredictor
from src.kaggle_data_loader import KaggleDataLoader
from src.database_manager import DatabaseManager
from src.utils import (
    load_data, create_risk_gauge_chart, create_sentiment_distribution_chart,
    create_feature_importance_chart, create_risk_distribution_chart,
    format_probability, get_risk_level_color, get_sentiment_color,
    get_customer_by_id, create_customer_list, check_model_files, add_customer_names
)
from src.adaptive_churn_engine import AdaptiveChurnEngine

# Page configuration
st.set_page_config(
    page_title="Digital Banking Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern responsive FinTech styling
def load_css():
    css = """
    <style>
        /* Hide Streamlit deploy button */
        .stDeployButton { display: none !important; visibility: hidden !important; opacity: 0 !important; }
        .stApp .stDeployButton { display: none !important; visibility: hidden !important; opacity: 0 !important; }
        [data-testid="stDeployButton"] { display: none !important; visibility: hidden !important; opacity: 0 !important; }
        .stAppHeader button[title="Deploy"] { display: none !important; visibility: hidden !important; opacity: 0 !important; }
        .stAppHeader .stDeployButton { display: none !important; visibility: hidden !important; opacity: 0 !important; }
        header button[aria-label="Deploy"] { display: none !important; visibility: hidden !important; opacity: 0 !important; }
        .stAppHeader > div:last-child > button { display: none !important; visibility: hidden !important; opacity: 0 !important; }
        .stAppHeader > div > button:last-child { display: none !important; visibility: hidden !important; opacity: 0 !important; }
        header > div > button:last-child { display: none !important; visibility: hidden !important; opacity: 0 !important; }
        
        /* JavaScript to remove deploy button */
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            setInterval(function() {
                const deployButtons = document.querySelectorAll('[data-testid="stDeployButton"], .stDeployButton, button[title="Deploy"]');
                deployButtons.forEach(btn => {
                    btn.style.display = 'none';
                    btn.style.visibility = 'hidden';
                    btn.style.opacity = '0';
                    btn.remove();
                });
            }, 100);
        });
        </script>
        
        /* Customer Overview Section Styling */
        .customer-overview {
            background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid #e1e8ed;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        }
        
        .customer-header {
            font-size: 1.4rem !important;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1rem !important;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #667eea;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .customer-metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .metric-card {
            flex: 1;
            min-width: 200px;
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            border-left: 4px solid #667eea;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.12);
        }
        
        .gauge-container {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid #e1e8ed;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .gauge-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        /* Column spacing fixes */
        .element-container:has([data-testid="stVerticalBlock"]) {
            padding-right: 1rem !important;
        }
        
        /* Ensure proper spacing between customer overview and gauge */
        div[data-testid="column"] > div:first-child {
            margin-right: 1rem !important;
        }
        
        /* Main background with gradient */
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            background-attachment: fixed;
        }
        
        .stApp {
            background: transparent;
        }
        
        /* Container styling */
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            margin: 1rem auto;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }
        
        /* Header styling */
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
            margin: 0;
        }
        
        /* Card styling */
        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
        }
        
        /* Metric card styling */
        .metric-card {
            background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
            margin-bottom: 1rem;
            border: 1px solid rgba(102, 126, 234, 0.1);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(102, 126, 234, 0.2);
        }
        
        /* Risk level styling */
        .risk-high {
            border-left: 5px solid #ff4b4b;
            background: linear-gradient(135deg, #fff5f5 0%, #ffffff 100%);
        }
        
        .risk-medium {
            border-left: 5px solid #ffa500;
            background: linear-gradient(135deg, #fffdf5 0%, #ffffff 100%);
        }
        
        .risk-low {
            border-left: 5px solid #00c851;
            background: linear-gradient(135deg, #f5fff5 0%, #ffffff 100%);
        }
        
        /* Sentiment styling */
        .sentiment-positive {
            border-left: 5px solid #00c851;
            background: linear-gradient(135deg, #f5fff5 0%, #ffffff 100%);
        }
        
        .sentiment-neutral {
            border-left: 5px solid #ffa500;
            background: linear-gradient(135deg, #fffdf5 0%, #ffffff 100%);
        }
        
        .sentiment-negative {
            border-left: 5px solid #ff4b4b;
            background: linear-gradient(135deg, #fff5f5 0%, #ffffff 100%);
        }
        
        /* Navigation sidebar styling */
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0B1C2D 0%, #1a2a3a 100%);
        }
        
        div[data-testid="stSidebar"] .sidebar-content {
            padding: 2rem 1rem;
        }
        
        /* Data table styling */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        /* Modern alert/info boxes */
        .stAlert {
            border-radius: 12px;
            border: none;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }
        
        /* Chart container styling */
        .element-container .stPlotlyChart {
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8rem;
            }
            
            .header p {
                font-size: 1rem;
            }
            
            .metric-value {
                font-size: 1.8rem;
            }
            
            .card {
                padding: 1.5rem;
            }
        }
    </style>
    """
    return css

# Add caching decorator for expensive operations
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_cached_data(file_path):
    """Cached version of load_data function"""
    return load_data(file_path)

@st.cache_resource(ttl=600)  # Cache for 10 minutes
def create_cached_preprocessor():
    """Cached version of preprocessor creation"""
    return DataPreprocessor()

@st.cache_resource(ttl=600)  # Cache for 10 minutes
def create_cached_sentiment_analyzer():
    """Cached version of sentiment analyzer creation"""
    return SentimentAnalyzer()

# Load custom CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'merged_data' not in st.session_state:
    st.session_state.merged_data = None
if 'adaptive_engine' not in st.session_state:
    st.session_state.adaptive_engine = None
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = None

# Sidebar navigation
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1e3c72 0%, #1FB6A6 100%); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(31, 182, 166, 0.3);'>
            <h2 style='color: white; margin: 0; font-weight: 700;'>🏦 Digital Banking</h2>
            <p style='color: #E8F5E8; margin: 0.5rem 0 0 0; font-size: 1rem; font-weight: 500;'>Churn Prediction System</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation buttons - single column layout for better mobile experience
        st.markdown("### 🧭 Navigation")
        
        nav_col = st.container()
        with nav_col:
            # Dashboard button
            dashboard_type = "primary" if st.session_state.get('current_page') == 'dashboard' else "secondary"
            if st.button("📊 Dashboard", type=dashboard_type, use_container_width=True):
                st.session_state.current_page = "dashboard"
                st.rerun()
            
            # Customer Analysis button
            customer_type = "primary" if st.session_state.get('current_page') == 'customer_analysis' else "secondary"
            if st.button("👥 Customer Analysis", type=customer_type, use_container_width=True):
                st.session_state.current_page = "customer_analysis"
                st.rerun()
            
            # Analytics button (new tab for data insights)
            analytics_type = "primary" if st.session_state.get('current_page') == 'analytics' else "secondary"
            if st.button("📈 Analytics", type=analytics_type, use_container_width=True):
                st.session_state.current_page = "analytics"
                st.rerun()
            
            # Reports button (new tab for generating reports)
            reports_type = "primary" if st.session_state.get('current_page') == 'reports' else "secondary"
            if st.button("📄 Reports", type=reports_type, use_container_width=True):
                st.session_state.current_page = "reports"
                st.rerun()
        
        st.markdown("---")
        
        # System Status Section
        st.markdown("### 📊 System Status")
        
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded")
        else:
            st.warning("⏳ Loading Data...")
        
        if st.session_state.model_trained:
            st.success("✅ Model Ready")
        else:
            st.warning("⏳ Training Model...")
        
        if st.session_state.merged_data is not None:
            st.info(f"📊 {len(st.session_state.merged_data)} Customers")
        
        st.markdown("---")
        
        # Refresh button (for re-initialization if needed)
        if st.button("🔄 Refresh System", type="secondary", use_container_width=True):
            # Reset session state
            st.session_state.data_loaded = False
            st.session_state.model_trained = False
            st.session_state.preprocessor = None
            st.session_state.sentiment_analyzer = None
            st.session_state.predictor = None
            st.session_state.merged_data = None
            st.rerun()
    
    # Get current page from session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "dashboard"
    
    return st.session_state.current_page

# Enhanced initialization function
def initialize_app():
    """Initialize the application with automatic data loading and model training"""
    if not st.session_state.data_loaded or not st.session_state.model_trained:
        # Create a placeholder for initialization messages
        init_placeholder = st.empty()
        
        with init_placeholder.container():
            with st.spinner("🚀 Initializing Customer Churn Prediction System..."):
                try:
                    # Check if model files exist
                    model_exists = os.path.exists("models/churn_model.pkl")
                    scaler_exists = os.path.exists("models/scaler.pkl")
                    data_exists = os.path.exists("data/customer_behavior.csv") and os.path.exists("data/customer_feedback.csv")
                    
                    # Initialize database
                    st.session_state.database = DatabaseManager()
                    
                    # Initialize components using cached functions
                    st.session_state.preprocessor = create_cached_preprocessor()
                    st.session_state.sentiment_analyzer = create_cached_sentiment_analyzer()
                    
                    # Load or train model
                    if model_exists and scaler_exists and data_exists:
                        st.info("� Loading existing model and data...")
                        
                        # Load existing data using cached functions
                        behavior_df = load_cached_data("data/customer_behavior.csv")
                        feedback_df = load_cached_data("data/customer_feedback.csv")
                        
                        if behavior_df is None or feedback_df is None:
                            st.error("❌ Failed to load data files. Please check if data files exist in the 'data' folder.")
                            return False
                        
                        # Add customer names to behavior data
                        behavior_df = add_customer_names(behavior_df)
                        st.info("👥 Added customer names to behavior data")
                        
                        # Load existing model
                        st.session_state.predictor = ChurnPredictor(model_type='random_forest')
                        st.session_state.predictor.load_model("models/churn_model.pkl", "models/scaler.pkl")
                        st.session_state.model_trained = True
                        
                        # Initialize adaptive churn engine
                        st.session_state.adaptive_engine = AdaptiveChurnEngine()
                        
                        # Store data in SQLite database (if not already stored)
                        st.info("💾 Ensuring data is in SQLite database...")
                        st.session_state.database.store_behavior_data(behavior_df)
                        st.session_state.database.store_feedback_data(feedback_df)
                        
                        # Preprocess and merge data
                        st.info("📊 Preprocessing and merging data...")
                        behavior_df = st.session_state.preprocessor.handle_missing_values(behavior_df)
                        feedback_df = st.session_state.preprocessor.handle_missing_values(feedback_df)
                        
                        # Sentiment analysis
                        sentiment_df = st.session_state.sentiment_analyzer.analyze_feedback(feedback_df)
                        
                        # Merge data
                        merged_df = st.session_state.preprocessor.merge_datasets(behavior_df, feedback_df)
                        merged_df = st.session_state.sentiment_analyzer.merge_sentiment_to_behavior(merged_df, sentiment_df)
                        
                        # Remove any duplicate customer IDs
                        if 'customer_id' in merged_df.columns:
                            initial_count = len(merged_df)
                            merged_df = merged_df.drop_duplicates(subset=['customer_id'], keep='first')
                            if len(merged_df) < initial_count:
                                st.info(f"🧹 Removed {initial_count - len(merged_df)} duplicate customer records")
                        
                        st.session_state.merged_data = merged_df
                        st.session_state.data_loaded = True
                        st.session_state.last_update_time = pd.Timestamp.now()
                        st.session_state.update_count = 1
                        
                        st.success("✅ System loaded successfully from existing files!")
                        
                    else:
                        st.info("🔄 Training new model with Kaggle datasets...")
                        
                        # Initialize data loader
                        st.session_state.data_loader = KaggleDataLoader()
                        
                        # Process all Kaggle datasets
                        st.info("📊 Processing Kaggle datasets...")
                        success = st.session_state.data_loader.process_and_save()
                        
                        if not success:
                            st.error("❌ Failed to process Kaggle datasets")
                            return False
                        
                        # Load data with error handling using cached functions
                        behavior_df = load_cached_data("data/customer_behavior.csv")
                        feedback_df = load_cached_data("data/customer_feedback.csv")
                        
                        if behavior_df is None or feedback_df is None:
                            st.error("❌ Failed to load data files. Please check if data files exist in the 'data' folder.")
                            return False
                        
                        # Add customer names to behavior data
                        behavior_df = add_customer_names(behavior_df)
                        st.info("👥 Added customer names to behavior data")
                        
                        # Store data in SQLite database
                        st.info("💾 Storing data in SQLite database...")
                        st.session_state.database.store_behavior_data(behavior_df)
                        st.session_state.database.store_feedback_data(feedback_df)
                        
                        # Validate data structure
                        required_behavior_cols = ['customer_id', 'customer_name', 'login_frequency', 'session_duration', 'inactivity_days', 'transaction_count', 'churn']
                        required_feedback_cols = ['customer_id', 'feedback_text']
                        
                        missing_behavior = [col for col in required_behavior_cols if col not in behavior_df.columns]
                        missing_feedback = [col for col in required_feedback_cols if col not in feedback_df.columns]
                        
                        if missing_behavior:
                            st.error(f"❌ Missing columns in behavior data: {missing_behavior}")
                            return False
                        
                        if missing_feedback:
                            st.error(f"❌ Missing columns in feedback data: {missing_feedback}")
                            return False
                        
                        # Preprocess data
                        st.info("📊 Preprocessing behavioral data...")
                        behavior_df = st.session_state.preprocessor.handle_missing_values(behavior_df)
                        
                        st.info("💭 Analyzing customer sentiment...")
                        feedback_df = st.session_state.preprocessor.handle_missing_values(feedback_df)
                        
                        # Sentiment analysis
                        sentiment_df = st.session_state.sentiment_analyzer.analyze_feedback(feedback_df)
                        
                        # Merge data
                        st.info("🔗 Merging behavioral and sentiment data...")
                        merged_df = st.session_state.preprocessor.merge_datasets(behavior_df, feedback_df)
                        merged_df = st.session_state.sentiment_analyzer.merge_sentiment_to_behavior(merged_df, sentiment_df)
                        
                        # Remove any duplicate customer IDs
                        if 'customer_id' in merged_df.columns:
                            initial_count = len(merged_df)
                            merged_df = merged_df.drop_duplicates(subset=['customer_id'], keep='first')
                            if len(merged_df) < initial_count:
                                st.info(f"🧹 Removed {initial_count - len(merged_df)} duplicate customer records")
                        
                        # Train model
                        st.info("🤖 Training Random Forest model...")
                        st.session_state.predictor = ChurnPredictor(model_type='random_forest')
                        st.session_state.predictor.train(merged_df)
                        
                        # Save model
                        os.makedirs("models", exist_ok=True)
                        st.session_state.predictor.save_model("models/churn_model.pkl", "models/scaler.pkl")
                        
                        # Store model performance in database
                        model_metrics = st.session_state.predictor.get_model_metrics()
                        if model_metrics:
                            st.session_state.database.store_model_performance(
                                model_type='random_forest',
                                accuracy=model_metrics.get('accuracy', 0),
                                precision=model_metrics.get('precision', 0),
                                recall=model_metrics.get('recall', 0),
                                f1_score=model_metrics.get('f1_score', 0),
                                training_samples=len(merged_df)
                            )
                        
                        # Store initial update summary
                        churn_rate = merged_df['churn'].mean() * 100
                        avg_credit_score = merged_df['credit_score'].mean()
                        avg_account_balance = merged_df['account_balance'].mean()
                        
                        st.session_state.database.store_update_summary(
                            update_count=1,
                            total_customers=len(merged_df),
                            churn_rate=churn_rate,
                            avg_credit_score=avg_credit_score,
                            avg_account_balance=avg_account_balance
                        )
                        
                        # Store data
                        st.session_state.merged_data = merged_df
                        st.session_state.data_loaded = True
                        st.session_state.model_trained = True
                        st.session_state.last_update_time = pd.Timestamp.now()
                        st.session_state.update_count = 1
                        
                        # Success message
                        st.success("✅ System initialized successfully!")
                        st.balloons()
                    
                    # Brief delay to show success message
                    time.sleep(2)
                
                except Exception as e:
                    st.error(f"❌ Error during initialization: {str(e)}")
                    st.info("🔧 Please check the error details and try restarting the application.")
                    return False
        
        # Clear the initialization messages after completion
        init_placeholder.empty()
        return True
    
    return True

def update_real_time_data():
    """Update data in real-time and retrain model if needed"""
    if st.session_state.data_loader is None:
        return False
    
    try:
        # Check if 1 minute has passed since last update
        current_time = pd.Timestamp.now()
        if st.session_state.last_update_time is None:
            time_since_update = pd.Timedelta(minutes=2)  # Force update on first run
        else:
            time_since_update = current_time - st.session_state.last_update_time
        
        if time_since_update >= pd.Timedelta(minutes=1):
            with st.spinner("🔄 Updating customer data..."):
                # Reprocess all Kaggle datasets
                success = st.session_state.data_loader.process_and_save()
                
                if not success:
                    st.error("❌ Failed to update data")
                    return False
                
                # Load updated data
                behavior_df = load_data("data/customer_behavior.csv")
                feedback_df = load_data("data/customer_feedback.csv")
                
                # Store updated data in database
                st.session_state.database.store_behavior_data(behavior_df)
                st.session_state.database.store_feedback_data(feedback_df)
                
                # Preprocess updated data
                behavior_df = st.session_state.preprocessor.handle_missing_values(behavior_df)
                feedback_df = st.session_state.preprocessor.handle_missing_values(feedback_df)
                
                # Sentiment analysis on updated feedback
                sentiment_df = st.session_state.sentiment_analyzer.analyze_feedback(feedback_df)
                
                # Merge updated data
                merged_df = st.session_state.preprocessor.merge_datasets(behavior_df, feedback_df)
                merged_df = st.session_state.sentiment_analyzer.merge_sentiment_to_behavior(merged_df, sentiment_df)
                
                # Remove any duplicate customer IDs
                if 'customer_id' in merged_df.columns:
                    initial_count = len(merged_df)
                    merged_df = merged_df.drop_duplicates(subset=['customer_id'], keep='first')
                    if len(merged_df) < initial_count:
                        print(f"🧹 Removed {initial_count - len(merged_df)} duplicate customer records in update")
                
                # Retrain model with updated data
                st.session_state.predictor.train(merged_df)
                
                # Save updated model
                st.session_state.predictor.save_model("models/churn_model.pkl", "models/scaler.pkl")
                
                # Store model performance in database
                model_metrics = st.session_state.predictor.get_model_metrics()
                if model_metrics:
                    st.session_state.database.store_model_performance(
                        model_type='random_forest',
                        accuracy=model_metrics.get('accuracy', 0),
                        precision=model_metrics.get('precision', 0),
                        recall=model_metrics.get('recall', 0),
                        f1_score=model_metrics.get('f1_score', 0),
                        training_samples=len(merged_df)
                    )
                
                # Store update summary in database
                churn_rate = merged_df['churn'].mean() * 100
                avg_credit_score = merged_df['credit_score'].mean()
                avg_account_balance = merged_df['account_balance'].mean()
                
                st.session_state.database.store_update_summary(
                    update_count=st.session_state.update_count + 1,
                    total_customers=len(merged_df),
                    churn_rate=churn_rate,
                    avg_credit_score=avg_credit_score,
                    avg_account_balance=avg_account_balance
                )
                
                # Update session state
                st.session_state.merged_data = merged_df
                st.session_state.last_update_time = current_time
                st.session_state.update_count += 1
                
                # Show update notification
                st.toast(f"🔄 Data Updated #{st.session_state.update_count} - Model retrained with {len(merged_df)} customers", icon="🔄")
                
                return True
                
    except Exception as e:
        st.error(f"❌ Error updating data: {str(e)}")
        return False
    
    return False

# Load and train data function (kept for backward compatibility)
def load_and_train_data():
    """Legacy function - now calls initialize_app()"""
    return initialize_app()

# Dashboard page
def dashboard_page():
    st.markdown("""
    <div class='header'>
        <h1>Digital Banking Churn Prediction</h1>
        <p>Behavior + Emotion Based Analytics for Customer Retention</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or not st.session_state.model_trained:
        st.warning("🔄 System is initializing... Please wait a moment.")
        return
    
    # Check for real-time updates
    update_real_time_data()
    
    # Get predictions
    predictions_df = st.session_state.predictor.predict_churn_probability(st.session_state.merged_data)
    
    # Real-time status indicator
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 🔄 Real-Time Analytics")
        if st.session_state.last_update_time:
            last_update = st.session_state.last_update_time.strftime('%H:%M:%S')
            
            # Get database stats
            db_stats = st.session_state.database.get_database_stats()
            
            st.markdown(f"""
            <div class='card'>
                <p><strong>🕐 Last Update:</strong> {last_update}</p>
                <p><strong>🔄 Update Count:</strong> {st.session_state.update_count}</p>
                <p><strong>📊 Customers Tracked:</strong> 17,243</p>
                <p><strong>⏱️ Update Frequency:</strong> Every 1 minute</p>
                <p><strong>💾 Database Records:</strong> {db_stats['behavior_records'] if db_stats else 0} behavior, {db_stats['feedback_records'] if db_stats else 0} feedback</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Live status indicator
        if st.session_state.last_update_time:
            time_since = pd.Timestamp.now() - st.session_state.last_update_time
            if time_since < pd.Timedelta(minutes=1):
                st.success("🟢 LIVE")
            else:
                st.warning("🟡 UPDATING...")
        else:
            st.error("🔴 OFFLINE")
    
    with col3:
        # Force update button
        if st.button("🔄 Force Update", type="secondary"):
            st.session_state.last_update_time = None  # Force update
            st.rerun()
    
    st.markdown("---")
    
    # Enhanced Key Metrics Section
    st.markdown("### 🎯 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(predictions_df)
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{total_customers}</div>
            <div class='metric-label'>Total Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_churn_prob = predictions_df['churn_probability'].mean()
        prob_str, prob_color = format_probability(avg_churn_prob)
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color: {prob_color}'>{prob_str}</div>
            <div class='metric-label'>Avg Churn Probability</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_risk_count = len(predictions_df[predictions_df['risk_level'] == 'High Risk'])
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color: #FF4444'>{high_risk_count}</div>
            <div class='metric-label'>High Risk Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        churn_rate = predictions_df['churn_prediction'].mean() * 100
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{churn_rate:.1f}%</div>
            <div class='metric-label'>Predicted Churn Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Real-time changes indicator
    if st.session_state.update_count > 1:
        st.info(f"🔄 Dashboard has been updated {st.session_state.update_count - 1} times with live data changes")
    
    # Detailed Analytics Section
    st.markdown("---")
    st.markdown("### 📊 Detailed Analytics")
    
    # Row 1: Risk and Sentiment Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Risk Level Distribution")
        risk_chart = create_risk_distribution_chart(predictions_df)
        st.plotly_chart(risk_chart, use_container_width=True)
        
        # Risk statistics
        risk_stats = predictions_df['risk_level'].value_counts()
        st.markdown("**Risk Breakdown:**")
        for risk_level, count in risk_stats.items():
            percentage = (count / len(predictions_df)) * 100
            color = get_risk_level_color(risk_level)
            st.markdown(f"<span style='color: {color}; font-weight: 600;'>{risk_level}:</span> {count} customers ({percentage:.1f}%)", unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 💭 Sentiment Analysis")
        sentiment_df = st.session_state.sentiment_analyzer.analyze_feedback(load_data("data/customer_feedback.csv"))
        sentiment_chart = create_sentiment_distribution_chart(sentiment_df)
        st.plotly_chart(sentiment_chart, use_container_width=True)
        
        # Sentiment statistics
        sentiment_stats = sentiment_df['sentiment_label'].value_counts()
        st.markdown("**Sentiment Breakdown:**")
        for sentiment, count in sentiment_stats.items():
            percentage = (count / len(sentiment_df)) * 100
            color = get_sentiment_color(sentiment)
            st.markdown(f"<span style='color: {color}; font-weight: 600;'>{sentiment}:</span> {count} customers ({percentage:.1f}%)", unsafe_allow_html=True)
    
    # Row 2: Behavioral Analytics
    st.markdown("#### 📈 Behavioral Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        # Login frequency vs churn probability
        fig = px.scatter(
            predictions_df,
            x='login_frequency',
            y='churn_probability',
            color='risk_level',
            title="Login Frequency vs Churn Probability",
            color_discrete_map={
                'Low Risk': '#1FB6A6',
                'Medium Risk': '#FFA500',
                'High Risk': '#FF4444'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Inactivity days distribution
        fig = px.histogram(
            predictions_df,
            x='inactivity_days',
            color='risk_level',
            title="Inactivity Days Distribution",
            color_discrete_map={
                'Low Risk': '#1FB6A6',
                'Medium Risk': '#FFA500',
                'High Risk': '#FF4444'
            },
            nbins=10
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.markdown("### 🤖 Model Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Feature Importance")
        feature_importance = st.session_state.predictor.get_feature_importance()
        if feature_importance is not None:
            importance_chart = create_feature_importance_chart(feature_importance)
            st.plotly_chart(importance_chart, use_container_width=True)
    
    with col2:
        # Model Performance Summary removed - keeping only feature importance chart
        pass
    
    # Customer Risk Table
    st.markdown("### 🚨 High-Risk Customers Alert")
    
    # Show customers with highest churn probability
    high_risk_customers = predictions_df[predictions_df['risk_level'] == 'High Risk'].sort_values('churn_probability', ascending=False)
    
    if not high_risk_customers.empty:
        st.warning(f"⚠️ Found {len(high_risk_customers)} high-risk customers requiring immediate attention")
        
        # Display top 10 high-risk customers
        display_cols = ['customer_id', 'customer_name', 'churn_probability', 'risk_level', 'sentiment_label', 'inactivity_days', 'login_frequency']
        top_risk_df = high_risk_customers[display_cols].head(10).copy()
        
        # Format churn_probability as percentage
        top_risk_df['churn_probability'] = top_risk_df['churn_probability'].apply(
            lambda x: f"{min(x * 100, 100):.0f}%" if x <= 1 else f"{min(x, 100):.0f}%"
        )
        
        # Format inactivity_days as plain numbers (no decimals)
        top_risk_df['inactivity_days'] = top_risk_df['inactivity_days'].apply(lambda x: f"{int(x)}")
        
        def color_risk(val):
            color = get_risk_level_color(val)
            return f'color: {color}; font-weight: 600'
        
        def color_sentiment(val):
            color = get_sentiment_color(val)
            return f'color: {color}; font-weight: 600'
        
        styled_df = top_risk_df.style.map(color_risk, subset=['risk_level'])
        styled_df = styled_df.map(color_sentiment, subset=['sentiment_label'])
        
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.success("✅ No high-risk customers detected in the current dataset")

# Customer analysis page
def customer_analysis_page():
    st.markdown("""
    <div class='header'>
        <h1>Customer Analysis</h1>
        <p>Detailed analysis of individual customer churn risk</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or not st.session_state.model_trained:
        st.warning("Please load data and train the model first using the sidebar.")
        return
    
    # Customer search
    st.markdown("### 🔍 Search Customer")
    
    # Search input
    search_query = st.text_input(
        "Enter Customer ID",
        placeholder="e.g., 15634602, 15647311, CB_1, TC_7590-VHVEG, etc.",
        help="Enter the customer ID to search and analyze"
    )
    
        
    # Search button
    col1, col2 = st.columns([1, 3])
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    with col2:
        st.markdown("")  # Spacer
    
    # Initialize search results
    if search_button and search_query:
        try:
            # Clean the input
            search_clean = search_query.strip()
            
            # Try to convert to int for numeric IDs, otherwise keep as string
            try:
                customer_id = int(search_clean)
            except ValueError:
                # Keep as string for prefixed IDs (CB_, TC_)
                customer_id = search_clean
            
            # Get customer data
            customer_data = get_customer_by_id(st.session_state.merged_data, customer_id)
            
            if customer_data is None:
                st.error(f"❌ Customer {search_query} not found in the dataset")
                st.info("💡 Make sure the customer ID exists in the current dataset")
                return
                
        except Exception as e:
            st.error(f"❌ Error searching for customer: {str(e)}")
            st.info("💡 Please check the customer ID format and try again")
            return
    else:
        # Show search instructions
        st.info("🔍 Enter a customer ID above and click Search to analyze")
        return
    
    # Get prediction
    customer_df = pd.DataFrame([customer_data])
    prediction_result = st.session_state.predictor.predict_churn_probability(customer_df)
    prediction = prediction_result.iloc[0]
    
    # Track customer behavior for adaptive scoring
    if st.session_state.adaptive_engine:
        st.session_state.adaptive_engine.track_customer_behavior(customer_id, customer_data)
        
        # Calculate adaptive score
        adaptive_result = st.session_state.adaptive_engine.calculate_adaptive_score(
            customer_id, customer_data, prediction['churn_probability'] / 100
        )
        
        # Get adaptive insights
        insights = st.session_state.adaptive_engine.get_adaptive_insights(customer_id, adaptive_result)
    else:
        adaptive_result = None
        insights = []
    
    # Customer overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        customer_name = customer_data.get('customer_name', 'Unknown')
        st.markdown(f"""
        <div class='customer-overview'>
            <div class='customer-header'>Customer Overview - ID: {customer_id} | Name: {customer_name}</div>
            <div class='customer-metrics'>
                <div class='metric-card'>
                    <div style='font-size: 1.5rem; font-weight: 700; color: #667eea;'>{customer_data['login_frequency']:.1f}</div>
                    <div style='color: #6c757d; font-size: 0.9rem;'>Login Frequency /week</div>
                </div>
                <div class='metric-card'>
                    <div style='font-size: 1.5rem; font-weight: 700; color: #667eea;'>{customer_data['session_duration']:.1f}</div>
                    <div style='color: #6c757d; font-size: 0.9rem;'>Session Duration (min)</div>
                </div>
                <div class='metric-card'>
                    <div style='font-size: 1.5rem; font-weight: 700; color: #667eea;'>{customer_data['inactivity_days']:.0f}</div>
                    <div style='color: #6c757d; font-size: 0.9rem;'>Inactivity Days</div>
                </div>
                <div class='metric-card'>
                    <div style='font-size: 1.5rem; font-weight: 700; color: #667eea;'>{customer_data['transaction_count']:.0f}</div>
                    <div style='color: #6c757d; font-size: 0.9rem;'>Transaction Count</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='gauge-container'>
            <div class='gauge-title'>Churn Risk Gauge</div>
        </div>
        """, unsafe_allow_html=True)
        gauge_chart = create_risk_gauge_chart(prediction['churn_probability'])
        st.plotly_chart(gauge_chart, use_container_width=True)
    
    # Risk and sentiment analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Assessment")
        risk_color = get_risk_level_color(prediction['risk_level'])
        
        # Show both base and adaptive scores if available
        if adaptive_result:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%); padding: 1rem; border-radius: 10px; border-left: 5px solid {risk_color}; margin-bottom: 1rem;'>
                <div style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;'>🎯 Risk Analysis</div>
                <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                    <span>Base Score:</span>
                    <span style='font-weight: 700;'>{prediction['churn_probability']:.1f}%</span>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <span>Adaptive Score:</span>
                    <span style='font-weight: 700; color: {risk_color};'>{adaptive_result['adaptive_score']*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%); padding: 1rem; border-radius: 10px; border-left: 5px solid {risk_color};'>
                <div style='font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;'>🎯 Risk Analysis</div>
                <div style='font-size: 1.5rem; font-weight: 700; color: {risk_color};'>{prediction['churn_probability']:.1f}%</div>
                <div style='color: #6c757d; font-size: 1rem;'>{prediction['risk_level']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Sentiment Analysis")
        sentiment_color = get_sentiment_color(prediction['sentiment_label'])
        st.markdown(f"""
        <div class='card'>
            <h4 style='color: {sentiment_color};'>{prediction['sentiment_label']}</h4>
            <p>Sentiment Score: <strong>{prediction['sentiment_score']:.2f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Customer feedback
    if 'feedback_text' in customer_data and pd.notna(customer_data['feedback_text']):
        st.markdown("### Customer Feedback")
        st.markdown(f"""
        <div class='card'>
            <p><em>"{customer_data['feedback_text']}"</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Retention Analysis Section
    st.markdown("---")
    st.markdown("### 🎯 Retention Analysis")
    
    # Show adaptive insights if available
    if adaptive_result and insights:
        st.markdown("#### 🧠 Adaptive Insights")
        
        for insight in insights:
            insight_color = {
                'critical': '#dc3545',
                'warning': '#ffc107', 
                'info': '#17a2b8'
            }.get(insight['type'], '#6c757d')
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%); padding: 1rem; border-radius: 10px; border-left: 5px solid {insight_color}; margin-bottom: 1rem;'>
                <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                    <div style='background: {insight_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin-right: 0.5rem;'>
                        {insight['type'].upper()}
                    </div>
                    <div style='font-weight: 600; color: #2c3e50;'>{insight['title']}</div>
                </div>
                <div style='color: #6c757d; margin-bottom: 0.5rem;'>{insight['description']}</div>
                <div style='background: #e9ecef; padding: 0.75rem; border-radius: 8px; font-size: 0.9rem;'>
                    <strong>💡 Recommendation:</strong> {insight['recommendation']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Get customer feedback data for analysis
    feedback_data = None
    if st.session_state.database:
        feedback_data = st.session_state.database.get_feedback_data()
    
    # Analyze feedback sentiment and generate retention recommendations
    sentiment_label = prediction.get('sentiment_label', 'Neutral')
    sentiment_score = prediction.get('sentiment_score', 0)
    churn_prob = prediction.get('churn_probability', 0)
    risk_level = prediction.get('risk_level', 'Unknown')
    
    # Determine retention priority and actions
    if churn_prob >= 0.7 or risk_level == 'High Risk':
        priority = "🔴 Critical - Immediate Action Required"
        priority_color = "#FF4444"
    elif churn_prob >= 0.4 or risk_level == 'Medium Risk':
        priority = "🟠 Moderate - Monitor Closely"
        priority_color = "#FFA500"
    else:
        priority = "🟢 Low - Maintain Engagement"
        priority_color = "#1FB6A6"
    
    # Generate recommendations based on sentiment and feedback
    recommendations = []
    
    if sentiment_label == 'Negative':
        recommendations.extend([
            "📞 Schedule immediate follow-up call to address concerns",
            "🎁 Offer personalized retention incentive or discount",
            "👤 Assign dedicated customer success manager"
        ])
    elif sentiment_label == 'Neutral':
        recommendations.extend([
            "📧 Send personalized engagement email",
            "📱 Promote app features to increase usage",
            "💡 Offer financial wellness tips"
        ])
    else:  # Positive
        recommendations.extend([
            "🌟 Encourage referral program participation",
            "🏆 Offer loyalty rewards or premium features",
            "📊 Share personalized financial insights"
        ])
    
    # Add behavior-based recommendations
    if customer_data.get('inactivity_days', 0) > 7:
        recommendations.append("⏰ Send re-engagement notification about recent app updates")
    
    if customer_data.get('login_frequency', 0) < 2:
        recommendations.append("📲 Push notification with personalized offers")
    
    # Display retention analysis card
    st.markdown(f"""
    <div class='card' style='border-left: 5px solid {priority_color};'>
        <h4 style='color: {priority_color};'>{priority}</h4>
        <p><strong>Retention Probability:</strong> {max(0, min(100, (1 - churn_prob) * 100)):.0f}%</p>
        <p><strong>Sentiment Impact:</strong> {sentiment_label} ({sentiment_score:.2f})</p>
        <p><strong>Risk Factors:</strong> {risk_level}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display recommended actions
    st.markdown("#### 💡 Recommended Retention Actions")
    for rec in recommendations:
        st.markdown(f"- {rec}")
    
    # Historical feedback trend (if available)
    if feedback_data is not None and not feedback_data.empty:
        customer_history = feedback_data[feedback_data['customer_id'] == customer_id]
        if len(customer_history) > 1:
            st.markdown("#### 📈 Feedback History")
            st.info(f"This customer has provided {len(customer_history)} feedback entries. Review trends for deeper insights.")

# Model metrics page
def model_metrics_page():
    st.markdown("""
    <div class='header'>
        <h1>Model Performance Metrics</h1>
        <p>Machine learning model evaluation and statistics</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or not st.session_state.model_trained:
        st.warning("Please load data and train the model first using the sidebar.")
        return
    
    # Model information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h4>Model Type</h4>
            <p><strong>Random Forest Classifier</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h4>Training Samples</h4>
            <p><strong>{}</strong></p>
        </div>
        """.format(len(st.session_state.merged_data)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card'>
            <h4>Features Used</h4>
            <p><strong>{}</strong></p>
        </div>
        """.format(len(st.session_state.predictor.feature_columns)), unsafe_allow_html=True)
    
    # Feature importance
    st.markdown("### Feature Importance")
    feature_importance = st.session_state.predictor.get_feature_importance()
    if feature_importance is not None:
        importance_chart = create_feature_importance_chart(feature_importance)
        st.plotly_chart(importance_chart, use_container_width=True)
        
        # Display feature importance table
        st.markdown("### Feature Importance Details")
        st.dataframe(feature_importance, use_container_width=True)
    
    # Model evaluation (would need test data for real evaluation)
    st.markdown("### Model Evaluation Summary")
    st.markdown("""
    <div class='card'>
        <p><strong>Note:</strong> This is a demonstration model trained on sample data. 
        In a production environment, the model would be evaluated on a separate test set 
        with metrics like accuracy, precision, recall, and F1-score.</p>
        
        <p><strong>Current Model Features:</strong></p>
        <ul>
            <li>Login Frequency</li>
            <li>Session Duration</li>
            <li>Inactivity Days</li>
            <li>Transaction Count</li>
            <li>Sentiment Score (from customer feedback)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# About page
def about_page():
    st.markdown("""
    <div class='header'>
        <h1>About This Project</h1>
        <p>Behavior + Emotion Based Customer Churn Prediction System</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Project Overview
        
        This project demonstrates a comprehensive customer churn prediction system for digital banking 
        that combines both behavioral analytics and emotional sentiment analysis.
        
        ### Key Features
        
        - **Behavioral Analytics**: Analyzes customer usage patterns including login frequency, 
          session duration, inactivity periods, and transaction counts.
        
        - **Sentiment Analysis**: Processes customer feedback using NLP techniques (VADER and TextBlob) 
          to extract emotional insights and sentiment scores.
        
        - **Machine Learning**: Uses Random Forest classifier to predict churn probability 
          based on combined behavioral and sentiment features.
        
        - **Professional Dashboard**: Clean, banking-grade UI with real-time risk assessment 
          and customer analytics.
        
        ### Technology Stack
        
        - **Backend**: Python 3.10+, pandas, numpy, scikit-learn
        - **NLP**: VADER Sentiment, TextBlob
        - **Frontend**: Streamlit with custom CSS styling
        - **Visualization**: Plotly for interactive charts
        """)
    
    with col2:
        st.markdown("""
        ### Project Structure
        
        ```
        project_root/
        ├── app.py
        ├── data/
        ├── models/
        ├── src/
        │   ├── preprocessing.py
        │   ├── sentiment.py
        │   ├── model.py
        │   └── utils.py
        ├── assets/
        └── requirements.txt
        ```
        
        ### How to Use
        
        1. Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
        
        2. Run the application:
        ```bash
        streamlit run app.py
        ```
        
        3. Click "Load & Train Model" in the sidebar
        
        4. Explore the dashboard and customer analysis
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Academic Project Information
    
    **Project Title**: Behavior + Emotion Based Customer Churn Prediction System for Digital Banking
    
    **Type**: College Minor Project
    
    **Objective**: Demonstrate practical application of machine learning and NLP techniques 
    in solving real-world business problems in the banking sector.
    
    **Innovation**: Combines traditional behavioral metrics with emotional sentiment analysis 
    for more accurate churn prediction.
    """)

# Analytics page (new)
def analytics_page():
    st.markdown("""
    <div class='header'>
        <h1>📊 Advanced Analytics</h1>
        <p>Deep Dive into Customer Behavior Patterns and Trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or not st.session_state.model_trained:
        st.warning("🔄 System is initializing... Please wait a moment.")
        return
    
    # Check for real-time updates
    update_real_time_data()
    
    # Get predictions with risk levels
    predictions_df = st.session_state.predictor.predict_churn_probability(st.session_state.merged_data)
    
    st.markdown("### 🔍 Customer Segmentation Analysis")
    
    if predictions_df is not None:
        df = predictions_df
        
        # Segmentation by churn risk and sentiment
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 Churn Risk Distribution")
            if 'risk_level' in df.columns:
                risk_counts = df['risk_level'].value_counts()
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    color=risk_counts.index,
                    color_discrete_map={
                        'Low Risk': '#1FB6A6',
                        'Medium Risk': '#FFA500',
                        'High Risk': '#FF4444'
                    }
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Risk level data not available")
        
        with col2:
            st.markdown("#### 💭 Sentiment Analysis")
            if 'sentiment_label' in df.columns:
                sentiment_counts = df['sentiment_label'].value_counts()
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'Positive': '#1FB6A6',
                        'Neutral': '#5A6C7D',
                        'Negative': '#FF4444'
                    }
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sentiment data not available")
        
        with col3:
            st.markdown("#### 🏦 Data Sources")
            if 'source' in df.columns:
                source_counts = df['source'].value_counts()
                fig = px.bar(
                    x=source_counts.index,
                    y=source_counts.values,
                    color=source_counts.index,
                    color_discrete_sequence=['#1FB6A6', '#0B1C2D', '#5A6C7D']
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Source data not available")
        
        st.markdown("### 📈 Key Factor Insights")
        st.markdown("<p style='color: #5A6C7D; font-size: 14px;'>Understanding what drives customer churn</p>", unsafe_allow_html=True)
        
        # Define available columns for correlation
        numeric_cols = ['login_frequency', 'session_duration', 'inactivity_days', 
                       'transaction_count', 'account_balance', 'credit_score', 'age', 'churn']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        # Calculate key correlations with churn
        if 'churn' in df.columns and len(available_cols) > 2:
            corr_matrix = df[available_cols].corr()
            churn_correlations = corr_matrix['churn'].drop('churn').sort_values(key=lambda x: abs(x), ascending=False)
            
            # Create 3 columns for insights
            insight_cols = st.columns(3)
            
            # Define user-friendly labels
            labels = {
                'login_frequency': 'Login Frequency',
                'session_duration': 'Session Duration',
                'inactivity_days': 'Days Inactive',
                'transaction_count': 'Transactions',
                'account_balance': 'Account Balance',
                'credit_score': 'Credit Score',
                'age': 'Customer Age'
            }
            
            # Top 3 factors affecting churn
            for idx, (col, corr_val) in enumerate(churn_correlations.head(3).items()):
                with insight_cols[idx]:
                    # Determine impact direction and icon
                    if corr_val > 0.3:
                        impact = "🔴 High Risk"
                        color = "#FF4444"
                        desc = f"Higher {labels.get(col, col)} → More likely to churn"
                    elif corr_val > 0:
                        impact = "🟡 Moderate Risk"
                        color = "#FFA500"
                        desc = f"Slight increase in churn risk"
                    elif corr_val > -0.3:
                        impact = "🟢 Low Protection"
                        color = "#1FB6A6"
                        desc = f"Slight decrease in churn risk"
                    else:
                        impact = "🟢 Strong Protection"
                        color = "#1FB6A6"
                        desc = f"Lower {labels.get(col, col)} → Less likely to churn"
                    
                    st.markdown(f"""
                    <div style='background: #f8f9fa; border-radius: 12px; padding: 15px; border-left: 4px solid {color}; margin-bottom: 10px;'>
                        <h4 style='margin: 0; color: #0B1C2D; font-size: 16px;'>{labels.get(col, col)}</h4>
                        <p style='margin: 5px 0; font-size: 13px; color: {color}; font-weight: 600;'>{impact}</p>
                        <p style='margin: 0; font-size: 12px; color: #5A6C7D; line-height: 1.4;'>{desc}</p>
                        <p style='margin: 8px 0 0 0; font-size: 11px; color: #888;'>Correlation: {corr_val:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Simple explanation
            st.markdown("""
            <div style='background: #E8F4F8; border-radius: 8px; padding: 15px; margin-top: 20px;'>
                <h5 style='margin: 0 0 10px 0; color: #0B1C2D;'>📖 How to Read This:</h5>
                <ul style='margin: 0; padding-left: 20px; color: #5A6C7D; font-size: 13px; line-height: 1.6;'>
                    <li><b>🔴 High Risk</b>: This factor strongly predicts customers leaving</li>
                    <li><b>🟡 Moderate Risk</b>: This factor has some impact on churn</li>
                    <li><b>🟢 Protection</b>: This factor helps keep customers from leaving</li>
                    <li><b>Correlation Score</b>: Ranges from -1 to +1 (higher absolute value = stronger relationship)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Insufficient data for correlation analysis")

# Reports page (new)
def reports_page():
    st.markdown("""
    <div class='header'>
        <h1>📄 Reports & Insights</h1>
        <p>Generate Comprehensive Reports and Export Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded or not st.session_state.model_trained:
        st.warning("🔄 System is initializing... Please wait a moment.")
        return
    
    # Get predictions with risk levels
    predictions_df = st.session_state.predictor.predict_churn_probability(st.session_state.merged_data)
    
    st.markdown("### 📊 Executive Summary Report")
    
    if predictions_df is not None:
        df = predictions_df
        
        # Key metrics summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(df)
            st.metric("Total Customers", f"{total_customers:,}")
        
        with col2:
            if 'churn' in df.columns:
                churn_rate = (df['churn'].mean() * 100)
                st.metric("Overall Churn Rate", f"{churn_rate:.1f}%", 
                         delta=f"{churn_rate - 20:.1f}%" if churn_rate != 20 else None)
            else:
                st.metric("Overall Churn Rate", "N/A")
        
        with col3:
            if 'risk_level' in df.columns:
                high_risk = len(df[df['risk_level'] == 'High Risk'])
                st.metric("High Risk Customers", f"{high_risk}")
            else:
                st.metric("High Risk Customers", "N/A")
        
        with col4:
            if 'sentiment_score' in df.columns:
                avg_sentiment = df['sentiment_score'].mean()
                st.metric("Avg Sentiment Score", f"{avg_sentiment:.2f}")
            else:
                st.metric("Avg Sentiment Score", "N/A")
        
        st.markdown("---")
        
        # Export options
        st.markdown("### 💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Customer Data (CSV)", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='customer_churn_analysis.csv',
                    mime='text/csv'
                )
        
        with col2:
            if st.button("📊 Export High-Risk Customers", use_container_width=True):
                if 'risk_level' in df.columns:
                    high_risk_df = df[df['risk_level'] == 'High Risk']
                    if not high_risk_df.empty:
                        csv = high_risk_df.to_csv(index=False)
                        st.download_button(
                            label="Download High-Risk CSV",
                            data=csv,
                            file_name='high_risk_customers.csv',
                            mime='text/csv'
                        )
                    else:
                        st.info("No high-risk customers found")
                else:
                    st.error("Risk level data not available")
        
        st.markdown("---")
        
        # Data preview with lazy loading
        st.markdown("### Data Preview")
        
        # Initialize session state for pagination
        if 'data_preview_page' not in st.session_state:
            st.session_state.data_preview_page = 1
        if 'data_preview_rows_per_page' not in st.session_state:
            st.session_state.data_preview_rows_per_page = 100
        
        # Pagination controls
        total_rows = len(df)
        total_pages = (total_rows + st.session_state.data_preview_rows_per_page - 1) // st.session_state.data_preview_rows_per_page
        
        col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
        
        with col1:
            if st.button("⬅️ Previous", disabled=st.session_state.data_preview_page <= 1):
                st.session_state.data_preview_page -= 1
                st.rerun()
        
        with col2:
            if st.button("Next ➡️", disabled=st.session_state.data_preview_page >= total_pages):
                st.session_state.data_preview_page += 1
                st.rerun()
        
        with col3:
            st.write(f"Page {st.session_state.data_preview_page} of {total_pages} | Total: {total_rows} customers")
        
        with col4:
            rows_per_page = st.selectbox(
                "Rows per page",
                [50, 100, 200, 500],
                index=1,
                key="rows_per_page_select"
            )
            if rows_per_page != st.session_state.data_preview_rows_per_page:
                st.session_state.data_preview_rows_per_page = rows_per_page
                st.session_state.data_preview_page = 1
                st.rerun()
        
        # Calculate slice indices
        start_idx = (st.session_state.data_preview_page - 1) * st.session_state.data_preview_rows_per_page
        end_idx = start_idx + st.session_state.data_preview_rows_per_page
        
        # Display data slice
        display_df = df.iloc[start_idx:end_idx].copy()
        
        # Format churn_probability as percentage if present
        if 'churn_probability' in display_df.columns:
            display_df['churn_probability'] = display_df['churn_probability'].apply(
                lambda x: f"{min(x * 100, 100):.0f}%" if x <= 1 else f"{min(x, 100):.0f}%"
            )
        
        # Format inactivity_days as plain numbers if present
        if 'inactivity_days' in display_df.columns:
            display_df['inactivity_days'] = display_df['inactivity_days'].apply(lambda x: f"{int(x)}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Quick navigation
        if total_pages > 10:
            st.markdown("**Quick Navigation:**")
            page_cols = st.columns(min(10, total_pages))
            for i, col in enumerate(page_cols):
                page_num = i + 1
                if col.button(f"Page {page_num}", key=f"quick_nav_{page_num}"):
                    st.session_state.data_preview_page = page_num
                    st.rerun()

# Main application
def main():
    # Auto-initialize the application on startup
    initialize_app()
    
    # Get selected page from sidebar
    page = sidebar()
    
    # Display selected page based on new navigation system
    if page == "dashboard":
        dashboard_page()
    elif page == "customer_analysis":
        customer_analysis_page()
    elif page == "analytics":
        analytics_page()
    elif page == "reports":
        reports_page()

if __name__ == "__main__":
    main()
