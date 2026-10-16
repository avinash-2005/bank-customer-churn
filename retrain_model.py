#!/usr/bin/env python3
"""
Retrain Model with Engineered Features
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from src.model import ChurnPredictor
from src.preprocessing import DataPreprocessor
from src.sentiment import SentimentAnalyzer
from src.utils import add_customer_names

def retrain_model():
    """Retrain model with new engineered features"""
    print("🚀 Retraining Model with Engineered Features...")
    
    # Load data
    print("📊 Loading data...")
    behavior_df = pd.read_csv('data/customer_behavior.csv')
    feedback_df = pd.read_csv('data/customer_feedback.csv')
    
    # Add customer names
    behavior_df = add_customer_names(behavior_df)
    
    # Basic data cleaning
    print("🧹 Cleaning data...")
    # Remove any rows with missing critical values
    behavior_df = behavior_df.dropna(subset=['login_frequency', 'session_duration', 'inactivity_days', 'transaction_count'])
    feedback_df = feedback_df.dropna(subset=['customer_id', 'feedback_text'])
    
    # Add sentiment features
    print("🧠 Analyzing sentiment...")
    sentiment_analyzer = SentimentAnalyzer()
    # Process each feedback text
    sentiment_scores = []
    sentiment_labels = []
    
    for text in feedback_df['feedback_text']:
        # Get VADER scores
        vader_result = sentiment_analyzer.analyze_sentiment_vader(text)
        # Get TextBlob scores
        textblob_result = sentiment_analyzer.analyze_sentiment_textblob(text)
        
        # Calculate combined sentiment
        sentiment_score = sentiment_analyzer.get_sentiment_score(
            vader_result['compound'], 
            textblob_result['polarity']
        )
        sentiment_label = sentiment_analyzer.get_sentiment_label(
            vader_result['compound'], 
            textblob_result['polarity']
        )
        
        sentiment_scores.append(sentiment_score)
        sentiment_labels.append(sentiment_label)
    
    feedback_df['sentiment_score'] = sentiment_scores
    feedback_df['sentiment_label'] = sentiment_labels
    
    # Merge datasets
    merged_df = pd.merge(behavior_df, feedback_df, on='customer_id', how='left')
    merged_df['sentiment_score'] = merged_df['sentiment_score'].fillna(0.5)
    
    print(f"📈 Final dataset shape: {merged_df.shape}")
    print(f"📋 Features: {list(merged_df.columns)}")
    
    # Initialize and train model
    print("\n🏋️ Training model with engineered features...")
    predictor = ChurnPredictor(model_type='random_forest')
    
    # Train model
    X_train, X_test, y_train, y_test = predictor.train(merged_df)
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    metrics, class_report, cm = predictor.evaluate_model(X_test, y_test)
    
    # Get feature importance
    feature_importance = predictor.get_feature_importance()
    if feature_importance is not None:
        print("\n🎯 Top 10 Feature Importance:")
        print(feature_importance.head(10))
    
    # Save enhanced model
    print("\n💾 Saving enhanced model...")
    predictor.save_model('models/enhanced_churn_model.pkl', 'models/enhanced_scaler.pkl')
    
    # Also backup old model
    os.rename('models/churn_model.pkl', 'models/churn_model_backup.pkl')
    os.rename('models/scaler.pkl', 'models/scaler_backup.pkl')
    
    # Copy enhanced model to main model names
    import shutil
    shutil.copy2('models/enhanced_churn_model.pkl', 'models/churn_model.pkl')
    shutil.copy2('models/enhanced_scaler.pkl', 'models/scaler.pkl')
    
    print(f"\n✅ Model Retrained Successfully!")
    print(f"📈 Accuracy: {metrics['accuracy']:.4f}")
    print(f"🎯 Precision: {metrics['precision']:.4f}")
    print(f"🔄 Recall: {metrics['recall']:.4f}")
    print(f"📊 F1-Score: {metrics['f1_score']:.4f}")
    print(f"\n💾 Models saved:")
    print(f"   - Main: models/churn_model.pkl")
    print(f"   - Backup: models/churn_model_backup.pkl")
    
    return predictor, metrics

if __name__ == "__main__":
    try:
        predictor, metrics = retrain_model()
        print("\n🎉 Retraining completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during retraining: {e}")
        import traceback
        traceback.print_exc()
