"""
ML Model Module
Handles model training, prediction, and evaluation for customer churn prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


class ChurnPredictor:
    """Class for training and predicting customer churn"""
    
    def __init__(self, model_type='random_forest'):
        """Initialize the churn predictor"""
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
        else:
            raise ValueError("model_type must be 'random_forest' or 'logistic_regression'")
    
    def create_engineered_features(self, df):
        """Create engineered features for better churn prediction"""
        df_engineered = df.copy()
        
        # 1. Engagement Ratio - How active customer is relative to inactivity
        df_engineered['engagement_ratio'] = (
            df_engineered['login_frequency'] / (df_engineered['inactivity_days'] + 1)
        )
        
        # 2. Transaction Intensity - Transactions per login session
        df_engineered['transaction_intensity'] = (
            df_engineered['transaction_count'] / (df_engineered['login_frequency'] + 0.1)
        )
        
        # 3. Session Efficiency - Transaction value per session minute
        if 'account_balance' in df_engineered.columns:
            df_engineered['session_efficiency'] = (
                df_engineered['account_balance'] / (df_engineered['session_duration'] + 1)
            )
        
        # 4. Risk Score Composite - Combined risk indicator
        df_engineered['risk_composite'] = (
            (df_engineered['inactivity_days'] * 0.4) +
            (df_engineered['login_frequency'] * -0.3) +
            (df_engineered['transaction_count'] * -0.2) +
            (df_engineered['session_duration'] * -0.1)
        )
        
        # 5. Customer Tenure Proxy - Based on accumulated activity
        df_engineered['activity_accumulation'] = (
            df_engineered['login_frequency'] * df_engineered['session_duration'] +
            df_engineered['transaction_count'] * 10
        )
        
        # 6. Behavioral Consistency - Variance in engagement patterns
        df_engineered['engagement_stability'] = (
            df_engineered['login_frequency'] / (df_engineered['session_duration'] + 1)
        )
        
        # 7. High-Risk Flags - Binary indicators for critical thresholds
        df_engineered['high_inactivity_flag'] = (df_engineered['inactivity_days'] > 30).astype(int)
        df_engineered['low_engagement_flag'] = (df_engineered['login_frequency'] < 1).astype(int)
        df_engineered['zero_transaction_flag'] = (df_engineered['transaction_count'] == 0).astype(int)
        
        print(f"✅ Created {len(df_engineered.columns) - len(df.columns)} new engineered features")
        return df_engineered
    
    def prepare_features(self, df):
        """Prepare features for model training/prediction"""
        # First create engineered features
        df = self.create_engineered_features(df)
        
        # Define feature columns including engineered ones
        feature_columns = [
            # Original features
            'login_frequency',
            'session_duration', 
            'inactivity_days',
            'transaction_count',
            
            # Engineered features
            'engagement_ratio',
            'transaction_intensity',
            'risk_composite',
            'activity_accumulation',
            'engagement_stability',
            'high_inactivity_flag',
            'low_engagement_flag',
            'zero_transaction_flag'
        ]
        
        # Add optional features if available
        if 'sentiment_score' in df.columns:
            feature_columns.append('sentiment_score')
        
        if 'session_efficiency' in df.columns:
            feature_columns.append('session_efficiency')
        
        if 'account_balance' in df.columns:
            feature_columns.append('account_balance')
        
        if 'credit_score' in df.columns:
            feature_columns.append('credit_score')
        
        if 'age' in df.columns:
            feature_columns.append('age')
        
        # Ensure all required columns exist
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.feature_columns = feature_columns
        X = df[feature_columns].copy()
        
        return X
    
    def train(self, df, target_column='churn', test_size=0.2, validation_split=True):
        """Train the churn prediction model"""
        print(f"Training {self.model_type} model...")
        
        # Prepare features
        X = self.prepare_features(df)
        y = df[target_column]
        
        # Split data
        if validation_split:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        print(f"Model trained successfully on {len(X_train)} samples")
        
        # Evaluate if validation split
        if X_test is not None:
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            print(f"Training accuracy: {train_score:.4f}")
            print(f"Test accuracy: {test_score:.4f}")
            
            # Calculate and cache metrics
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Cache metrics for later use
            self._cached_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return X_train_scaled, X_test_scaled, y_train, y_test
        
        return X_train_scaled, None, y_train, None
    
    def predict(self, df):
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def predict_churn_probability(self, df):
        """Predict churn probability and risk level"""
        predictions, probabilities = self.predict(df)
        
        # Extract churn probability (probability of class 1)
        churn_probabilities = probabilities[:, 1]
        
        # Convert to percentage
        churn_percentages = churn_probabilities * 100
        
        # Classify risk levels
        risk_levels = []
        for prob in churn_percentages:
            if prob < 30:
                risk_levels.append("Low Risk")
            elif prob < 70:
                risk_levels.append("Medium Risk")
            else:
                risk_levels.append("High Risk")
        
        # Create results dataframe
        results_df = df.copy()
        results_df['churn_prediction'] = predictions
        results_df['churn_probability'] = churn_percentages
        results_df['risk_level'] = risk_levels
        
        return results_df
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Create metrics dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print("Model Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(class_report)
        
        return metrics, class_report, cm
    
    def get_model_metrics(self):
        """Get current model metrics"""
        if not self.is_trained:
            return None
        
        # Store metrics during training for quick access
        if hasattr(self, '_cached_metrics'):
            return self._cached_metrics
        
        # If no cached metrics, return None (metrics would be calculated during evaluation)
        return None
    
    def get_feature_importance(self):
        """Get feature importance (for Random Forest)"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance
        else:
            print("Feature importance not available for this model type")
            return None
    
    def save_model(self, model_path, scaler_path):
        """Save trained model and scaler"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        
        # Save model and scaler
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path, scaler_path):
        """Load trained model and scaler"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        # Load model and scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True
        
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")


# Utility function for complete model training pipeline
def train_churn_model(df, model_type='random_forest', save_models=True):
    """Complete model training pipeline"""
    print("Starting churn model training pipeline...")
    
    # Initialize predictor
    predictor = ChurnPredictor(model_type=model_type)
    
    # Train model
    X_train, X_test, y_train, y_test = predictor.train(df)
    
    # Evaluate model
    if X_test is not None:
        metrics, class_report, cm = predictor.evaluate_model(X_test, y_test)
        
        # Get feature importance
        feature_importance = predictor.get_feature_importance()
        if feature_importance is not None:
            print("\nFeature Importance:")
            print(feature_importance)
    
    # Save model if requested
    if save_models:
        predictor.save_model(
            "../models/churn_model_01.pkl",
            "../models/scaleri65t3_01.pkl"
        )
    
    return predictor, metrics if X_test is not None else None


# Utility function for loading and using trained model
def load_and_predict(df, model_path="../models/churn_model.pkl", scaler_path="../models/scaler.pkl"):
    """Load trained model and make predictions"""
    predictor = ChurnPredictor()
    predictor.load_model(model_path, scaler_path)
    
    # Make predictions
    results = predictor.predict_churn_probability(df)
    
    return results, predictor


if __name__ == "__main__":
    # Test the model training pipeline
    print("Testing churn model training pipeline...")
    
    # Load and preprocess data
    from preprocessing import preprocess_pipeline
    from sentiment import sentiment_analysis_pipeline
    
    # Preprocess data
    X, y, merged_df, preprocessor, feature_columns = preprocess_pipeline(
        "../data/customer_behavior.csv",
        "../data/customer_feedback.csv"
    )
    
    if merged_df is not None:
        # Add sentiment features
        sentiment_df, sentiment_summary = sentiment_analysis_pipeline(
            "../data/customer_feedback.csv", 
            merged_df
        )
        
        # Train model
        predictor, metrics = train_churn_model(sentiment_df, model_type='random_forest')
        
        print(f"\nModel training completed successfully!")
        if metrics:
            print(f"Final model accuracy: {metrics['accuracy']:.4f}")
