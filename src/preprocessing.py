"""
Data Preprocessing Module
Handles data cleaning, missing values, and feature normalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
for resource, pkg in [
    ('tokenizers/punkt', 'punkt'),
    ('tokenizers/punkt_tab', 'punkt_tab'),
    ('corpora/stopwords', 'stopwords')
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


class DataPreprocessor:
    """Class for preprocessing customer behavior and feedback data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.stop_words = set(stopwords.words('english'))
        
    def load_behavioral_data(self, file_path):
        """Load customer behavioral data from CSV"""
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded behavioral data: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading behavioral data: {e}")
            return None
    
    def load_feedback_data(self, file_path):
        """Load customer feedback data from CSV"""
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded feedback data: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading feedback data: {e}")
            return None
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Check for missing values
        missing_values = df.isnull().sum()
        print(f"Missing values before handling:\n{missing_values}")
        
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        print("Missing values handled successfully")
        return df
    
    def normalize_features(self, df, feature_columns, fit=True):
        """Normalize numerical features using StandardScaler"""
        if fit:
            # Fit and transform the data
            normalized_features = self.scaler.fit_transform(df[feature_columns])
        else:
            # Only transform the data (for new data)
            normalized_features = self.scaler.transform(df[feature_columns])
        
        # Create a new dataframe with normalized features
        normalized_df = df.copy()
        for i, col in enumerate(feature_columns):
            normalized_df[col] = normalized_features[:, i]
        
        print(f"Features normalized: {feature_columns}")
        return normalized_df
    
    def clean_text(self, text):
        """Clean text data by removing punctuation, stopwords, and converting to lowercase"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Join back to string
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def preprocess_feedback_text(self, df, text_column='feedback_text'):
        """Preprocess feedback text data"""
        df_copy = df.copy()
        
        # Apply text cleaning
        df_copy['cleaned_text'] = df_copy[text_column].apply(self.clean_text)
        
        print(f"Text preprocessing completed for {len(df_copy)} records")
        return df_copy
    
    def merge_datasets(self, behavior_df, feedback_df):
        """Merge behavioral and feedback datasets on customer_id"""
        # Merge datasets
        merged_df = pd.merge(behavior_df, feedback_df, on='customer_id', how='left')
        
        # Handle missing feedback for customers without feedback
        if 'feedback_text' in merged_df.columns:
            merged_df['feedback_text'] = merged_df['feedback_text'].fillna('No feedback available')
        
        print(f"Merged dataset shape: {merged_df.shape}")
        return merged_df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        # Define feature columns
        feature_columns = [
            'login_frequency', 
            'session_duration', 
            'inactivity_days', 
            'transaction_count'
        ]
        
        # Add sentiment_score if available (will be added later from sentiment analysis)
        if 'sentiment_score' in df.columns:
            feature_columns.append('sentiment_score')
        
        # Extract features
        X = df[feature_columns].copy()
        y = df['churn'] if 'churn' in df.columns else None
        
        print(f"Features prepared: {feature_columns}")
        return X, y, feature_columns


# Utility function for complete preprocessing pipeline
def preprocess_pipeline(behavior_file, feedback_file):
    """Complete preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    
    # Load data
    behavior_df = preprocessor.load_behavioral_data(behavior_file)
    feedback_df = preprocessor.load_feedback_data(feedback_file)
    
    if behavior_df is None or feedback_df is None:
        return None, None, None
    
    # Handle missing values
    behavior_df = preprocessor.handle_missing_values(behavior_df)
    feedback_df = preprocessor.handle_missing_values(feedback_df)
    
    # Preprocess feedback text
    feedback_df = preprocessor.preprocess_feedback_text(feedback_df)
    
    # Merge datasets
    merged_df = preprocessor.merge_datasets(behavior_df, feedback_df)
    
    # Prepare features
    X, y, feature_columns = preprocessor.prepare_features(merged_df)
    
    return X, y, merged_df, preprocessor, feature_columns


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing preprocessing pipeline...")
    X, y, merged_df, preprocessor, feature_columns = preprocess_pipeline(
        "../data/customer_behavior.csv",
        "../data/customer_feedback.csv"
    )
    
    if X is not None:
        print(f"Preprocessing successful!")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Feature columns: {feature_columns}")
