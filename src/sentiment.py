"""
Sentiment Analysis Module
Performs NLP sentiment analysis on customer feedback using VADER and TextBlob
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
for resource, pkg in [
    ('tokenizers/punkt', 'punkt'),
    ('tokenizers/punkt_tab', 'punkt_tab'),
    ('corpora/stopwords', 'stopwords'),
    ('sentiment/vader_lexicon.zip', 'vader_lexicon')
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


class SentimentAnalyzer:
    """Class for performing sentiment analysis on customer feedback"""
    
    def __init__(self):
        """Initialize sentiment analyzers"""
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean text data by removing punctuation, stopwords, and converting to lowercase"""
        if pd.isna(text) or text == "":
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
    
    def analyze_sentiment_vader(self, text):
        """Analyze sentiment using VADER"""
        if pd.isna(text) or text == "":
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        
        # Get VADER scores
        scores = self.vader_analyzer.polarity_scores(str(text))
        return scores
    
    def analyze_sentiment_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        if pd.isna(text) or text == "":
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        # Get TextBlob scores
        blob = TextBlob(str(text))
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def get_sentiment_label(self, vader_compound, textblob_polarity):
        """Get sentiment label based on VADER and TextBlob scores"""
        # Use VADER compound score as primary indicator
        if vader_compound >= 0.05:
            return "Positive"
        elif vader_compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    def get_sentiment_score(self, vader_compound, textblob_polarity):
        """Get normalized sentiment score (-1 to 1)"""
        # Combine VADER and TextBlob scores
        # VADER compound is already normalized (-1 to 1)
        # TextBlob polarity is also normalized (-1 to 1)
        
        # Weight VADER more heavily as it's specifically tuned for social media/text
        combined_score = 0.7 * vader_compound + 0.3 * textblob_polarity
        
        # Ensure score is within bounds
        combined_score = max(-1.0, min(1.0, combined_score))
        
        return combined_score
    
    def analyze_feedback(self, df, text_column='feedback_text'):
        """Analyze sentiment for all feedback in the dataframe"""
        results = []
        
        for idx, row in df.iterrows():
            text = row[text_column]
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Analyze with VADER
            vader_scores = self.analyze_sentiment_vader(text)
            
            # Analyze with TextBlob
            textblob_scores = self.analyze_sentiment_textblob(text)
            
            # Get combined sentiment
            sentiment_label = self.get_sentiment_label(
                vader_scores['compound'], 
                textblob_scores['polarity']
            )
            
            sentiment_score = self.get_sentiment_score(
                vader_scores['compound'], 
                textblob_scores['polarity']
            )
            
            # Store results
            result = {
                'customer_id': row['customer_id'],
                'original_text': text,
                'cleaned_text': cleaned_text,
                'vader_compound': vader_scores['compound'],
                'vader_positive': vader_scores['pos'],
                'vader_neutral': vader_scores['neu'],
                'vader_negative': vader_scores['neg'],
                'textblob_polarity': textblob_scores['polarity'],
                'textblob_subjectivity': textblob_scores['subjectivity'],
                'sentiment_label': sentiment_label,
                'sentiment_score': sentiment_score
            }
            
            results.append(result)
        
        # Convert to DataFrame
        sentiment_df = pd.DataFrame(results)
        
        print(f"Sentiment analysis completed for {len(sentiment_df)} records")
        print(f"Sentiment distribution:\n{sentiment_df['sentiment_label'].value_counts()}")
        
        return sentiment_df
    
    def get_sentiment_summary(self, sentiment_df):
        """Get summary statistics of sentiment analysis"""
        summary = {
            'total_customers': len(sentiment_df),
            'positive_count': len(sentiment_df[sentiment_df['sentiment_label'] == 'Positive']),
            'neutral_count': len(sentiment_df[sentiment_df['sentiment_label'] == 'Neutral']),
            'negative_count': len(sentiment_df[sentiment_df['sentiment_label'] == 'Negative']),
            'avg_sentiment_score': sentiment_df['sentiment_score'].mean(),
            'sentiment_std': sentiment_df['sentiment_score'].std()
        }
        
        # Calculate percentages
        summary['positive_percentage'] = (summary['positive_count'] / summary['total_customers']) * 100
        summary['neutral_percentage'] = (summary['neutral_count'] / summary['total_customers']) * 100
        summary['negative_percentage'] = (summary['negative_count'] / summary['total_customers']) * 100
        
        return summary
    
    def merge_sentiment_to_behavior(self, behavior_df, sentiment_df):
        """Merge sentiment analysis results with behavioral data"""
        # Select only relevant columns from sentiment analysis
        sentiment_features = sentiment_df[['customer_id', 'sentiment_label', 'sentiment_score']]
        
        # Merge with behavioral data
        merged_df = pd.merge(behavior_df, sentiment_features, on='customer_id', how='left')
        
        # Handle missing sentiment for customers without feedback
        merged_df['sentiment_label'] = merged_df['sentiment_label'].fillna('Neutral')
        merged_df['sentiment_score'] = merged_df['sentiment_score'].fillna(0.0)
        
        print(f"Merged sentiment features with behavioral data: {merged_df.shape}")
        return merged_df


# Utility function for complete sentiment analysis pipeline
def sentiment_analysis_pipeline(feedback_file, behavior_df=None):
    """Complete sentiment analysis pipeline"""
    analyzer = SentimentAnalyzer()
    
    # Load feedback data
    feedback_df = pd.read_csv(feedback_file)
    
    # Perform sentiment analysis
    sentiment_df = analyzer.analyze_feedback(feedback_df)
    
    # Get summary
    summary = analyzer.get_sentiment_summary(sentiment_df)
    print("Sentiment Analysis Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Merge with behavioral data if provided
    if behavior_df is not None:
        merged_df = analyzer.merge_sentiment_to_behavior(behavior_df, sentiment_df)
        return sentiment_df, merged_df, summary
    
    return sentiment_df, summary


if __name__ == "__main__":
    # Test the sentiment analysis pipeline
    print("Testing sentiment analysis pipeline...")
    
    # Test with sample feedback data
    sentiment_df, summary = sentiment_analysis_pipeline("../data/customer_feedback.csv")
    
    print(f"\nSentiment analysis completed successfully!")
    print(f"Analyzed {len(sentiment_df)} customer feedback entries")
    
    # Display sample results
    print(f"\nSample sentiment analysis results:")
    print(sentiment_df[['customer_id', 'sentiment_label', 'sentiment_score']].head())
