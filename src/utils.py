"""
Utility Functions Module
Contains helper functions for the churn prediction system
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib

# Customer name generation
def generate_customer_names(count):
    """Generate realistic customer names"""
    first_names = [
        'James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda',
        'William', 'Elizabeth', 'David', 'Barbara', 'Richard', 'Susan', 'Joseph', 'Jessica',
        'Thomas', 'Sarah', 'Charles', 'Karen', 'Christopher', 'Nancy', 'Daniel', 'Lisa',
        'Matthew', 'Betty', 'Anthony', 'Helen', 'Mark', 'Sandra', 'Donald', 'Donna',
        'Steven', 'Carol', 'Paul', 'Ruth', 'Andrew', 'Sharon', 'Joshua', 'Michelle',
        'Kenneth', 'Laura', 'Kevin', 'Sarah', 'Brian', 'Kimberly', 'George', 'Deborah',
        'Edward', 'Dorothy', 'Ronald', 'Lisa', 'Timothy', 'Nancy', 'Jason', 'Karen',
        'Jeffrey', 'Betty', 'Ryan', 'Helen', 'Jacob', 'Sandra', 'Gary', 'Donna',
        'Nicholas', 'Carol', 'Eric', 'Ruth', 'Jonathan', 'Sharon', 'Stephen', 'Michelle',
        'Larry', 'Laura', 'Justin', 'Sarah', 'Scott', 'Kimberly', 'Brandon', 'Deborah',
        'Benjamin', 'Dorothy', 'Samuel', 'Lisa', 'Frank', 'Nancy', 'Raymond', 'Donna',
        'Alexander', 'Sharon', 'Patrick', 'Michelle', 'Jack', 'Dorothy', 'Dennis', 'Lisa'
    ]
    
    last_names = [
        'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
        'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
        'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
        'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker',
        'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill',
        'Flores', 'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell',
        'Mitchell', 'Carter', 'Roberts', 'Gomez', 'Phillips', 'Evans', 'Turner',
        'Diaz', 'Parker', 'Cruz', 'Edwards', 'Collins', 'Stewart', 'Sanchez', 'Morris',
        'Rogers', 'Reed', 'Cook', 'Morgan', 'Bell', 'Murphy', 'Bailey', 'Rivera',
        'Cooper', 'Richardson', 'Cox', 'Howard', 'Ward', 'Torres', 'Peterson', 'Gray',
        'Ramirez', 'James', 'Watson', 'Brooks', 'Kelly', 'Sanders', 'Price', 'Bennett',
        'Wood', 'Barnes', 'Ross', 'Henderson', 'Coleman', 'Jenkins', 'Perry', 'Powell',
        'Long', 'Patterson', 'Hughes', 'Flores', 'Washington', 'Butler', 'Simmons', 'Foster',
        'Gonzales', 'Bryant', 'Alexander', 'Russell', 'Griffin', 'Diaz', 'Hayes', 'Myers'
    ]
    
    names = []
    for i in range(count):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        names.append(f"{first_name} {last_name}")
    
    return names

def add_customer_names(df):
    """Add customer names to dataframe based on customer_id"""
    df_with_names = df.copy()
    
    # Create a mapping of customer_id to name (consistent for same ID)
    unique_ids = df_with_names['customer_id'].unique()
    name_mapping = {}
    
    # Generate names for each unique customer ID
    names = generate_customer_names(len(unique_ids))
    for i, customer_id in enumerate(unique_ids):
        name_mapping[customer_id] = names[i]
    
    # Add customer_name column
    df_with_names['customer_name'] = df_with_names['customer_id'].map(name_mapping)
    
    # Reorder columns to put customer_name after customer_id
    cols = list(df_with_names.columns)
    cols.insert(cols.index('customer_id') + 1, cols.pop(cols.index('customer_name')))
    df_with_names = df_with_names[cols]
    
    return df_with_names


def load_data(file_path):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def save_data(df, file_path):
    """Save dataframe to CSV file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        df.to_csv(file_path, index=False)
        print(f"Successfully saved data to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")
        return False


def create_risk_gauge_chart(probability, title="Churn Risk"):
    """Create a gauge chart for churn probability visualization"""
    # Determine color based on risk level
    if probability < 30:
        color = "#1FB6A6"  # Teal (Low Risk)
    elif probability < 70:
        color = "#FFA500"  # Orange (Medium Risk)
    else:
        color = "#FF4444"  # Red (High Risk)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "#E8F5E8"},
                {'range': [30, 70], 'color': "#FFF4E6"},
                {'range': [70, 100], 'color': "#FFEBEE"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "#0B1C2D", "family": "Arial, sans-serif"}
    )
    
    return fig


def create_sentiment_distribution_chart(sentiment_data):
    """Create a pie chart for sentiment distribution"""
    sentiment_counts = sentiment_data['sentiment_label'].value_counts()
    
    colors = {
        'Positive': '#1FB6A6',
        'Neutral': '#5A6C7D', 
        'Negative': '#FF4444'
    }
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        color=sentiment_counts.index,
        color_discrete_map=colors,
        title="Customer Sentiment Distribution"
    )
    
    fig.update_layout(
        font={'color': "#0B1C2D", "family": "Arial, sans-serif"},
        showlegend=True
    )
    
    return fig


def create_feature_importance_chart(feature_importance_df):
    """Create a bar chart for feature importance"""
    fig = px.bar(
        feature_importance_df.head(10),
        x='importance',
        y='feature',
        orientation='h',
        title="Top 10 Feature Importance",
        color='importance',
        color_continuous_scale='teal'
    )
    
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Features",
        font={'color': "#0B1C2D", "family": "Arial, sans-serif"},
        height=400
    )
    
    return fig


def create_risk_distribution_chart(predictions_df):
    """Create a bar chart for risk level distribution"""
    risk_counts = predictions_df['risk_level'].value_counts()
    
    colors = {
        'Low Risk': '#1FB6A6',
        'Medium Risk': '#FFA500',
        'High Risk': '#FF4444'
    }
    
    fig = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        color=risk_counts.index,
        color_discrete_map=colors,
        title="Customer Risk Level Distribution"
    )
    
    fig.update_layout(
        xaxis_title="Risk Level",
        yaxis_title="Number of Customers",
        font={'color': "#0B1C2D", "family": "Arial, sans-serif"},
        showlegend=False
    )
    
    return fig


def create_customer_metrics_cards(customer_data):
    """Create metrics cards for customer analysis"""
    metrics = {
        'Login Frequency': f"{customer_data['login_frequency']:.1f} per week",
        'Session Duration': f"{customer_data['session_duration']:.1f} minutes",
        'Inactivity Days': f"{customer_data['inactivity_days']:.0f} days",
        'Transaction Count': f"{customer_data['transaction_count']:.0f} transactions"
    }
    
    return metrics


def format_probability(probability):
    """Format probability as percentage with appropriate color"""
    percentage = f"{probability:.1f}%"
    
    if probability < 30:
        color = "#1FB6A6"  # Teal
    elif probability < 70:
        color = "#FFA500"  # Orange
    else:
        color = "#FF4444"  # Red
    
    return percentage, color


def get_risk_level_color(risk_level):
    """Get color for risk level"""
    colors = {
        'Low Risk': '#1FB6A6',
        'Medium Risk': '#FFA500', 
        'High Risk': '#FF4444'
    }
    return colors.get(risk_level, '#5A6C7D')


def get_sentiment_color(sentiment_label):
    """Get color for sentiment label"""
    colors = {
        'Positive': '#1FB6A6',
        'Neutral': '#5A6C7D',
        'Negative': '#FF4444'
    }
    return colors.get(sentiment_label, '#5A6C7D')


def validate_input_data(df, required_columns):
    """Validate that dataframe contains required columns"""
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True


def create_summary_statistics(df):
    """Create summary statistics for the dataset"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    summary = {
        'total_customers': len(df),
        'numeric_features': len(numeric_columns),
        'missing_values': df.isnull().sum().sum(),
        'churn_rate': df['churn'].mean() * 100 if 'churn' in df.columns else None
    }
    
    # Add statistics for numeric columns
    for col in numeric_columns:
        if col in df.columns:
            summary[f'{col}_mean'] = df[col].mean()
            summary[f'{col}_std'] = df[col].std()
    
    return summary


def export_model_results(results_df, file_path):
    """Export model prediction results to CSV"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Add timestamp to results
        results_df['analysis_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        results_df.to_csv(file_path, index=False)
        print(f"Results exported to {file_path}")
        return True
    except Exception as e:
        print(f"Error exporting results: {e}")
        return False


def check_model_files(model_path, scaler_path):
    """Check if model files exist and are valid"""
    model_exists = os.path.exists(model_path)
    scaler_exists = os.path.exists(scaler_path)
    
    if not model_exists:
        print(f"Model file not found: {model_path}")
    if not scaler_exists:
        print(f"Scaler file not found: {scaler_path}")
    
    return model_exists and scaler_exists


def get_customer_by_id(df, customer_id):
    """Get customer data by ID - handles both string and integer IDs"""
    # Try searching with the original type first
    customer_data = df[df['customer_id'] == customer_id]
    
    # If not found and customer_id is int, try as string
    if customer_data.empty and isinstance(customer_id, int):
        customer_data = df[df['customer_id'] == str(customer_id)]
    
    # If not found and customer_id is string that looks like int, try as int
    if customer_data.empty and isinstance(customer_id, str):
        try:
            customer_id_int = int(customer_id)
            customer_data = df[df['customer_id'] == customer_id_int]
        except ValueError:
            pass  # Not a numeric string, keep as is
    
    if customer_data.empty:
        return None
    
    return customer_data.iloc[0]


def create_customer_list(df):
    """Create formatted customer list for dropdown"""
    customer_options = []
    
    for _, row in df.iterrows():
        customer_info = f"Customer {row['customer_id']}"
        if 'churn' in row:
            status = "Churned" if row['churn'] == 1 else "Active"
            customer_info += f" ({status})"
        
        customer_options.append(customer_info)
    
    return customer_options


def format_large_number(number):
    """Format large numbers for display"""
    if number >= 1000000:
        return f"{number/1000000:.1f}M"
    elif number >= 1000:
        return f"{number/1000:.1f}K"
    else:
        return f"{number:.0f}"


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test data loading
    df = load_data("../data/customer_behavior.csv")
    if df is not None:
        print("Data loading test passed")
        
        # Test summary statistics
        summary = create_summary_statistics(df)
        print("Summary statistics:", summary)
        
        # Test customer list creation
        customer_list = create_customer_list(df)
        print(f"Created customer list with {len(customer_list)} options")
    
    print("Utility functions test completed")
