"""
Kaggle Dataset Loader Module
Loads and processes multiple Kaggle datasets for banking project
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class KaggleDataLoader:
    """Loads and processes multiple Kaggle datasets for banking churn prediction"""
    
    def __init__(self):
        self.data_dir = "kaggle_data"
        os.makedirs("data", exist_ok=True)
    
    def load_churn_modelling(self):
        """Load Churn_Modelling.csv dataset"""
        try:
            df = pd.read_csv(f"{self.data_dir}/Churn_Modelling.csv")
            print(f"✅ Loaded Churn_Modelling dataset: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading Churn_Modelling: {e}")
            return None
    
    def load_customers_behavior(self):
        """Load Customers_Behavior.csv dataset"""
        try:
            df = pd.read_csv(f"{self.data_dir}/Customers_Behavior.csv")
            print(f"✅ Loaded Customers_Behavior dataset: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading Customers_Behavior: {e}")
            return None
    
    def load_telco_churn(self):
        """Load Telco-Customer-Churn.csv dataset"""
        try:
            df = pd.read_csv(f"{self.data_dir}/Telco-Customer-Churn.csv")
            print(f"✅ Loaded Telco-Customer-Churn dataset: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading Telco-Customer-Churn: {e}")
            return None
    
    def combine_datasets(self):
        """Combine all three datasets into unified banking dataset"""
        print("🔄 Combining all Kaggle datasets...")
        
        # Load all datasets
        churn_df = self.load_churn_modelling()
        behavior_df = self.load_customers_behavior()
        telco_df = self.load_telco_churn()
        
        banking_data = []
        
        # Process Churn_Modelling dataset
        if churn_df is not None:
            for _, row in churn_df.iterrows():
                # Use original CustomerId
                customer_id = row['CustomerId']
                banking_data.append({
                    'customer_id': customer_id,
                    'login_frequency': np.random.randint(5, 25),
                    'session_duration': row['Tenure'] * 30,
                    'inactivity_days': np.where(row['IsActiveMember'] == 0, 
                                              np.random.exponential(20), 
                                              np.random.exponential(5)),
                    'transaction_count': row['NumOfProducts'] * 12,
                    'churn': row['Exited'],
                    'account_balance': row['Balance'],
                    'credit_score': row['CreditScore'],
                    'age': row['Age'],
                    'source': 'Churn_Modelling'
                })
        
        # Process Customers_Behavior dataset
        if behavior_df is not None:
            for _, row in behavior_df.iterrows():
                # Use original CustomerID with prefix
                customer_id = f"CB_{row['CustomerID']}"
                # Create churn based on spending score and loyalty
                churn_prob = (row['Spending Score (1-100)'] < 30) * 0.3 + \
                            (row['Loyalty Years'] < 3) * 0.2 + \
                            (row['Preferred Category'] == 'Budget') * 0.1
                
                banking_data.append({
                    'customer_id': customer_id,
                    'login_frequency': np.random.randint(3, 20),
                    'session_duration': row['Loyalty Years'] * 25,
                    'inactivity_days': np.random.exponential(8),
                    'transaction_count': row['Annual Income (k$)'] / 2,
                    'churn': int(churn_prob > 0.3),
                    'account_balance': row['Estimated Savings (k$)'] * 1000,
                    'credit_score': row['Credit Score'],
                    'age': row['Age'],
                    'source': 'Customers_Behavior'
                })
        
        # Process Telco-Customer-Churn dataset
        if telco_df is not None:
            for _, row in telco_df.iterrows():
                # Map telco features to banking
                tenure_months = float(pd.to_numeric(row['tenure'], errors='coerce') if pd.notna(row['tenure']) else 0)
                monthly_charges = float(pd.to_numeric(row['MonthlyCharges'], errors='coerce') if pd.notna(row['MonthlyCharges']) else 0)
                
                # Use original customerID with prefix
                customer_id = f"TC_{row['customerID']}"
                
                banking_data.append({
                    'customer_id': customer_id,
                    'login_frequency': 30 if row['Contract'] == 'Month-to-month' else 15,
                    'session_duration': tenure_months * 30,
                    'inactivity_days': np.random.exponential(10),
                    'transaction_count': int(monthly_charges * 2),
                    'churn': 1 if row['Churn'] == 'Yes' else 0,
                    'account_balance': np.random.normal(5000, 2000),
                    'credit_score': np.random.normal(650, 100),
                    'age': np.random.normal(40, 12),
                    'source': 'Telco-Customer-Churn'
                })
        
        # Create DataFrame and remove duplicates based on customer_id
        df = pd.DataFrame(banking_data)
        
        # Remove duplicate customer IDs, keeping first occurrence
        if not df.empty and 'customer_id' in df.columns:
            initial_count = len(df)
            df = df.drop_duplicates(subset=['customer_id'], keep='first')
            removed_count = initial_count - len(df)
            if removed_count > 0:
                print(f"🧹 Removed {removed_count} duplicate customer IDs")
        
        return df
    
    def generate_feedback(self, banking_df):
        """Generate feedback based on customer profiles"""
        feedback_data = []
        
        for _, row in banking_df.iterrows():
            if row['churn'] == 1:
                # Churned customers - negative feedback
                if row['credit_score'] < 600:
                    feedback = np.random.choice([
                        "Too many fees for basic services.",
                        "The app lacks some essential features.",
                        "Having trouble with bill payments feature."
                    ])
                elif row['age'] > 50:
                    feedback = np.random.choice([
                        "The user interface could be more intuitive.",
                        "Login process is sometimes slow.",
                        "Having some issues with the latest update."
                    ])
                else:
                    feedback = np.random.choice([
                        "The mobile app keeps crashing on my device.",
                        "Need better fraud detection mechanisms.",
                        "I'm concerned about the recent security changes."
                    ])
            else:
                # Active customers - positive feedback
                if row['credit_score'] > 700:
                    feedback = np.random.choice([
                        "Excellent banking experience overall!",
                        "Transaction processing is very fast and reliable.",
                        "Biometric login works perfectly!"
                    ])
                else:
                    feedback = np.random.choice([
                        "The app is user-friendly and efficient.",
                        "Very satisfied with the service quality.",
                        "Customer service was very helpful today."
                    ])
            
            feedback_data.append({
                'customer_id': row['customer_id'],
                'feedback_text': feedback
            })
        
        return pd.DataFrame(feedback_data)
    
    def process_and_save(self):
        """Process all datasets and save for banking project"""
        print("� Processing all Kaggle datasets...")
        
        # Combine all datasets
        banking_df = self.combine_datasets()
        
        if banking_df.empty:
            print("❌ No data loaded from any dataset")
            return False
        
        print(f"📊 Combined dataset shape: {banking_df.shape}")
        
        # Generate feedback
        print("💭 Generating customer feedback...")
        feedback_df = self.generate_feedback(banking_df)
        
        # Save processed datasets
        behavior_path = "data/customer_behavior.csv"
        feedback_path = "data/customer_feedback.csv"
        
        banking_df.to_csv(behavior_path, index=False)
        feedback_df.to_csv(feedback_path, index=False)
        
        print(f"✅ Saved banking behavior data: {behavior_path}")
        print(f"✅ Saved banking feedback data: {feedback_path}")
        
        # Display dataset summary
        print("\n📋 Combined Dataset Summary:")
        print(f"   Total Customers: {len(banking_df)}")
        print(f"   Churn Rate: {banking_df['churn'].mean()*100:.1f}%")
        print(f"   Avg Credit Score: {banking_df['credit_score'].mean():.0f}")
        print(f"   Avg Account Balance: ${banking_df['account_balance'].mean():.0f}")
        print(f"   Avg Age: {banking_df['age'].mean():.1f} years")
        
        # Show source distribution
        if 'source' in banking_df.columns:
            print(f"\n📊 Data Sources:")
            for source, count in banking_df['source'].value_counts().items():
                print(f"   {source}: {count} customers")
        
        return True


def main():
    """Main function to test the data loader"""
    loader = KaggleDataLoader()
    success = loader.process_and_save()
    
    if success:
        print("\n🎉 All Kaggle datasets successfully integrated!")
    else:
        print("\n❌ Failed to integrate Kaggle datasets")


if __name__ == "__main__":
    main()
