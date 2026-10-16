"""
SQLite Database Manager Module
Stores and manages real-time customer churn data in SQLite3
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os

class DatabaseManager:
    """Manages SQLite3 database for customer churn prediction data"""
    
    def __init__(self, db_path="data/churn_database.db"):
        self.db_path = db_path
        os.makedirs("data", exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create customer_behavior table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS customer_behavior (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    customer_name TEXT,
                    login_frequency REAL,
                    session_duration REAL,
                    inactivity_days REAL,
                    transaction_count INTEGER,
                    churn INTEGER,
                    account_balance REAL,
                    credit_score INTEGER,
                    age INTEGER,
                    source TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Check if customer_name column exists, add if it doesn't
            cursor.execute("PRAGMA table_info(customer_behavior)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'customer_name' not in columns:
                cursor.execute("ALTER TABLE customer_behavior ADD COLUMN customer_name TEXT")
            
            # Create customer_feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS customer_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    feedback_text TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create model_performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    training_samples INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create data_updates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    update_count INTEGER,
                    total_customers INTEGER,
                    churn_rate REAL,
                    avg_credit_score REAL,
                    avg_account_balance REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Database initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
    
    def store_behavior_data(self, df):
        """Store customer behavior data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Add timestamp column if not present
            df_to_store = df.copy()
            df_to_store['timestamp'] = datetime.now()
            
            # Store data
            df_to_store.to_sql('customer_behavior', conn, if_exists='append', index=False)
            
            conn.close()
            print(f"✅ Stored {len(df_to_store)} behavior records in database")
            return True
            
        except Exception as e:
            print(f"❌ Error storing behavior data: {e}")
            return False
    
    def store_feedback_data(self, df):
        """Store customer feedback data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Add timestamp column if not present
            df_to_store = df.copy()
            df_to_store['timestamp'] = datetime.now()
            
            # Store data
            df_to_store.to_sql('customer_feedback', conn, if_exists='append', index=False)
            
            conn.close()
            print(f"✅ Stored {len(df_to_store)} feedback records in database")
            return True
            
        except Exception as e:
            print(f"❌ Error storing feedback data: {e}")
            return False
    
    def store_model_performance(self, model_type, accuracy, precision, recall, f1_score, training_samples):
        """Store model performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_performance 
                (model_type, accuracy, precision, recall, f1_score, training_samples)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (model_type, accuracy, precision, recall, f1_score, training_samples))
            
            conn.commit()
            conn.close()
            print(f"✅ Stored model performance metrics")
            return True
            
        except Exception as e:
            print(f"❌ Error storing model performance: {e}")
            return False
    
    def store_update_summary(self, update_count, total_customers, churn_rate, avg_credit_score, avg_account_balance):
        """Store data update summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO data_updates 
                (update_count, total_customers, churn_rate, avg_credit_score, avg_account_balance)
                VALUES (?, ?, ?, ?, ?)
            ''', (update_count, total_customers, churn_rate, avg_credit_score, avg_account_balance))
            
            conn.commit()
            conn.close()
            print(f"✅ Stored update summary #{update_count}")
            return True
            
        except Exception as e:
            print(f"❌ Error storing update summary: {e}")
            return False
    
    def get_latest_behavior_data(self, limit=1000):
        """Get latest behavior data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT * FROM customer_behavior 
                ORDER BY timestamp DESC 
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(limit,))
            conn.close()
            
            return df
            
        except Exception as e:
            print(f"❌ Error retrieving behavior data: {e}")
            return None
    
    def get_feedback_data(self, limit=1000):
        """Get feedback data from database (alias for get_latest_feedback_data)"""
        return self.get_latest_feedback_data(limit)
    
    def get_latest_feedback_data(self, limit=1000):
        """Get latest feedback data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT * FROM customer_feedback 
                ORDER BY timestamp DESC 
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(limit,))
            conn.close()
            
            return df
            
        except Exception as e:
            print(f"❌ Error retrieving feedback data: {e}")
            return None
    
    def get_model_performance_history(self):
        """Get model performance history"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            df = pd.read_sql_query('''
                SELECT * FROM model_performance 
                ORDER BY timestamp DESC
            ''', conn)
            
            conn.close()
            return df
            
        except Exception as e:
            print(f"❌ Error retrieving model performance: {e}")
            return None
    
    def get_update_history(self):
        """Get data update history"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            df = pd.read_sql_query('''
                SELECT * FROM data_updates 
                ORDER BY timestamp DESC
            ''', conn)
            
            conn.close()
            return df
            
        except Exception as e:
            print(f"❌ Error retrieving update history: {e}")
            return None
    
    def get_database_stats(self):
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # Get table counts
            cursor.execute("SELECT COUNT(*) FROM customer_behavior")
            stats['behavior_records'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM customer_feedback")
            stats['feedback_records'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM model_performance")
            stats['model_records'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM data_updates")
            stats['update_records'] = cursor.fetchone()[0]
            
            # Get latest update time
            cursor.execute("SELECT MAX(timestamp) FROM customer_behavior")
            latest_behavior = cursor.fetchone()[0]
            
            cursor.execute("SELECT MAX(timestamp) FROM customer_feedback")
            latest_feedback = cursor.fetchone()[0]
            
            stats['latest_behavior'] = latest_behavior
            stats['latest_feedback'] = latest_feedback
            
            conn.close()
            return stats
            
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return None
    
    def clear_old_data(self, days_to_keep=30):
        """Clear data older than specified days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete old behavior data
            cursor.execute('''
                DELETE FROM customer_behavior 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days_to_keep))
            
            # Delete old feedback data
            cursor.execute('''
                DELETE FROM customer_feedback 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days_to_keep))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Cleared data older than {days_to_keep} days")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing old data: {e}")
            return False


def main():
    """Test the database manager"""
    db = DatabaseManager()
    
    # Test database stats
    stats = db.get_database_stats()
    if stats:
        print("\n📊 Database Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    # Test update history
    history = db.get_update_history()
    if history is not None and not history.empty:
        print(f"\n📈 Update History: {len(history)} records")
        print(history.head())


if __name__ == "__main__":
    main()
