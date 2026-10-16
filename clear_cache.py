"""
Clear cached data files to force regeneration with UnifiedCustomerID
"""
import os
import shutil

data_files = [
    "data/customer_behavior.csv",
    "data/customer_feedback.csv",
    "data/churn_database.db"
]

print("🗑️ Clearing cached data files...")
for file in data_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"✅ Deleted: {file}")
    else:
        print(f"ℹ️ Not found: {file}")

print("\n🔄 Next step: Restart the Streamlit app to regenerate data with UnifiedCustomerID")
print("Run: streamlit run app.py")
