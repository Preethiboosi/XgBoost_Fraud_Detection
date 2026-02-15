import pandas as pd
import xgboost as xgb
import joblib
import os

def train_fraud_model():
    print("🚀 Starting Upgraded Training Pipeline...")
    
    data_path = 'data/fraud_data.csv'
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found!")
        return

    # UPGRADE: Load only necessary columns to save memory (Pro move)
    # Most tutorials load the whole thing; we only load what we need.
    cols_to_use = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'is_fraud']
    
    try:
        print("📂 Loading and filtering dataset...")
        df = pd.read_csv(data_path, usecols=cols_to_use)
    except ValueError:
        print("⚠️ Column mismatch! Checking for alternative names...")
        # Sometimes Kaggle uses merch_lat/merch_long instead of lat/long
        df = pd.read_csv(data_path) 
        # You can manually check columns with: print(df.columns)

    # Prepare features
    features = ['amt', 'lat', 'long', 'city_pop', 'unix_time'] 
    X = df[features].fillna(0) # Handle any missing values
    y = df['is_fraud']
    
    # THE UPGRADE: Calculate Scale Position Weight
    fraud_count = sum(y == 1)
    legal_count = sum(y == 0)
    ratio = legal_count / fraud_count
    
    print(f"📊 Data Stats: {legal_count} Legit, {fraud_count} Fraud. Ratio: {ratio:.2f}")

    # Initialize XGBoost with specific Fraud-Detection settings
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=ratio, # Focuses on catching fraud
        use_label_encoder=False,
        eval_metric='aucpr',    # Best metric for fraud detection
        tree_method='hist'      # Faster training for large datasets
    )

    print("🧠 Training the XGBoost Engine (this may take a minute)...")
    model.fit(X, y)

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save the brain
    joblib.dump(model, 'models/xgboost_model.pkl')
    print("✅ Success! Model saved in models/xgboost_model.pkl")

if __name__ == "__main__":
    train_fraud_model()
