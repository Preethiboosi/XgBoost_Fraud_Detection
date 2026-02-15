#The realtime project
import joblib
import pandas as pd

def get_prediction(data_dict):
    # Load the saved model
    try:
        model = joblib.load('models/xgboost_model.pkl')
    except:
        return {"error": "Model file not found. Run train.py first."}

    # Convert input to DataFrame
    df = pd.DataFrame([data_dict])
    
    # Get Probability
    prob = model.predict_proba(df)[0][1]
    prediction = 1 if prob > 0.5 else 0
    
    return {
        "fraud_probability": f"{prob*100:.2f}%",
        "is_fraud": bool(prediction),
        "risk_level": "High" if prob > 0.8 else "Medium" if prob > 0.3 else "Low"
    }