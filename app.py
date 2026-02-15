from fastapi import FastAPI
import joblib
import pandas as pd

# 1. Initialize the App
app = FastAPI(title="Real-Time Fraud Detection Engine")

# 2. Load the trained brain
# We use a try-except block so the app doesn't crash if the file is missing
try:
    model = joblib.load('models/xgboost_model.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running. Go to /docs for the interface."}

# 3. The Prediction Route (This is what will show up in your browser)
@app.post("/predict")
def predict(amt: float, lat: float, long: float, city_pop: int, unix_time: int):
    if model is None:
        return {"error": "Model not found. Please train the model first."}
    
    # Create the data structure the model expects
    input_data = pd.DataFrame([[amt, lat, long, city_pop, unix_time]], 
                               columns=['amt', 'lat', 'long', 'city_pop', 'unix_time'])
    
    # Get the prediction
    probability = model.predict_proba(input_data)[0][1]
    is_fraud = bool(probability > 0.5)
    
    return {
        "fraud_probability": f"{probability*100:.2f}%",
        "is_fraud": is_fraud,
        "recommendation": "BLOCK TRANSACTION" if is_fraud else "ALLOW TRANSACTION"
    }