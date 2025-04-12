import os
from dotenv import load_dotenv
import pandas as pd
from kusa.client import SecureDatasetClient

load_dotenv()
model_path = "secure_model_sklearn.model"
TRAINING_FRAMEWORK = "sklearn"

load_dotenv()

def predict_rain_tomorrow(input_data):
    """Make prediction using the trained model"""
    PUBLIC_ID = os.getenv("PUBLIC_ID")
    SECRET_KEY = os.getenv("SECRET_KEY")
    
    client = SecureDatasetClient(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
    
    # 1. Initialize with same configuration as training
    initialization = client.initialize()
    
    # 3. Load the trained model
    client.load_model(model_path, training_framework=TRAINING_FRAMEWORK)
    
    # 4. Create DataFrame with same structure as training data
    
    # Step 4: Make prediction
    print("🔮 Making prediction...")
     
    prediction = client.predict(pd.DataFrame([input_data]))
    print(f"📊 Prediction: {'Yes' if prediction[0] == 1 else 'No'} (RainTomorrow)")
   

if __name__ == "__main__":
    # Example usage with data from your screenshot (first row)
    input_data = {
        'Date': '2009-08-01',
        'Location': 'Melbourneairport',
        'MinTemp': 13.4,
        'MaxTemp': 22.9,
        'Rainfall': 0.6,
        'Evaporation': 4.8,
        'Sunshine': 8.4,
        'WindGustDir': 'W',
        'WindGustSpeed': 44,
        'WindDir9am': 'W',
        'WindDir3pm': 'WWW',
        'WindSpeed9am': 20,
        'WindSpeed3pm': 24,
        'Humidity9am': 71,
        'Humidity3pm': 22,
        'Pressure9am': 1007.7,
        'Pressure3pm': 1007.1,
        'Cloud9am': 5,
        'Cloud3pm': 16.9,
        'Temp9am': 21.8,
        'Temp3pm': 20.1,
        'RainToday': 'No',
        'RainTomorrow': None  # This is what we're predicting
    }
    
    predict_rain_tomorrow(input_data)
