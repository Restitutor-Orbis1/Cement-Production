from fastapi import FastAPI, HTTPException
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = FastAPI()

# Load the trained model using pickle
with open('random_forest_model.pkl', 'rb') as model_file:
    loaded_rf_model = pickle.load(model_file)

scaler = MinMaxScaler()  # Load the MinMaxScaler used during training

def preprocess_input(input_data):
    # Convert input data to the format expected by the model
    # Apply the same MinMaxScaler used during training
    # Here, we assume input_data is a dictionary with keys P1 to P11
    features = [input_data[f'P{i}'] for i in range(1, 12)]
    scaled_features = scaler.transform([features])  # Scale the features
    return scaled_features

@app.post("/predict/")
def predict(data: dict):
    try:
        # Preprocess input data
        processed_data = preprocess_input(data)
        
        # Make predictions
        prediction = loaded_rf_model.predict(processed_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
#