from fastapi import FastAPI
import pickle
import pandas as pd

app=FastAPI()

model = pickle.load(open("housing_model.pkl", "rb"))
@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/predict")
def predict(data: dict):
    # Extract input values
    x = float(data.get("x", 0.0))
    dataset = data.get("dataset", "I")
    
    # Map dataset string to dummy columns expected by the model
    features = {
        "x": x,
        "dataset_I": dataset == "I",
        "dataset_II": dataset == "II",
        "dataset_III": dataset == "III",
        "dataset_IV": dataset == "IV"
    }
    
    # Create DataFrame with exact column names and order
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]

    # Status thresholds adjusted for the Anscombe's quartet values (approx 3.0 to 13.0)
    if prediction < 6.0:
        status = "low"
    elif prediction < 10.0:
        status = "medium"
    else:
        status = "high"

    return {
        "prediction": float(prediction),
        "status": status
    }

