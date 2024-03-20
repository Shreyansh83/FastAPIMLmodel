# Import necessary libraries
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd

# Create a FastAPI instance
app = FastAPI()

# Load the pre-trained model from a pickle file
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

# Define the root endpoint
@app.get("/")
def index():
    # Return a simple message
    return {"message": "Hello"}

# Define an endpoint that takes a name as a parameter
@app.get("/{name}")
def get_name(name: str):
    # Return a welcome message with the provided name
    return {"Welcome ": f'{name}'}

# Define an endpoint for predicting whether a bank note is fake or not
@app.post("/predict")
def predict_bank_note(data: BankNote):
    # Convert the data into a model-friendly format
    data = data.model_dump()
    
    # Extract the features from the data
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    
    # Use the model to make a prediction
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    
    # Interpret the prediction
    if prediction[0] > 0.5:
        prediction = "Fake note"
    else:
        prediction = "Its a Bank note"
    
    # Return the prediction
    return {
        "prediction": prediction
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)