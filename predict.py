import requests
import argparse
import torch
import json
import numpy as np
from src.model import LSTMModel  # Import your model class

API_URL = "http://127.0.0.1:8000/predict"
MODEL_PATH = "models/lstm_model.pth"


def predict_from_api(features):
    """Send data to FastAPI endpoint for prediction."""
    response = requests.post(API_URL, json={"features": features})
    return response.json()


def predict_locally(features):
    """Load the model and predict without API."""
    model = LSTMModel(input_size=7, hidden_size=50, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Convert features to tensor
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor).item()

    return {"predicted_price": prediction}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run local prediction instead of API request",
    )
    args = parser.parse_args()

    # Example: Load 60 days of stock data
    stock_data = np.random.rand(60, 7).tolist()  # Replace with real preprocessed data

    if args.local:
        result = predict_locally(stock_data)
    else:
        result = predict_from_api(stock_data)

    print(json.dumps(result, indent=2))
