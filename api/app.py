import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import StockPredictor
from src.trainer import StockTrainer
from src.utils.data_models import TrainingConfig

app = FastAPI(
    title="Stock Price Prediction API",
    description="An AI-powered API to predict future stock prices using LSTM model.",
    version="1.0.0"
)
predictor = StockPredictor()

trainer = StockTrainer(
     training_config=TrainingConfig(epochs=10, batch_size=32, learning_rate=0.001, seq_length=60),
    file_path="data/AAPL_stock_data.csv"
)

class StockRequest(BaseModel):
    features: list

@app.get("/heathcheck")
def helthcheck():
    """Check if the API server is alive"""
    return {"status": "API is running perfectly"}

@app.post("/predict")
def predict_stock(data: StockRequest):
    prediction = predictor.predict(data.features)
    return {"predicted_price": prediction}

@app.post("/train")
def retrain_model():
    """Retrain the LSTM model with updated stock data."""
    trainer.train()
    return {"status": "Model retrained and saved successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host = "0.0.0.0",
        port = 8001,
        reload= True
    )