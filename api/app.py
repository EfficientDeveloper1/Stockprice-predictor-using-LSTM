from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import StockPredictor

app = FastAPI()
predictor = StockPredictor()


class StockRequest(BaseModel):
    features: list


@app.post("/predict")
def predict_stock(data: StockRequest):
    prediction = predictor.predict(data.features)
    return {"predicted_price": prediction}
