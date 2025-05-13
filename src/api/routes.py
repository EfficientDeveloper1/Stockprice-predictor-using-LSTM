from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from ..inference import StockPredictor

router = APIRouter()
class PredictRequest(BaseModel):
    ticker: str
    days: Optional[int] = Query(1, ge=1, le=100, description="Number of days to predict")
@router.get("/predict/{ticker}")
async def predict_stock(
    ticker: str,
    days: Optional[int] = Query(1, ge=1, le=100, description="Number of days to predict")
):
    """
    Get stock price prediction for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., NVDA, AAPL)
        days: Number of days to predict (1-30)
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        predictor = StockPredictor(ticker=ticker)
        
        if days == 1:
            result = predictor.get_prediction_with_confidence()
        else:
            predictions = predictor.predict_next_n_days(n_days=days)
            result = {
                "predictions": predictions,
                "timestamp": datetime.now().isoformat()
            }
            
        return {
            "ticker": ticker,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict")
async def predict_stock(request: PredictRequest):
    """
    Get stock price prediction for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., NVDA, AAPL)
        days: Number of days to predict (1-30)
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        predictor = StockPredictor(ticker=request.ticker)
        
        if request.days == 1:
            result = predictor.get_prediction_with_confidence()
        else:
            predictions = predictor.predict_next_n_days(n_days=request.days)
            result = {
                "predictions": predictions,
                "timestamp": datetime.now().isoformat()
            }
            
        return {
            "ticker": request.ticker,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    } 