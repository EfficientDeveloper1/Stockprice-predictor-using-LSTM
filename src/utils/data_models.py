from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class StockDataPoint:
    """Represents a single stock data entry."""
    open: float
    high: float
    low: float
    close: float
    volume: float
    sma_10: float
    ema_10: float
    rsi: float
    macd: float

@dataclass
class ProcessedStockData:
    """Stores processed stock data ready for model input."""
    features: List[List[float]]  # Nested list representing the time-series window
    target: float  # The close price as the prediction target

@dataclass
class TrainingConfig:
    """Holds hyperparameters for model training."""
    epochs: int
    batch_size: int
    learning_rate: float
    seq_length: int

@dataclass
class EvaluationMetrics:
    """Stores model evaluation metrics."""
    mse: float
    rmse: float

@dataclass
class PredictionInput:
    """Represents input format for making stock predictions."""
    features: List[List[float]]  # 2D array of the last 60 days' stock features

@dataclass
class PredictionOutput:
    """Stores the model's prediction result."""
    predicted_price: float

