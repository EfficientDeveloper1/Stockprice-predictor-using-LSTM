from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class StockDataPoint:
    """Represents a single stock data entry."""
    date: str  # Stock trading date
    open: float
    high: float
    low: float
    close: float
    volume: float
    ticker: str  # Stock symbol (e.g., AAPL, TSLA)

@dataclass
class ProcessedStockData:
    """Stores processed stock data ready for model input."""
    features: List[List[float]]  # Nested list representing the time-series window
    target: float  # The close price as the prediction target

@dataclass
class TrainingConfig:
    """Holds hyperparameters for model training."""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    seq_length: int = 60


@dataclass
class EvaluationMetrics:
    """Stores model evaluation metrics."""
    mse: float
    rmse: float
    mae: float = None
    r2: float = None
    directional_accuracy: float = None

@dataclass
class PredictionInput:
    """Represents input format for making stock predictions."""
    features: List[List[float]]  # 2D array of the last 60 days' stock features

@dataclass
class PredictionOutput:
    """Stores the model's prediction result."""
    predicted_price: float

