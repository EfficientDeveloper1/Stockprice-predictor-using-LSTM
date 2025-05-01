import torch
import ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from src.model import LSTMModel
from src.data_preprocessing import StockDataProcessor
import yfinance as yf
from datetime import datetime, timedelta


class StockPredictor(StockDataProcessor):
    def __init__(
        self, 
        model_path: str = "models/lstm_nvda_stock.pth", 
        file_path: Optional[str] = None, 
        seq_length: int = 60,
        ticker: str = "NVDA"
    ):
        super().__init__(file_path, seq_length)
        self.ticker = ticker
        
        # Automatic Device Selection for Apple Silicon Compatibility
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")

        # Load LSTM Model
        self.model = LSTMModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Fit the scaler with initial data
        self._fit_scaler()

    def _fit_scaler(self):
        """Fit the scaler with historical data."""
        # Get historical data for fitting
        df = self.get_latest_stock_data(days=120)  # Get more data for better scaling
        
        # Add technical indicators
        df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)
        df["EMA_10"] = ta.trend.ema_indicator(df["Close"], window=10)
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
        df["MACD"] = ta.trend.macd(df["Close"])
        
        bb_indicator = ta.volatility.BollingerBands(close=df["Close"], window=20)
        df["BB_High"] = bb_indicator.bollinger_hband()
        df["BB_Low"] = bb_indicator.bollinger_lband()
        df["BB_Middle"] = bb_indicator.bollinger_mavg()
        df["BB_Width"] = bb_indicator.bollinger_wband()
        
        df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
        df["Stoch_K"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"])
        df["Stoch_D"] = ta.momentum.stoch_signal(df["High"], df["Low"], df["Close"])
        df["CCI"] = ta.trend.cci(df["High"], df["Low"], df["Close"])
        df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
        
        df.bfill(inplace=True)
        
        # Select required columns
        input_data = df[["Open", "High", "Low", "Close", "Volume", 
                        "SMA_10", "EMA_10", "RSI", "MACD", 
                        "BB_High", "BB_Low", "BB_Middle", "BB_Width",
                        "ATR", "Stoch_K", "Stoch_D", "CCI", "OBV"]]
        
        # Fit the scaler
        self.scaler.fit(input_data)

    def get_latest_stock_data(self, days: int = 60) -> pd.DataFrame:
        """
        Fetch the latest stock data using yfinance.
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame containing the stock data
        """
        end_date = datetime.now()
        # Fetch more data than needed to ensure we have enough for the sequence length
        start_date = end_date - timedelta(days=max(days * 2, 120))  # Fetch at least 120 days
        
        stock = yf.Ticker(self.ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")
            
        # Ensure we have enough data points
        if len(df) < self.seq_length:
            raise ValueError(f"Not enough historical data available for {self.ticker}. Need at least {self.seq_length} days, but only got {len(df)} days.")
            
        return df

    def prepare_input_data(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Prepare input data for prediction by adding technical indicators and scaling.
        
        Args:
            df: DataFrame containing stock data
            
        Returns:
            Tensor ready for model input
        """
        # Add technical indicators
        df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)
        df["EMA_10"] = ta.trend.ema_indicator(df["Close"], window=10)
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
        df["MACD"] = ta.trend.macd(df["Close"])
        
        bb_indicator = ta.volatility.BollingerBands(close=df["Close"], window=20)
        df["BB_High"] = bb_indicator.bollinger_hband()
        df["BB_Low"] = bb_indicator.bollinger_lband()
        df["BB_Middle"] = bb_indicator.bollinger_mavg()
        df["BB_Width"] = bb_indicator.bollinger_wband()
        
        df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
        df["Stoch_K"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"])
        df["Stoch_D"] = ta.momentum.stoch_signal(df["High"], df["Low"], df["Close"])
        df["CCI"] = ta.trend.cci(df["High"], df["Low"], df["Close"])
        df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
        
        df.bfill(inplace=True)
        
        # Select required columns
        input_data = df[["Open", "High", "Low", "Close", "Volume", 
                        "SMA_10", "EMA_10", "RSI", "MACD", 
                        "BB_High", "BB_Low", "BB_Middle", "BB_Width",
                        "ATR", "Stoch_K", "Stoch_D", "CCI", "OBV"]]
        
        # Scale the data
        scaled_data = self.scaler.transform(input_data)
        return torch.FloatTensor(scaled_data).unsqueeze(0)

    def predict(self, input_data: Optional[pd.DataFrame] = None) -> float:
        """
        Make a single prediction for the next day's closing price.
        
        Args:
            input_data: Optional DataFrame containing stock data. If None, latest data will be fetched.
            
        Returns:
            Predicted closing price
        """
        if input_data is None:
            input_data = self.get_latest_stock_data()
            
        if len(input_data) < self.seq_length:
            raise ValueError(f"Input data must have at least {self.seq_length} days of data")
            
        input_tensor = self.prepare_input_data(input_data)
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_tensor).cpu().numpy()
            
        # Create a dummy array with the same shape as the input data
        # but with zeros except for the Close price column
        dummy_data = np.zeros((1, input_tensor.shape[2]))
        dummy_data[0, 3] = prediction[0][0]  # Close price is at index 3
        
        # Inverse transform the prediction
        unscaled_prediction = self.scaler.inverse_transform(dummy_data)
        return float(unscaled_prediction[0, 3])  # Return the unscaled Close price

    def predict_next_n_days(self, n_days: int = 5) -> List[float]:
        """
        Make predictions for the next n days.
        
        Args:
            n_days: Number of days to predict
            
        Returns:
            List of predicted closing prices
        """
        predictions = []
        current_data = self.get_latest_stock_data()
        
        for _ in range(n_days):
            prediction = self.predict(current_data)
            predictions.append(prediction)
            
            # Update the current data with the prediction
            new_row = current_data.iloc[-1].copy()
            new_row["Close"] = prediction
            new_row["Open"] = prediction
            new_row["High"] = prediction
            new_row["Low"] = prediction
            current_data = pd.concat([current_data.iloc[1:], pd.DataFrame([new_row])])
            
        return predictions

    def get_prediction_with_confidence(self) -> Dict[str, Any]:
        """
        Get prediction with confidence interval.
        
        Returns:
            Dictionary containing prediction and confidence interval
        """
        prediction = self.predict()
        
        # Calculate confidence interval (this is a simple example)
        # In a real scenario, you might want to use more sophisticated methods
        confidence_interval = {
            "lower": prediction * 0.95,  # 5% below prediction
            "upper": prediction * 1.05   # 5% above prediction
        }
        
        return {
            "prediction": prediction,
            "confidence_interval": confidence_interval,
            "timestamp": datetime.now().isoformat()
        }
    


