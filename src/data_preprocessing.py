import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta  # Technical Analysis Library

from src.utils.data_models import StockDataPoint


class StockDataProcessor:
    def __init__(self, file_path=None, seq_length=60):
        self.file_path = file_path
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()

    def load_data(self):
        """Loads stock data and applies technical indicators."""
        df = pd.read_csv(self.file_path)
        df = df[["Open", "High", "Low", "Close", "Volume"]]

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
        return df

    def preprocess_data(self, df):
        """Fits the scaler and transforms the entire dataset (used during training)."""
        scaled_data = self.scaler.fit_transform(df)
        return torch.FloatTensor(scaled_data)

    def create_sequences(self, data):
        """Creates sequences for LSTM training."""
        sequences, labels = [], [] # X samples and y labels
        for i in range(len(data) - self.seq_length):
            sequences.append(data[i : i + self.seq_length])
            labels.append(data[i + self.seq_length][3])  # Predicting Close price
        # print(f"Sequences: {sequences} \n labels: {labels}" )
        return torch.stack(sequences), torch.FloatTensor(labels).view(-1, 1)

    def get_data(self):
        """Loads, processes, and splits the data into train/test sets."""
        df = self.load_data()
        processed_data = self.preprocess_data(df)
        X, y = self.create_sequences(processed_data)
        train_size = int(len(X) * 0.8)
        # print(f"Length of the Training set >> {train_size} samples")
        return X[:train_size], y[:train_size], X[train_size:], y[train_size:]

    def transform(self, new_data):
        """Preprocesses new data for inference (without re-fitting the scaler)."""
        df = pd.DataFrame(
            new_data,
            columns=[
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "SMA_10",
                "EMA_10",
                "RSI",
                "MACD",
                "BB_High",
                "BB_Low",
                "BB_Width",
                "BB_Middle",
                "ATR",
                "Stoch_k",
                "Stoch_D",
                "CCI",
                "OBV",
            ],
        )
        df.bfill(inplace=True)

        # Ensure the scaler is already fitted (use previously learned scaling parameters)
        scaled_data = self.scaler.transform(df)
        return torch.FloatTensor(scaled_data).unsqueeze(
            0
        )  # Add batch dimension for LSTM


class StockDataProcessorMutipleOutput:
    def __init__(self, file_path: str, seq_length: int = 60):
        self.file_path = file_path
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()
    
    def load_data(self) -> pd.DataFrame:
        """Loads and processes stock market data"""
        df = pd.read_csv(self.file_path)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
        
        # 📌 Add new stock indicators
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd(df['Close'])

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

        df.bfill(inplace=True)  # Fill missing values
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> torch.Tensor:
        """Normalizes and converts data to tensors"""
        scaled_data = self.scaler.fit_transform(df)
        return torch.FloatTensor(scaled_data)

    def create_sequences(self, data: torch.Tensor):
        """Creates sequences for LSTM training"""
        sequences, labels = [], []
        for i in range(len(data) - self.seq_length):
            sequences.append(data[i:i+self.seq_length])
            labels.append(data[i+self.seq_length])  # Predict all features
        return torch.stack(sequences), torch.FloatTensor(labels)

    def get_data(self):
        df = self.load_data()
        processed_data = self.preprocess_data(df)
        X, y = self.create_sequences(processed_data)
        train_size = int(len(X) * 0.8)
        return X[:train_size], y[:train_size], X[train_size:], y[train_size:]