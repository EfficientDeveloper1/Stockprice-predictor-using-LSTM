import torch
import ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Any
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from src.model import LSTMModel
from src.data_preprocessing import StockDataProcessor


class StockPredictor(StockDataProcessor):
    def __init__(
        self, model_path="models/lstm_model.pth", file_path=None, seq_length=60
    ):
        super().__init__(file_path, seq_length)
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

    def predict(self, raw_input_data):
        """Preprocesses input and makes stock price predictions."""
        input_data = self.transform(raw_input_data)
        input_tensor = input_data.to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor).cpu().numpy()

        return float(prediction[0][0])
    


@dataclass
class EvaluationMetrics:
    mse: float
    rmse: float
    mae: float = None
    r2: float = None

class StockPredictor2(StockDataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def load_trained_model(self):
        """Load a pre-trained model for prediction"""
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()
        return self.model
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        Args:
            input_data: numpy array of shape (seq_length, n_features)
        Returns:
            predicted_close_price: numpy array
        """
        if not hasattr(self, 'model'):
            self.load_trained_model()
            
        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        return prediction.cpu().numpy().flatten()
    
    def evaluate_model(self) -> Tuple[EvaluationMetrics, Dict[str, Any]]:
        """Evaluate model and return metrics with visualization data"""
        _, _, X_test, y_test = self.get_data()
        test_loader = DataLoader(
            TensorDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False
        )
        
        metrics = self.evaluate(test_loader)
        
        # Get full predictions for visualization
        predictions, actuals = self._get_full_predictions(test_loader)
        
        # Inverse transform the predictions and actuals
        predictions = self._inverse_transform_predictions(predictions)
        actuals = self._inverse_transform_predictions(actuals)
        
        # Calculate additional metrics
        from sklearn.metrics import mean_absolute_error, r2_score
        metrics.mae = mean_absolute_error(actuals, predictions)
        metrics.r2 = r2_score(actuals, predictions)
        
        visualization_data = {
            'predictions': predictions,
            'actuals': actuals,
            'dates': self._get_test_dates()  # Implement this method to get dates for test set
        }
        
        return metrics, visualization_data
    
    def _get_full_predictions(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get all predictions and actual values from data loader"""
        predictions, actuals = [], []
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch).cpu().numpy()
                predictions.extend(y_pred.flatten())
                actuals.extend(y_batch.cpu().numpy().flatten())
        return np.array(predictions), np.array(actuals)
    
    def _inverse_transform_predictions(self, values: np.ndarray) -> np.ndarray:
        """Inverse transform predictions to original price scale"""
        # Create dummy array with same shape as original features
        dummy = np.zeros((len(values), len(self.scaler.feature_names_in_)))
        dummy[:, 3] = values  # Close price is at index 3
        return self.scaler.inverse_transform(dummy)[:, 3]
    
    def _get_test_dates(self) -> np.ndarray:
        """Get dates corresponding to test set"""
        df = self.load_data()
        train_size = int(len(df) * 0.8)
        test_dates = df.index[train_size + self.seq_length:]
        return test_dates.values
    
    def plot_predictions(self, visualization_data: Dict[str, Any]):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(15, 6))
        plt.plot(visualization_data['dates'], visualization_data['actuals'], label='Actual Prices')
        plt.plot(visualization_data['dates'], visualization_data['predictions'], label='Predicted Prices')
        plt.title('Stock Price Prediction vs Actual')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()
    
    def prepare_api_input(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare input data for API prediction
        Args:
            raw_data: dictionary containing Open, High, Low, Close, Volume values
        Returns:
            processed_data: numpy array ready for model prediction
        """
        # Create DataFrame from input
        df = pd.DataFrame([raw_data])
        
        # Add technical indicators (same as in load_data)
        df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)
        df["EMA_10"] = ta.trend.ema_indicator(df["Close"], window=10)
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
        df["MACD"] = ta.trend.macd(df["Close"])
        
        # Scale the data
        scaled_data = self.scaler.transform(df)
        return scaled_data
    
    def api_predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        API-friendly prediction method
        Args:
            input_data: dictionary containing stock data features
        Returns:
            prediction_result: dictionary with prediction and metadata
        """
        try:
            # Prepare input
            processed_input = self.prepare_api_input(input_data)
            
            # Make prediction (assuming we have seq_length=1 for API predictions)
            prediction = self.predict(processed_input)
            
            # Inverse transform to get actual price
            prediction_price = self._inverse_transform_predictions(prediction)
            
            return {
                'success': True,
                'prediction': float(prediction_price[0]),
                'currency': 'USD',  # Can be parameterized
                'model_version': '1.0'  # Can be dynamic
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
