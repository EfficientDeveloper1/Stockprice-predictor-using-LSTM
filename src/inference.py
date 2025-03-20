import torch
import numpy as np
from src.model import LSTMModel
from src.data_preprocessing import StockDataProcessor


class StockPredictor(StockDataProcessor):
    def __init__(
        self, model_path="models/lstm_model.pth", file_path=None, seq_length=60
    ):
        super().__init__(file_path, seq_length)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
