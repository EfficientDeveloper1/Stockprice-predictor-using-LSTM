import torch
import numpy as np
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
