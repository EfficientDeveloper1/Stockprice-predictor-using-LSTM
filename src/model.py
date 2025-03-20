import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=9, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    


class LSTMModelMutipleOutput(nn.Module):
    def __init__(self, input_size: int = 9, hidden_size: int = 64, num_layers: int = 2, output_size: int = 5):
        """
        LSTM Model for Stock Prediction
        input_size: Number of features in the dataset
        hidden_size: Number of LSTM neurons
        num_layers: Number of stacked LSTM layers
        output_size: Number of values we want to predict (Close, SMA, EMA, RSI, MACD)
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Predict next day's indicators
