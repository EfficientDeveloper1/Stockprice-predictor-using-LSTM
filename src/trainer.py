import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import numpy as np
from src.model import LSTMModel
from src.data_preprocessing import StockDataProcessor, StockDataProcessorMutipleOutput
from src.utils.data_models import EvaluationMetrics
from src.config.logging_config import logger


class StockTrainer(StockDataProcessor):
    def __init__(
        self,
        file_path: str,
        model_save_path: str = "models/lstm_stock_price_model_NVDA.pth",
        batch_size: int = 32,
        epochs: int = 50,
        lr: float = 0.001,
        seq_length: int = 60,
    ):
        """
        Initialize the StockTrainer class with the given parameters.

        Parameters:
        - file_path (str): The path to the CSV file containing stock data.
        - model_save_path (str): The path where the trained LSTM model will be saved. Default is "models/lstm_model.pth".
        - batch_size (int): The batch size for training and testing. Default is 32.
        - epochs (int): The number of training epochs. Default is 50.
        - lr (float): The learning rate for the optimizer. Default is 0.001.
        - seq_length (int): The length of the input sequence for the LSTM model. Default is 60.

        Returns:
        - None
        """
        super().__init__(file_path, seq_length)
        
        # Automatic Device Selection for Apple Silicon Compatibility
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Using device: {self.device}")
        self.model = LSTMModel().to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model_save_path = model_save_path

    def train(self):
        """Train the LSTM model using preprocessed stock data."""
        X_train, y_train, X_test, y_test = self.get_data()

        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.info(
                f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(train_loader):.4f}"
            )

        torch.save(self.model.state_dict(), self.model_save_path)
        logger.info(f"Model saved to {self.model_save_path}")

        # Evaluate model after training
        self.evaluate(test_loader)

    def evaluate(self, test_loader: DataLoader) -> EvaluationMetrics:
        """Evaluate the trained model on the test set using the test DataLoader."""
        self.model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for (
                X_batch,
                y_batch,
            ) in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch).cpu().numpy()
                predictions.extend(y_pred.flatten())
                actuals.extend(y_batch.cpu().numpy().flatten())

        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)

        logger.info("Model Evaluation:")
        logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

        return EvaluationMetrics(mse=mse, rmse=rmse)


class StockTrainerMutipleOutput(StockDataProcessorMutipleOutput):
    def __init__(
        self,
        file_path: str,
        model_save_path: str = "models/lstm_model.pth",
        batch_size: int = 32,
        epochs: int = 50,
        lr: float = 0.001,
        seq_length: int = 60,
    ):
        super().__init__(file_path, seq_length)
        # Automatic Device Selection for Apple Silicon Compatibility
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Using device: {self.device}")
        self.model = LSTMModel().to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model_save_path = model_save_path

    def train(self):
        X_train, y_train, X_test, y_test = self.get_data()
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)  # Predicting multiple values
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(
                f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(train_loader):.4f}"
            )

        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")
