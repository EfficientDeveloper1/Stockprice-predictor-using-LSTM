import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error,
)

from src.model import LSTMModel
from src.data_preprocessing import StockDataProcessor, StockDataProcessorMutipleOutput
from src.utils.data_models import EvaluationMetrics, TrainingConfig
from src.config.logging_config import logger


class StockTrainer(StockDataProcessor):
    def __init__(
        self,
        training_config: TrainingConfig,
        file_path: str,
        model_save_path: str = "models/lstm_stock_price_model_apple.pth",
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
        super().__init__(file_path, training_config.seq_length)

        # Automatic Device Selection for Apple Silicon Compatibility
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Using device: {self.device}")
        self.model = LSTMModel().to(self.device)
        self.epochs = training_config.epochs
        self.lr = training_config.learning_rate
        self.batch_size = training_config.batch_size
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
        self.evaluate_with_plot(test_loader)
        # self.evaluate(test_loader)

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

    def evaluate_with_plot(
        self, test_loader: DataLoader
    ) -> Tuple[EvaluationMetrics, plt.Figure]:
        """
        Evaluate the trained model on the test set using the test DataLoader.
        Returns evaluation metrics and a matplotlib Figure object with plots.
        """
        self.model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch).cpu().numpy()
                predictions.extend(y_pred.flatten())
                actuals.extend(y_batch.cpu().numpy().flatten())

        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)

        # Calculate directional accuracy
        actual_changes = np.diff(actuals)
        predicted_changes = np.diff(predictions)
        directional_accuracy = (
            np.mean((actual_changes * predicted_changes) > 0) * 100
        )  # percentage

        logger.info("Model Evaluation:")
        logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"Directional Accuracy: {directional_accuracy:.2f}%")

        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Actual vs Predicted values
        ax1.plot(actuals, label="Actual", color="blue", alpha=0.7)
        ax1.plot(predictions, label="Predicted", color="red", alpha=0.5)
        ax1.set_title("Actual vs Predicted Values")
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Scatter plot of Actual vs Predicted with perfect fit line
        ax2.scatter(actuals, predictions, alpha=0.5)
        ax2.plot(
            [min(actuals), max(actuals)],
            [min(actuals), max(actuals)],
            "--",
            color="red",
            linewidth=2,
        )
        ax2.set_title("Actual vs Predicted Scatter Plot")
        ax2.set_xlabel("Actual Values")
        ax2.set_ylabel("Predicted Values")
        ax2.grid(True)

        plt.tight_layout()

        return (
            EvaluationMetrics(
                mse=mse,
                rmse=rmse,
                mae=mae,
                r2=r2,
                directional_accuracy=directional_accuracy,
            ),
            fig,
        )


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
