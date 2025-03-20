import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import numpy as np
from src.model import LSTMModel
from src.data_preprocessing import StockDataProcessor


class StockTrainer(StockDataProcessor):  
    def __init__(
        self,
        file_path,
        model_save_path="models/lstm_model.pth",
        batch_size=32,
        epochs=50,
        lr=0.001,
        seq_length=60,
    ):
        super().__init__(file_path, seq_length)  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LSTMModel().to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model_save_path = model_save_path

    def train(self):
        """Train the LSTM model using preprocessed stock data."""
        X_train, y_train, X_test, y_test = (
            self.get_data()
        )  # Directly use inherited method

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

            print(
                f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(train_loader):.4f}"
            )

        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

        # Evaluate model after training
        self.evaluate(test_loader)

    def evaluate(self, test_loader):
        """Evaluate the trained model on the test set using the test DataLoader."""
        self.model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for (
                X_batch,
                y_batch,
            ) in (
                test_loader
            ):  
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch).cpu().numpy()
                predictions.extend(y_pred.flatten())
                actuals.extend(y_batch.cpu().numpy().flatten())

        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)

        print("\nModel Evaluation:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

        return mse, rmse
