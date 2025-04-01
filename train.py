from src.trainer import StockTrainer
from src.utils.data_models import TrainingConfig

# Set up training configuration
training_config = TrainingConfig(
    epochs=10
)

trainer = StockTrainer(
    training_config=training_config,
    model_save_path="models/lstm_tsla_stock.pth",
    file_path="data/NVDA_stock_data.csv")

trainer.train()  # Start training
