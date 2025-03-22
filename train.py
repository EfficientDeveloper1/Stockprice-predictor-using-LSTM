from src.trainer import StockTrainer

trainer = StockTrainer(file_path="data/NVDA_stock_data.csv", epochs=20, lr=0.0001)
trainer.train()  # Start training
