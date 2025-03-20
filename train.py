from src.trainer import StockTrainer

trainer = StockTrainer(file_path="data/AAPL_stock_data.csv", epochs=20, lr=0.0005)
trainer.train()  # Start training
