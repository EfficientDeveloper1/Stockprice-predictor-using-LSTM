from src.trainer import StockTrainer

trainer = StockTrainer(file_path="data/stock_prices.csv", epochs=100, lr=0.0005)
trainer.train()  # Start training
