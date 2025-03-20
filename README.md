# LSTM Stock Price Prediction API

## Overview
This project utilizes a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical data. It includes:
- **Data Processing**: Extracts key features and applies technical indicators.
- **Model Training**: Uses PyTorch to train an LSTM model.
- **Inference API**: A FastAPI-based service for real-time predictions.
- **Scalability**: Organized structure for easy expansion and deployment.

---

## Project Structure
```
stock_prediction/
│── data/
│   ├── AAPL.csv  # Stock price dataset
│   └── AMZN_stock_data.csv
│   └── NVDA_stock_data.csv
│── models/
│   ├── lstm_model.pth  # Trained model
│── src/
│   ├── data_preprocessing.py  # Data preparation
│   ├── model.py  # LSTM model definition
│   ├── trainer.py  # Training logic
│   ├── inference.py  # Model inference
│── api/
│   ├── app.py  # FastAPI application
│── train.py  # Train the model
│── predict.py  # Run inference
│── requirements.txt  # Dependencies
│── README.md  # Documentation
```

---

## Setup Instructions
### **1. Install Dependencies**
Ensure you have Python 3.8+ installed, then:
#### create python environment 
* Unix/Mac
```sh
python3 -m venv stock_prediction
```
* Windows
```sh
python -m venv stock_prediction
```
#### Activate the environment
* Unix/Mac
```sh
source stock_prediction/bin/activate
```
* Windows
```sh
stock_prediction\Scripts\activate
```
#### Install dependencies

```sh
pip install -r requirements.txt
```

### **2. Prepare Data**
Place your stock price dataset (CSV format) in the `data/` directory.
Ensure the CSV contains at least:
- `Open`, `High`, `Low`, `Close`, `Volume`

### **3. Train the Model**
To train the LSTM model on the stock dataset:
```sh
python train.py
```
The trained model will be saved in `models/lstm_stock_price_model.pth`.

### **4. Run the API**
Start the FastAPI server:
```sh
uvicorn api.app:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

---

## API Usage
### **1. Make Predictions**
Send a POST request to the API with 60 days of stock data:
```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"features": [[... 60 days of data ...]]}'
```
**Example Response:**
```json
{
  "predicted_price": 150.25
}
```

---

## Deployment
### **Deploy to Hugging Face Spaces**
1. Save your model to Hugging Face Model Hub.
2. Create a `Gradio` or `FastAPI` app for user interaction.

### **Deploy to AWS/GCP**
1. Use **AWS SageMaker** or **Google Cloud AI** for deployment.
2. Deploy FastAPI with **Docker & AWS Lambda**.

---

## Future Improvements
- Implement more stock market indicators.
- Deploy as a **SaaS service**.
- Add a **Gradio UI** for easy interaction.


