import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import ta
import plotly.express as px
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Stock Price Predictor")
st.markdown("""
    This application predicts future stock prices using an LSTM model.
    Enter a stock ticker symbol and select the number of days to predict.
""")

# Sidebar for input
with st.sidebar:
    st.header("Input Parameters")
    ticker = st.text_input("Stock Ticker", value="NVDA", help="Enter a valid stock ticker symbol (e.g., NVDA, AAPL)")
    
    col1, col2 = st.columns(2)
    with col1:
        days = st.slider("Days to Predict", min_value=1, max_value=30, value=5, help="Number of days to predict into the future")
    with col2:
        historical_days = st.slider("Historical Days", min_value=30, max_value=365, value=60, help="Number of historical days to display")
    
    st.markdown("### Technical Indicators")
    show_rsi = st.checkbox("RSI", value=True)
    show_macd = st.checkbox("MACD", value=True)
    show_bollinger = st.checkbox("Bollinger Bands", value=True)
    
    predict_button = st.button("Predict")

# Main content
if predict_button:
    try:
        # Show loading spinner
        with st.spinner("Fetching predictions..."):
            # Get historical data for plotting
            end_date = datetime.now()
            start_date = end_date - timedelta(days=historical_days)
            stock = yf.Ticker(ticker)
            historical_data = stock.history(start=start_date, end=end_date)
            
            # Calculate technical indicators
            if show_rsi:
                historical_data['RSI'] = ta.momentum.rsi(historical_data['Close'], window=14)
            
            if show_macd:
                macd = ta.trend.MACD(historical_data['Close'])
                historical_data['MACD'] = macd.macd()
                historical_data['MACD_Signal'] = macd.macd_signal()
                historical_data['MACD_Hist'] = macd.macd_diff()
            
            if show_bollinger:
                bb = ta.volatility.BollingerBands(historical_data['Close'])
                historical_data['BB_High'] = bb.bollinger_hband()
                historical_data['BB_Low'] = bb.bollinger_lband()
                historical_data['BB_Mid'] = bb.bollinger_mavg()
            
            # Get predictions from API
            api_url = f"http://localhost:8000/api/v1/predict/{ticker}?days={days}"
            response = requests.get(api_url)
            
            if response.status_code == 200:
                data = response.json()
                predictions = data["data"]["predictions"]
                
                # Create prediction dates
                last_date = historical_data.index[-1]
                prediction_dates = [last_date + timedelta(days=i+1) for i in range(days)]
                
                # Create DataFrames for plotting
                historical_df = pd.DataFrame({
                    'Date': historical_data.index,
                    'Close': historical_data['Close']
                })
                
                prediction_df = pd.DataFrame({
                    'Date': prediction_dates,
                    'Close': predictions
                })
                
                # Create subplots
                num_subplots = 1 + sum([show_rsi, show_macd])
                fig = make_subplots(
                    rows=num_subplots, 
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(["Price"] + ["RSI"] if show_rsi else []) + (["MACD"] if show_macd else [])
                )
                
                # Add price data
                fig.add_trace(
                    go.Scatter(
                        x=historical_df['Date'],
                        y=historical_df['Close'],
                        name='Historical',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
                # Add predictions
                fig.add_trace(
                    go.Scatter(
                        x=prediction_df['Date'],
                        y=prediction_df['Close'],
                        name='Predictions',
                        line=dict(color='red', dash='dash')
                    ),
                    row=1, col=1
                )
                
                # Add Bollinger Bands if selected
                if show_bollinger:
                    fig.add_trace(
                        go.Scatter(
                            x=historical_data.index,
                            y=historical_data['BB_High'],
                            name='BB High',
                            line=dict(color='gray', dash='dot'),
                            opacity=0.5
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=historical_data.index,
                            y=historical_data['BB_Low'],
                            name='BB Low',
                            line=dict(color='gray', dash='dot'),
                            opacity=0.5,
                            fill='tonexty'
                        ),
                        row=1, col=1
                    )
                
                # Add RSI if selected
                if show_rsi:
                    fig.add_trace(
                        go.Scatter(
                            x=historical_data.index,
                            y=historical_data['RSI'],
                            name='RSI',
                            line=dict(color='purple')
                        ),
                        row=2 if show_rsi else 1, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2 if show_rsi else 1, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2 if show_rsi else 1, col=1)
                
                # Add MACD if selected
                if show_macd:
                    row = 3 if show_rsi and show_macd else 2 if show_rsi or show_macd else 1
                    fig.add_trace(
                        go.Scatter(
                            x=historical_data.index,
                            y=historical_data['MACD'],
                            name='MACD',
                            line=dict(color='blue')
                        ),
                        row=row, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=historical_data.index,
                            y=historical_data['MACD_Signal'],
                            name='Signal',
                            line=dict(color='orange')
                        ),
                        row=row, col=1
                    )
                    fig.add_trace(
                        go.Bar(
                            x=historical_data.index,
                            y=historical_data['MACD_Hist'],
                            name='Histogram',
                            marker_color='gray'
                        ),
                        row=row, col=1
                    )
                
                # Update layout
                fig.update_layout(
                    title=f"{ticker} Stock Analysis",
                    height=800,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Display prediction details
                st.markdown("### Prediction Details")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Historical Data")
                    st.dataframe(historical_df.tail(), use_container_width=True)
                
                with col2:
                    st.markdown("#### Predictions")
                    st.dataframe(prediction_df, use_container_width=True)
                
                # Display prediction metrics
                st.markdown("### Prediction Metrics")
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    st.metric(
                        "Current Price",
                        f"${historical_df['Close'].iloc[-1]:.2f}"
                    )
                
                with metrics_col2:
                    st.metric(
                        "Next Day Prediction",
                        f"${predictions[0]:.2f}",
                        f"{((predictions[0] - historical_df['Close'].iloc[-1]) / historical_df['Close'].iloc[-1] * 100):.2f}%"
                    )
                
                with metrics_col3:
                    st.metric(
                        f"{days}-Day Prediction",
                        f"${predictions[-1]:.2f}",
                        f"{((predictions[-1] - historical_df['Close'].iloc[-1]) / historical_df['Close'].iloc[-1] * 100):.2f}%"
                    )
                
                with metrics_col4:
                    if show_rsi:
                        current_rsi = historical_data['RSI'].iloc[-1]
                        rsi_color = "red" if current_rsi > 70 else "green" if current_rsi < 30 else "gray"
                        st.metric(
                            "Current RSI",
                            f"{current_rsi:.2f}",
                            delta_color=rsi_color
                        )
                
                # Add export functionality
                st.markdown("### Export Data")
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    if st.button("Export Historical Data"):
                        historical_df.to_csv(f"{ticker}_historical_data.csv")
                        st.success(f"Historical data exported to {ticker}_historical_data.csv")
                
                with export_col2:
                    if st.button("Export Predictions"):
                        prediction_df.to_csv(f"{ticker}_predictions.csv")
                        st.success(f"Predictions exported to {ticker}_predictions.csv")
                
            else:
                st.error(f"Error fetching predictions: {response.text}")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Stock Price Predictor using LSTM Model</p>
    </div>
""", unsafe_allow_html=True) 