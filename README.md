# 🍷 LSTM-finewine-streamlit application

Welcome to the Fine Wine Analysis and Forecasting application.
This app demonstrates various time series analysis techniques applied to fine wine data. 

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

**Navigate through the pages using the sidebar:**

1.  **EDA:** Explore the raw price trends, log prices, returns, time-series decomposition, and returns distributions.
2.  **HS VaR:** Calculate Historical Simulation Value at Risk for different fine wines and investment parameters.
3.  **ARIMA & SARIMA:** View predictions and evaluations using ARIMA and SARIMA models.
4.  **LSTM & DeepVaR:** Explore forecasts using Long Short-Term Memory networks and compare Historical VaR with DeepVaR estimates.

*Data Source: Top 37 Most Expensive Fine Wines (Monthly Prices: USD)*

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
