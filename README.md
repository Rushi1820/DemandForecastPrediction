# DemandForecastPrediction

# Overview
The Demand Forecasting System aims to predict future sales for a company's products based on historical sales data. This system uses an LSTM (Long Short-Term Memory) neural network to forecast the quantity sold over the next 15 weeks for selected products, helping the company optimize its inventory and supply chain efficiency.

# Key Features:
Predicts demand for the next 1–15 weeks.
Utilizes LSTM for time-series forecasting.
Displays the results with visualizations including historical data plots, training/validation error plots, moving averages, and forecasted trends.

# Project Structure
The project consists of several functions that handle data preprocessing, building the LSTM model, forecasting sales, and visualizing the results.

# 1. Data Loading and Preprocessing:
The load_data() function reads the CSV files containing sales and customer data and combines them into a single DataFrame for further analysis. The merge() function is used to combine customer and product information.

# 2. Exploratory Data Analysis (EDA):
The get_top_products() function identifies the top 10 products by quantity sold and revenue generated. This allows the system to focus on forecasting for the top-selling products.

# 3. Building the LSTM Model:
The LSTM model is built using the build_lstm_model() function. This function creates a Sequential model with two LSTM layers, Dropout for regularization, and a Dense layer for output.

# 4. Training and Forecasting:
The run_forecast() function trains the model using the input data, performs time series forecasting for the selected stock code, and generates the forecast for the next 1–15 weeks. This function also tracks the training and validation loss to ensure the model’s performance is evaluated.

# 5. Visualizations:
Various plots are generated to provide insights:

# Historical Data Plot: Shows past sales trends.
Training/Validation Loss Plot: Displays the model's training and validation errors.
Moving Average Plot: Shows the 7-day moving average of sales.
Forecast Plot: Visualizes both historical data and future forecasted sales

# Streamlit live server:
   https://demandforecastprediction-rushi.streamlit.app/
   
