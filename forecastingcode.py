import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st

# Function to create LSTM sequences
def create_lstm_sequences(data, time_steps=3):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Forecasting function for LSTM
def forecast_lstm(model, data, time_steps=3, forecast_length=15):
    predictions = []
    current_input = data[-time_steps:]  # Use the last `time_steps` data points to start the forecast
    for _ in range(forecast_length):
        current_input = current_input.reshape((1, time_steps, 1))  # Reshape for LSTM input
        next_pred = model.predict(current_input)
        predictions.append(next_pred[0, 0])
        current_input = np.append(current_input[:, 1:, :], next_pred.reshape((1, 1, 1)), axis=1)
    return np.array(predictions)

# Building the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Load and preprocess the data
def load_data():
    df1 = pd.read_csv("C:/Users/grvn1/Downloads/Usecase_DemadForecasting1/Usecase_DemadForecasting1/Transactional_data_retail_01.csv")
    df2 = pd.read_csv("C:/Users/grvn1/Downloads/Usecase_DemadForecasting1/Usecase_DemadForecasting1/Transactional_data_retail_02.csv")
    customer_df = pd.read_csv("C:/Users/grvn1/Downloads/Usecase_DemadForecasting1/Usecase_DemadForecasting1/CustomerDemographics.csv")
    product_df = pd.read_csv("C:/Users/grvn1/Downloads/Usecase_DemadForecasting1/Usecase_DemadForecasting1/ProductInfo.csv")

    # Combine transactional data
    df = pd.concat([df1, df2])
    
    # Convert 'InvoiceDate' to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    # Merge with customer and product info
    df = df.merge(customer_df, on='Customer ID', how='left')
    df = df.merge(product_df, on='StockCode', how='left')

    return df

# Perform EDA and identify top products
def get_top_products(df):
    # Calculate total quantity sold and revenue
    df['Revenue'] = df['Quantity'] * df['Price']
    
    top_10_stock_codes = df.groupby('StockCode')['Quantity'].sum().nlargest(10).index.tolist()
    top_10_revenue_products = df.groupby('StockCode')['Revenue'].sum().nlargest(10).index.tolist()
    
    return top_10_stock_codes, top_10_revenue_products

# Main forecasting function
def run_forecast(stock_code, forecast_length):
    # Load data
    df = load_data()

    # Filter data for the selected stock code
    stock_data = df[df['StockCode'] == stock_code]

    # Group by date and sum quantities sold
    daily_sales = stock_data.groupby('InvoiceDate').agg({'Quantity': 'sum'}).reset_index()

    # Check if we have enough data
    if daily_sales.empty:
        st.error(f"No data found for Stock Code: {stock_code}. Please try a different code.")
        return None

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    historical_data_scaled = scaler.fit_transform(daily_sales['Quantity'].values.reshape(-1, 1))

    # Prepare the sequences for LSTM
    X, y = create_lstm_sequences(historical_data_scaled, time_steps=3)

    # Reshape the data for LSTM input
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Build the model
    model = build_lstm_model(input_shape=(X.shape[1], 1))

    # Train the model with validation split to track training and validation error rates
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Perform forecasting
    forecast_scaled = forecast_lstm(model, historical_data_scaled, time_steps=3, forecast_length=forecast_length)

    # Inverse transform to get the original scale
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1))

    # Prepare results for display
    st.subheader(f"Forecasted values for the next {forecast_length} weeks for Stock Code {stock_code}:")
    forecast_df = pd.DataFrame(forecast, columns=['Forecasted Quantity'])
    st.write(forecast_df)

    # Visualizations

    # 1. Plot Historical Data
    st.subheader("Historical Data of Sales Quantity")
    plt.figure(figsize=(10, 6))
    plt.plot(daily_sales['InvoiceDate'], daily_sales['Quantity'], label="Historical Data", color='blue')
    plt.title(f"Historical Sales Data for Stock Code {stock_code}")
    plt.xlabel("Date")
    plt.ylabel("Quantity Sold")
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)

    # 2. Plot Training and Testing Error
    st.subheader("Training and Validation Loss Over Epochs")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title("Training and Validation Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # 3. Moving Average to Smooth the Trend
    st.subheader("Moving Average of Sales Quantity (7-day window)")
    daily_sales['7-day MA'] = daily_sales['Quantity'].rolling(window=7).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(daily_sales['InvoiceDate'], daily_sales['7-day MA'], label="7-day Moving Average", color='green')
    plt.title(f"7-Day Moving Average for Stock Code {stock_code}")
    plt.xlabel("Date")
    plt.ylabel("Smoothed Quantity Sold")
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)
    

    # 5. Forecast Plot with Future Dates
    st.subheader("Forecasted Data vs Historical Data")
    plt.figure(figsize=(10, 6))
    plt.plot(daily_sales['InvoiceDate'], daily_sales['Quantity'], label="Historical Data", color='blue')
    future_dates = pd.date_range(start=daily_sales['InvoiceDate'].iloc[-1] + pd.Timedelta(weeks=1), periods=forecast_length, freq='W')
    plt.plot(future_dates, forecast, label="Forecasted Data", color='orange')
    plt.title(f"Forecast for Stock Code: {stock_code} for Next {forecast_length} Weeks")
    plt.xlabel("Date")
    plt.ylabel("Quantity Sold")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


# Streamlit App
def main():
    st.title("Demand Forecasting System")

    df = load_data()  # Load data to display top products
    top_stock_codes, _ = get_top_products(df)

    stock_code = st.selectbox("Select Stock Code", top_stock_codes)
    forecast_length = st.slider("Select Forecast Length (Weeks)", min_value=1, max_value=15, value=5)

    if st.button("Run Forecast"):
        run_forecast(stock_code, forecast_length)

if __name__ == "__main__":
    main()
