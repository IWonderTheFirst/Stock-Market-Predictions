import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

import yfinance as yf
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM

import stock_models
from stock_models import lstm_model_1, lstm_model_2, lstm_model_3, conv_model, hybrid_model, lstm_model_4, lstm_model_5, bidirectional_lstm_model 

def get_prediction(stock):
    # Download stock data
    df = yf.download(stock, start='2014-01-01', end=datetime.now())

    # Filter to get only the 'Close' column
    data = df.filter(['Close'])
    dataset = data.values
    
    # Define training data length
    training_data_len = int(np.ceil(len(dataset) * 0.8))

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Split the data into training and testing sets
    train_data = scaled_data[0:training_data_len, :]

    # Prepare the training data
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the training data to numpy arrays and reshape
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Prepare the test data
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    # Convert the test data to numpy arrays and reshape
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    # Define input shape for the model
    INPUT_SHAPE = (x_train.shape[1], 1)
    
    # Load the LSTM model
    model = lstm_model_1(INPUT_SHAPE)
    model.fit(x_train, y_train, batch_size=1, epochs=10, validation_data=(x_test, y_test))

    # Get the model's predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print(f"RMSE for test data: {rmse}")

    # Get the last 60 days of data
    last_60_days = scaled_data[-60:]
    x_input = np.array(last_60_days)
    x_input = np.reshape(x_input, (1, x_input.shape[0], 1))

    # Get the predicted scaled price for the next day
    pred_scaled = model.predict(x_input)

    # Undo the scaling to get the actual price
    prediction = scaler.inverse_transform(pred_scaled)

    return prediction[0][0]

# Example usage:
print(get_prediction('AAPL'))
