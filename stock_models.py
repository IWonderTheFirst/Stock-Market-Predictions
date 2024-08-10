import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional, GRU
from tensorflow.keras.regularizers import l2

def lstm_model_1(input_shape):  # Base model, rmse=6
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        LSTM(64, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def lstm_model_2(input_shape):  # Reduced overfitting with higher dropout, rmse=22
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.5),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def lstm_model_3(input_shape):  # Decrease LSTM units with dropout and activation, rmse=20
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.5),
        LSTM(32, return_sequences=False),
        Dropout(0.5),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def lstm_model_4(input_shape):  # Kernel regularizers, rmse=18
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        LSTM(32, return_sequences=False, kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(25, activation="relu", kernel_regularizer=l2(0.01)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def lstm_model_5(input_shape):  # Base model, rmse=15
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.02)),
        LSTM(64, return_sequences=False, kernel_regularizer=l2(0.02)),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model


def conv_model(input_shape):  # Increased LSTM units with dropout, rmse=14
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model


def hybrid_model(input_shape):  # Increased LSTM units with dropout, rmse=12
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        LSTM(50, return_sequences=True),
        Dropout(0.3),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def bidirectional_lstm_model(input_shape):  # RMSE ~ Expected to improve
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.3),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def gru_model(input_shape):  # RMSE ~ Expected to improve or match LSTM 1
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        GRU(64, return_sequences=False),
        Dropout(0.3),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def stacked_lstm_model(input_shape):  # RMSE ~ Expected to improve
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.4),
        LSTM(32, return_sequences=False),
        Dropout(0.5),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def tuned_hybrid_model(input_shape):  # RMSE ~ Expected to improve
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        LSTM(50, return_sequences=True, kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        LSTM(25, return_sequences=False, kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(50, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model
