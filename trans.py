import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# Function to download stock data
def download_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, interval='5m')
    return df

# Function to prepare the dataset for training
def prepare_data(df, sequence_length=20):
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    X = []
    y = []
    
    for i in range(sequence_length, len(data)):
        X.append(data_scaled[i-sequence_length:i, 0])
        y.append(1 if data[i, 0] > data[i-1, 0] else 0)  # 1: price up, 0: price down
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshaping data to fit the Transformer model input shape
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Define the Transformer model
def build_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=64)(inputs, inputs)
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Main execution
if __name__ == "__main__":
    ticker = 'AAPL'  # Example: Apple Inc.
    start_date = '2022-01-01'
    end_date = '2022-12-31'

    # Download the data
    data = download_data(ticker, start_date, end_date)

    # Prepare the data
    X, y, scaler = prepare_data(data)

    # Build the model
    model = build_transformer_model(X.shape[1:])
    
    # Train the model
    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X, y)
    print(f"Model Loss: {loss}, Model Accuracy: {accuracy}")
