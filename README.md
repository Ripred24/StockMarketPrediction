# StockMarketPrediction
This project builds and compares various machine learning and deep learning models to predict future stock index prices. We forecast three major market indexes: S&P 500, Dow Jones Industrial Average, and NASDAQ Composite. By enriching the datasets with technical indicators and applying a variety of models, we assess which approaches work best for financial time series prediction.

Packages/Tech Needed:
Python 3
yfinance (for data extraction)
pandas, numpy, matplotlib
scikit-learn
TensorFlow/Keras
statsmodels (for ARIMA)
Facebook Prophet

Code:
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow statsmodels prophet

Structure:
ARIMA.ipynb (ARIMA modeling for stock index prediction)
baseline_models.ipynb (Naïve and simple moving average baselines)
FBProphet.ipynb (Forecasting with Facebook Prophet)
LSTM_NN.ipynb (Deep learning LSTM model with technical indicators)
MLP_NN.ipynb (Neural network MLP model with technical indicators)


