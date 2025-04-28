# StockMarketPrediction

## Project Description
This project builds and compares various machine learning and deep learning models to predict future stock index prices. We forecast three major market indexes: **S&P 500**, **Dow Jones Industrial Average**, and **NASDAQ Composite**. By enriching the datasets with technical indicators and applying a variety of models, we assess which approaches work best for financial time series prediction.

---

## Packages/Technologies Needed
- Python 3
- yfinance (for data extraction)
- pandas, numpy, matplotlib
- scikit-learn
- TensorFlow/Keras
- statsmodels (for ARIMA)
- Facebook Prophet

### Install all dependencies:
```bash
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow statsmodels prophet
```

---

## Project Structure
```
/
├── ARIMA.ipynb            # ARIMA modeling for stock index prediction
├── baseline_models.ipynb  # Naïve and simple moving average baselines
├── FBProphet.ipynb        # Forecasting with Facebook Prophet
├── LSTM_NN.ipynb          # Deep learning LSTM model with technical indicators
├── MLP_NN.ipynb           # Neural network MLP model with technical indicators
```

---

## How to Run
1. Clone or download the repository.
2. Install the required Python packages.
3. Open and run the notebooks in the following suggested order:
   - `baseline_models.ipynb` (baseline metrics)
   - `ARIMA.ipynb` (traditional statistical model)
   - `FBProphet.ipynb` (Facebook Prophet modeling)
   - `MLP_NN.ipynb` (basic deep learning MLP)
   - `LSTM_NN.ipynb` (advanced deep learning LSTM)

Each notebook will automatically fetch the latest 10 years of stock data using `yfinance`.

---

## Notes
---

