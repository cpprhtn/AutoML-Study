import autokeras as ak
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')

def Training_Model(data_path, max_trials=50, epochs=20):
    data_with_volatility = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')

    X = data_with_volatility.iloc[:-1].values
    y = data_with_volatility['Consumption'].shift(-1).dropna().values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    regressor = ak.StructuredDataRegressor(max_trials=max_trials, objective="val_mean_squared_error", overwrite=True, directory=".", seed=42, tuner="greedy")
    regressor.fit(X_scaled[:-60], y[:-60], epochs=epochs)

    X_test_scaled = scaler.transform(X[-60:])

    print(X_test_scaled)

    predictions_test = regressor.predict(X_test_scaled).flatten()

    test_result = pd.DataFrame({
        'Date': data_with_volatility.index[-60:],
        'Actual Consumption': data_with_volatility['Consumption'][-60:].values,
        'Predicted Consumption': predictions_test
    })

    best_model = regressor.export_model()

    best_model.save(data_path[:-4])

    plt.figure(figsize=(12, 6))
    plt.plot(data_with_volatility.index, data_with_volatility['Consumption'], label='Actual Consumption', color='blue')
    plt.plot(test_result['Date'], test_result['Predicted Consumption'], label='Predicted Consumption (Test)', color='red', linestyle='dashed')
    plt.title('Power Consumption Prediction Results (Test)')
    plt.xlabel('Date')
    plt.ylabel('Power Consumption (Kwh)')
    plt.legend()
    plt.grid(True)
    plt.savefig(data_path[:-4]+".png")

    return "{data_path} 데이터셋 학습이 완료되었습니다."
