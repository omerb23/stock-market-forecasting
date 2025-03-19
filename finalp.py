import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Şirket listesi
companies = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'TSLA', 'NVDA', 'BRK-B', 'META', 'JNJ', 'V']

# Bugünkü tarihi al
today = datetime.today().strftime('%Y-%m-%d')

# Veri alma ve özellik mühendisliği
def fetch_and_engineer_features(company):
    print(f"{company} için veri işleniyor...")
    data = yf.download(company, start=(datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d'), end=today)
    
    data['Daily_Change'] = ((data['Close'] - data['Open']) / data['Open']) * 100
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['Volatility_10'] = data['Close'].rolling(window=10).std()
    data['Momentum_10'] = data['Close'] - data['Close'].shift(10)

    data.dropna(inplace=True)
    return data[['Daily_Change', 'MA_10', 'MA_50', 'Volatility_10', 'Momentum_10']]

# Tüm şirketlerin verilerini işleme
all_data = []
for company in companies:
    engineered_data = fetch_and_engineer_features(company)
    engineered_data.columns = [f"{company}_{col}" for col in engineered_data.columns]
    all_data.append(engineered_data)

merged_data = pd.concat(all_data, axis=1)
merged_data.dropna(inplace=True)

# Hedef değişken: AAPL'nin kapanış fiyatı
target_data = yf.download("AAPL", start=(datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d'), end=today)['Close']
target_data = target_data[target_data.index.isin(merged_data.index)]

# Bağımsız ve bağımlı değişkenlerin ayrılması
X = merged_data
y = target_data

# Normalizasyon
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

y_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

# Zaman serisi oluşturma
def create_sequences(X, y, timesteps=30):
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i+timesteps])
        y_seq.append(y[i+timesteps])
    return np.array(X_seq), np.array(y_seq)

timesteps = 30
X_seq, y_seq = create_sequences(X_scaled, y_scaled, timesteps)

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# GRU Modeli Tanımlama
def build_gru_model():
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(timesteps, X_train.shape[2])),
        GRU(32, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Optimize edilmiş eğitim süreci
prev_mse = float('inf')
tolerance = 0.001
max_iterations = 50
best_model = None
best_mse = float('inf')
model_path = "best_gru_model.h5"

for i in range(1, max_iterations + 1):
    print(f"Iterasyon {i}: Model Eğitiliyor...")
    model = build_gru_model()
    model.fit(X_train, y_train, epochs=45, batch_size=64, verbose=0)  

    y_pred = model.predict(X_test, verbose=0)
    y_pred_rescaled = y_scaler.inverse_transform(y_pred)
    y_test_rescaled = y_scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    print(f"Iterasyon {i} - MSE: {mse:.4f}")

    if mse < best_mse:
        best_mse = mse
        best_model = model
        model.save(model_path)
        print("Yeni en iyi model kaydedildi.")

    if abs(prev_mse - mse) < tolerance:
        print("MSE değişimi küçük, eğitim durduruluyor.")
        break
    
    prev_mse = mse

print(f"En iyi model {best_mse:.4f} MSE ile kaydedildi.")

# Kaydedilen modeli yükleme
best_model = load_model(model_path)

# Geleceğe yönelik tahmin yapma
def predict_future(model, data, n_steps, y_scaler):
    predictions = []
    current_input = data.copy()

    for _ in range(n_steps):
        prediction = model.predict(current_input, verbose=0)
        prediction_rescaled = y_scaler.inverse_transform(prediction)
        predictions.append(prediction_rescaled[0, 0])

        new_feature = np.roll(current_input, shift=-1, axis=1)
        new_feature[:, -1, :] = prediction
        current_input = new_feature

    return predictions

last_data = X_scaled[-timesteps:]
last_data_reshaped = last_data.reshape((1, timesteps, last_data.shape[1]))

future_predictions = predict_future(best_model, last_data_reshaped, n_steps=7, y_scaler=y_scaler)
prediction_dates = [(datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]

for date, price in zip(prediction_dates, future_predictions):
    print(f"{date} için tahmin edilen fiyat: {price:.2f} USD")

# Sonuçları görselleştirme
plt.figure(figsize=(14, 7))
plt.plot(prediction_dates, future_predictions, label='Gelecek Tahminleri', color='green', linestyle='dashed', marker='o', alpha=0.7)
plt.title(f'Gelecek 1 Hafta İçin Tahminler ({today} ve sonrası)', fontsize=16)
plt.xlabel('Tarih', fontsize=12)
plt.ylabel('Kapanış Fiyatı (USD)', fontsize=12)
plt.xticks(rotation=45)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()
