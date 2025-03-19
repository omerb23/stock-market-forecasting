import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
import os
import time

class StockPredictor:
    def __init__(self, companies, target_company='AAPL', years=5):
        self.companies = companies
        self.target_company = target_company
        self.years = years
        # Sabit tarih aralığı kullan
        self.start_date = datetime(2020, 1, 1)
        self.end_date = datetime(2022, 12, 31)
        self.model = None
        self.X_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.best_model_path = 'best_model.h5'
        self.history = None
        
        # Tarihleri string formatına çevir
        self.start_date_str = self.start_date.strftime('%Y-%m-%d')
        self.end_date_str = self.end_date.strftime('%Y-%m-%d')
        print(f"Veri aralığı: {self.start_date_str} - {self.end_date_str}")

    def fetch_data(self, company):
        """Hisse senedi verilerini çeker"""
        try:
            print(f"{company} için veri işleniyor...")
            
            # Ticker'ı oluştur
            ticker = yf.Ticker(company)
            
            # Veriyi indir
            data = ticker.history(
                start=self.start_date_str,
                end=self.end_date_str,
                interval="1d",
                prepost=False,
                actions=False,
                auto_adjust=True
            )
            
            if not data.empty:
                print(f"{company} verisi başarıyla indirildi.")
                return data
            else:
                print(f"{company} için veri bulunamadı.")
                return None
                
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
            return None

    def fetch_and_engineer_features(self, company):
        """Veriyi çeker ve özellik mühendisliği yapar"""
        data = self.fetch_data(company)
        if data is None:
            return None
            
        try:
            # Günlük yüzde değişim
            data['Daily_Change'] = ((data['Close'] - data['Open']) / data['Open']) * 100

            # Hareketli Ortalamalar
            data['MA_10'] = data['Close'].rolling(window=10).mean()
            data['MA_50'] = data['Close'].rolling(window=50).mean()

            # Volatilite (10 günlük standart sapma)
            data['Volatility_10'] = data['Close'].rolling(window=10).std()

            # Momentum (10 günlük)
            data['Momentum_10'] = data['Close'] - data['Close'].shift(10)

            # Eksik verileri temizleme
            data.dropna(inplace=True)
            
            if len(data) < 50:  # Minimum veri kontrolü
                print(f"{company} için yeterli veri yok.")
                return None
                
            return data[['Daily_Change', 'MA_10', 'MA_50', 'Volatility_10', 'Momentum_10']]
            
        except Exception as e:
            print(f"Veri işleme hatası ({company}): {str(e)}")
            return None

    def prepare_data(self):
        """Veriyi model için hazırlar"""
        print("\nVeri hazırlama başlıyor...")
        
        # Hedef şirket verilerini al
        target_data = self.fetch_data(self.target_company)
        if target_data is None:
            raise ValueError(f"{self.target_company} için veri alınamadı!")
            
        # Hedef değişkeni hazırla
        target = target_data['Close'].values.reshape(-1, 1)
        target = self.y_scaler.fit_transform(target)
        
        # Diğer şirketlerin verilerini al
        features = []
        for company in self.companies:
            if company != self.target_company:
                print(f"\n{company} verisi işleniyor...")
                data = self.fetch_data(company)
                if data is not None:
                    # Teknik göstergeleri hesapla
                    data['Daily_Change'] = ((data['Close'] - data['Open']) / data['Open']) * 100
                    data['MA_10'] = data['Close'].rolling(window=10).mean()
                    data['MA_50'] = data['Close'].rolling(window=50).mean()
                    data['Volatility_10'] = data['Close'].rolling(window=10).std()
                    data['Momentum_10'] = data['Close'] - data['Close'].shift(10)
                    data.dropna(inplace=True)
                    
                    if len(data) >= 50:  # Minimum veri kontrolü
                        features.append(data[['Daily_Change', 'MA_10', 'MA_50', 'Volatility_10', 'Momentum_10']].values)
                        print(f"{company} verisi başarıyla eklendi.")
                    else:
                        print(f"{company} için yeterli veri yok.")
                else:
                    print(f"{company} verisi kullanılamıyor, bu şirket atlanıyor.")
        
        if not features:
            raise ValueError("Hiçbir şirket için veri alınamadı!")
            
        # Özellik matrisini oluştur
        X = np.column_stack(features)
        X = self.X_scaler.fit_transform(X)
        
        # Veriyi eğitim ve test setlerine ayır
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = target[:train_size], target[train_size:]
        
        # LSTM için veriyi yeniden şekillendir
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        print(f"Toplam veri noktası sayısı: {len(X)}")
        return X_train, X_test, y_train, y_test

    def create_sequences(self, X, y, timesteps=30):
        X_seq, y_seq = [], []
        for i in range(len(X) - timesteps):
            X_seq.append(X[i:i+timesteps])
            y_seq.append(y[i+timesteps])
        return np.array(X_seq), np.array(y_seq)

    def build_model(self, timesteps, n_features, gru_units_1=64, gru_units_2=32, dropout_rate=0.2):
        model = Sequential()
        model.add(GRU(gru_units_1, return_sequences=True, input_shape=(timesteps, n_features)))
        model.add(Dropout(dropout_rate))
        model.add(GRU(gru_units_2, return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def create_callbacks(self):
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                self.best_model_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        return callbacks

    def random_search(self, n_trials=10, timesteps=30):
        best_score = float('inf')
        best_params = None
        
        param_distributions = {
            'gru_units_1': randint(32, 128),
            'gru_units_2': randint(16, 64),
            'dropout_rate': uniform(0.1, 0.3),
            'batch_size': randint(32, 128),
            'learning_rate': uniform(0.0001, 0.01)
        }
        
        X, y = self.prepare_data()
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.values.reshape(-1, 1))
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, timesteps)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for trial in range(n_trials):
            print(f"\nTrial {trial + 1}/{n_trials}")
            
            # Rastgele parametreleri seç
            params = {k: v.rvs() for k, v in param_distributions.items()}
            print("Parametreler:", params)
            
            # K-fold cross validation
            fold_scores = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_seq)):
                X_train, X_val = X_seq[train_idx], X_seq[val_idx]
                y_train, y_val = y_seq[train_idx], y_seq[val_idx]
                
                model = self.build_model(
                    timesteps, 
                    X_train.shape[2],
                    gru_units_1=params['gru_units_1'],
                    gru_units_2=params['gru_units_2'],
                    dropout_rate=params['dropout_rate']
                )
                
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                    loss='mean_squared_error'
                )
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=params['batch_size'],
                    callbacks=self.create_callbacks(),
                    verbose=0
                )
                
                fold_scores.append(min(history.history['val_loss']))
            
            avg_score = np.mean(fold_scores)
            print(f"Ortalama Validation Loss: {avg_score:.4f}")
            
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
                print("Yeni en iyi parametreler bulundu!")
        
        return best_params, best_score

    def train(self, epochs=45, batch_size=64, timesteps=30, optimize=True):
        if optimize:
            print("Hiperparametre optimizasyonu başlıyor...")
            best_params, best_score = self.random_search(n_trials=10, timesteps=timesteps)
            print("\nEn iyi parametreler:", best_params)
            print("En iyi skor:", best_score)
            
            # En iyi parametrelerle modeli eğit
            X, y = self.prepare_data()
            X_scaled = self.X_scaler.fit_transform(X)
            y_scaled = self.y_scaler.fit_transform(y.values.reshape(-1, 1))
            X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, timesteps)
            
            X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
            
            self.model = self.build_model(
                timesteps, 
                X_train.shape[2],
                gru_units_1=best_params['gru_units_1'],
                gru_units_2=best_params['gru_units_2'],
                dropout_rate=best_params['dropout_rate']
            )
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
                loss='mean_squared_error'
            )
            
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=best_params['batch_size'],
                callbacks=self.create_callbacks(),
                verbose=1
            )
        else:
            # Orijinal eğitim kodu
            X, y = self.prepare_data()
            X_scaled = self.X_scaler.fit_transform(X)
            y_scaled = self.y_scaler.fit_transform(y.values.reshape(-1, 1))
            X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, timesteps)
            X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
            
            self.model = self.build_model(timesteps, X_train.shape[2])
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=self.create_callbacks(),
                verbose=1
            )
        
        return X_test, y_test

    def predict_future(self, n_steps=7, timesteps=30):
        X, _ = self.prepare_data()
        X_scaled = self.X_scaler.transform(X)
        last_data = X_scaled[-timesteps:]
        last_data_reshaped = last_data.reshape((1, timesteps, last_data.shape[1]))

        predictions = []
        current_input = last_data_reshaped.copy()

        for _ in range(n_steps):
            prediction = self.model.predict(current_input, verbose=0)
            prediction_rescaled = self.y_scaler.inverse_transform(prediction)
            predictions.append(prediction_rescaled[0, 0])

            new_feature = np.roll(current_input, shift=-1, axis=1)
            new_feature[:, -1, :] = prediction
            current_input = new_feature

        return predictions

    def evaluate_model(self, X_test, y_test):
        y_test_rescaled = self.y_scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_rescaled = self.y_scaler.inverse_transform(y_pred)

        metrics = {
            'MAE': mean_absolute_error(y_test_rescaled, y_pred_rescaled),
            'MSE': mean_squared_error(y_test_rescaled, y_pred_rescaled),
            'RMSE': np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled)),
            'R2': r2_score(y_test_rescaled, y_pred_rescaled),
            'MAPE': mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)
        }

        huber_loss_fn = tf.keras.losses.Huber()
        metrics['Huber Loss'] = huber_loss_fn(y_test_rescaled, y_pred_rescaled).numpy().mean()

        return metrics

    def plot_predictions(self, predictions):
        prediction_dates = [(datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
        
        plt.figure(figsize=(14, 7))
        plt.plot(prediction_dates, predictions, label='Gelecek Tahminleri', 
                color='green', linestyle='dashed', marker='o', alpha=0.7)
        plt.title(f'Gelecek 1 Hafta İçin Tahminler ({self.today} ve sonrası)', fontsize=16)
        plt.xlabel('Tarih', fontsize=12)
        plt.ylabel('Kapanış Fiyatı (USD)', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.show()

    def plot_training_history(self):
        if self.history is None:
            print("Henüz eğitim yapılmadı!")
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(self.history.history['loss'], label='Eğitim Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Eğitim Geçmişi')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

def fetch_stock_data(symbol):
    """
    Belirtilen hisse senedi için veri çeker
    """
    try:
        print(f"{symbol} için veri çekiliyor...")
        
        # Son 1 yıllık veri için tarih hesaplama
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Tarihleri string formatına çevirme
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        print(f"Tarih aralığı: {start_date_str} - {end_date_str}")
        
        # Ticker oluştur ve veriyi çek
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            start=start_date_str,
            end=end_date_str,
            interval="1d"
        )
        
        if not data.empty:
            print(f"{symbol} için veri başarıyla çekildi.")
            print(f"Toplam {len(data)} günlük veri alındı.")
            return data
        else:
            print(f"{symbol} için veri bulunamadı.")
            return None
            
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return None

def main():
    # Test edilecek hisse senedi
    symbol = "THYAO.IS"
    
    # Veriyi çek
    data = fetch_stock_data(symbol)
    
    # Sonuçları göster
    if data is not None:
        print("\nİlk 5 satır:")
        print(data.head())
        print("\nSon 5 satır:")
        print(data.tail())

if __name__ == "__main__":
    main() 