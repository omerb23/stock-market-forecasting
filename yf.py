import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, make_scorer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List, Tuple, Dict, Optional
import gc

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Şirket listesi
companies = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'TSLA', 'NVDA', 'BRK-B', 'META', 'JNJ', 'V',  
            'RHM.DE', 'PFE','AMD', 'INTC', 'QCOM', 'AVGO', 'TSM','JPM', 'BAC', 'WFC', 'GS', 'C','XOM',
            'CVX', 'BP', 'SHEL', 'KO', 'PEP', 'MCD', 'SBUX', 'NKE','LMT', 'BA', 'RTX', 'GD', 'NOC']

# Bugünkü tarihi al
today = datetime.today().strftime('%Y-%m-%d')

def fetch_company_data(company: str) -> Optional[pd.DataFrame]:
    """Paralel veri indirme için yardımcı fonksiyon"""
    try:
        data = yf.download(company, start=(datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d'), end=today)
        if data.empty:
            logger.warning(f"{company} için veri bulunamadı")
            return None
            
        data['Daily_Change'] = ((data['Close'] - data['Open']) / data['Open']) * 100
        data['MA_10'] = data['Close'].rolling(window=10).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['Volatility_10'] = data['Close'].rolling(window=10).std()
        data['Momentum_10'] = data['Close'] - data['Close'].shift(10)
        data.dropna(inplace=True)
        
        return data[['Daily_Change', 'MA_10', 'MA_50', 'Volatility_10', 'Momentum_10']]
    except Exception as e:
        logger.error(f"{company} verisi indirilirken hata: {str(e)}")
        return None

def fetch_and_engineer_features(company: str) -> Optional[pd.DataFrame]:
    """Paralel veri indirme ve özellik mühendisliği"""
    logger.info(f"{company} için veri işleniyor...")
    return fetch_company_data(company)

def process_companies_parallel(companies: List[str], max_workers: int = 5) -> pd.DataFrame:
    """Şirket verilerini paralel olarak işle"""
    all_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_company = {executor.submit(fetch_and_engineer_features, company): company 
                           for company in companies}
        
        for future in as_completed(future_to_company):
            company = future_to_company[future]
            try:
                data = future.result()
                if data is not None:
                    data.columns = [f"{company}_{col}" for col in data.columns]
                    all_data.append(data)
            except Exception as e:
                logger.error(f"{company} işlenirken hata: {str(e)}")
    
    if not all_data:
        raise ValueError("Hiçbir şirket verisi işlenemedi")
        
    merged_data = pd.concat(all_data, axis=1)
    merged_data.dropna(inplace=True)
    return merged_data

# Bellek optimizasyonu için veri temizleme
def clean_memory():
    """Belleği temizle"""
    gc.collect()
    tf.keras.backend.clear_session()

class StockPredictor:
    def __init__(self, companies: List[str], target_company: str = 'AAPL', years: int = 5):
        self.companies = companies
        self.target_company = target_company
        self.years = years
        self.start_date = datetime(2000, 1, 1)
        self.end_date = datetime(2025, 3, 14)
        self.model = None
        self.X_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.best_model_path = 'best_model.h5'
        self.history = None
        self.timesteps = 30

    def fetch_data(self, company: str) -> Optional[pd.DataFrame]:
        """Hisse senedi verilerini çeker"""
        try:
            ticker = yf.Ticker(company)
            data = ticker.history(start=self.start_date, end=self.end_date)
            if data.empty:
                logger.warning(f"{company} için veri bulunamadı")
                return None
            return data
        except Exception as e:
            logger.error(f"{company} verisi indirilirken hata: {str(e)}")
            return None

    def hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray) -> None:
        """Hiperparametre optimizasyonu yapar"""
        try:
            # Hiperparametre grid'i
            param_grid = {
                'gru_units_1': [32, 64, 128],
                'gru_units_2': [16, 32, 64],
                'dropout_rate': [0.1, 0.2, 0.3],
                'batch_size': [32, 64, 128],
                'learning_rate': [0.001, 0.01, 0.1]
            }

            # Veriyi eğitim ve test olarak böl
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            best_score = float('inf')
            best_params = None
            best_model = None

            # Manuel grid search
            for units_1 in param_grid['gru_units_1']:
                for units_2 in param_grid['gru_units_2']:
                    for dropout_rate in param_grid['dropout_rate']:
                        for batch_size in param_grid['batch_size']:
                            for learning_rate in param_grid['learning_rate']:
                                print(f"\nParametreler test ediliyor: units_1={units_1}, units_2={units_2}, "
                                      f"dropout={dropout_rate}, batch_size={batch_size}, lr={learning_rate}")
                                
                                # Model oluştur
                                model = Sequential([
                                    GRU(units_1, return_sequences=True, input_shape=(self.timesteps, X.shape[2])),
                                    Dropout(dropout_rate),
                                    GRU(units_2, return_sequences=False),
                                    Dropout(dropout_rate),
                                    Dense(16, activation='relu'),
                                    Dense(1)
                                ])
                                
                                # Optimizer'ı özelleştir
                                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                                model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
                                
                                # Early stopping ekle
                                early_stopping = tf.keras.callbacks.EarlyStopping(
                                    monitor='val_loss',
                                    patience=5,
                                    restore_best_weights=True
                                )
                                
                                # Modeli eğit
                                history = model.fit(
                                    X_train, y_train,
                                    epochs=20,  # Daha az epoch ile test et
                                    batch_size=batch_size,
                                    validation_data=(X_test, y_test),
                                    callbacks=[early_stopping],
                                    verbose=0
                                )
                                
                                # En iyi skoru kontrol et
                                val_loss = min(history.history['val_loss'])
                                if val_loss < best_score:
                                    best_score = val_loss
                                    best_params = {
                                        'gru_units_1': units_1,
                                        'gru_units_2': units_2,
                                        'dropout_rate': dropout_rate,
                                        'batch_size': batch_size,
                                        'learning_rate': learning_rate
                                    }
                                    best_model = model
                                    print(f"Yeni en iyi skor: {best_score:.4f}")
            
            print("\nEn iyi parametreler:", best_params)
            print("En iyi skor:", best_score)
            
            # En iyi modeli kaydet
            self.model = best_model
            self.model.save(self.best_model_path)
            
        except Exception as e:
            logger.error(f"Hiperparametre optimizasyonu sırasında hata: {str(e)}")
            raise

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Veriyi model için hazırlar"""
        try:
            # Hedef şirket verisini al
            target_data = self.fetch_data(self.target_company)
            if target_data is None:
                raise ValueError(f"{self.target_company} için veri bulunamadı")

            # Hedef şirketin tarih aralığını belirle
            target_dates = target_data.index
            start_date = target_dates.min()
            end_date = target_dates.max()

            # Hedef değişkeni hazırla
            target = target_data['Close'].values.reshape(-1, 1)
            target = self.y_scaler.fit_transform(target)

            features = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_company = {executor.submit(self.fetch_data, company): company 
                                   for company in self.companies if company != self.target_company}
                
                for future in as_completed(future_to_company):
                    company = future_to_company[future]
                    try:
                        data = future.result()
                        if data is not None and len(data) > 0:
                            # Veriyi hedef şirketin tarih aralığına göre filtrele
                            data = data.loc[start_date:end_date].copy()  # .copy() ekledik
                            
                            if len(data) > 0:
                                # Pandas uyarılarını önlemek için .loc kullanıyoruz
                                data.loc[:, 'Daily_Change'] = ((data['Close'] - data['Open']) / data['Open']) * 100
                                data.loc[:, 'MA_10'] = data['Close'].rolling(window=10).mean()
                                data.loc[:, 'MA_50'] = data['Close'].rolling(window=50).mean()
                                data.loc[:, 'Volatility_10'] = data['Close'].rolling(window=10).std()
                                data.loc[:, 'Momentum_10'] = data['Close'] - data['Close'].shift(10)
                                data.dropna(inplace=True)

                                if len(data) >= 50:
                                    features.append(data[['Daily_Change', 'MA_10', 'MA_50', 'Volatility_10', 'Momentum_10']].values)
                                else:
                                    logger.warning(f"{company} için yeterli veri yok, atlanıyor.")
                            else:
                                logger.warning(f"{company} için hedef tarih aralığında veri yok, atlanıyor.")
                    except Exception as e:
                        logger.error(f"{company} işlenirken hata: {str(e)}")

            if not features:
                raise ValueError("Hiçbir özellik eklenemedi")

            # Tüm özellikleri aynı boyuta getir
            min_length = min(feature.shape[0] for feature in features)
            features = [feature[:min_length] for feature in features]
            target = target[:min_length]

            X = np.column_stack(features)
            X = self.X_scaler.fit_transform(X)

            X_seq, y_seq = self.create_sequences(X, target)
            return X_seq, y_seq

        except Exception as e:
            logger.error(f"Veri hazırlama hatası: {str(e)}")
            raise

    def create_sequences(self, X: np.ndarray, y: np.ndarray, timesteps: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Zaman serisi dizilerini oluşturur"""
        X_seq, y_seq = [], []
        for i in range(len(X) - timesteps):
            X_seq.append(X[i:i + timesteps])
            y_seq.append(y[i + timesteps])
        return np.array(X_seq), np.array(y_seq)

    def build_model(self, timesteps: int, n_features: int) -> Sequential:
        """GRU modelini oluşturur"""
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(timesteps, n_features)),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    def train(self, epochs: int = 45, batch_size: int = 64, timesteps: int = 30, optimize: bool = True) -> None:
        """Modeli eğitir"""
        try:
            X_seq, y_seq = self.prepare_data()
            
            if optimize:
                print("Hiperparametre optimizasyonu başlıyor...")
                self.hyperparameter_optimization(X_seq, y_seq)
                print("Hiperparametre optimizasyonu tamamlandı.")
            else:
                self.model = self.build_model(timesteps, X_seq.shape[2])
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                
                self.history = self.model.fit(
                    X_seq, y_seq,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1
                )
            
            clean_memory()
            
        except Exception as e:
            logger.error(f"Model eğitimi sırasında hata: {str(e)}")
            raise

    def predict_future(self, n_steps: int = 5) -> Dict[str, float]:
        """Gelecek tahminleri yapar"""
        try:
            if self.model is None:
                self.model = load_model(self.best_model_path)
            
            X_seq, _ = self.prepare_data()
            last_data = X_seq[-1:]
            
            predictions = []
            current_input = last_data.copy()
            prediction_dates = []
            current_date = datetime.today()
            
            # Son işlem gününün fiyatını al
            last_price = self.y_scaler.inverse_transform(last_data[:, -1, :1])[0, 0]
            print(f"Son işlem günü fiyatı: {last_price:.2f} USD")
            
            for i in range(n_steps):
                next_date = current_date + timedelta(days=i+1)
                day_of_week = next_date.weekday()  # 0=Pazartesi, 6=Pazar
                day_name = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar'][day_of_week]
                
                print(f"\nTahmin yapılıyor: {next_date.strftime('%Y-%m-%d')} ({day_name})")
                
                # Hafta sonu günleri için son işlem gününün fiyatını kullan
                if day_of_week >= 5:  # Cumartesi (5) veya Pazar (6)
                    print(f"Hafta sonu günü - Son işlem günü fiyatı kullanılıyor: {last_price:.2f} USD")
                    predictions.append(last_price)
                else:
                    prediction = self.model.predict(current_input, verbose=0)
                    prediction_rescaled = self.y_scaler.inverse_transform(prediction)
                    predicted_price = prediction_rescaled[0, 0]
                    print(f"İş günü - Model tahmini: {predicted_price:.2f} USD")
                    predictions.append(predicted_price)
                    last_price = predicted_price  # Son işlem gününün fiyatını güncelle
                
                prediction_dates.append(next_date.strftime('%Y-%m-%d'))
                
                # Sadece iş günlerinde model tahminini güncelle
                if day_of_week < 5:
                    new_feature = np.roll(current_input, shift=-1, axis=1)
                    new_feature[:, -1, :] = prediction
                    current_input = new_feature
            
            return dict(zip(prediction_dates, predictions))
            
        except Exception as e:
            logger.error(f"Tahmin sırasında hata: {str(e)}")
            raise

def predict_stock(target_company: str) -> Dict[str, float]:
    """Belirli bir hisse senedi için tahmin yapar"""
    try:
        predictor = StockPredictor(companies, target_company)
        
        # Eğer model dosyası yoksa, önce modeli eğit
        if not os.path.exists(predictor.best_model_path):
            print("Model eğitiliyor...")
            predictor.train(epochs=45, batch_size=64, timesteps=30, optimize=True)
            print("Model eğitimi tamamlandı.")
        
        # Modeli yükle
        predictor.model = load_model(predictor.best_model_path)
        
        # Tahminleri yap
        predictions = predictor.predict_future(n_steps=7)  # 7 günlük tahmin (hafta sonları dahil)
        return predictions
    except Exception as e:
        logger.error(f"Tahmin hatası: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Tahminler
        print("\nTahminler yapılıyor...")
        predictions = predict_stock('AAPL')
        print("\nTahmin Sonuçları:")
        for date, price in predictions.items():
            day_of_week = datetime.strptime(date, '%Y-%m-%d').weekday()
            day_name = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar'][day_of_week]
            print(f"{date} ({day_name}) için tahmin edilen fiyat: {price:.2f} USD")
    except Exception as e:
        logger.error(f"Program çalıştırılırken hata: {str(e)}")