import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def veri_cek(sembol, baslangic_tarihi, bitis_tarihi):
    """Hisse senedi verilerini çeker"""
    try:
        hisse = yf.Ticker(sembol)
        veri = hisse.history(start=baslangic_tarihi, end=bitis_tarihi)
        return veri
    except Exception as e:
        print(f"Veri çekme hatası: {e}")
        return None

def veri_hazirla(veri):
    """Veriyi model için hazırlar"""
    if veri is None or veri.empty:
        return None, None
        
    # Teknik indikatörler ekle
    veri['SMA_20'] = veri['Close'].rolling(window=20).mean()
    veri['SMA_50'] = veri['Close'].rolling(window=50).mean()
    
    # NaN değerleri temizle
    veri = veri.dropna()
    
    # Özellikler ve hedef değişken
    X = veri[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50']]
    y = veri['Close'].shift(-1)  # Bir sonraki günün kapanış fiyatı
    
    # Son satırı kaldır (NaN değer)
    X = X[:-1]
    y = y[:-1]
    
    return X, y

def model_egit(X, y):
    """Modeli eğitir"""
    if X is None or y is None:
        return None, None, None, None
        
    # Veriyi ölçeklendir
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Eğitim ve test verilerini ayır
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Modeli eğit
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, scaler, X_test, y_test

def tahmin_yap(model, scaler, X):
    """Tahmin yapar"""
    if model is None or scaler is None or X is None:
        return None
        
    X_scaled = scaler.transform(X)
    tahmin = model.predict(X_scaled)
    return tahmin

def sonuclari_goster(y_test, tahminler):
    """Sonuçları görselleştirir"""
    if y_test is None or tahminler is None:
        return
        
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Gerçek Değerler')
    plt.plot(y_test.index, tahminler, label='Tahminler')
    plt.title('Hisse Senedi Fiyat Tahminleri')
    plt.xlabel('Tarih')
    plt.ylabel('Fiyat')
    plt.legend()
    plt.show()

def main():
    # Örnek kullanım
    sembol = "THYAO.IS"  # Türk Hava Yolları
    baslangic = "2023-01-01"
    bitis = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Veri çekiliyor... ({sembol})")
    veri = veri_cek(sembol, baslangic, bitis)
    
    if veri is not None and not veri.empty:
        print("Veri hazırlanıyor...")
        X, y = veri_hazirla(veri)
        
        if X is not None and y is not None:
            print("Model eğitiliyor...")
            model, scaler, X_test, y_test = model_egit(X, y)
            
            if model is not None:
                print("Tahminler yapılıyor...")
                tahminler = tahmin_yap(model, scaler, X_test)
                
                if tahminler is not None:
                    print("Sonuçlar görselleştiriliyor...")
                    sonuclari_goster(y_test, tahminler)
                else:
                    print("Tahmin yapılamadı!")
            else:
                print("Model eğitilemedi!")
        else:
            print("Veri hazırlanamadı!")
    else:
        print("Veri çekilemedi!")

if __name__ == "__main__":
    main() 