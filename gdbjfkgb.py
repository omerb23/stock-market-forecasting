import yfinance as yf
from datetime import datetime, timedelta

try:
    # Yahoo Finance veri çekme (Google hissesi için)
    googl = yf.Ticker("GOOGL")
    # Son 1 günlük veri
    veri = googl.history(period="1d", interval="1h")
    
    if not veri.empty:
        fiyat = veri.iloc[-1]["Close"]
        print(f"Google'ın güncel fiyatı: {fiyat:.2f} USD")
        print(f"Son güncelleme: {veri.index[-1]}")
        
        # Son 5 veriyi göster
        print("\nSon 5 veri:")
        print(veri.tail())
    else:
        print("Veri bulunamadı. Piyasa kapalı olabilir.")
        
except Exception as e:
    print(f"Bir hata oluştu: {e}")