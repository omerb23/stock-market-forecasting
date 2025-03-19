# Hisse Senedi Fiyat Tahmin Modeli

Bu proje, hisse senedi fiyatlarını tahmin etmek için makine öğrenmesi kullanır.

## Özellikler

- Yahoo Finance'den hisse senedi verisi çekme
- Teknik indikatörler (SMA-20, SMA-50)
- Veri ön işleme ve normalizasyon
- Lineer regresyon modeli
- Görselleştirme

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Programı çalıştırın:
```bash
python stock_predictor.py
```

## Kullanım

Program varsayılan olarak Türk Hava Yolları (THYAO.IS) hissesi için tahmin yapar. Farklı bir hisse senedi için `main()` fonksiyonundaki `sembol` değişkenini değiştirebilirsiniz.

## Notlar

- Model basit bir lineer regresyon kullanır ve sadece eğitim amaçlıdır
- Gerçek yatırım kararları için kullanılmamalıdır
- Tahminler geçmiş verilere dayanır ve geleceği garanti etmez 