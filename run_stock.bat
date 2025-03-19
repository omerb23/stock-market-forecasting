@echo off
chcp 65001 > nul
echo Program başlatılıyor...

REM Masaüstüne git
cd /d "C:\Users\bilal\OneDrive\Masaüstü"

REM Sanal ortamı aktifleştir
if not exist .venv (
    echo Sanal ortam oluşturuluyor...
    python -m venv .venv
    if errorlevel 1 (
        echo Sanal ortam oluşturulamadı!
        pause
        exit /b 1
    )
)

REM Sanal ortamı aktifleştir
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Sanal ortam aktifleştirilemedi!
    pause
    exit /b 1
)

REM Gerekli kütüphaneleri yükle
echo Gerekli kütüphaneler yükleniyor...
pip install yfinance pandas numpy scikit-learn tensorflow matplotlib scipy
if errorlevel 1 (
    echo Kütüphaneler yüklenirken hata oluştu!
    pause
    exit /b 1
)

REM Python kodunu çalıştır
python stock_prediction.py
if errorlevel 1 (
    echo Program çalıştırılırken bir hata oluştu!
    pause
    exit /b 1
)

pause 