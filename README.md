Borsa Tahmin Aracı
Bu proje, yapay zeka ve makine öğrenmesi yöntemlerini kullanarak borsa tahminleri yapmayı amaçlamaktadır. Yahoo Finance üzerinden elde edilen 25 yıllık veri ile eğitilmiş bir model kullanarak, kullanıcılara tahmin edilmek istenen hisse senedinin son 30 günlük borsa verilerini analiz ederek doğru ve tutarlı tahminler sunar.

Özellikler:
Son 30 günlük borsa verilerini analiz ederek hisse senedi tahminleri yapar.

Kullanıcılar, anlık olarak en çok işlem gören 5 hissenin durumunu takip edebilir.

Hisse etiketlerini seçerek tek tuşla tahmin yapılabilir.

Kolay okunabilir grafiklerle tahmin sonuçları görselleştirilir.

Kullanılan Teknolojiler:
Python

Yapay Zeka ve Makine Öğrenmesi Algoritmaları

Yahoo Finance API

Matplotlib ve Seaborn (veri görselleştirme)

Kurulum:
Depoyu klonlayın:

bash
Kopyala
Düzenle
git clone https://github.com/omerb23/stock-market-forecasting.git
Gerekli kütüphaneleri yükleyin:

bash
Kopyala
Düzenle
pip install -r requirements.txt
Modeli eğitmek için yf.py dosyasını çalıştırın.

Tahmin yapmak için runner.py dosyasını kullanabilirsiniz.

Arayüze ulaşmak için ise prediction.py dosaysını çalıştırabilirsiniz.
