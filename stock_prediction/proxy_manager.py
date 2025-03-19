import requests
import random
from fake_useragent import UserAgent
import time
import yfinance as yf
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import pandas_datareader as pdr
from pandas_datareader import data as pdr_data
from datetime import datetime
import asyncio
import aiohttp
import concurrent.futures
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

# SSL uyarılarını kapat
urllib3.disable_warnings(InsecureRequestWarning)

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProxyType(Enum):
    HTTP = "http"
    HTTPS = "https"
    SOCKS4 = "socks4"
    SOCKS5 = "socks5"

@dataclass
class ProxyInfo:
    proxy: Dict[str, str]
    type: ProxyType
    speed: float
    anonymity_level: str
    last_check: datetime
    success_count: int
    fail_count: int
    is_active: bool

class ProxyManager:
    def __init__(self):
        self.proxies: List[ProxyInfo] = []
        self.ua = UserAgent()
        self.proxy_sources = {
            ProxyType.HTTP: [
                'https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt',
                'https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/http.txt',
                'https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt',
                'https://raw.githubusercontent.com/sunny9577/proxy-scraper/master/proxies.txt'
            ],
            ProxyType.SOCKS4: [
                'https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks4.txt',
                'https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/socks4.txt'
            ],
            ProxyType.SOCKS5: [
                'https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks5.txt',
                'https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/socks5.txt',
                'https://raw.githubusercontent.com/hookzof/socks5_list/master/proxy.txt'
            ]
        }
        self.load_proxies()
        self.start_proxy_monitor()
        
    def load_proxies(self):
        """Proxy listelerini yükler ve doğrular"""
        logger.info("Proxy listeleri yükleniyor...")
        
        for proxy_type, sources in self.proxy_sources.items():
            for source in sources:
                try:
                    response = requests.get(source, timeout=10)
                    if response.status_code == 200:
                        proxy_list = response.text.strip().split('\n')
                        for proxy in proxy_list:
                            if proxy.strip():
                                proxy_dict = {
                                    'http': f'{proxy_type.value}://{proxy.strip()}',
                                    'https': f'{proxy_type.value}://{proxy.strip()}'
                                }
                                self.proxies.append(ProxyInfo(
                                    proxy=proxy_dict,
                                    type=proxy_type,
                                    speed=0.0,
                                    anonymity_level="unknown",
                                    last_check=datetime.now(),
                                    success_count=0,
                                    fail_count=0,
                                    is_active=True
                                ))
                except Exception as e:
                    logger.error(f"Proxy kaynağı yüklenirken hata: {str(e)}")
                    continue
        
        if not self.proxies:
            logger.warning("Hiç proxy bulunamadı, doğrudan bağlantı kullanılacak.")
        else:
            logger.info(f"Toplam {len(self.proxies)} proxy yüklendi.")
            self.verify_proxies()
    
    async def check_proxy_async(self, proxy_info: ProxyInfo) -> Tuple[bool, float, str]:
        """Proxy'yi asenkron olarak kontrol eder"""
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'https://api64.ipify.org?format=json',
                    proxy=proxy_info.proxy['http'],
                    timeout=10,
                    headers={'User-Agent': self.get_random_user_agent()},
                    ssl=False
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        speed = time.time() - start_time
                        anonymity_level = self.check_anonymity_level(data.get('ip', ''))
                        return True, speed, anonymity_level
        except Exception as e:
            logger.debug(f"Proxy kontrol hatası: {str(e)}")
        return False, 0.0, "unknown"
    
    def verify_proxies(self):
        """Tüm proxy'leri doğrular"""
        logger.info("Proxy'ler doğrulanıyor...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        tasks = [self.check_proxy_async(proxy) for proxy in self.proxies]
        results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        
        for proxy_info, (is_working, speed, anonymity) in zip(self.proxies, results):
            if isinstance(results, Exception):
                proxy_info.is_active = False
                proxy_info.fail_count += 1
            else:
                proxy_info.speed = speed
                proxy_info.anonymity_level = anonymity
                proxy_info.last_check = datetime.now()
                if is_working:
                    proxy_info.success_count += 1
                else:
                    proxy_info.fail_count += 1
                    if proxy_info.fail_count >= 3:
                        proxy_info.is_active = False
        
        # Çalışmayan proxy'leri listeden çıkar
        self.proxies = [p for p in self.proxies if p.is_active]
        logger.info(f"Doğrulama sonrası {len(self.proxies)} aktif proxy kaldı.")
    
    def check_anonymity_level(self, ip: str) -> str:
        """Proxy'nin anonimlik seviyesini kontrol eder"""
        try:
            response = requests.get(
                'https://api.ipify.org?format=json',
                timeout=5,
                headers={'User-Agent': self.get_random_user_agent()}
            )
            if response.status_code == 200:
                real_ip = response.json()['ip']
                if real_ip == ip:
                    return "transparent"
                return "anonymous"
        except:
            pass
        return "elite"
    
    def get_random_proxy(self) -> Optional[ProxyInfo]:
        """En iyi performans gösteren proxy'lerden birini seçer"""
        active_proxies = [p for p in self.proxies if p.is_active]
        if not active_proxies:
            return None
            
        # Başarı oranı ve hıza göre sırala
        sorted_proxies = sorted(
            active_proxies,
            key=lambda x: (x.success_count / (x.success_count + x.fail_count + 1), -x.speed)
        )
        
        # En iyi 3 proxy'den rastgele birini seç
        return random.choice(sorted_proxies[:3]) if sorted_proxies else None
    
    def get_random_user_agent(self) -> str:
        """Rastgele bir User-Agent döndürür"""
        return self.ua.random
    
    def start_proxy_monitor(self):
        """Proxy'leri düzenli olarak kontrol eder"""
        def monitor():
            while True:
                self.verify_proxies()
                time.sleep(3600)  # Her saat başı kontrol et
        
        # Arka planda çalıştır
        thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        thread.submit(monitor)
    
    def get_yfinance_data(self, symbol: str, start_date: str = "2000-01-01", 
                         end_date: str = "2025-03-01", max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Proxy kullanarak Yahoo Finance'den veri çeker"""
        for attempt in range(max_retries):
            try:
                proxy_info = self.get_random_proxy()
                if not proxy_info:
                    logger.warning("Çalışan proxy bulunamadı, doğrudan bağlantı deneniyor...")
                    return yf.Ticker(symbol).history(start=start_date, end=end_date)

                session = requests.Session()
                session.proxies = proxy_info.proxy
                session.verify = False
                session.headers.update({
                    'User-Agent': self.get_random_user_agent(),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                })

                yf.pdr_override()
                yf.set_tz_cache_location('yfinance.cache')
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    prepost=False,
                    actions=False,
                    auto_adjust=True
                )

                if not data.empty:
                    logger.info(f"{symbol} verisi başarıyla çekildi!")
                    proxy_info.success_count += 1
                    return data
                else:
                    logger.warning(f"{symbol} için veri bulunamadı. (Deneme {attempt + 1}/{max_retries})")
                    proxy_info.fail_count += 1
                    time.sleep(2 ** attempt)  # Exponential backoff

            except Exception as e:
                logger.error(f"Hata oluştu: {str(e)}")
                if proxy_info:
                    proxy_info.fail_count += 1
                if attempt < max_retries - 1:
                    logger.info(f"Yeniden deneniyor... (Deneme {attempt + 1}/{max_retries})")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"{symbol} için veri çekilemedi.")
                    return None

        return None

    def get_google_finance_data(self, symbol: str, start_date: str = "2000-01-01", 
                              end_date: str = "2025-03-01", max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Google Finance'den veri çeker"""
        for attempt in range(max_retries):
            try:
                proxy_info = self.get_random_proxy()
                if not proxy_info:
                    logger.warning("Çalışan proxy bulunamadı, doğrudan bağlantı deneniyor...")
                    return pdr_data.get_data_google(symbol, start=start_date, end=end_date)

                session = requests.Session()
                session.proxies = proxy_info.proxy
                session.verify = False
                session.headers.update({
                    'User-Agent': self.get_random_user_agent(),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                })

                data = pdr_data.get_data_google(
                    symbol,
                    start=start_date,
                    end=end_date,
                    retry_count=3,
                    timeout=30,
                    pause=2.0
                )

                if not data.empty:
                    logger.info(f"{symbol} verisi başarıyla çekildi!")
                    proxy_info.success_count += 1
                    return data
                else:
                    logger.warning(f"{symbol} için veri bulunamadı. (Deneme {attempt + 1}/{max_retries})")
                    proxy_info.fail_count += 1
                    time.sleep(2 ** attempt)

            except Exception as e:
                logger.error(f"Hata oluştu: {str(e)}")
                if proxy_info:
                    proxy_info.fail_count += 1
                if attempt < max_retries - 1:
                    logger.info(f"Yeniden deneniyor... (Deneme {attempt + 1}/{max_retries})")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"{symbol} için veri çekilemedi.")
                    return None

        return None

# Test kodu
if __name__ == "__main__":
    proxy_manager = ProxyManager()
    
    # Test için bir hisse senedi seç
    symbol = "THYAO.IS"  # Türk Hava Yolları
    start_date = "2000-01-01"
    end_date = "2025-03-01"
    
    logger.info(f"\n{symbol} için veri çekiliyor...")
    logger.info(f"Tarih aralığı: {start_date} - {end_date}")
    
    # Önce Google Finance'i dene
    data = proxy_manager.get_google_finance_data(symbol, start_date, end_date)
    
    if data is None or data.empty:
        logger.info("\nGoogle Finance başarısız oldu, Yahoo Finance deneniyor...")
        data = proxy_manager.get_yfinance_data(symbol, start_date, end_date)
    
    if data is not None and not data.empty:
        logger.info("\nVeri başarıyla çekildi!")
        logger.info("\nİlk 5 satır:")
        print(data.head())
        logger.info(f"\nVeri boyutu: {data.shape}")
    else:
        logger.error("\nVeri çekilemedi!") 