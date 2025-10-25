# test_visual_system.py
"""
Görsel Sistem Testi - Chart + Zones
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trend_detector import TrendDetector
from core.zone_detector import ZoneDetector
from position_agent import CoinSelector, SelectorConfig

def plot_analysis(coin, close, high, low, zones, trend_result):
    """Coin analizini görselleştir"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 1. Fiyat ve Zone'lar
    ax1.plot(close, label='Close Price', linewidth=1, alpha=0.7)
    ax1.plot(high, label='High', linewidth=0.5, alpha=0.5, color='green')
    ax1.plot(low, label='Low', linewidth=0.5, alpha=0.5, color='red')
    
    # Zone'ları çiz
    colors = ['red', 'blue', 'orange', 'purple', 'brown']
    for i, zone in enumerate(zones[:5]):  # İlk 5 zone
        color = colors[i % len(colors)]
        ax1.axhspan(zone.price_low, zone.price_high, alpha=0.3, color=color, 
                   label=f'Zone {i+1} (Q:{zone.quality:.1f})')
    
    ax1.set_title(f'{coin["symbol"]} - Price & Zones\nTrend: {trend_result.direction} (Conf: {trend_result.confidence:.2f})')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Zone Kaliteleri
    zone_qualities = [z.quality for z in zones]
    zone_touches = [z.touch_count for z in zones]
    
    x_pos = np.arange(len(zones))
    bars = ax2.bar(x_pos, zone_qualities, alpha=0.7, color='skyblue')
    
    # Bar'ları kaliteye göre renklendir
    for i, bar in enumerate(bars):
        if zone_qualities[i] >= 7:
            bar.set_color('green')
        elif zone_qualities[i] >= 5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
        
        # Üzerine dokunma sayısını yaz
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{zone_touches[i]}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_title('Zone Quality Scores')
    ax2.set_xlabel('Zone Index')
    ax2.set_ylabel('Quality (0-10)')
    ax2.set_xticks(x_pos)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=4, color='red', linestyle='--', alpha=0.5, label='Min Quality (4)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'output/{coin["symbol"]}_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print("🚀 Görsel Sistem Testi Başlatılıyor...\n")
    
    # Çıktı klasörü oluştur
    os.makedirs('output', exist_ok=True)
    
    # Coin seçimi
    print("🪙 Coin Seçimi Yapılıyor...")
    try:
        selector_config = SelectorConfig()
        selector = CoinSelector(config=selector_config)
        coins = selector.load_or_update_pool(force=False)[:2]
        print(f"✅ Seçilen Coin'ler: {[c['symbol'] for c in coins]}")
    except Exception as e:
        print(f"❌ Coin seçimi hatası: {e}")
        coins = [
            {"symbol": "BTCUSDT", "volume": 12597718381, "atr": 3.61},
            {"symbol": "ETHUSDT", "volume": 14672593838, "atr": 6.34},
        ]
    
    # Her coin için görsel analiz
    for coin in coins:
        print(f"\n📊 {coin['symbol']} Görsel Analiz Hazırlanıyor...")
        
        try:
            # Fiyat verisi
            np.random.seed(42)
            base_price = 50000 if coin['symbol'] == 'BTCUSDT' else 3000
            close = base_price + np.cumsum(np.random.randn(200) * coin['atr']/100 * base_price)
            high = close + np.abs(np.random.randn(200) * base_price * 0.02)
            low = close - np.abs(np.random.randn(200) * base_price * 0.02)
            
            # Analiz
            trend_detector = TrendDetector()
            trend_result = trend_detector.detect(close, high, low)
            
            zone_detector = ZoneDetector()
            zones = zone_detector.detect_zones(high, low, close, timeframe="1H")
            
            # Görselleştir
            plot_analysis(coin, close, high, low, zones, trend_result)
            
            print(f"✅ {coin['symbol']} chart kaydedildi: output/{coin['symbol']}_analysis.png")
            
        except Exception as e:
            print(f"❌ {coin['symbol']} görselleştirme hatası: {e}")
    
    print(f"\n🎉 TÜM GÖRSELLER HAZIR!")
    print("📁 'output/' klasöründeki PNG dosyalarını kontrol edin")

if __name__ == "__main__":
    main()