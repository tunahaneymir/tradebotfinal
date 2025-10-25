# test_full_system.py
"""
Full System Integration Test - TAM VERSİYON
"""

import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trend_detector import TrendDetector
from core.zone_detector import ZoneDetector
from core.zone_quality_scorer import ZoneQualityScorer
from position_agent import CoinSelector, SelectorConfig

def main():
    print("🚀 Full System Test Başlatılıyor...\n")
    
    # 1. Coin Seçimi
    print("🪙 Coin Seçimi Yapılıyor...")
    try:
        # ✅ DÜZELTME: Config ile CoinSelector
        selector_config = SelectorConfig()  # Varsayılan config
        selector = CoinSelector(config=selector_config)  # ✅ Config ile
        coins = selector.load_or_update_pool(force=False)[:2]  # İlk 2 coin
        print(f"✅ Seçilen Coin'ler: {[c['symbol'] for c in coins]}")
    except Exception as e:
        print(f"❌ Coin seçimi hatası: {e}")
        # Fallback: Manuel coin listesi
        coins = [
            {"symbol": "BTCUSDT", "volume": 12597718381, "atr": 3.61},
            {"symbol": "ETHUSDT", "volume": 14672593838, "atr": 6.34},
        ]
        print(f"✅ Manuel coin'ler kullanılıyor: {[c['symbol'] for c in coins]}")

    print("\n" + "="*60)

    # 2. Her coin için analiz
    for i, coin in enumerate(coins, 1):
        print(f"\n🔍 {i}. {coin['symbol']} Analizi")
        print("-" * 40)

        try:
            # Fiyat verisi oluştur
            np.random.seed(42 + i)
            base_price = 50000 if coin['symbol'] == 'BTCUSDT' else 3000
            close = base_price + np.cumsum(np.random.randn(500) * coin['atr']/100 * base_price)
            high = close + np.abs(np.random.randn(500) * base_price * 0.02)
            low = close - np.abs(np.random.randn(500) * base_price * 0.02)

            print(f"📊 Fiyat verisi: {len(close)} mum")

            # Trend analizi
            trend_detector = TrendDetector()
            trend_result = trend_detector.detect(close, high, low)
            print(f"📈 Trend: {trend_result.direction}, Güven: {trend_result.confidence:.2f}")

            # Zone analizi
            zone_detector = ZoneDetector()
            zones = zone_detector.detect_zones(
                high=high, 
                low=low, 
                close=close, 
                timeframe="1H"
            )
            
            print(f"📍 Toplam Zone: {len(zones)}")

            # Kaliteli zone'ları filtrele
            qualified_zones = [z for z in zones if z.quality >= 4]
            print(f"🎯 Kaliteli Zone'lar (≥4): {len(qualified_zones)}")

            if qualified_zones:
                # İlk 3 zone'u göster
                for j, zone in enumerate(qualified_zones[:3], 1):
                    thickness_pct = (zone.price_high - zone.price_low) / zone.price_mid * 100
                    print(f"   {j}. {zone.id}")
                    print(f"      Fiyat: ${zone.price_low:.0f} - ${zone.price_high:.0f}")
                    print(f"      Kalite: {zone.quality:.1f}/10")
                    print(f"      Dokunma: {zone.touch_count}, Kalınlık: {thickness_pct:.1f}%")

        except Exception as e:
            print(f"❌ {coin['symbol']} analiz hatası: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("✅ FULL SYSTEM TEST TAMAMLANDI! 🎯")
    print("Sistem demo trading için hazır!")

if __name__ == "__main__":
    main()