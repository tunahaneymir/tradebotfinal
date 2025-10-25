"""
Trend Detector - Parça 1 (FINAL VERSION)
Based on: pa-strateji2 Parça 1

Features:
- EMA20/50 trend detection
- Sideways market filter
- 4H timeframe
- ML features for RL agent
- Adaptive parameters (ATR-based)
- Config integration
"""

from __future__ import annotations
from typing import Literal, Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class TrendResult:
    """Trend detection sonucu"""
    direction: Literal["UP", "DOWN", "SIDEWAYS"]
    ema_20: float
    ema_50: float
    ema_distance_pct: float
    atr_ratio: float
    price_range_pct: float
    confidence: float  # 0-1 arası trend güveni
    slope_up: bool
    slope_down: bool
    slope_strength: float = 0.0
    sideways_signal_count: int = 0
    
    def get_ml_features(self) -> Dict[str, float]:
        """
        ML modeli için feature dictionary
        RL agent bu feature'ları kullanarak öğrenecek
        """
        return {
            # === Trend Direction ===
            'trend_direction_numeric': (
                1.0 if self.direction == "UP" 
                else -1.0 if self.direction == "DOWN" 
                else 0.0
            ),
            'trend_confidence': self.confidence,
            
            # === EMA Features ===
            'ema_distance_ratio': self.ema_distance_pct,
            'ema_20_value': self.ema_20,
            'ema_50_value': self.ema_50,
            'ema_cross_position': (
                (self.ema_20 - self.ema_50) / self.ema_50 
                if self.ema_50 != 0 else 0.0
            ),
            
            # === Slope Features ===
            'ema_slope_up': float(self.slope_up),
            'ema_slope_down': float(self.slope_down),
            'ema_slope_strength': self.slope_strength,
            
            # === Volatility Features ===
            'atr_ratio': self.atr_ratio,
            'price_range_ratio': self.price_range_pct,
            
            # === Market State ===
            'is_sideways': float(self.direction == "SIDEWAYS"),
            'is_trending': float(self.direction != "SIDEWAYS"),
            'sideways_signal_count': float(self.sideways_signal_count),
            
            # === Trend Strength Composite ===
            'trend_strength': self._calculate_trend_strength(),
        }
    
    def _calculate_trend_strength(self) -> float:
        """
        Genel trend gücü (0-1)
        RL için önemli composite feature
        """
        if self.direction == "SIDEWAYS":
            return 0.0
        
        strength = 0.0
        
        # Confidence contribution (40%)
        strength += self.confidence * 0.4
        
        # EMA distance contribution (30%)
        ema_dist_normalized = min(self.ema_distance_pct / 0.05, 1.0)  # %5 = max
        strength += ema_dist_normalized * 0.3
        
        # Slope strength contribution (30%)
        strength += self.slope_strength * 0.3
        
        return min(strength, 1.0)


class TrendDetector:
    """
    4H Timeframe Trend Detection
    
    Rules:
    - UPTREND: price > EMA20 > EMA50 + EMA20 slope up
    - DOWNTREND: price < EMA20 < EMA50 + EMA20 slope down
    - SIDEWAYS: 3 koşuldan 2'si:
        1. EMA distance < 0.5%
        2. ATR ratio < 0.6%
        3. Price range (8 candles) < 8%
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize TrendDetector
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        # Load from config or use defaults
        trend_config = config.get('trend', {}) if config else {}
        sideways_config = trend_config.get('sideways', {}) if trend_config else {}
        
        self.ema_fast = trend_config.get('ema_fast', 20)
        self.ema_slow = trend_config.get('ema_slow', 50)
        self.sideways_ema_distance = sideways_config.get('ema_distance_pct', 0.005)
        self.sideways_atr_ratio = sideways_config.get('atr_ratio', 0.006)
        self.sideways_range = sideways_config.get('range_pct', 0.08)
        self.ema_slope_lookback = 5  # Fixed
    
    def detect(
        self, 
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: Optional[np.ndarray] = None
    ) -> TrendResult:
        """
        Trend tespiti yap
        
        Args:
            close: Close prices
            high: High prices
            low: Low prices
            atr: ATR values (optional, hesaplanır)
            
        Returns:
            TrendResult with ML features
        """
        # EMA hesapla
        ema_20_values = self.calculate_ema(close, self.ema_fast)
        ema_50_values = self.calculate_ema(close, self.ema_slow)
        ema_20 = ema_20_values[-1]
        ema_50 = ema_50_values[-1]
        
        current_price = close[-1]
        
        # EMA slope kontrolü
        slope_up = self._check_slope_up(ema_20_values)
        slope_down = self._check_slope_down(ema_20_values)
        slope_strength = self._calculate_slope_strength(ema_20_values)
        
        # ATR hesapla
        if atr is None:
            atr = self.calculate_atr(high, low, close)
        current_atr = atr[-1]
        
        # Sideways kriterleri hesapla
        ema_distance_pct = abs(ema_20 - ema_50) / current_price
        atr_ratio = current_atr / current_price
        price_range_pct = self._calculate_price_range(high, low, current_price)
        
        # Sideways kontrolü
        sideways_signals = 0
        if ema_distance_pct < self.sideways_ema_distance:
            sideways_signals += 1
        if atr_ratio < self.sideways_atr_ratio:
            sideways_signals += 1
        if price_range_pct < self.sideways_range:
            sideways_signals += 1
        
        # Trend belirleme
        if sideways_signals >= 2:
            direction = "SIDEWAYS"
            confidence = sideways_signals / 3.0
        elif current_price > ema_20 > ema_50 and slope_up:
            direction = "UP"
            confidence = self._calculate_trend_confidence(
                current_price, ema_20, ema_50, slope_up
            )
        elif current_price < ema_20 < ema_50 and slope_down:
            direction = "DOWN"
            confidence = self._calculate_trend_confidence(
                current_price, ema_20, ema_50, slope_down
            )
        else:
            direction = "SIDEWAYS"
            confidence = 0.3
        
        return TrendResult(
            direction=direction,
            ema_20=ema_20,
            ema_50=ema_50,
            ema_distance_pct=ema_distance_pct,
            atr_ratio=atr_ratio,
            price_range_pct=price_range_pct,
            confidence=confidence,
            slope_up=slope_up,
            slope_down=slope_down,
            slope_strength=slope_strength,
            sideways_signal_count=sideways_signals
        )
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Exponential Moving Average hesapla
        
        Args:
            prices: Fiyat dizisi
            period: EMA periyodu
            
        Returns:
            EMA değerleri
        """
        ema = np.zeros_like(prices, dtype=float)
        multiplier = 2 / (period + 1)
        
        # İlk değer SMA
        ema[period-1] = np.mean(prices[:period])
        
        # EMA hesaplama
        for i in range(period, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema
    
    def calculate_atr(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        period: int = 14
    ) -> np.ndarray:
        """
        Average True Range hesapla
        
        Args:
            high: High fiyatları
            low: Low fiyatları
            close: Close fiyatları
            period: ATR periyodu
            
        Returns:
            ATR değerleri
        """
        # True Range hesapla
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        
        # ATR = TR'nin EMA'sı
        atr = self.calculate_ema(tr, period)
        
        return atr
    
    def _check_slope_up(self, ema_values: np.ndarray) -> bool:
        """EMA20 yükseliyor mu?"""
        if len(ema_values) < self.ema_slope_lookback:
            return False
        
        recent_emas = ema_values[-self.ema_slope_lookback:]
        
        # Son 5 değerin en az 4'ü artan trend (60% threshold)
        increasing_count = 0
        for i in range(1, len(recent_emas)):
            if recent_emas[i] > recent_emas[i-1]:
                increasing_count += 1
        
        return increasing_count >= (self.ema_slope_lookback - 1) * 0.6
    
    def _check_slope_down(self, ema_values: np.ndarray) -> bool:
        """EMA20 düşüyor mu?"""
        if len(ema_values) < self.ema_slope_lookback:
            return False
        
        recent_emas = ema_values[-self.ema_slope_lookback:]
        
        # Son 5 değerin en az 4'ü azalan trend (60% threshold)
        decreasing_count = 0
        for i in range(1, len(recent_emas)):
            if recent_emas[i] < recent_emas[i-1]:
                decreasing_count += 1
        
        return decreasing_count >= (self.ema_slope_lookback - 1) * 0.6
    
    def _calculate_slope_strength(self, ema_values: np.ndarray) -> float:
        """
        EMA slope gücü hesapla (0-1)
        
        Args:
            ema_values: EMA değerleri
            
        Returns:
            Slope strength (0 = yok, 1 = çok güçlü)
        """
        if len(ema_values) < self.ema_slope_lookback:
            return 0.0
        
        recent_emas = ema_values[-self.ema_slope_lookback:]
        
        # Kaç tanesinde artış/azalış var?
        increasing_count = 0
        decreasing_count = 0
        
        for i in range(1, len(recent_emas)):
            if recent_emas[i] > recent_emas[i-1]:
                increasing_count += 1
            elif recent_emas[i] < recent_emas[i-1]:
                decreasing_count += 1
        
        max_count = max(increasing_count, decreasing_count)
        total_comparisons = len(recent_emas) - 1
        
        # Normalize: tamamı aynı yönde ise 1.0
        strength = max_count / total_comparisons
        
        return strength
    
    def _calculate_price_range(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        current_price: float,
        lookback: int = 8
    ) -> float:
        """Son N mumun fiyat aralığı"""
        recent_high = np.max(high[-lookback:])
        recent_low = np.min(low[-lookback:])
        
        price_range = (recent_high - recent_low) / current_price
        return price_range
    
    def _calculate_trend_confidence(
        self,
        price: float,
        ema_20: float,
        ema_50: float,
        slope_confirmed: bool
    ) -> float:
        """Trend güven skoru (0-1)"""
        confidence = 0.5
        
        # EMA sıralaması
        if price > ema_20 > ema_50:
            confidence += 0.2
        elif price < ema_20 < ema_50:
            confidence += 0.2
        
        # Slope onayı
        if slope_confirmed:
            confidence += 0.2
        
        # EMA mesafesi (daha geniş = daha güçlü trend)
        ema_distance = abs(ema_20 - ema_50) / price
        if ema_distance > 0.02:  # >%2
            confidence += 0.1
        
        return min(confidence, 1.0)


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test data oluştur
    np.random.seed(42)
    
    # Config örneği
    config = {
        'trend': {
            'ema_fast': 20,
            'ema_slow': 50,
            'sideways': {
                'ema_distance_pct': 0.005,
                'atr_ratio': 0.006,
                'range_pct': 0.08
            }
        }
    }
    
    # Uptrend simulation
    close = np.cumsum(np.random.randn(100) * 10 + 5) + 50000
    high = close + np.random.rand(100) * 20
    low = close - np.random.rand(100) * 20
    
    # Detector oluştur
    detector = TrendDetector(config)
    
    # Trend tespit et
    result = detector.detect(close, high, low)
    
    print("\n" + "="*60)
    print("TREND DETECTION RESULT")
    print("="*60)
    print(f"Direction: {result.direction}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"\nEMA 20: ${result.ema_20:,.2f}")
    print(f"EMA 50: ${result.ema_50:,.2f}")
    print(f"Current Price: ${close[-1]:,.2f}")
    print(f"\nEMA Distance: {result.ema_distance_pct*100:.2f}%")
    print(f"ATR Ratio: {result.atr_ratio*100:.2f}%")
    print(f"Price Range: {result.price_range_pct*100:.2f}%")
    print(f"\nSlope Up: {result.slope_up}")
    print(f"Slope Down: {result.slope_down}")
    print(f"Slope Strength: {result.slope_strength:.2f}")
    print(f"Sideways Signals: {result.sideways_signal_count}/3")
    print("="*60)
    
    # ML Features
    print("\n" + "="*60)
    print("ML FEATURES FOR RL AGENT")
    print("="*60)
    features = result.get_ml_features()
    for key, value in features.items():
        print(f"{key:30s}: {value:8.4f}")
    print("="*60 + "\n")