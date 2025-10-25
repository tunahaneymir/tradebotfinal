"""
Zone Detector - Parça 1 (FINAL VERSION)
Based on: pa-strateji2 Parça 1

Features:
- ZigZag++ detection
- Swing High/Low detection
- Zone merging & filtering
- Zone quality scoring (0-10)
- Multi-timeframe support
- Adaptive parameters (ATR-based)
- Lookback periods
- Config integration
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class Zone:
    """Zone bilgisi"""
    id: str
    price_low: float
    price_high: float
    price_mid: float
    touch_count: int
    thickness_pct: float
    last_touch_index: int
    creation_index: int
    timeframe: str
    method: str  # "zigzag", "swing", "both"
    quality: float = 0.0
    days_since_last_touch: float = 0.0
    
    def get_ml_features(self) -> Dict[str, float]:
        """ML modeli için zone feature'ları"""
        return {
            'zone_quality': self.quality,
            'zone_touch_count': float(self.touch_count),
            'zone_thickness_pct': self.thickness_pct,
            'zone_days_since_touch': self.days_since_last_touch,
            'zone_is_fresh': float(self.days_since_last_touch < 7),
            'zone_is_thin': float(self.thickness_pct < 0.01),  # <%1
            'zone_method_both': float(self.method == "both"),
            'zone_age_candles': float(self.last_touch_index - self.creation_index),
        }


@dataclass
class ZigZagPoint:
    """ZigZag pivot noktası"""
    index: int
    price: float
    type: str  # "high" or "low"


class ZoneDetector:
    """
    Zone Detection Engine
    
    Methods:
    - ZigZag++: Swing point detection
    - Swing HL: Confirmation
    - Dual confirmation: Both methods agree
    - Quality scoring: 0-10 rating system
    - Adaptive parameters based on volatility
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ZoneDetector
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        # Load from config or use defaults
        zone_config = config.get('zones', {}) if config else {}
        zigzag_config = zone_config.get('zigzag', {})
        swing_config = zone_config.get('swing', {})
        lookback_config = zone_config.get('lookback', {})
        
        # Base parameters
        self.base_zigzag_depth = zigzag_config.get('depth', 12)
        self.base_zigzag_deviation = zigzag_config.get('deviation', 5)
        self.zigzag_backstep = zigzag_config.get('backstep', 2)
        self.base_swing_strength = swing_config.get('strength', 5)
        
        # Zone criteria
        self.min_touches = zone_config.get('min_touches', 2)
        self.max_thickness_pct = zone_config.get('max_thickness_pct', 1.5) / 100  # Convert to decimal
        self.merge_tolerance_pct = 0.015  # %1.5
        self.min_quality = zone_config.get('min_quality', 4.0)
        
        # Lookback periods
        self.lookback_periods = {
            '4H': lookback_config.get('4H', 720),
            '1H': lookback_config.get('1H', 600),
            '15M': lookback_config.get('15M', 400)
        }
        
        # Adaptive multipliers (will be set per detection)
        self.current_zigzag_depth = self.base_zigzag_depth
        self.current_zigzag_deviation = self.base_zigzag_deviation
        self.current_swing_strength = self.base_swing_strength
    
    def detect_zones(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr: Optional[np.ndarray] = None,
        timeframe: str = "4H",
        method: str = "both"
    ) -> List[Zone]:
        """
        Zone detection ana method
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            atr: ATR values (for adaptive params)
            timeframe: Timeframe (4H, 1H, 15M)
            method: "zigzag", "swing", or "both"
            
        Returns:
            List of quality-scored zones
        """
        # Apply lookback period
        lookback = self.lookback_periods.get(timeframe, 600)
        high = high[-lookback:]
        low = low[-lookback:]
        close = close[-lookback:]
        if atr is not None:
            atr = atr[-lookback:]
        
        # Calculate adaptive parameters
        self._apply_adaptive_parameters(close, high, low, atr, timeframe)
        
        zones = []
        
        if method in ["zigzag", "both"]:
            zigzag_zones = self._detect_zigzag_zones(high, low, close, timeframe)
            zones.extend(zigzag_zones)
        
        if method in ["swing", "both"]:
            swing_zones = self._detect_swing_zones(high, low, timeframe)
            zones.extend(swing_zones)
        
        # Merge overlapping zones
        zones = self._merge_zones(zones, close[-1])
        
        # Filter by criteria
        zones = self._filter_zones(zones)
        
        # Quality scoring
        from .zone_quality_scorer import ZoneQualityScorer
        scorer = ZoneQualityScorer()
        
        current_index = len(close) - 1
        candle_minutes = self._get_candle_minutes(timeframe)
        
        for zone in zones:
            zone.quality = scorer.calculate_quality(
                zone=zone,
                current_index=current_index,
                candle_time_minutes=candle_minutes
            )
        
        # Filter by minimum quality
        zones = [z for z in zones if z.quality >= self.min_quality]
        
        # Sort by quality first, then proximity
        zones = sorted(zones, key=lambda z: (-z.quality, abs(z.price_mid - close[-1])))
        
        return zones
    
    def _apply_adaptive_parameters(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: Optional[np.ndarray],
        timeframe: str
    ):
        """
        Volatiliteye göre parametreleri ayarla
        """
        # ATR hesapla veya kullan
        if atr is None:
            atr = self._calculate_atr(high, low, close)
        
        current_price = close[-1]
        current_atr = atr[-1]
        atr_percent = (current_atr / current_price) * 100
        
        # ATR-based multiplier
        if atr_percent > 8.0:
            atr_mult = 1.5  # Çok volatil
        elif atr_percent > 5.0:
            atr_mult = 1.2  # Orta-yüksek
        elif atr_percent > 3.0:
            atr_mult = 1.0  # Normal
        else:
            atr_mult = 0.8  # Düşük volatilite
        
        # Timeframe multiplier
        tf_multipliers = {
            '4H': 1.5,   # Daha geniş bakış
            '1H': 1.0,   # Normal
            '15M': 0.7   # Daha dar bakış
        }
        tf_mult = tf_multipliers.get(timeframe, 1.0)
        
        # Apply multipliers
        self.current_zigzag_depth = int(self.base_zigzag_depth * atr_mult * tf_mult)
        self.current_zigzag_deviation = int(self.base_zigzag_deviation * atr_mult)
        self.current_swing_strength = int(self.base_swing_strength * atr_mult)
        
        # Apply limits
        self.current_zigzag_depth = max(5, min(30, self.current_zigzag_depth))
        self.current_zigzag_deviation = max(2, min(15, self.current_zigzag_deviation))
        self.current_swing_strength = max(3, min(15, self.current_swing_strength))
    
    def _calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Calculate ATR"""
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        
        # EMA of TR
        ema = np.zeros_like(tr, dtype=float)
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(tr)):
            ema[i] = (tr[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema
    
    def _get_candle_minutes(self, timeframe: str) -> int:
        """Timeframe'den dakika hesapla"""
        mapping = {
            "15M": 15,
            "1H": 60,
            "4H": 240,
            "1D": 1440
        }
        return mapping.get(timeframe, 240)
    
    # ═══════════════════════════════════════════════════════════
    # ZIGZAG++ DETECTION
    # ═══════════════════════════════════════════════════════════
    
    def _detect_zigzag_zones(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        timeframe: str
    ) -> List[Zone]:
        """ZigZag++ ile zone detection"""
        pivots = self._calculate_zigzag(high, low)
        
        zones = []
        for pivot in pivots:
            zone_low = pivot.price * (1 - self.max_thickness_pct / 2)
            zone_high = pivot.price * (1 + self.max_thickness_pct / 2)
            
            touches = self._count_touches(high, low, zone_low, zone_high)
            last_touch = self._find_last_touch(high, low, zone_low, zone_high, len(close) - 1)
            
            if touches >= self.min_touches:
                zone = Zone(
                    id=f"zz_{timeframe}_{pivot.index}",
                    price_low=zone_low,
                    price_high=zone_high,
                    price_mid=pivot.price,
                    touch_count=touches,
                    thickness_pct=(zone_high - zone_low) / pivot.price,
                    last_touch_index=last_touch,
                    creation_index=pivot.index,
                    timeframe=timeframe,
                    method="zigzag"
                )
                zones.append(zone)
        
        return zones
    
    def _calculate_zigzag(
        self,
        high: np.ndarray,
        low: np.ndarray
    ) -> List[ZigZagPoint]:
        """ZigZag++ pivot calculation"""
        pivots = []
        last_pivot_type = None
        last_pivot_price = 0.0
        last_pivot_index = 0
        
        depth = self.current_zigzag_depth
        deviation = self.current_zigzag_deviation / 100
        backstep = self.zigzag_backstep
        
        for i in range(depth, len(high) - depth):
            # High pivot kontrolü
            is_high_pivot = True
            for j in range(1, depth + 1):
                if high[i] <= high[i - j] or high[i] <= high[i + j]:
                    is_high_pivot = False
                    break
            
            # Low pivot kontrolü
            is_low_pivot = True
            for j in range(1, depth + 1):
                if low[i] >= low[i - j] or low[i] >= low[i + j]:
                    is_low_pivot = False
                    break
            
            # Pivot tespit edildi
            if is_high_pivot and (last_pivot_type != "high" or 
                                 (high[i] - last_pivot_price) / last_pivot_price > deviation):
                if i - last_pivot_index >= backstep:
                    pivots.append(ZigZagPoint(index=i, price=high[i], type="high"))
                    last_pivot_type = "high"
                    last_pivot_price = high[i]
                    last_pivot_index = i
            
            elif is_low_pivot and (last_pivot_type != "low" or 
                                  (last_pivot_price - low[i]) / last_pivot_price > deviation):
                if i - last_pivot_index >= backstep:
                    pivots.append(ZigZagPoint(index=i, price=low[i], type="low"))
                    last_pivot_type = "low"
                    last_pivot_price = low[i]
                    last_pivot_index = i
        
        return pivots
    
    # ═══════════════════════════════════════════════════════════
    # SWING HIGH/LOW DETECTION
    # ═══════════════════════════════════════════════════════════
    
    def _detect_swing_zones(
        self,
        high: np.ndarray,
        low: np.ndarray,
        timeframe: str
    ) -> List[Zone]:
        """Swing High/Low ile zone detection"""
        zones = []
        
        # Swing Highs
        swing_highs = self._find_swing_highs(high)
        for idx, price in swing_highs:
            zone_low = price * (1 - self.max_thickness_pct / 2)
            zone_high = price * (1 + self.max_thickness_pct / 2)
            touches = self._count_touches(high, low, zone_low, zone_high)
            last_touch = self._find_last_touch(high, low, zone_low, zone_high, len(high) - 1)
            
            if touches >= self.min_touches:
                zone = Zone(
                    id=f"sw_{timeframe}_{idx}_h",
                    price_low=zone_low,
                    price_high=zone_high,
                    price_mid=price,
                    touch_count=touches,
                    thickness_pct=(zone_high - zone_low) / price,
                    last_touch_index=last_touch,
                    creation_index=idx,
                    timeframe=timeframe,
                    method="swing"
                )
                zones.append(zone)
        
        # Swing Lows
        swing_lows = self._find_swing_lows(low)
        for idx, price in swing_lows:
            zone_low = price * (1 - self.max_thickness_pct / 2)
            zone_high = price * (1 + self.max_thickness_pct / 2)
            touches = self._count_touches(high, low, zone_low, zone_high)
            last_touch = self._find_last_touch(high, low, zone_low, zone_high, len(low) - 1)
            
            if touches >= self.min_touches:
                zone = Zone(
                    id=f"sw_{timeframe}_{idx}_l",
                    price_low=zone_low,
                    price_high=zone_high,
                    price_mid=price,
                    touch_count=touches,
                    thickness_pct=(zone_high - zone_low) / price,
                    last_touch_index=last_touch,
                    creation_index=idx,
                    timeframe=timeframe,
                    method="swing"
                )
                zones.append(zone)
        
        return zones
    
    def _find_swing_highs(self, high: np.ndarray) -> List[Tuple[int, float]]:
        """Swing High noktaları bul"""
        swings = []
        strength = self.current_swing_strength
        
        for i in range(strength, len(high) - strength):
            is_swing = True
            
            for j in range(1, strength + 1):
                if high[i] <= high[i - j] or high[i] <= high[i + j]:
                    is_swing = False
                    break
            
            if is_swing:
                swings.append((i, high[i]))
        
        return swings
    
    def _find_swing_lows(self, low: np.ndarray) -> List[Tuple[int, float]]:
        """Swing Low noktaları bul"""
        swings = []
        strength = self.current_swing_strength
        
        for i in range(strength, len(low) - strength):
            is_swing = True
            
            for j in range(1, strength + 1):
                if low[i] >= low[i - j] or low[i] >= low[i + j]:
                    is_swing = False
                    break
            
            if is_swing:
                swings.append((i, low[i]))
        
        return swings
    
    # ═══════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════
    
    def _count_touches(
        self,
        high: np.ndarray,
        low: np.ndarray,
        zone_low: float,
        zone_high: float
    ) -> int:
        """Zone'a kaç kez dokunulmuş?"""
        touches = 0
        
        for i in range(len(high)):
            if low[i] <= zone_high and high[i] >= zone_low:
                touches += 1
        
        return touches
    
    def _find_last_touch(
        self,
        high: np.ndarray,
        low: np.ndarray,
        zone_low: float,
        zone_high: float,
        current_index: int
    ) -> int:
        """Zone'un son dokunma index'i"""
        last_touch = 0
        
        for i in range(len(high)):
            if low[i] <= zone_high and high[i] >= zone_low:
                last_touch = i
        
        return last_touch
    
    def _merge_zones(self, zones: List[Zone], current_price: float) -> List[Zone]:
        """Çakışan zone'ları birleştir"""
        if not zones:
            return zones
        
        zones = sorted(zones, key=lambda z: z.price_mid)
        
        merged = []
        current = zones[0]
        
        for next_zone in zones[1:]:
            overlap = min(current.price_high, next_zone.price_high) - max(current.price_low, next_zone.price_low)
            overlap_pct = overlap / current_price
            
            if overlap > 0 and overlap_pct > self.merge_tolerance_pct:
                current = Zone(
                    id=f"merged_{current.id}_{next_zone.id}",
                    price_low=min(current.price_low, next_zone.price_low),
                    price_high=max(current.price_high, next_zone.price_high),
                    price_mid=(current.price_mid + next_zone.price_mid) / 2,
                    touch_count=max(current.touch_count, next_zone.touch_count),
                    thickness_pct=(max(current.price_high, next_zone.price_high) - 
                                 min(current.price_low, next_zone.price_low)) / current_price,
                    last_touch_index=max(current.last_touch_index, next_zone.last_touch_index),
                    creation_index=min(current.creation_index, next_zone.creation_index),
                    timeframe=current.timeframe,
                    method="both" if current.method != next_zone.method else current.method
                )
            else:
                merged.append(current)
                current = next_zone
        
        merged.append(current)
        return merged
    
    def _filter_zones(self, zones: List[Zone]) -> List[Zone]:
        """Kriterlere uymayan zone'ları filtrele"""
        filtered = []
        
        for zone in zones:
            if zone.touch_count < self.min_touches:
                continue
            
            if zone.thickness_pct > self.max_thickness_pct:
                continue
            
            filtered.append(zone)
        
        return filtered


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random
    
    np.random.seed(42)
    n = 500
    
    # Config örneği
    config = {
        'zones': {
            'zigzag': {
                'depth': 12,
                'deviation': 5,
                'backstep': 2
            },
            'swing': {
                'strength': 5
            },
            'min_touches': 2,
            'max_thickness_pct': 1.5,
            'min_quality': 4,
            'lookback': {
                '4H': 720,
                '1H': 600,
                '15M': 400
            }
        }
    }
    
    # Test data
    base = 50000 + np.cumsum(np.random.randn(n) * 50)
    high = base + np.random.rand(n) * 100 + 50
    low = base - np.random.rand(n) * 100 - 50
    close = (high + low) / 2 + np.random.randn(n) * 20
    
    # Detector oluştur
    detector = ZoneDetector(config)
    
    # Zone tespit et
    zones = detector.detect_zones(high, low, close, timeframe="4H", method="both")
    
    print(f"\n✅ Found {len(zones)} zones")
    print(f"Current price: ${close[-1]:,.2f}\n")
    
    for i, zone in enumerate(zones[:5], 1):
        print(f"[{i}] {zone.id}")
        print(f"    Range: ${zone.price_low:,.0f} - ${zone.price_high:,.0f}")
        print(f"    Quality: {zone.quality:.1f}/10")
        print(f"    Touches: {zone.touch_count}, Method: {zone.method}\n")