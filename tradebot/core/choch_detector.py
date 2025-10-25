"""
ChoCH Detector - Parça 2
Based on: pa-strateji2 Parça 2

Change of Character (ChoCH) Detection:
- Market structure break detection
- LONG ChoCH: Lower structure breaks upward (Last LH broken)
- SHORT ChoCH: Higher structure breaks downward (Last HL broken)
- Strength calculation (body distance + volume)
- Body close confirmation required
"""

from __future__ import annotations
from typing import Literal, Optional, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ChoCHResult:
    """ChoCH detection result"""
    detected: bool
    direction: Optional[Literal["LONG", "SHORT"]]
    breakout_price: float
    breakout_index: int
    broken_level: float  # Last LH or HL that was broken
    strength: float  # 0.0-1.0
    body_score: float
    volume_score: float
    
    def get_ml_features(self) -> Dict[str, float]:
        """ML features for RL agent"""
        return {
            'choch_detected': float(self.detected),
            'choch_strength': self.strength if self.detected else 0.0,
            'choch_body_score': self.body_score if self.detected else 0.0,
            'choch_volume_score': self.volume_score if self.detected else 0.0,
            'choch_direction': (
                1.0 if self.direction == "LONG"
                else -1.0 if self.direction == "SHORT"
                else 0.0
            ),
        }


@dataclass
class SwingPoint:
    """Swing High or Swing Low point"""
    index: int
    price: float
    type: Literal["HIGH", "LOW"]


class ChoCHDetector:
    """
    Change of Character Detector
    
    Detects market structure breaks:
    - LONG: Downtrend structure breaks (LL-LH pattern breaks upward)
    - SHORT: Uptrend structure breaks (HH-HL pattern breaks downward)
    
    Rules:
    1. Identify market structure (Higher/Lower points)
    2. Find last significant High/Low
    3. Wait for breakout with body close
    4. Calculate strength (body + volume)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ChoCH Detector
        
        Args:
            config: Configuration dictionary
        """
        entry_config = config.get('entry', {}) if config else {}
        choch_config = entry_config.get('choch', {})
        
        self.min_strength = choch_config.get('min_strength', 0.4)
        self.swing_lookback = 50  # Look back for swing points
        self.min_swing_gap = 5    # Minimum candles between swings
    
    def detect(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        open_prices: np.ndarray,
        volume: np.ndarray,
        direction: Literal["LONG", "SHORT"]
    ) -> ChoCHResult:
        """
        Detect ChoCH for given direction
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            open_prices: Open prices
            volume: Volume data
            direction: Direction to check ("LONG" or "SHORT")
            
        Returns:
            ChoCHResult
        """
        if direction == "LONG":
            return self._detect_long_choch(high, low, close, open_prices, volume)
        else:
            return self._detect_short_choch(high, low, close, open_prices, volume)
    
    def _detect_long_choch(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        open_prices: np.ndarray,
        volume: np.ndarray
    ) -> ChoCHResult:
        """
        Detect LONG ChoCH (Downtrend break)
        
        Logic:
        1. Find Lower Highs (LH) in downtrend
        2. Get last LH (most recent high in down structure)
        3. Check if new candle breaks above last LH with BODY CLOSE
        4. Calculate strength
        """
        # Find swing highs (potential Lower Highs)
        swing_highs = self._find_swing_highs(high, low)
        
        if len(swing_highs) < 2:
            return self._no_choch_result()
        
        # Get last Lower High
        # In downtrend: each high should be lower than previous
        last_lh = self._find_last_lower_high(swing_highs)
        
        if last_lh is None:
            return self._no_choch_result()
        
        last_lh_price = last_lh.price
        last_lh_index = last_lh.index
        
        # Check recent candles for breakout
        # Look at candles after last LH
        for i in range(last_lh_index + 1, len(close)):
            candle_high = high[i]
            candle_close = close[i]
            
            # Breakout conditions:
            # 1. High touched/passed the last LH (wick OK)
            # 2. CLOSE above last LH (body close required!)
            if candle_high >= last_lh_price and candle_close > last_lh_price:
                # ChoCH detected!
                strength = self._calculate_strength(
                    breakout_candle_index=i,
                    broken_level=last_lh_price,
                    high=high,
                    low=low,
                    close=close,
                    open_prices=open_prices,
                    volume=volume
                )
                
                return ChoCHResult(
                    detected=True,
                    direction="LONG",
                    breakout_price=candle_close,
                    breakout_index=i,
                    broken_level=last_lh_price,
                    strength=strength['total'],
                    body_score=strength['body_score'],
                    volume_score=strength['volume_score']
                )
        
        # No ChoCH found
        return self._no_choch_result()
    
    def _detect_short_choch(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        open_prices: np.ndarray,
        volume: np.ndarray
    ) -> ChoCHResult:
        """
        Detect SHORT ChoCH (Uptrend break)
        
        Logic:
        1. Find Higher Lows (HL) in uptrend
        2. Get last HL (most recent low in up structure)
        3. Check if new candle breaks below last HL with BODY CLOSE
        4. Calculate strength
        """
        # Find swing lows (potential Higher Lows)
        swing_lows = self._find_swing_lows(high, low)
        
        if len(swing_lows) < 2:
            return self._no_choch_result()
        
        # Get last Higher Low
        last_hl = self._find_last_higher_low(swing_lows)
        
        if last_hl is None:
            return self._no_choch_result()
        
        last_hl_price = last_hl.price
        last_hl_index = last_hl.index
        
        # Check recent candles for breakout
        for i in range(last_hl_index + 1, len(close)):
            candle_low = low[i]
            candle_close = close[i]
            
            # Breakout conditions:
            # 1. Low touched/passed the last HL (wick OK)
            # 2. CLOSE below last HL (body close required!)
            if candle_low <= last_hl_price and candle_close < last_hl_price:
                # ChoCH detected!
                strength = self._calculate_strength(
                    breakout_candle_index=i,
                    broken_level=last_hl_price,
                    high=high,
                    low=low,
                    close=close,
                    open_prices=open_prices,
                    volume=volume
                )
                
                return ChoCHResult(
                    detected=True,
                    direction="SHORT",
                    breakout_price=candle_close,
                    breakout_index=i,
                    broken_level=last_hl_price,
                    strength=strength['total'],
                    body_score=strength['body_score'],
                    volume_score=strength['volume_score']
                )
        
        return self._no_choch_result()
    
    def _find_swing_highs(
        self,
        high: np.ndarray,
        low: np.ndarray,
        strength: int = 3
    ) -> List[SwingPoint]:
        """
        Find swing high points
        
        Args:
            high: High prices
            low: Low prices
            strength: Bars on each side to confirm swing
            
        Returns:
            List of swing high points
        """
        swings = []
        lookback = min(len(high), self.swing_lookback)
        start = len(high) - lookback
        
        for i in range(start + strength, len(high) - strength):
            is_swing = True
            
            # Check left and right
            for j in range(1, strength + 1):
                if high[i] <= high[i - j] or high[i] <= high[i + j]:
                    is_swing = False
                    break
            
            if is_swing:
                # Check minimum gap from last swing
                if not swings or (i - swings[-1].index) >= self.min_swing_gap:
                    swings.append(SwingPoint(
                        index=i,
                        price=high[i],
                        type="HIGH"
                    ))
        
        return swings
    
    def _find_swing_lows(
        self,
        high: np.ndarray,
        low: np.ndarray,
        strength: int = 3
    ) -> List[SwingPoint]:
        """Find swing low points"""
        swings = []
        lookback = min(len(low), self.swing_lookback)
        start = len(low) - lookback
        
        for i in range(start + strength, len(low) - strength):
            is_swing = True
            
            for j in range(1, strength + 1):
                if low[i] >= low[i - j] or low[i] >= low[i + j]:
                    is_swing = False
                    break
            
            if is_swing:
                if not swings or (i - swings[-1].index) >= self.min_swing_gap:
                    swings.append(SwingPoint(
                        index=i,
                        price=low[i],
                        type="LOW"
                    ))
        
        return swings
    
    def _find_last_lower_high(self, swing_highs: List[SwingPoint]) -> Optional[SwingPoint]:
        """
        Find last Lower High in downtrend structure
        
        In downtrend: Each high is lower than previous
        We want the most recent high
        """
        if len(swing_highs) < 2:
            return None
        
        # Check if we have a downtrend structure (LH pattern)
        # At least last 2 highs should show lower structure
        if swing_highs[-1].price < swing_highs[-2].price:
            # Confirmed lower structure
            return swing_highs[-1]  # Return most recent (last LH)
        
        # Check if previous structure shows LH
        if len(swing_highs) >= 3:
            if swing_highs[-2].price < swing_highs[-3].price:
                return swing_highs[-2]
        
        return None
    
    def _find_last_higher_low(self, swing_lows: List[SwingPoint]) -> Optional[SwingPoint]:
        """
        Find last Higher Low in uptrend structure
        
        In uptrend: Each low is higher than previous
        We want the most recent low
        """
        if len(swing_lows) < 2:
            return None
        
        # Check if we have an uptrend structure (HL pattern)
        if swing_lows[-1].price > swing_lows[-2].price:
            return swing_lows[-1]  # Return most recent (last HL)
        
        if len(swing_lows) >= 3:
            if swing_lows[-2].price > swing_lows[-3].price:
                return swing_lows[-2]
        
        return None
    
    def _calculate_strength(
        self,
        breakout_candle_index: int,
        broken_level: float,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        open_prices: np.ndarray,
        volume: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate ChoCH strength (0.0-1.0)
        
        Components:
        1. Body Distance Score (40%): How far body closed past level
        2. Volume Score (60%): Volume confirmation
        
        Returns:
            Dict with total, body_score, volume_score
        """
        i = breakout_candle_index
        
        # ═══════════════════════════════════════════════════════════
        # 1. BODY DISTANCE SCORE (0.0 - 0.4)
        # ═══════════════════════════════════════════════════════════
        candle_close = close[i]
        candle_range = abs(high[i] - low[i])
        
        if candle_range == 0:
            body_ratio = 0.0
        else:
            body_distance = abs(candle_close - broken_level)
            body_ratio = body_distance / candle_range
        
        body_score = min(body_ratio, 1.0) * 0.4
        
        # ═══════════════════════════════════════════════════════════
        # 2. VOLUME SCORE (0.0 - 0.6)
        # ═══════════════════════════════════════════════════════════
        current_volume = volume[i]
        
        # Average volume of last 20 candles (excluding current)
        lookback = min(20, i)
        if lookback > 0:
            avg_volume = np.mean(volume[i-lookback:i])
        else:
            avg_volume = current_volume
        
        if avg_volume == 0:
            volume_ratio = 1.0
        else:
            volume_ratio = current_volume / avg_volume
        
        # Normalize: 2x volume = max score (0.6)
        volume_normalized = min(volume_ratio / 2.0, 1.0)
        volume_score = volume_normalized * 0.6
        
        # ═══════════════════════════════════════════════════════════
        # TOTAL STRENGTH
        # ═══════════════════════════════════════════════════════════
        total_strength = body_score + volume_score
        
        return {
            'total': min(total_strength, 1.0),
            'body_score': body_score,
            'volume_score': volume_score
        }
    
    def _no_choch_result(self) -> ChoCHResult:
        """Return empty ChoCH result"""
        return ChoCHResult(
            detected=False,
            direction=None,
            breakout_price=0.0,
            breakout_index=-1,
            broken_level=0.0,
            strength=0.0,
            body_score=0.0,
            volume_score=0.0
        )


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    
    # Config
    config = {
        'entry': {
            'choch': {
                'min_strength': 0.4
            }
        }
    }
    
    # Create detector
    detector = ChoCHDetector(config)
    
    # ═════════════════════════════════════════════════════════
    # TEST 1: LONG ChoCH (Downtrend break)
    # ═════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("TEST 1: LONG ChoCH (Downtrend Structure Break)")
    print("="*60)
    
    # Simulate downtrend: LL-LH pattern
    n = 100
    
    # Create downtrend with LL-LH structure
    prices = []
    highs = []
    lows = []
    
    # First swing: LL at 1600, LH at 2000
    prices.extend(np.linspace(2000, 1600, 20))
    prices.extend(np.linspace(1600, 2000, 15))
    
    # Second swing: LL at 800, LH at 1500
    prices.extend(np.linspace(2000, 800, 20))
    prices.extend(np.linspace(800, 1500, 15))
    
    # Third swing: LL at 700 (zone entry)
    prices.extend(np.linspace(1500, 700, 15))
    
    # ChoCH: Break above last LH (1500)
    prices.extend(np.linspace(700, 1600, 15))  # Break 1500!
    
    close = np.array(prices)
    high = close + np.random.rand(len(close)) * 50
    low = close - np.random.rand(len(close)) * 50
    open_prices = close + np.random.randn(len(close)) * 20
    volume = np.random.rand(len(close)) * 1000 + 500
    
    # Detect LONG ChoCH
    result = detector.detect(high, low, close, open_prices, volume, direction="LONG")
    
    print(f"\nChoCH Detected: {result.detected}")
    if result.detected:
        print(f"Direction: {result.direction}")
        print(f"Breakout Price: ${result.breakout_price:.2f}")
        print(f"Broken Level (Last LH): ${result.broken_level:.2f}")
        print(f"Strength: {result.strength:.2f}")
        print(f"  ├─ Body Score: {result.body_score:.2f}")
        print(f"  └─ Volume Score: {result.volume_score:.2f}")
        
        if result.strength >= 0.4:
            print(f"\n✅ ChoCH CONFIRMED (Strength >= 0.4)")
        else:
            print(f"\n⚠️  ChoCH WEAK (Strength < 0.4)")
    
    # ML Features
    print("\n" + "="*60)
    print("ML FEATURES")
    print("="*60)
    features = result.get_ml_features()
    for key, value in features.items():
        print(f"{key:25s}: {value:.4f}")
    
    print("\n" + "="*60 + "\n")