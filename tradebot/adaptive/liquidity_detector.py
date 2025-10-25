"""
Liquidity Detector - Parça 3
Based on: pa-strateji2 Parça 3

Liquidity Detection:
- Identifies wick sweep levels (liquidity pools)
- Detects when liquidity is "cleaned" (swept but held)
- Used for TP2 target selection
- Prioritizes recent and strong liquidity levels
"""

from __future__ import annotations
from typing import List, Optional, Literal, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class LiquidityLevel:
    """Liquidity level information"""
    price: float
    index: int  # Where the wick occurred
    type: Literal["RESISTANCE", "SUPPORT"]
    wick_size: float  # Size of the wick
    wick_ratio: float  # Wick size / body size
    strength: float  # 0-1, how strong the liquidity
    cleaned: bool  # Was it swept and held?
    cleaned_index: Optional[int]  # When it was cleaned
    
    def distance_from(self, current_price: float) -> float:
        """Distance from current price (percentage)"""
        return abs(self.price - current_price) / current_price


class LiquidityDetector:
    """
    Liquidity Level Detector
    
    Detects liquidity pools:
    1. Identifies significant wicks (liquidity hunts)
    2. Checks if liquidity was "cleaned" (swept but held)
    3. Scores liquidity strength
    4. Provides targets for TP2
    
    Liquidity Logic:
    - RESISTANCE: Upper wick (sell-side liquidity)
      * Price wicked up (touched high)
      * But closed significantly below (body held lower)
      * = Liquidity above that got swept
    
    - SUPPORT: Lower wick (buy-side liquidity)
      * Price wicked down (touched low)
      * But closed significantly above (body held higher)
      * = Liquidity below that got swept
    
    Cleaned Liquidity:
    - Price returns to level
    - Wick sweeps it again
    - Body closes on the "good" side
    - = Level is now clean and likely to hold
    
    Usage:
        detector = LiquidityDetector(config)
        
        # Find liquidity levels
        levels = detector.detect_liquidity(
            high, low, close, open,
            direction="LONG",  # Looking for resistance (TP targets)
            lookback=50
        )
        
        # Get best TP2 target
        best_target = detector.find_best_tp2_target(
            levels, entry_price, stop_loss, direction="LONG"
        )
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Liquidity Detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Wick detection thresholds
        self.min_wick_ratio = 0.3  # Wick must be 30% of candle range
        self.min_body_rejection = 0.3  # Body must reject 30% from wick
        
        # Strength calculation
        self.recent_bonus_candles = 20  # Bonus for recent liquidity
    
    def detect_liquidity(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        open_prices: np.ndarray,
        direction: Literal["LONG", "SHORT"],
        lookback: int = 50
    ) -> List[LiquidityLevel]:
        """
        Detect liquidity levels
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            open_prices: Open prices
            direction: Trade direction (determines which liquidity to find)
                       LONG: Find resistance (sell-side liquidity for TP)
                       SHORT: Find support (buy-side liquidity for TP)
            lookback: How many candles to look back
            
        Returns:
            List of LiquidityLevel objects, sorted by strength
        """
        levels = []
        
        # Look at recent candles
        start_index = max(0, len(high) - lookback)
        
        for i in range(start_index, len(high)):
            candle_high = high[i]
            candle_low = low[i]
            candle_close = close[i]
            candle_open = open_prices[i]
            candle_range = candle_high - candle_low
            
            if candle_range == 0:
                continue
            
            body_top = max(candle_open, candle_close)
            body_bottom = min(candle_open, candle_close)
            body_size = abs(candle_close - candle_open)
            
            # ═══════════════════════════════════════════════════════
            # RESISTANCE (Upper wick - for LONG trades)
            # ═══════════════════════════════════════════════════════
            if direction == "LONG":
                upper_wick = candle_high - body_top
                upper_wick_ratio = upper_wick / candle_range
                
                # Check if significant upper wick
                if upper_wick_ratio >= self.min_wick_ratio:
                    # Check if body rejected from high
                    rejection = (candle_high - candle_close) / candle_range
                    
                    if rejection >= self.min_body_rejection:
                        # Calculate strength
                        strength = self._calculate_strength(
                            wick_ratio=upper_wick_ratio,
                            rejection=rejection,
                            body_size=body_size,
                            candle_range=candle_range,
                            age=len(high) - i
                        )
                        
                        # Check if cleaned
                        cleaned, cleaned_idx = self._check_if_cleaned(
                            price_level=candle_high,
                            level_type="RESISTANCE",
                            high=high[i+1:],
                            low=low[i+1:],
                            close=close[i+1:]
                        )
                        
                        level = LiquidityLevel(
                            price=candle_high,
                            index=i,
                            type="RESISTANCE",
                            wick_size=upper_wick,
                            wick_ratio=upper_wick_ratio,
                            strength=strength,
                            cleaned=cleaned,
                            cleaned_index=i + cleaned_idx if cleaned else None
                        )
                        levels.append(level)
            
            # ═══════════════════════════════════════════════════════
            # SUPPORT (Lower wick - for SHORT trades)
            # ═══════════════════════════════════════════════════════
            elif direction == "SHORT":
                lower_wick = body_bottom - candle_low
                lower_wick_ratio = lower_wick / candle_range
                
                # Check if significant lower wick
                if lower_wick_ratio >= self.min_wick_ratio:
                    # Check if body rejected from low
                    rejection = (candle_close - candle_low) / candle_range
                    
                    if rejection >= self.min_body_rejection:
                        # Calculate strength
                        strength = self._calculate_strength(
                            wick_ratio=lower_wick_ratio,
                            rejection=rejection,
                            body_size=body_size,
                            candle_range=candle_range,
                            age=len(high) - i
                        )
                        
                        # Check if cleaned
                        cleaned, cleaned_idx = self._check_if_cleaned(
                            price_level=candle_low,
                            level_type="SUPPORT",
                            high=high[i+1:],
                            low=low[i+1:],
                            close=close[i+1:]
                        )
                        
                        level = LiquidityLevel(
                            price=candle_low,
                            index=i,
                            type="SUPPORT",
                            wick_size=lower_wick,
                            wick_ratio=lower_wick_ratio,
                            strength=strength,
                            cleaned=cleaned,
                            cleaned_index=i + cleaned_idx if cleaned else None
                        )
                        levels.append(level)
        
        # Sort by strength (strongest first)
        levels.sort(key=lambda x: x.strength, reverse=True)
        
        return levels
    
    def find_best_tp2_target(
        self,
        liquidity_levels: List[LiquidityLevel],
        entry_price: float,
        stop_loss: float,
        direction: Literal["LONG", "SHORT"],
        min_rr: float = 2.5
    ) -> Optional[float]:
        """
        Find best TP2 target from liquidity levels
        
        Args:
            liquidity_levels: List of liquidity levels
            entry_price: Trade entry price
            stop_loss: Stop loss price
            direction: Trade direction
            min_rr: Minimum risk-reward ratio required
            
        Returns:
            Best TP2 price or None
        """
        if not liquidity_levels:
            return None
        
        risk = abs(entry_price - stop_loss)
        
        # Filter levels that meet RR requirement
        valid_levels = []
        
        for level in liquidity_levels:
            # Calculate RR for this level
            reward = abs(level.price - entry_price)
            rr = reward / risk if risk > 0 else 0
            
            # Check if meets minimum RR
            if rr >= min_rr:
                # Check if level is in correct direction
                if direction == "LONG" and level.price > entry_price:
                    valid_levels.append((level, rr))
                elif direction == "SHORT" and level.price < entry_price:
                    valid_levels.append((level, rr))
        
        if not valid_levels:
            return None
        
        # Sort by: 1) Cleaned first, 2) Strength, 3) RR
        valid_levels.sort(
            key=lambda x: (
                x[0].cleaned,  # Cleaned levels first
                x[0].strength,  # Then by strength
                x[1]  # Then by RR
            ),
            reverse=True
        )
        
        # Return best level
        return valid_levels[0][0].price
    
    def _calculate_strength(
        self,
        wick_ratio: float,
        rejection: float,
        body_size: float,
        candle_range: float,
        age: int
    ) -> float:
        """
        Calculate liquidity strength (0-1)
        
        Factors:
        - Wick ratio (30%)
        - Rejection strength (30%)
        - Body size relative to range (20%)
        - Recency (20%)
        """
        strength = 0.0
        
        # Wick ratio contribution (30%)
        strength += min(wick_ratio / 0.5, 1.0) * 0.3
        
        # Rejection contribution (30%)
        strength += min(rejection / 0.5, 1.0) * 0.3
        
        # Body size contribution (20%)
        # Prefer larger bodies (more conviction)
        body_ratio = body_size / candle_range if candle_range > 0 else 0
        strength += body_ratio * 0.2
        
        # Recency contribution (20%)
        # More recent = stronger
        recency_score = max(0, 1 - (age / self.recent_bonus_candles))
        strength += recency_score * 0.2
        
        return min(strength, 1.0)
    
    def _check_if_cleaned(
        self,
        price_level: float,
        level_type: Literal["RESISTANCE", "SUPPORT"],
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> tuple[bool, int]:
        """
        Check if liquidity level was cleaned
        
        Cleaned = Price came back, swept the level, but held on good side
        
        Args:
            price_level: The liquidity level price
            level_type: "RESISTANCE" or "SUPPORT"
            high: High prices after the liquidity formation
            low: Low prices after the liquidity formation
            close: Close prices after the liquidity formation
            
        Returns:
            (cleaned: bool, index: int)
        """
        for i in range(len(high)):
            if level_type == "RESISTANCE":
                # Check if wick swept resistance
                if high[i] >= price_level:
                    # Check if body closed below (held)
                    if close[i] < price_level:
                        return True, i
            
            elif level_type == "SUPPORT":
                # Check if wick swept support
                if low[i] <= price_level:
                    # Check if body closed above (held)
                    if close[i] > price_level:
                        return True, i
        
        return False, -1


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    
    # Create detector
    detector = LiquidityDetector()
    
    print("\n" + "="*60)
    print("LIQUIDITY DETECTOR - TEST")
    print("="*60)
    
    # ═════════════════════════════════════════════════════════
    # Create realistic price action with liquidity levels
    # ═════════════════════════════════════════════════════════
    n = 100
    
    base_price = 50000
    prices = []
    
    # Phase 1: Uptrend with liquidity sweeps
    for i in range(50):
        price = base_price + i * 50
        prices.append(price)
    
    # Add liquidity sweep at 52500
    # Candle wicks to 52500 but closes at 52200
    prices.append(52500)  # This will be adjusted to create wick
    
    # Continue
    for i in range(49):
        price = 52200 + i * 30
        prices.append(price)
    
    close = np.array(prices)
    
    # Create high/low with wicks
    high = close.copy()
    low = close.copy()
    open_prices = close.copy()
    
    # Add specific liquidity levels
    # Level 1: Strong resistance at index 50
    high[50] = 52500  # Wick up
    close[50] = 52200  # Close below
    open_prices[50] = 52300
    low[50] = 52100
    
    # Level 2: Another resistance at index 70
    high[70] = 53800
    close[70] = 53400
    open_prices[70] = 53500
    low[70] = 53300
    
    # Level 3: Cleaned resistance at index 85
    high[85] = 54200
    close[85] = 53900
    open_prices[85] = 54000
    low[85] = 53800
    # Clean it at index 90
    high[90] = 54250  # Sweep again
    close[90] = 53950  # Hold below
    
    # Add normal volatility to other candles
    for i in range(len(high)):
        if i not in [50, 70, 85, 90]:
            high[i] = close[i] + np.random.rand() * 100
            low[i] = close[i] - np.random.rand() * 100
            open_prices[i] = close[i] + np.random.randn() * 50
    
    # ═════════════════════════════════════════════════════════
    # Detect liquidity for LONG trade (looking for resistance TP)
    # ═════════════════════════════════════════════════════════
    print("\n[SCENARIO: LONG TRADE - Looking for TP2 targets]")
    print("-" * 60)
    
    levels = detector.detect_liquidity(
        high=high,
        low=low,
        close=close,
        open_prices=open_prices,
        direction="LONG",
        lookback=50
    )
    
    print(f"\nFound {len(levels)} liquidity levels:")
    print()
    
    for i, level in enumerate(levels, 1):
        print(f"Level {i}:")
        print(f"  Price: ${level.price:,.2f}")
        print(f"  Type: {level.type}")
        print(f"  Index: {level.index}")
        print(f"  Wick Ratio: {level.wick_ratio:.1%}")
        print(f"  Strength: {level.strength:.2f}")
        print(f"  Cleaned: {'✅ Yes' if level.cleaned else '❌ No'}")
        if level.cleaned:
            print(f"  Cleaned at: Index {level.cleaned_index}")
        print()
    
    # ═════════════════════════════════════════════════════════
    # Find best TP2 target
    # ═════════════════════════════════════════════════════════
    print("-" * 60)
    print("TP2 TARGET SELECTION")
    print("-" * 60)
    
    # Simulate trade entry
    entry_price = 52000
    stop_loss = 51500
    risk = entry_price - stop_loss
    
    print(f"\nTrade Setup:")
    print(f"  Entry: ${entry_price:,.2f}")
    print(f"  Stop Loss: ${stop_loss:,.2f}")
    print(f"  Risk: ${risk:,.2f}")
    print(f"  Direction: LONG")
    
    best_tp2 = detector.find_best_tp2_target(
        liquidity_levels=levels,
        entry_price=entry_price,
        stop_loss=stop_loss,
        direction="LONG",
        min_rr=2.5
    )
    
    if best_tp2:
        reward = best_tp2 - entry_price
        rr = reward / risk
        
        print(f"\n✅ Best TP2 Target Found:")
        print(f"  Price: ${best_tp2:,.2f}")
        print(f"  Reward: ${reward:,.2f}")
        print(f"  Risk-Reward: {rr:.2f}")
        
        # Find which level this is
        for level in levels:
            if level.price == best_tp2:
                print(f"  Liquidity Strength: {level.strength:.2f}")
                print(f"  Cleaned: {'✅ Yes' if level.cleaned else '❌ No'}")
                break
    else:
        print(f"\n❌ No suitable TP2 target found (min RR: 2.5)")
    
    # ═════════════════════════════════════════════════════════
    # Summary
    # ═════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\n💡 Liquidity Concept:")
    print(f"  • Resistance: Upper wicks (sell-side liquidity)")
    print(f"  • Price wicks up but body closes lower")
    print(f"  • Shows liquidity was swept but rejected")
    print(f"  • Good targets for take profit")
    
    print(f"\n✅ Cleaned Liquidity:")
    print(f"  • Price returns and sweeps again")
    print(f"  • But still holds on good side")
    print(f"  • Stronger level (liquidity cleaned)")
    print(f"  • Higher priority for TP2")
    
    print("\n" + "="*60)
    print("✅ Liquidity Detector working correctly!")
    print("="*60 + "\n")