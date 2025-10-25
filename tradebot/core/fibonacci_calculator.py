"""
Fibonacci Calculator - Parça 2
Based on: pa-strateji2 Parça 2

Fibonacci Retracement Levels:
- 0.705 (Optimal Entry Point - OEP)
- 0.618 (Golden Ratio)
- Calculate from ChoCH swing range
- LONG: Calculate from swing low to ChoCH high
- SHORT: Calculate from swing high to ChoCH low
"""

from __future__ import annotations
from typing import Literal, Optional, Dict, List
from dataclasses import dataclass
import numpy as np


@dataclass
class FibonacciLevels:
    """Fibonacci retracement levels"""
    swing_low: float
    swing_high: float
    swing_range: float
    direction: Literal["LONG", "SHORT"]
    
    # Fibonacci levels
    fib_0: float      # 0% (ChoCH breakout point)
    fib_0705: float   # 70.5% retracement (OEP)
    fib_0618: float   # 61.8% retracement (Golden)
    fib_1: float      # 100% (Swing extreme)
    
    # Level info
    primary_level: float     # 0.705 (priority)
    secondary_level: float   # 0.618 (backup)
    
    def get_ml_features(self) -> Dict[str, float]:
        """ML features for RL agent"""
        return {
            'fib_swing_range': self.swing_range,
            'fib_0705_level': self.fib_0705,
            'fib_0618_level': self.fib_0618,
            'fib_levels_distance': abs(self.fib_0705 - self.fib_0618),
            'fib_direction': 1.0 if self.direction == "LONG" else -1.0,
        }
    
    def is_price_at_level(
        self,
        current_price: float,
        level: Literal["0.705", "0.618"],
        tolerance: float = 0.002  # 0.2% tolerance
    ) -> bool:
        """
        Check if current price touched a Fibonacci level
        
        Args:
            current_price: Current price to check
            level: Which level to check ("0.705" or "0.618")
            tolerance: Price tolerance as percentage (default 0.2%)
            
        Returns:
            True if price at level
        """
        if level == "0.705":
            target_price = self.fib_0705
        else:
            target_price = self.fib_0618
        
        distance_pct = abs(current_price - target_price) / target_price
        
        return distance_pct <= tolerance
    
    def get_entry_signal(
        self,
        current_price: float,
        tolerance: float = 0.002
    ) -> Optional[Dict[str, any]]:
        """
        Check if price gives entry signal at any Fib level
        
        Priority:
        1. Check 0.705 first (primary)
        2. If missed, check 0.618 (secondary)
        
        Returns:
            Entry signal dict or None
        """
        # Priority 1: 0.705 level
        if self.is_price_at_level(current_price, "0.705", tolerance):
            return {
                'entry': True,
                'price': self.fib_0705,
                'level': '0.705',
                'quality': 'EXCELLENT'
            }
        
        # Priority 2: 0.618 level
        if self.is_price_at_level(current_price, "0.618", tolerance):
            return {
                'entry': True,
                'price': self.fib_0618,
                'level': '0.618',
                'quality': 'GOOD'
            }
        
        return None
    
    def missed_both_levels(self, current_price: float) -> bool:
        """
        Check if price missed both Fibonacci levels
        
        LONG: Price went below both levels
        SHORT: Price went above both levels
        
        Returns:
            True if both levels missed
        """
        if self.direction == "LONG":
            # For LONG, price retraces DOWN to Fib levels
            # Missed = went below both levels
            return current_price < min(self.fib_0705, self.fib_0618)
        else:
            # For SHORT, price retraces UP to Fib levels
            # Missed = went above both levels
            return current_price > max(self.fib_0705, self.fib_0618)


class FibonacciCalculator:
    """
    Fibonacci Retracement Calculator
    
    Calculates Fibonacci levels after ChoCH:
    - LONG: From swing low (before ChoCH) to ChoCH high
    - SHORT: From swing high (before ChoCH) to ChoCH low
    
    Levels:
    - 0.705: Optimal Entry Point (OEP) - Priority 1
    - 0.618: Golden Ratio - Priority 2
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Fibonacci Calculator
        
        Args:
            config: Configuration dictionary
        """
        entry_config = config.get('entry', {}) if config else {}
        fib_config = entry_config.get('fibonacci', {})
        
        # Fibonacci levels to use
        self.levels = fib_config.get('levels', [0.705, 0.618])
        
        # Default levels
        self.primary_level = 0.705   # OEP
        self.secondary_level = 0.618  # Golden
    
    def calculate(
        self,
        direction: Literal["LONG", "SHORT"],
        choch_breakout_price: float,
        swing_extreme_price: float
    ) -> FibonacciLevels:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            direction: Trade direction ("LONG" or "SHORT")
            choch_breakout_price: ChoCH breakout candle extreme
                                  LONG: High of breakout candle
                                  SHORT: Low of breakout candle
            swing_extreme_price: Last swing extreme before ChoCH
                                LONG: Swing low before ChoCH
                                SHORT: Swing high before ChoCH
        
        Returns:
            FibonacciLevels object
        """
        if direction == "LONG":
            return self._calculate_long_fib(
                choch_high=choch_breakout_price,
                swing_low=swing_extreme_price
            )
        else:
            return self._calculate_short_fib(
                choch_low=choch_breakout_price,
                swing_high=swing_extreme_price
            )
    
    def _calculate_long_fib(
        self,
        choch_high: float,
        swing_low: float
    ) -> FibonacciLevels:
        """
        Calculate Fibonacci for LONG setup
        
        After bullish ChoCH:
        - Swing Low: Last low before ChoCH (100% level)
        - ChoCH High: Breakout candle high (0% level)
        - Price retraces DOWN from ChoCH high toward swing low
        
        Formula:
        Fib Level = Swing Low + (Range * (1 - Fib %))
        
        Example:
        Swing Low: $700
        ChoCH High: $1600
        Range: $900
        
        Fib 0.705 = $700 + ($900 * 0.295) = $965.50
        Fib 0.618 = $700 + ($900 * 0.382) = $1043.80
        """
        swing_range = choch_high - swing_low
        
        # Calculate levels (price retraces from high to low)
        fib_0 = choch_high  # 0% = ChoCH high (no retracement)
        fib_0705 = swing_low + (swing_range * (1 - 0.705))  # 29.5% from swing low
        fib_0618 = swing_low + (swing_range * (1 - 0.618))  # 38.2% from swing low
        fib_1 = swing_low   # 100% = Full retracement to swing low
        
        return FibonacciLevels(
            swing_low=swing_low,
            swing_high=choch_high,
            swing_range=swing_range,
            direction="LONG",
            fib_0=fib_0,
            fib_0705=fib_0705,
            fib_0618=fib_0618,
            fib_1=fib_1,
            primary_level=fib_0705,
            secondary_level=fib_0618
        )
    
    def _calculate_short_fib(
        self,
        choch_low: float,
        swing_high: float
    ) -> FibonacciLevels:
        """
        Calculate Fibonacci for SHORT setup
        
        After bearish ChoCH:
        - Swing High: Last high before ChoCH (100% level)
        - ChoCH Low: Breakout candle low (0% level)
        - Price retraces UP from ChoCH low toward swing high
        
        Formula:
        Fib Level = Swing High - (Range * (1 - Fib %))
        
        Example:
        Swing High: $2000
        ChoCH Low: $1100
        Range: $900
        
        Fib 0.705 = $2000 - ($900 * 0.295) = $1734.50
        Fib 0.618 = $2000 - ($900 * 0.382) = $1656.20
        """
        swing_range = swing_high - choch_low
        
        # Calculate levels (price retraces from low to high)
        fib_0 = choch_low   # 0% = ChoCH low (no retracement)
        fib_0705 = swing_high - (swing_range * (1 - 0.705))
        fib_0618 = swing_high - (swing_range * (1 - 0.618))
        fib_1 = swing_high  # 100% = Full retracement to swing high
        
        return FibonacciLevels(
            swing_low=choch_low,
            swing_high=swing_high,
            swing_range=swing_range,
            direction="SHORT",
            fib_0=fib_0,
            fib_0705=fib_0705,
            fib_0618=fib_0618,
            fib_1=fib_1,
            primary_level=fib_0705,
            secondary_level=fib_0618
        )
    
    def find_swing_extreme(
        self,
        high: np.ndarray,
        low: np.ndarray,
        choch_index: int,
        direction: Literal["LONG", "SHORT"],
        lookback: int = 30
    ) -> float:
        """
        Find swing extreme before ChoCH
        
        Args:
            high: High prices
            low: Low prices
            choch_index: Index where ChoCH occurred
            direction: Trade direction
            lookback: How many candles to look back
            
        Returns:
            Swing extreme price (low for LONG, high for SHORT)
        """
        start_index = max(0, choch_index - lookback)
        
        if direction == "LONG":
            # Find lowest low before ChoCH
            swing_extreme = np.min(low[start_index:choch_index])
        else:
            # Find highest high before ChoCH
            swing_extreme = np.max(high[start_index:choch_index])
        
        return swing_extreme


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Config
    config = {
        'entry': {
            'fibonacci': {
                'levels': [0.705, 0.618]
            }
        }
    }
    
    # Create calculator
    calc = FibonacciCalculator(config)
    
    print("\n" + "="*60)
    print("FIBONACCI CALCULATOR - EXAMPLES")
    print("="*60)
    
    # ═════════════════════════════════════════════════════════
    # EXAMPLE 1: LONG Setup
    # ═════════════════════════════════════════════════════════
    print("\n[EXAMPLE 1: LONG SETUP]")
    print("-" * 60)
    
    # Scenario from strategy:
    # Swing Low: $700 (LL before ChoCH)
    # ChoCH High: $1600 (breakout candle high)
    
    long_fib = calc.calculate(
        direction="LONG",
        choch_breakout_price=1600,
        swing_extreme_price=700
    )
    
    print(f"\nSwing Range:")
    print(f"  Swing Low (100%): ${long_fib.swing_low:,.2f}")
    print(f"  ChoCH High (0%):  ${long_fib.swing_high:,.2f}")
    print(f"  Range: ${long_fib.swing_range:,.2f}")
    
    print(f"\nFibonacci Levels (Price retraces DOWN):")
    print(f"  0% (ChoCH High):  ${long_fib.fib_0:,.2f}")
    print(f"  70.5% (OEP):      ${long_fib.fib_0705:,.2f} ⭐ PRIMARY")
    print(f"  61.8% (Golden):   ${long_fib.fib_0618:,.2f} ✓ SECONDARY")
    print(f"  100% (Swing Low): ${long_fib.fib_1:,.2f}")
    
    # Test entry signals
    print(f"\nEntry Signal Tests:")
    
    test_prices = [965, 1044, 1200, 650]
    for price in test_prices:
        signal = long_fib.get_entry_signal(price, tolerance=0.002)
        if signal:
            print(f"  ${price:,.0f} → ✅ ENTRY at {signal['level']} ({signal['quality']})")
        else:
            missed = long_fib.missed_both_levels(price)
            if missed:
                print(f"  ${price:,.0f} → ❌ MISSED both levels (Cancel setup)")
            else:
                print(f"  ${price:,.0f} → ⏳ Waiting for retracement")
    
    # ═════════════════════════════════════════════════════════
    # EXAMPLE 2: SHORT Setup
    # ═════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("[EXAMPLE 2: SHORT SETUP]")
    print("-" * 60)
    
    # Scenario:
    # Swing High: $2000 (HH before ChoCH)
    # ChoCH Low: $1100 (breakout candle low)
    
    short_fib = calc.calculate(
        direction="SHORT",
        choch_breakout_price=1100,
        swing_extreme_price=2000
    )
    
    print(f"\nSwing Range:")
    print(f"  Swing High (100%): ${short_fib.swing_high:,.2f}")
    print(f"  ChoCH Low (0%):    ${short_fib.swing_low:,.2f}")
    print(f"  Range: ${short_fib.swing_range:,.2f}")
    
    print(f"\nFibonacci Levels (Price retraces UP):")
    print(f"  0% (ChoCH Low):    ${short_fib.fib_0:,.2f}")
    print(f"  70.5% (OEP):       ${short_fib.fib_0705:,.2f} ⭐ PRIMARY")
    print(f"  61.8% (Golden):    ${short_fib.fib_0618:,.2f} ✓ SECONDARY")
    print(f"  100% (Swing High): ${short_fib.fib_1:,.2f}")
    
    # Test entry signals
    print(f"\nEntry Signal Tests:")
    
    test_prices = [1735, 1656, 1500, 2100]
    for price in test_prices:
        signal = short_fib.get_entry_signal(price, tolerance=0.002)
        if signal:
            print(f"  ${price:,.0f} → ✅ ENTRY at {signal['level']} ({signal['quality']})")
        else:
            missed = short_fib.missed_both_levels(price)
            if missed:
                print(f"  ${price:,.0f} → ❌ MISSED both levels (Cancel setup)")
            else:
                print(f"  ${price:,.0f} → ⏳ Waiting for retracement")
    
    # ═════════════════════════════════════════════════════════
    # EXAMPLE 3: Real-time monitoring simulation
    # ═════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("[EXAMPLE 3: REAL-TIME MONITORING]")
    print("-" * 60)
    
    print("\nScenario: LONG setup, monitoring price retracement")
    print(f"ChoCH confirmed at $1600")
    print(f"Waiting for retracement to Fib levels...")
    
    # Simulate price movement
    retracement_prices = [1600, 1500, 1400, 1200, 1044, 1000, 965, 950]
    
    print(f"\nPrice Movement:")
    for i, price in enumerate(retracement_prices, 1):
        signal = long_fib.get_entry_signal(price, tolerance=0.002)
        
        print(f"\nCandle {i}: ${price:,.0f}")
        
        if signal:
            print(f"  🎯 ENTRY SIGNAL!")
            print(f"     Level: {signal['level']}")
            print(f"     Quality: {signal['quality']}")
            print(f"     Entry Price: ${signal['price']:,.2f}")
            break
        else:
            if long_fib.missed_both_levels(price):
                print(f"  ❌ Setup CANCELED - Missed both Fib levels")
                break
            else:
                # Check distance to levels
                dist_705 = abs(price - long_fib.fib_0705) / long_fib.fib_0705 * 100
                dist_618 = abs(price - long_fib.fib_0618) / long_fib.fib_0618 * 100
                
                if dist_705 < dist_618:
                    print(f"  ⏳ Approaching 0.705 level (${long_fib.fib_0705:,.2f})")
                    print(f"     Distance: {dist_705:.2f}%")
                else:
                    print(f"  ⏳ Approaching 0.618 level (${long_fib.fib_0618:,.2f})")
                    print(f"     Distance: {dist_618:.2f}%")
    
    # ═════════════════════════════════════════════════════════
    # ML Features
    # ═════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("ML FEATURES (LONG Setup)")
    print("="*60)
    
    features = long_fib.get_ml_features()
    for key, value in features.items():
        print(f"{key:30s}: {value:,.4f}")
    
    print("\n" + "="*60)
    print("✅ Fibonacci Calculator working correctly!")
    print("="*60 + "\n")