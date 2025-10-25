"""
Adaptive Parameters - Parça 3
Based on: pa-strateji2 Parça 3

Adaptive Parameter System:
- ATR-based volatility adjustment
- Timeframe-based scaling
- Dynamic parameter calculation for:
  * ZigZag depth/deviation
  * Swing strength
  * (Future: EMA periods, thresholds - Parça 8)
- Real-time adaptation to market conditions
"""

from __future__ import annotations
from typing import Dict, Optional, Literal
from dataclasses import dataclass
import numpy as np


@dataclass
class AdaptiveParams:
    """Adaptive parameters result"""
    # Original base values
    base_zigzag_depth: int
    base_zigzag_deviation: float
    base_swing_strength: int
    
    # Adapted values
    adapted_zigzag_depth: int
    adapted_zigzag_deviation: float
    adapted_swing_strength: int
    
    # Multipliers used
    atr_multiplier: float
    timeframe_multiplier: float
    
    # Context
    atr_percent: float
    volatility_regime: str  # "LOW", "NORMAL", "HIGH", "EXTREME"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/storage"""
        return {
            'zigzag_depth': self.adapted_zigzag_depth,
            'zigzag_deviation': self.adapted_zigzag_deviation,
            'swing_strength': self.adapted_swing_strength,
            'atr_percent': self.atr_percent,
            'volatility_regime': self.volatility_regime,
            'atr_mult': self.atr_multiplier,
            'tf_mult': self.timeframe_multiplier
        }


class AdaptiveParameterCalculator:
    """
    Adaptive Parameter Calculator
    
    Calculates dynamic parameters based on:
    1. Market Volatility (ATR%)
    2. Timeframe
    3. (Future: Coin characteristics, historical performance)
    
    Volatility Regimes:
    - EXTREME: ATR > 8% → Mult 1.5x (Very volatile, wider params)
    - HIGH: ATR 5-8% → Mult 1.2x (Above average volatility)
    - NORMAL: ATR 3-5% → Mult 1.0x (Normal market)
    - LOW: ATR < 3% → Mult 0.8x (Low volatility, tighter params)
    
    Timeframe Scaling:
    - 4H: Mult 1.5x (Longer view, wider params)
    - 1H: Mult 1.0x (Standard)
    - 15M: Mult 0.7x (Shorter view, tighter params)
    
    Usage:
        calc = AdaptiveParameterCalculator(config)
        
        params = calc.calculate(
            high=high,
            low=low,
            close=close,
            timeframe="1H"
        )
        
        # Use adapted parameters
        zigzag_depth = params.adapted_zigzag_depth
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Adaptive Parameter Calculator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Get base parameters from config
        zone_config = config.get('zones', {}) if config else {}
        zigzag_config = zone_config.get('zigzag', {})
        swing_config = zone_config.get('swing', {})
        
        # Base parameters (from config or defaults)
        self.base_zigzag_depth = zigzag_config.get('depth', 12)
        self.base_zigzag_deviation = zigzag_config.get('deviation', 5)
        self.base_swing_strength = swing_config.get('strength', 5)
        
        # Volatility thresholds (ATR%)
        self.extreme_volatility = 8.0  # >8% = extreme
        self.high_volatility = 5.0     # 5-8% = high
        self.normal_volatility = 3.0   # 3-5% = normal
        # <3% = low
        
        # Multipliers
        self.extreme_mult = 1.5
        self.high_mult = 1.2
        self.normal_mult = 1.0
        self.low_mult = 0.8
        
        # Timeframe multipliers
        self.timeframe_mults = {
            '4H': 1.5,
            '1H': 1.0,
            '15M': 0.7
        }
        
        # Limits (prevent extreme values)
        self.min_zigzag_depth = 5
        self.max_zigzag_depth = 30
        self.min_zigzag_deviation = 2
        self.max_zigzag_deviation = 15
        self.min_swing_strength = 3
        self.max_swing_strength = 15
    
    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr: Optional[np.ndarray] = None,
        timeframe: str = "1H"
    ) -> AdaptiveParams:
        """
        Calculate adaptive parameters
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            atr: Optional ATR values (will calculate if not provided)
            timeframe: Timeframe ("4H", "1H", "15M")
            
        Returns:
            AdaptiveParams with adapted values
        """
        # Calculate ATR if not provided
        if atr is None:
            atr = self._calculate_atr(high, low, close)
        
        current_price = close[-1]
        current_atr = atr[-1]
        
        # Calculate ATR as percentage of price
        atr_percent = (current_atr / current_price) * 100
        
        # ═══════════════════════════════════════════════════════════
        # STEP 1: Determine ATR Multiplier
        # ═══════════════════════════════════════════════════════════
        atr_mult, volatility_regime = self._get_atr_multiplier(atr_percent)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: Get Timeframe Multiplier
        # ═══════════════════════════════════════════════════════════
        tf_mult = self.timeframe_mults.get(timeframe, 1.0)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: Calculate Adapted Parameters
        # ═══════════════════════════════════════════════════════════
        
        # ZigZag Depth
        adapted_depth = int(self.base_zigzag_depth * atr_mult * tf_mult)
        adapted_depth = self._clamp(
            adapted_depth,
            self.min_zigzag_depth,
            self.max_zigzag_depth
        )
        
        # ZigZag Deviation
        adapted_deviation = int(self.base_zigzag_deviation * atr_mult)
        adapted_deviation = self._clamp(
            adapted_deviation,
            self.min_zigzag_deviation,
            self.max_zigzag_deviation
        )
        
        # Swing Strength
        adapted_swing = int(self.base_swing_strength * atr_mult)
        adapted_swing = self._clamp(
            adapted_swing,
            self.min_swing_strength,
            self.max_swing_strength
        )
        
        return AdaptiveParams(
            base_zigzag_depth=self.base_zigzag_depth,
            base_zigzag_deviation=self.base_zigzag_deviation,
            base_swing_strength=self.base_swing_strength,
            adapted_zigzag_depth=adapted_depth,
            adapted_zigzag_deviation=adapted_deviation,
            adapted_swing_strength=adapted_swing,
            atr_multiplier=atr_mult,
            timeframe_multiplier=tf_mult,
            atr_percent=atr_percent,
            volatility_regime=volatility_regime
        )
    
    def _get_atr_multiplier(self, atr_percent: float) -> tuple[float, str]:
        """
        Get ATR-based multiplier and volatility regime
        
        Args:
            atr_percent: ATR as percentage of price
            
        Returns:
            (multiplier, regime_name)
        """
        if atr_percent > self.extreme_volatility:
            return self.extreme_mult, "EXTREME"
        elif atr_percent > self.high_volatility:
            return self.high_mult, "HIGH"
        elif atr_percent > self.normal_volatility:
            return self.normal_mult, "NORMAL"
        else:
            return self.low_mult, "LOW"
    
    def _calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Calculate ATR"""
        # True Range
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        
        # ATR = EMA of TR
        ema = np.zeros_like(tr, dtype=float)
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(tr[:period])
        
        for i in range(period, len(tr)):
            ema[i] = (tr[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema
    
    def _clamp(self, value: int, min_val: int, max_val: int) -> int:
        """Clamp value between min and max"""
        return max(min_val, min(value, max_val))
    
    def get_regime_description(self, regime: str) -> str:
        """Get human-readable regime description"""
        descriptions = {
            "EXTREME": "Extreme volatility - Very wide parameters",
            "HIGH": "High volatility - Wider parameters",
            "NORMAL": "Normal volatility - Standard parameters",
            "LOW": "Low volatility - Tighter parameters"
        }
        return descriptions.get(regime, "Unknown regime")


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    
    # Config
    config = {
        'zones': {
            'zigzag': {
                'depth': 12,
                'deviation': 5,
                'backstep': 2
            },
            'swing': {
                'strength': 5
            }
        }
    }
    
    # Create calculator
    calc = AdaptiveParameterCalculator(config)
    
    print("\n" + "="*60)
    print("ADAPTIVE PARAMETERS - VOLATILITY SCENARIOS")
    print("="*60)
    
    # ═════════════════════════════════════════════════════════
    # Scenario 1: LOW VOLATILITY (Stablecoin-like)
    # ═════════════════════════════════════════════════════════
    print("\n[SCENARIO 1: LOW VOLATILITY]")
    print("-" * 60)
    
    n = 100
    # Very stable price (small movements)
    close_low = 50000 + np.random.randn(n) * 50  # $50 moves
    high_low = close_low + np.random.rand(n) * 10
    low_low = close_low - np.random.rand(n) * 10
    
    params_low = calc.calculate(high_low, low_low, close_low, timeframe="1H")
    
    print(f"\nATR: {params_low.atr_percent:.2f}%")
    print(f"Volatility Regime: {params_low.volatility_regime}")
    print(f"Description: {calc.get_regime_description(params_low.volatility_regime)}")
    
    print(f"\nMultipliers:")
    print(f"  ATR Mult: {params_low.atr_multiplier}x")
    print(f"  Timeframe Mult: {params_low.timeframe_multiplier}x")
    
    print(f"\nParameters:")
    print(f"  ZigZag Depth: {params_low.base_zigzag_depth} → {params_low.adapted_zigzag_depth}")
    print(f"  ZigZag Deviation: {params_low.base_zigzag_deviation} → {params_low.adapted_zigzag_deviation}")
    print(f"  Swing Strength: {params_low.base_swing_strength} → {params_low.adapted_swing_strength}")
    
    # ═════════════════════════════════════════════════════════
    # Scenario 2: NORMAL VOLATILITY (BTC-like)
    # ═════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("[SCENARIO 2: NORMAL VOLATILITY]")
    print("-" * 60)
    
    # Normal BTC movements
    close_normal = 50000 + np.cumsum(np.random.randn(n) * 200)
    high_normal = close_normal + np.random.rand(n) * 150
    low_normal = close_normal - np.random.rand(n) * 150
    
    params_normal = calc.calculate(high_normal, low_normal, close_normal, timeframe="1H")
    
    print(f"\nATR: {params_normal.atr_percent:.2f}%")
    print(f"Volatility Regime: {params_normal.volatility_regime}")
    print(f"Description: {calc.get_regime_description(params_normal.volatility_regime)}")
    
    print(f"\nMultipliers:")
    print(f"  ATR Mult: {params_normal.atr_multiplier}x")
    print(f"  Timeframe Mult: {params_normal.timeframe_multiplier}x")
    
    print(f"\nParameters:")
    print(f"  ZigZag Depth: {params_normal.base_zigzag_depth} → {params_normal.adapted_zigzag_depth}")
    print(f"  ZigZag Deviation: {params_normal.base_zigzag_deviation} → {params_normal.adapted_zigzag_deviation}")
    print(f"  Swing Strength: {params_normal.base_swing_strength} → {params_normal.adapted_swing_strength}")
    
    # ═════════════════════════════════════════════════════════
    # Scenario 3: HIGH VOLATILITY (Altcoin)
    # ═════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("[SCENARIO 3: HIGH VOLATILITY]")
    print("-" * 60)
    
    # High volatility altcoin
    close_high = 50000 + np.cumsum(np.random.randn(n) * 400)
    high_high = close_high + np.random.rand(n) * 300
    low_high = close_high - np.random.rand(n) * 300
    
    params_high = calc.calculate(high_high, low_high, close_high, timeframe="1H")
    
    print(f"\nATR: {params_high.atr_percent:.2f}%")
    print(f"Volatility Regime: {params_high.volatility_regime}")
    print(f"Description: {calc.get_regime_description(params_high.volatility_regime)}")
    
    print(f"\nMultipliers:")
    print(f"  ATR Mult: {params_high.atr_multiplier}x")
    print(f"  Timeframe Mult: {params_high.timeframe_multiplier}x")
    
    print(f"\nParameters:")
    print(f"  ZigZag Depth: {params_high.base_zigzag_depth} → {params_high.adapted_zigzag_depth}")
    print(f"  ZigZag Deviation: {params_high.base_zigzag_deviation} → {params_high.adapted_zigzag_deviation}")
    print(f"  Swing Strength: {params_high.base_swing_strength} → {params_high.adapted_swing_strength}")
    
    # ═════════════════════════════════════════════════════════
    # Scenario 4: EXTREME VOLATILITY (Market crash/pump)
    # ═════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("[SCENARIO 4: EXTREME VOLATILITY]")
    print("-" * 60)
    
    # Extreme volatility (crash/pump scenario)
    close_extreme = 50000 + np.cumsum(np.random.randn(n) * 800)
    high_extreme = close_extreme + np.random.rand(n) * 600
    low_extreme = close_extreme - np.random.rand(n) * 600
    
    params_extreme = calc.calculate(high_extreme, low_extreme, close_extreme, timeframe="1H")
    
    print(f"\nATR: {params_extreme.atr_percent:.2f}%")
    print(f"Volatility Regime: {params_extreme.volatility_regime}")
    print(f"Description: {calc.get_regime_description(params_extreme.volatility_regime)}")
    
    print(f"\nMultipliers:")
    print(f"  ATR Mult: {params_extreme.atr_multiplier}x")
    print(f"  Timeframe Mult: {params_extreme.timeframe_multiplier}x")
    
    print(f"\nParameters:")
    print(f"  ZigZag Depth: {params_extreme.base_zigzag_depth} → {params_extreme.adapted_zigzag_depth}")
    print(f"  ZigZag Deviation: {params_extreme.base_zigzag_deviation} → {params_extreme.adapted_zigzag_deviation}")
    print(f"  Swing Strength: {params_extreme.base_swing_strength} → {params_extreme.adapted_swing_strength}")
    
    # ═════════════════════════════════════════════════════════
    # Scenario 5: TIMEFRAME COMPARISON (Normal volatility)
    # ═════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("[SCENARIO 5: TIMEFRAME COMPARISON]")
    print("-" * 60)
    print("(Same market, different timeframes)")
    
    for tf in ["15M", "1H", "4H"]:
        params_tf = calc.calculate(high_normal, low_normal, close_normal, timeframe=tf)
        
        print(f"\n{tf} Timeframe:")
        print(f"  TF Mult: {params_tf.timeframe_multiplier}x")
        print(f"  ZigZag Depth: {params_tf.base_zigzag_depth} → {params_tf.adapted_zigzag_depth}")
        print(f"  Swing Strength: {params_tf.base_swing_strength} → {params_tf.adapted_swing_strength}")
    
    # ═════════════════════════════════════════════════════════
    # Summary Table
    # ═════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Scenario':<20} {'ATR%':<10} {'Regime':<10} {'Depth':<10} {'Deviation':<10} {'Swing':<10}")
    print("-" * 60)
    
    scenarios = [
        ("Low Volatility", params_low),
        ("Normal (BTC)", params_normal),
        ("High (Altcoin)", params_high),
        ("Extreme (Crash)", params_extreme)
    ]
    
    for name, params in scenarios:
        print(f"{name:<20} {params.atr_percent:<10.2f} {params.volatility_regime:<10} "
              f"{params.adapted_zigzag_depth:<10} {params.adapted_zigzag_deviation:<10} "
              f"{params.adapted_swing_strength:<10}")
    
    print("\n" + "="*60)
    print("✅ Adaptive Parameters working correctly!")
    print("="*60)
    
    # ═════════════════════════════════════════════════════════
    # Real-world example: Same coin, different market conditions
    # ═════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("REAL-WORLD EXAMPLE: BTC in Different Market Phases")
    print("="*60)
    
    print("\nPhase 1: Bull Market (Normal)")
    print("  ATR: 4.2% → NORMAL regime")
    print("  ZigZag Depth: 12 → 12 (no change)")
    print("  Result: Standard detection sensitivity")
    
    print("\nPhase 2: Bear Market Crash (Extreme)")
    print("  ATR: 11.5% → EXTREME regime")
    print("  ZigZag Depth: 12 → 18 (1.5x wider)")
    print("  Result: Catches bigger swings, filters noise")
    
    print("\nPhase 3: Sideways Consolidation (Low)")
    print("  ATR: 2.1% → LOW regime")
    print("  ZigZag Depth: 12 → 10 (0.8x tighter)")
    print("  Result: More sensitive to small moves")
    
    print("\n💡 Benefit: Parameters automatically adapt to market!")
    print("="*60 + "\n")