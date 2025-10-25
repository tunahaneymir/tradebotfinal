"""
Entry System - Parça 2
Based on: pa-strateji2 Parça 2

Entry System Orchestrator:
1. Checks trend alignment (from TrendDetector)
2. Checks zone presence (from ZoneDetector)
3. Detects ChoCH (from ChoCHDetector)
4. Calculates Fibonacci (from FibonacciCalculator)
5. Waits for Fibonacci retest
6. Validates entry conditions
7. Returns entry signal with all data
"""

from __future__ import annotations
from typing import Optional, Dict, Literal
from dataclasses import dataclass
import numpy as np

from .trend_detector import TrendDetector, TrendResult
from .zone_detector import ZoneDetector, Zone
from .choch_detector import ChoCHDetector, ChoCHResult
from .fibonacci_calculator import FibonacciCalculator, FibonacciLevels


@dataclass
class EntrySignal:
    """Complete entry signal with all components"""
    # Signal status
    ready: bool
    action: Literal["ENTER", "WAIT", "CANCEL"]
    
    # Components
    trend: TrendResult
    zone: Optional[Zone]
    choch: ChoCHResult
    fibonacci: Optional[FibonacciLevels]
    
    # Entry details
    direction: Optional[Literal["LONG", "SHORT"]]
    entry_price: float
    entry_level: Optional[str]  # "0.705" or "0.618"
    entry_quality: Optional[str]  # "EXCELLENT" or "GOOD"
    
    # Validation
    trend_aligned: bool
    zone_valid: bool
    choch_strong: bool
    fib_touched: bool
    
    # Stop loss (calculated here)
    stop_loss: float
    risk_per_unit: float
    
    # Message
    message: str
    
    def get_ml_features(self) -> Dict[str, float]:
        """Aggregate ML features from all components"""
        features = {}
        
        # Trend features
        features.update(self.trend.get_ml_features())
        
        # Zone features
        if self.zone:
            features.update(self.zone.get_ml_features())
        
        # ChoCH features
        features.update(self.choch.get_ml_features())
        
        # Fibonacci features
        if self.fibonacci:
            features.update(self.fibonacci.get_ml_features())
        
        # Entry-specific features
        features.update({
            'entry_ready': float(self.ready),
            'entry_trend_aligned': float(self.trend_aligned),
            'entry_zone_valid': float(self.zone_valid),
            'entry_choch_strong': float(self.choch_strong),
            'entry_fib_touched': float(self.fib_touched),
            'entry_risk_per_unit': self.risk_per_unit,
        })
        
        return features


class EntrySystem:
    """
    Entry System Orchestrator
    
    Complete PA entry logic:
    1. Trend must be aligned (4H)
    2. Price must be in valid zone (1H)
    3. ChoCH must be confirmed (15M)
    4. Fibonacci retest must occur (15M)
    5. All conditions validated
    
    Usage:
        entry_sys = EntrySystem(config)
        signal = entry_sys.check_entry(
            high, low, close, open, volume,
            direction="LONG"
        )
        
        if signal.ready and signal.action == "ENTER":
            # Execute trade!
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Entry System
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize sub-components
        self.trend_detector = TrendDetector(config)
        self.zone_detector = ZoneDetector(config)
        self.choch_detector = ChoCHDetector(config)
        self.fib_calculator = FibonacciCalculator(config)
        
        # Entry configuration
        entry_config = config.get('entry', {}) if config else {}
        self.min_choch_strength = entry_config.get('choch', {}).get('min_strength', 0.4)
        
        # Stop loss configuration
        sl_config = entry_config.get('stop_loss', {})
        self.stop_buffer_pct = sl_config.get('buffer_pct', 0.005)  # 0.5%
    
    def check_entry(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        open_prices: np.ndarray,
        volume: np.ndarray,
        direction: Literal["LONG", "SHORT"],
        current_zone: Optional[Zone] = None
    ) -> EntrySignal:
        """
        Check if entry conditions are met
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            open_prices: Open prices
            volume: Volume data
            direction: Desired trade direction
            current_zone: Optional pre-detected zone
            
        Returns:
            EntrySignal with all components
        """
        # ═══════════════════════════════════════════════════════════
        # STEP 1: Check Trend (4H)
        # ═══════════════════════════════════════════════════════════
        trend = self.trend_detector.detect(close, high, low)
        
        trend_aligned = self._check_trend_alignment(trend, direction)
        
        if not trend_aligned:
            return self._create_signal(
                action="CANCEL",
                message=f"Trend not aligned. Trend: {trend.direction}, Want: {direction}",
                trend=trend,
                trend_aligned=False
            )
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: Check Zone (1H)
        # ═══════════════════════════════════════════════════════════
        if current_zone is None:
            zones = self.zone_detector.detect_zones(
                high, low, close,
                timeframe="1H",
                method="both"
            )
            
            if not zones:
                return self._create_signal(
                    action="CANCEL",
                    message="No valid zones found",
                    trend=trend,
                    trend_aligned=True,
                    zone_valid=False
                )
            
            # Get best zone (highest quality, closest to price)
            current_zone = zones[0]
        
        # Check if price in zone
        current_price = close[-1]
        price_in_zone = self._is_price_in_zone(current_price, current_zone)
        
        if not price_in_zone:
            return self._create_signal(
                action="WAIT",
                message=f"Price not in zone. Current: ${current_price:.2f}, Zone: ${current_zone.price_low:.2f}-${current_zone.price_high:.2f}",
                trend=trend,
                zone=current_zone,
                trend_aligned=True,
                zone_valid=False
            )
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: Detect ChoCH (15M)
        # ═══════════════════════════════════════════════════════════
        choch = self.choch_detector.detect(
            high, low, close, open_prices, volume,
            direction=direction
        )
        
        if not choch.detected:
            return self._create_signal(
                action="WAIT",
                message="No ChoCH detected yet. Waiting for market structure break",
                trend=trend,
                zone=current_zone,
                choch=choch,
                trend_aligned=True,
                zone_valid=True,
                choch_strong=False
            )
        
        # Check ChoCH strength
        choch_strong = choch.strength >= self.min_choch_strength
        
        if not choch_strong:
            return self._create_signal(
                action="WAIT",
                message=f"ChoCH too weak. Strength: {choch.strength:.2f}, Min: {self.min_choch_strength}",
                trend=trend,
                zone=current_zone,
                choch=choch,
                trend_aligned=True,
                zone_valid=True,
                choch_strong=False
            )
        
        # ═══════════════════════════════════════════════════════════
        # STEP 4: Calculate Fibonacci (15M)
        # ═══════════════════════════════════════════════════════════
        
        # Find swing extreme before ChoCH
        swing_extreme = self.fib_calculator.find_swing_extreme(
            high, low,
            choch_index=choch.breakout_index,
            direction=direction,
            lookback=30
        )
        
        # Get ChoCH extreme (high for LONG, low for SHORT)
        if direction == "LONG":
            choch_extreme = high[choch.breakout_index]
        else:
            choch_extreme = low[choch.breakout_index]
        
        # Calculate Fibonacci levels
        fibonacci = self.fib_calculator.calculate(
            direction=direction,
            choch_breakout_price=choch_extreme,
            swing_extreme_price=swing_extreme
        )
        
        # ═══════════════════════════════════════════════════════════
        # STEP 5: Check Fibonacci Retest
        # ═══════════════════════════════════════════════════════════
        
        entry_signal = fibonacci.get_entry_signal(current_price, tolerance=0.002)
        
        if entry_signal is None:
            # Check if missed both levels
            if fibonacci.missed_both_levels(current_price):
                return self._create_signal(
                    action="CANCEL",
                    message="Missed both Fibonacci levels. Setup canceled",
                    trend=trend,
                    zone=current_zone,
                    choch=choch,
                    fibonacci=fibonacci,
                    trend_aligned=True,
                    zone_valid=True,
                    choch_strong=True,
                    fib_touched=False
                )
            else:
                return self._create_signal(
                    action="WAIT",
                    message=f"Waiting for Fibonacci retest. Current: ${current_price:.2f}, Target: ${fibonacci.fib_0705:.2f} or ${fibonacci.fib_0618:.2f}",
                    trend=trend,
                    zone=current_zone,
                    choch=choch,
                    fibonacci=fibonacci,
                    trend_aligned=True,
                    zone_valid=True,
                    choch_strong=True,
                    fib_touched=False
                )
        
        # ═══════════════════════════════════════════════════════════
        # STEP 6: Calculate Stop Loss
        # ═══════════════════════════════════════════════════════════
        
        stop_loss = self._calculate_stop_loss(
            direction=direction,
            swing_extreme=swing_extreme
        )
        
        # Calculate risk
        entry_price = entry_signal['price']
        risk_per_unit = abs(entry_price - stop_loss)
        
        # ═══════════════════════════════════════════════════════════
        # ALL CONDITIONS MET - ENTRY READY!
        # ═══════════════════════════════════════════════════════════
        
        return EntrySignal(
            ready=True,
            action="ENTER",
            trend=trend,
            zone=current_zone,
            choch=choch,
            fibonacci=fibonacci,
            direction=direction,
            entry_price=entry_price,
            entry_level=entry_signal['level'],
            entry_quality=entry_signal['quality'],
            trend_aligned=True,
            zone_valid=True,
            choch_strong=True,
            fib_touched=True,
            stop_loss=stop_loss,
            risk_per_unit=risk_per_unit,
            message=f"✅ ENTRY READY! {direction} @ ${entry_price:.2f} (Fib {entry_signal['level']}, {entry_signal['quality']})"
        )
    
    def _check_trend_alignment(
        self,
        trend: TrendResult,
        direction: Literal["LONG", "SHORT"]
    ) -> bool:
        """Check if trend aligns with desired direction"""
        if direction == "LONG":
            return trend.direction == "UP"
        else:
            return trend.direction == "DOWN"
    
    def _is_price_in_zone(self, price: float, zone: Zone) -> bool:
        """Check if price is within zone"""
        return zone.price_low <= price <= zone.price_high
    
    def _calculate_stop_loss(
        self,
        direction: Literal["LONG", "SHORT"],
        swing_extreme: float
    ) -> float:
        """
        Calculate stop loss
        
        Rule: Last swing wick + buffer
        LONG: Below swing low
        SHORT: Above swing high
        """
        if direction == "LONG":
            # Stop below swing low
            stop_loss = swing_extreme * (1 - self.stop_buffer_pct)
        else:
            # Stop above swing high
            stop_loss = swing_extreme * (1 + self.stop_buffer_pct)
        
        return stop_loss
    
    def _create_signal(
        self,
        action: Literal["ENTER", "WAIT", "CANCEL"],
        message: str,
        trend: Optional[TrendResult] = None,
        zone: Optional[Zone] = None,
        choch: Optional[ChoCHResult] = None,
        fibonacci: Optional[FibonacciLevels] = None,
        trend_aligned: bool = False,
        zone_valid: bool = False,
        choch_strong: bool = False,
        fib_touched: bool = False
    ) -> EntrySignal:
        """Create entry signal with default values"""
        # Create empty ChoCH if not provided
        if choch is None:
            choch = ChoCHResult(
                detected=False,
                direction=None,
                breakout_price=0.0,
                breakout_index=-1,
                broken_level=0.0,
                strength=0.0,
                body_score=0.0,
                volume_score=0.0
            )
        
        # Create empty trend if not provided
        if trend is None:
            # This shouldn't happen in normal flow
            trend = TrendResult(
                direction="SIDEWAYS",
                ema_20=0.0,
                ema_50=0.0,
                ema_distance_pct=0.0,
                atr_ratio=0.0,
                price_range_pct=0.0,
                confidence=0.0,
                slope_up=False,
                slope_down=False
            )
        
        return EntrySignal(
            ready=False,
            action=action,
            trend=trend,
            zone=zone,
            choch=choch,
            fibonacci=fibonacci,
            direction=None,
            entry_price=0.0,
            entry_level=None,
            entry_quality=None,
            trend_aligned=trend_aligned,
            zone_valid=zone_valid,
            choch_strong=choch_strong,
            fib_touched=fib_touched,
            stop_loss=0.0,
            risk_per_unit=0.0,
            message=message
        )


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    
    # Config
    config = {
        'trend': {
            'ema_fast': 20,
            'ema_slow': 50,
            'sideways': {
                'ema_distance_pct': 0.005,
                'atr_ratio': 0.006,
                'range_pct': 0.08
            }
        },
        'zones': {
            'zigzag': {'depth': 12, 'deviation': 5, 'backstep': 2},
            'swing': {'strength': 5},
            'min_touches': 2,
            'max_thickness_pct': 1.5,
            'min_quality': 4,
            'lookback': {'4H': 720, '1H': 600, '15M': 400}
        },
        'entry': {
            'choch': {'min_strength': 0.4},
            'fibonacci': {'levels': [0.705, 0.618]},
            'stop_loss': {'buffer_pct': 0.005}
        }
    }
    
    # Create entry system
    entry_sys = EntrySystem(config)
    
    print("\n" + "="*60)
    print("ENTRY SYSTEM - COMPLETE FLOW TEST")
    print("="*60)
    
    # Simulate complete entry scenario
    n = 200
    
    # Create uptrend with zone and ChoCH
    prices = []
    
    # Phase 1: Uptrend to zone
    prices.extend(np.linspace(48000, 52000, 50))
    
    # Phase 2: Retracement to zone (downtrend mini)
    prices.extend(np.linspace(52000, 49000, 50))  # LL-LH structure
    
    # Phase 3: ChoCH (break upward)
    prices.extend(np.linspace(49000, 51500, 30))
    
    # Phase 4: Fibonacci retracement
    prices.extend(np.linspace(51500, 49800, 40))  # Retracing to Fib
    
    # Phase 5: Continuation
    prices.extend(np.linspace(49800, 53000, 30))
    
    close = np.array(prices)
    high = close + np.random.rand(len(close)) * 100
    low = close - np.random.rand(len(close)) * 100
    open_prices = close + np.random.randn(len(close)) * 50
    volume = np.random.rand(len(close)) * 1000 + 500
    
    # Test at different points
    test_points = [100, 130, 160, 170]
    
    for point in test_points:
        print(f"\n{'='*60}")
        print(f"CHECKPOINT: Candle {point} (Price: ${close[point]:,.2f})")
        print(f"{'='*60}")
        
        signal = entry_sys.check_entry(
            high=high[:point+1],
            low=low[:point+1],
            close=close[:point+1],
            open_prices=open_prices[:point+1],
            volume=volume[:point+1],
            direction="LONG"
        )
        
        print(f"\nAction: {signal.action}")
        print(f"Message: {signal.message}")
        
        if signal.action == "ENTER":
            print(f"\n🎯 ENTRY SIGNAL DETAILS:")
            print(f"   Direction: {signal.direction}")
            print(f"   Entry Price: ${signal.entry_price:,.2f}")
            print(f"   Entry Level: Fib {signal.entry_level} ({signal.entry_quality})")
            print(f"   Stop Loss: ${signal.stop_loss:,.2f}")
            print(f"   Risk per Unit: ${signal.risk_per_unit:,.2f}")
            print(f"\n   Validations:")
            print(f"   ├─ Trend Aligned: {signal.trend_aligned}")
            print(f"   ├─ Zone Valid: {signal.zone_valid}")
            print(f"   ├─ ChoCH Strong: {signal.choch_strong}")
            print(f"   └─ Fib Touched: {signal.fib_touched}")
            break
    
    print("\n" + "="*60)
    print("✅ Entry System test complete!")
    print("="*60 + "\n")