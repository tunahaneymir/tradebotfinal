"""
Re-entry Manager - Parça 3
Based on: pa-strateji2 Parça 3

Re-entry System:
- Allows second attempt after stop loss
- Requires new ChoCH and new Fibonacci retest
- Reduced risk (50% of original)
- Maximum 2 re-entries per zone
- Zone validation before re-entry
- Cooldown period (2 candles minimum)
"""

from __future__ import annotations
from typing import Optional, Dict, Literal, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from core import ChoCHDetector, FibonacciCalculator, Zone


@dataclass
class TradeHistory:
    """Record of a completed trade"""
    trade_id: str
    zone_id: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    result: Literal["WIN", "LOSS", "STOP_LOSS"]
    pnl: float
    exit_time: datetime
    exit_index: int  # Candle index when exited
    reason: str


@dataclass
class ReentryEligibility:
    """Re-entry eligibility check result"""
    eligible: bool
    reason: str
    conditions_met: Dict[str, bool]
    wait_candles: int  # How many more candles to wait
    risk_reduction: float  # Risk multiplier (0.5 = 50% risk)


class ReentryManager:
    """
    Re-entry Management System
    
    Manages second attempts after stop loss:
    1. Validates zone is still valid
    2. Enforces cooldown period (2 candles)
    3. Requires new ChoCH formation
    4. Requires new Fibonacci retest
    5. Reduces risk to 50%
    6. Limits to max 2 re-entries per zone
    
    Usage:
        manager = ReentryManager(config)
        
        # After stop loss
        eligibility = manager.check_reentry_eligibility(
            last_trade=trade,
            zone=zone,
            current_index=150,
            high=high,
            low=low,
            close=close,
            ...
        )
        
        if eligibility.eligible:
            # Proceed with re-entry at 50% risk
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Re-entry Manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Re-entry configuration
        reentry_config = config.get('reentry', {}) if config else {}
        
        self.enabled = reentry_config.get('enabled', True)
        self.wait_candles = reentry_config.get('wait_candles', 2)  # 30 min @ 15M
        self.max_attempts = reentry_config.get('max_attempts', 2)  # Max 2 re-entries
        self.require_new_choch = reentry_config.get('require_new_choch', True)
        self.require_new_fib = reentry_config.get('require_new_fib', True)
        self.risk_reduction = 0.5  # 50% risk on re-entry
        
        # Initialize detectors
        self.choch_detector = ChoCHDetector(config)
        self.fib_calculator = FibonacciCalculator(config)
    
    def check_reentry_eligibility(
        self,
        last_trade: TradeHistory,
        zone: Zone,
        current_index: int,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        open_prices: np.ndarray,
        volume: np.ndarray,
        zone_attempts: int = 1  # How many attempts on this zone so far
    ) -> ReentryEligibility:
        """
        Check if re-entry is eligible
        
        Args:
            last_trade: Previous trade that stopped out
            zone: Zone being traded
            current_index: Current candle index
            high: High prices
            low: Low prices
            close: Close prices
            open_prices: Open prices
            volume: Volume data
            zone_attempts: Total attempts on this zone
            
        Returns:
            ReentryEligibility with all checks
        """
        conditions = {}
        
        # ═══════════════════════════════════════════════════════════
        # CHECK 0: System enabled?
        # ═══════════════════════════════════════════════════════════
        if not self.enabled:
            return ReentryEligibility(
                eligible=False,
                reason="Re-entry system disabled in config",
                conditions_met={},
                wait_candles=0,
                risk_reduction=1.0
            )
        
        # ═══════════════════════════════════════════════════════════
        # CHECK 1: Was last trade a stop loss?
        # ═══════════════════════════════════════════════════════════
        if last_trade.result != "STOP_LOSS":
            return ReentryEligibility(
                eligible=False,
                reason="Last trade was not stopped out (result: {})".format(last_trade.result),
                conditions_met={'stopped': False},
                wait_candles=0,
                risk_reduction=1.0
            )
        
        conditions['stopped'] = True
        
        # ═══════════════════════════════════════════════════════════
        # CHECK 2: Cooldown period passed?
        # ═══════════════════════════════════════════════════════════
        candles_since_exit = current_index - last_trade.exit_index
        
        if candles_since_exit < self.wait_candles:
            return ReentryEligibility(
                eligible=False,
                reason=f"Cooldown period: wait {self.wait_candles - candles_since_exit} more candles",
                conditions_met={'time_passed': False},
                wait_candles=self.wait_candles - candles_since_exit,
                risk_reduction=self.risk_reduction
            )
        
        conditions['time_passed'] = True
        
        # ═══════════════════════════════════════════════════════════
        # CHECK 3: Zone still valid? (Not broken with body close)
        # ═══════════════════════════════════════════════════════════
        zone_valid = self._check_zone_validity(
            zone=zone,
            close=close,
            last_trade_direction=last_trade.direction,
            exit_index=last_trade.exit_index
        )
        
        if not zone_valid:
            return ReentryEligibility(
                eligible=False,
                reason="Zone invalidated (body close broke zone)",
                conditions_met={'zone_valid': False},
                wait_candles=0,
                risk_reduction=self.risk_reduction
            )
        
        conditions['zone_valid'] = True
        
        # ═══════════════════════════════════════════════════════════
        # CHECK 4: New ChoCH formed? (if required)
        # ═══════════════════════════════════════════════════════════
        if self.require_new_choch:
            new_choch = self.choch_detector.detect(
                high=high[last_trade.exit_index:],
                low=low[last_trade.exit_index:],
                close=close[last_trade.exit_index:],
                open_prices=open_prices[last_trade.exit_index:],
                volume=volume[last_trade.exit_index:],
                direction=last_trade.direction
            )
            
            if not new_choch.detected:
                return ReentryEligibility(
                    eligible=False,
                    reason="No new ChoCH formed since stop loss",
                    conditions_met={'new_choch': False},
                    wait_candles=0,
                    risk_reduction=self.risk_reduction
                )
            
            if new_choch.strength < 0.4:
                return ReentryEligibility(
                    eligible=False,
                    reason=f"New ChoCH too weak (strength: {new_choch.strength:.2f})",
                    conditions_met={'new_choch': False},
                    wait_candles=0,
                    risk_reduction=self.risk_reduction
                )
            
            conditions['new_choch'] = True
        
        # ═══════════════════════════════════════════════════════════
        # CHECK 5: New Fibonacci retest? (if required)
        # ═══════════════════════════════════════════════════════════
        if self.require_new_fib:
            # Check if price has retested a Fib level since stop
            # This is a simplified check - actual entry_system will do full validation
            current_price = close[-1]
            
            # For now, just check if price is reasonable for re-entry
            # (not too far from zone)
            price_in_zone = zone.price_low <= current_price <= zone.price_high
            
            if not price_in_zone:
                # Check if approaching zone
                distance_to_zone = min(
                    abs(current_price - zone.price_low),
                    abs(current_price - zone.price_high)
                ) / current_price
                
                if distance_to_zone > 0.02:  # More than 2% away
                    return ReentryEligibility(
                        eligible=False,
                        reason=f"Price too far from zone ({distance_to_zone*100:.1f}% away)",
                        conditions_met={'fib_retest': False},
                        wait_candles=0,
                        risk_reduction=self.risk_reduction
                    )
            
            conditions['fib_retest'] = True
        
        # ═══════════════════════════════════════════════════════════
        # CHECK 6: Max attempts not exceeded?
        # ═══════════════════════════════════════════════════════════
        if zone_attempts >= self.max_attempts:
            return ReentryEligibility(
                eligible=False,
                reason=f"Max attempts reached ({self.max_attempts}) on this zone",
                conditions_met={'attempts_ok': False},
                wait_candles=0,
                risk_reduction=self.risk_reduction
            )
        
        conditions['attempts_ok'] = True
        
        # ═══════════════════════════════════════════════════════════
        # ALL CONDITIONS MET - RE-ENTRY ELIGIBLE!
        # ═══════════════════════════════════════════════════════════
        return ReentryEligibility(
            eligible=True,
            reason=f"✅ Re-entry approved with {self.risk_reduction*100:.0f}% risk",
            conditions_met=conditions,
            wait_candles=0,
            risk_reduction=self.risk_reduction
        )
    
    def _check_zone_validity(
        self,
        zone: Zone,
        close: np.ndarray,
        last_trade_direction: Literal["LONG", "SHORT"],
        exit_index: int
    ) -> bool:
        """
        Check if zone is still valid (not broken)
        
        A zone is broken if:
        - LONG: Any candle closed BELOW zone bottom after stop
        - SHORT: Any candle closed ABOVE zone top after stop
        
        Args:
            zone: Zone to check
            close: Close prices
            last_trade_direction: Direction of last trade
            exit_index: When last trade exited
            
        Returns:
            True if zone still valid
        """
        # Check candles after exit
        closes_after_exit = close[exit_index:]
        
        if last_trade_direction == "LONG":
            # LONG: Zone broken if body closed below zone
            if np.any(closes_after_exit < zone.price_low):
                return False
        else:
            # SHORT: Zone broken if body closed above zone
            if np.any(closes_after_exit > zone.price_high):
                return False
        
        return True
    
    def record_trade(
        self,
        trade_id: str,
        zone_id: str,
        direction: Literal["LONG", "SHORT"],
        entry_price: float,
        exit_price: float,
        result: Literal["WIN", "LOSS", "STOP_LOSS"],
        pnl: float,
        exit_index: int,
        reason: str
    ) -> TradeHistory:
        """
        Record a completed trade
        
        Args:
            trade_id: Unique trade ID
            zone_id: Zone ID this trade was on
            direction: Trade direction
            entry_price: Entry price
            exit_price: Exit price
            result: Trade result
            pnl: Profit/Loss
            exit_index: Candle index when exited
            reason: Exit reason
            
        Returns:
            TradeHistory object
        """
        return TradeHistory(
            trade_id=trade_id,
            zone_id=zone_id,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            result=result,
            pnl=pnl,
            exit_time=datetime.now(),
            exit_index=exit_index,
            reason=reason
        )


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from dataclasses import dataclass
    
    np.random.seed(42)
    
    # Mock Zone
    @dataclass
    class MockZone:
        id: str
        price_low: float
        price_high: float
        price_mid: float
        touch_count: int
        thickness_pct: float
        last_touch_index: int
        creation_index: int
        timeframe: str
        method: str
        quality: float = 8.0
        days_since_last_touch: float = 2.0
    
    # Config
    config = {
        'reentry': {
            'enabled': True,
            'wait_candles': 2,
            'max_attempts': 2,
            'require_new_choch': True,
            'require_new_fib': True
        },
        'entry': {
            'choch': {'min_strength': 0.4},
            'fibonacci': {'levels': [0.705, 0.618]}
        }
    }
    
    # Create manager
    manager = ReentryManager(config)
    
    print("\n" + "="*60)
    print("RE-ENTRY MANAGER - TEST SCENARIOS")
    print("="*60)
    
    # Create mock zone
    zone = MockZone(
        id="BTC_1H_50000",
        price_low=50000,
        price_high=50100,
        price_mid=50050,
        touch_count=2,
        thickness_pct=0.002,
        last_touch_index=50,
        creation_index=30,
        timeframe="1H",
        method="both"
    )
    
    # Simulate price data
    n = 200
    
    # Phase 1: Trade entry and stop
    prices = []
    prices.extend(np.linspace(51000, 50050, 50))  # Approach zone
    prices.extend(np.linspace(50050, 49500, 20))  # Stop loss at 49500
    
    # Phase 2: After stop (cooldown)
    prices.extend(np.linspace(49500, 49800, 10))  # Waiting period
    
    # Phase 3: New setup forming
    prices.extend(np.linspace(49800, 50800, 30))  # New ChoCH
    prices.extend(np.linspace(50800, 50100, 20))  # Fib retest
    
    # Phase 4: Continuation
    prices.extend(np.linspace(50100, 51500, 70))
    
    close = np.array(prices)
    high = close + 50
    low = close - 50
    open_prices = close + np.random.randn(len(close)) * 20
    volume = np.random.rand(len(close)) * 1000 + 500
    
    # Record first trade (stopped out)
    first_trade = manager.record_trade(
        trade_id="TRADE_001",
        zone_id=zone.id,
        direction="LONG",
        entry_price=50050,
        exit_price=49500,
        result="STOP_LOSS",
        pnl=-550,
        exit_index=70,
        reason="Stop loss hit"
    )
    
    print(f"\n📊 First Trade:")
    print(f"   Entry: ${first_trade.entry_price:,.2f}")
    print(f"   Exit: ${first_trade.exit_price:,.2f}")
    print(f"   Result: {first_trade.result}")
    print(f"   PnL: ${first_trade.pnl:,.2f}")
    print(f"   Exit Index: {first_trade.exit_index}")
    
    # Test re-entry eligibility at different points
    test_points = [
        (71, "Immediately after stop"),
        (72, "1 candle later"),
        (73, "2 candles later (cooldown done)"),
        (100, "After new setup forms"),
        (120, "At Fib retest")
    ]
    
    for index, description in test_points:
        print(f"\n{'='*60}")
        print(f"CHECK: {description} (Candle {index})")
        print(f"Price: ${close[index]:,.2f}")
        print(f"{'='*60}")
        
        eligibility = manager.check_reentry_eligibility(
            last_trade=first_trade,
            zone=zone,
            current_index=index,
            high=high[:index+1],
            low=low[:index+1],
            close=close[:index+1],
            open_prices=open_prices[:index+1],
            volume=volume[:index+1],
            zone_attempts=1
        )
        
        print(f"\nEligible: {eligibility.eligible}")
        print(f"Reason: {eligibility.reason}")
        
        if eligibility.wait_candles > 0:
            print(f"Wait: {eligibility.wait_candles} more candles")
        
        if eligibility.eligible:
            print(f"Risk Reduction: {eligibility.risk_reduction*100:.0f}% (Use {eligibility.risk_reduction*100:.0f}% of normal risk)")
        
        print(f"\nConditions:")
        for condition, met in eligibility.conditions_met.items():
            status = "✅" if met else "❌"
            print(f"  {status} {condition}")
        
        if eligibility.eligible:
            print(f"\n🎯 RE-ENTRY APPROVED!")
            print(f"   Use 50% risk (1% instead of 2%)")
            print(f"   This is attempt #2 on this zone")
            break
    
    # ═════════════════════════════════════════════════════════
    # Test max attempts limit
    # ═════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("TEST: Max Attempts Limit")
    print(f"{'='*60}")
    
    eligibility_max = manager.check_reentry_eligibility(
        last_trade=first_trade,
        zone=zone,
        current_index=150,
        high=high,
        low=low,
        close=close,
        open_prices=open_prices,
        volume=volume,
        zone_attempts=2  # Already did 2 attempts
    )
    
    print(f"\nAttempts: 2 (max allowed)")
    print(f"Eligible: {eligibility_max.eligible}")
    print(f"Reason: {eligibility_max.reason}")
    
    print("\n" + "="*60)
    print("✅ Re-entry Manager test complete!")
    print("="*60 + "\n")