"""
Anti-Revenge Trading System - Prevents emotional revenge trades

Implements coin-based cooldown system to prevent rapid trading after losses.

Cooldown Levels:
1. Single loss: 15 min (soft warning)
2. Large loss: 30 min (hard block)
3. Double loss: 45 min + risk reduction
4. Triple loss: 2 hours + quality increase + global warning
5. Daily limit: Rest of day (stop trading)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class CoinState:
    """Track state for each coin"""
    symbol: str
    
    # Trade history
    last_trade_time: Optional[datetime] = None
    last_trade_result: Optional[Literal["WIN", "LOSS"]] = None
    last_trade_pnl: float = 0.0
    
    # Loss tracking
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    
    # Cooldown state
    cooldown_active: bool = False
    cooldown_until: Optional[datetime] = None
    cooldown_level: Optional[str] = None
    cooldown_reason: Optional[str] = None
    
    # Next trade adjustments
    next_trade_risk_multiplier: float = 1.0
    next_trade_quality_min: Optional[float] = None
    
    # Status
    status: str = "READY"
    warnings: List[str] = field(default_factory=list)


@dataclass
class TradeHistory:
    """Single trade record"""
    trade_id: str
    symbol: str
    timestamp: datetime
    result: Literal["WIN", "LOSS"]
    pnl_percent: float
    setup_score: float


class AntiRevengeManager:
    """
    Revenge Trade Prevention System
    
    Prevents emotional revenge trading through:
    - Coin-specific cooldowns
    - Risk reduction after losses
    - Quality threshold increases
    - Recovery protocols
    """
    
    def __init__(self):
        # Coin states
        self.coin_states: Dict[str, CoinState] = {}
        
        # Cooldown durations (minutes)
        self.SINGLE_LOSS_COOLDOWN = 15
        self.LARGE_LOSS_COOLDOWN = 30
        self.DOUBLE_LOSS_COOLDOWN = 45
        self.TRIPLE_LOSS_COOLDOWN = 120  # 2 hours
        
        # Loss thresholds
        self.LARGE_LOSS_THRESHOLD = -1.2  # %
        
        # Risk reductions
        self.DOUBLE_LOSS_RISK_REDUCTION = 0.75  # 75% of normal
        self.TRIPLE_LOSS_RISK_REDUCTION = 0.50  # 50% of normal
        
        # Quality increases
        self.TRIPLE_LOSS_QUALITY_INCREASE = 2  # +2 points
        
        # Global tracking
        self.daily_loss_count = 0
        self.daily_loss_percent = 0.0
        self.daily_loss_limit = -6.0  # %
        self.max_losses_per_day = 3
        
        # Recovery tracking
        self.global_warning_active = False
    
    def check_cooldown(
        self, 
        symbol: str, 
        current_time: Optional[datetime] = None
    ) -> Dict:
        """
        Check if coin is in cooldown
        
        Returns:
            Dict with:
            - in_cooldown: bool
            - level: str
            - remaining_minutes: int
            - reason: str
            - allowed: bool
            - adjustments: Dict (risk, quality)
        """
        
        if current_time is None:
            current_time = datetime.now()
        
        # Get or create coin state
        state = self._get_coin_state(symbol)
        
        # Check if cooldown expired
        if state.cooldown_active and state.cooldown_until:
            if current_time >= state.cooldown_until:
                # Cooldown expired
                self._clear_cooldown(symbol)
                state = self._get_coin_state(symbol)
        
        # Calculate remaining time
        remaining_minutes = 0
        if state.cooldown_active and state.cooldown_until:
            remaining = state.cooldown_until - current_time
            remaining_minutes = max(0, int(remaining.total_seconds() / 60))
        
        return {
            'in_cooldown': state.cooldown_active,
            'level': state.cooldown_level,
            'remaining_minutes': remaining_minutes,
            'reason': state.cooldown_reason,
            'allowed': not state.cooldown_active,
            'status': state.status,
            'warnings': state.warnings,
            'adjustments': {
                'risk_multiplier': state.next_trade_risk_multiplier,
                'quality_min': state.next_trade_quality_min
            }
        }
    
    def on_trade_closed(
        self, 
        trade: TradeHistory,
        current_time: Optional[datetime] = None
    ) -> Dict:
        """
        Update state when trade closes
        
        Returns:
            Dict with cooldown info and adjustments
        """
        
        if current_time is None:
            current_time = datetime.now()
        
        symbol = trade.symbol
        state = self._get_coin_state(symbol)
        
        # Update trade history
        state.last_trade_time = current_time
        state.last_trade_result = trade.result
        state.last_trade_pnl = trade.pnl_percent
        
        # ════════════════════════════════════
        # HANDLE WIN
        # ════════════════════════════════════
        if trade.result == "WIN":
            state.consecutive_wins += 1
            state.consecutive_losses = 0
            
            # Reset adjustments after win
            state.next_trade_risk_multiplier = 1.0
            state.next_trade_quality_min = None
            state.warnings = []
            
            # Clear global warning if back to normal
            if state.consecutive_wins >= 2:
                self.global_warning_active = False
            
            return {
                'cooldown_applied': False,
                'message': 'Win - state reset',
                'consecutive_wins': state.consecutive_wins,
                'adjustments': {
                    'risk_multiplier': 1.0,
                    'quality_min': None
                }
            }
        
        # ════════════════════════════════════
        # HANDLE LOSS
        # ════════════════════════════════════
        else:
            state.consecutive_losses += 1
            state.consecutive_wins = 0
            
            # Update daily loss tracking
            self.daily_loss_count += 1
            self.daily_loss_percent += trade.pnl_percent
            
            # Check daily limit
            if self._check_daily_limit_reached():
                return self._apply_daily_limit()
            
            # Apply appropriate cooldown level
            if state.consecutive_losses == 1:
                return self._apply_single_loss_cooldown(state, trade, current_time)
            elif state.consecutive_losses == 2:
                return self._apply_double_loss_cooldown(state, trade, current_time)
            elif state.consecutive_losses >= 3:
                return self._apply_triple_loss_cooldown(state, trade, current_time)
    
    def detect_revenge_pattern(
        self,
        symbol: str,
        setup_score: float,
        current_time: Optional[datetime] = None
    ) -> Dict:
        """
        Detect revenge trading patterns
        
        Returns:
            Dict with:
            - is_revenge: bool
            - score: int
            - signals: List[str]
            - reason: str
        """
        
        if current_time is None:
            current_time = datetime.now()
        
        state = self._get_coin_state(symbol)
        
        if state.last_trade_result != "LOSS":
            return {
                'is_revenge': False,
                'score': 0,
                'signals': [],
                'reason': 'Last trade was not a loss'
            }
        
        signals = []
        revenge_score = 0
        
        # Signal 1: Rapid trading after loss
        if state.last_trade_time:
            time_diff = (current_time - state.last_trade_time).total_seconds() / 60
            if time_diff < 15:
                signals.append('rapid_after_loss')
                revenge_score += 60
        
        # Signal 2: Quality degradation
        # (Assumes we have access to previous setup score)
        if setup_score < 50:
            signals.append('low_quality_setup')
            revenge_score += 40
        
        # Signal 3: Consecutive losses
        if state.consecutive_losses >= 2:
            signals.append('losing_streak')
            revenge_score += 50
        
        # Signal 4: In cooldown
        if state.cooldown_active:
            signals.append('cooldown_violation')
            revenge_score += 70
        
        is_revenge = revenge_score >= 60
        
        return {
            'is_revenge': is_revenge,
            'score': revenge_score,
            'signals': signals,
            'reason': f"Revenge pattern: {', '.join(signals)}" if is_revenge else "No revenge detected",
            'consecutive_losses': state.consecutive_losses
        }
    
    # ════════════════════════════════════
    # INTERNAL METHODS
    # ════════════════════════════════════
    
    def _get_coin_state(self, symbol: str) -> CoinState:
        """Get or create coin state"""
        if symbol not in self.coin_states:
            self.coin_states[symbol] = CoinState(symbol=symbol)
        return self.coin_states[symbol]
    
    def _clear_cooldown(self, symbol: str):
        """Clear cooldown for coin"""
        state = self._get_coin_state(symbol)
        state.cooldown_active = False
        state.cooldown_until = None
        state.cooldown_level = None
        state.cooldown_reason = None
        state.status = "READY"
    
    def _apply_single_loss_cooldown(
        self, 
        state: CoinState, 
        trade: TradeHistory,
        current_time: datetime
    ) -> Dict:
        """Apply level 1 cooldown (soft warning)"""
        
        # Check if large loss
        if trade.pnl_percent <= self.LARGE_LOSS_THRESHOLD:
            return self._apply_large_loss_cooldown(state, trade, current_time)
        
        # Soft cooldown (15 min)
        state.cooldown_active = True
        state.cooldown_until = current_time + timedelta(minutes=self.SINGLE_LOSS_COOLDOWN)
        state.cooldown_level = "single_loss"
        state.cooldown_reason = "Single loss - soft cooldown"
        state.status = "🟡 SOFT COOLDOWN"
        state.warnings = ["First loss - 15 min cooldown"]
        
        return {
            'cooldown_applied': True,
            'level': 'single_loss',
            'duration_minutes': self.SINGLE_LOSS_COOLDOWN,
            'message': 'Single loss detected - 15 minute soft cooldown',
            'adjustments': {
                'risk_multiplier': 1.0,  # No reduction yet
                'quality_min': None
            }
        }
    
    def _apply_large_loss_cooldown(
        self,
        state: CoinState,
        trade: TradeHistory,
        current_time: datetime
    ) -> Dict:
        """Apply large loss cooldown (30 min, hard block)"""
        
        state.cooldown_active = True
        state.cooldown_until = current_time + timedelta(minutes=self.LARGE_LOSS_COOLDOWN)
        state.cooldown_level = "large_loss"
        state.cooldown_reason = f"Large loss ({trade.pnl_percent:.1f}%) - hard block"
        state.status = "🔴 HARD COOLDOWN"
        state.warnings = [f"Large loss {trade.pnl_percent:.1f}% - 30 min block"]
        
        return {
            'cooldown_applied': True,
            'level': 'large_loss',
            'duration_minutes': self.LARGE_LOSS_COOLDOWN,
            'message': f'Large loss {trade.pnl_percent:.1f}% - 30 minute hard cooldown',
            'adjustments': {
                'risk_multiplier': 1.0,
                'quality_min': None
            }
        }
    
    def _apply_double_loss_cooldown(
        self,
        state: CoinState,
        trade: TradeHistory,
        current_time: datetime
    ) -> Dict:
        """Apply level 3 cooldown (45 min + risk reduction)"""
        
        state.cooldown_active = True
        state.cooldown_until = current_time + timedelta(minutes=self.DOUBLE_LOSS_COOLDOWN)
        state.cooldown_level = "double_loss"
        state.cooldown_reason = "2 consecutive losses - recovery mode"
        state.status = "🔴 RECOVERY MODE"
        state.warnings = ["2 losses - 45 min cooldown", "Risk reduced to 75%"]
        
        # Risk reduction
        state.next_trade_risk_multiplier = self.DOUBLE_LOSS_RISK_REDUCTION
        
        return {
            'cooldown_applied': True,
            'level': 'double_loss',
            'duration_minutes': self.DOUBLE_LOSS_COOLDOWN,
            'message': '2 consecutive losses - 45 min cooldown + risk reduction',
            'adjustments': {
                'risk_multiplier': self.DOUBLE_LOSS_RISK_REDUCTION,
                'quality_min': None
            }
        }
    
    def _apply_triple_loss_cooldown(
        self,
        state: CoinState,
        trade: TradeHistory,
        current_time: datetime
    ) -> Dict:
        """Apply level 4 cooldown (2 hours + quality increase + global warning)"""
        
        state.cooldown_active = True
        state.cooldown_until = current_time + timedelta(minutes=self.TRIPLE_LOSS_COOLDOWN)
        state.cooldown_level = "triple_loss"
        state.cooldown_reason = "3+ consecutive losses - critical recovery"
        state.status = "⚠️ CRITICAL"
        state.warnings = [
            "3+ losses - 2 hour cooldown",
            "Risk reduced to 50%",
            "Quality minimum raised +2"
        ]
        
        # Heavy risk reduction
        state.next_trade_risk_multiplier = self.TRIPLE_LOSS_RISK_REDUCTION
        
        # Quality increase
        state.next_trade_quality_min = self.TRIPLE_LOSS_QUALITY_INCREASE
        
        # Global warning
        self.global_warning_active = True
        
        return {
            'cooldown_applied': True,
            'level': 'triple_loss',
            'duration_minutes': self.TRIPLE_LOSS_COOLDOWN,
            'message': '3+ losses - 2 hour cooldown + heavy restrictions',
            'global_warning': True,
            'adjustments': {
                'risk_multiplier': self.TRIPLE_LOSS_RISK_REDUCTION,
                'quality_min': self.TRIPLE_LOSS_QUALITY_INCREASE
            }
        }
    
    def _check_daily_limit_reached(self) -> bool:
        """Check if daily loss limit reached"""
        return (
            self.daily_loss_count >= self.max_losses_per_day or
            self.daily_loss_percent <= self.daily_loss_limit
        )
    
    def _apply_daily_limit(self) -> Dict:
        """Apply daily limit (stop trading)"""
        return {
            'cooldown_applied': True,
            'level': 'daily_limit',
            'duration_minutes': 9999,  # Rest of day
            'message': 'Daily loss limit reached - TRADING STOPPED',
            'daily_losses': self.daily_loss_count,
            'daily_loss_percent': self.daily_loss_percent,
            'adjustments': {
                'risk_multiplier': 0.0,  # No trading
                'quality_min': None
            }
        }
    
    def reset_daily_counters(self):
        """Reset daily counters (call at 00:00 UTC)"""
        self.daily_loss_count = 0
        self.daily_loss_percent = 0.0
        self.global_warning_active = False


# ════════════════════════════════════════
# USAGE EXAMPLES
# ════════════════════════════════════════

def example_usage():
    """Example revenge prevention scenarios"""
    
    manager = AntiRevengeManager()
    
    print("=" * 70)
    print("ANTI-REVENGE SYSTEM EXAMPLES")
    print("=" * 70)
    print()
    
    # ════════════════════════════════════════
    # SCENARIO 1: Single Loss
    # ════════════════════════════════════════
    print("SCENARIO 1: Single Loss on BTCUSDT")
    print("-" * 70)
    
    trade1 = TradeHistory(
        trade_id="T001",
        symbol="BTCUSDT",
        timestamp=datetime.now(),
        result="LOSS",
        pnl_percent=-1.5,
        setup_score=75
    )
    
    result = manager.on_trade_closed(trade1)
    print(f"Cooldown Applied: {result['cooldown_applied']}")
    print(f"Level: {result.get('level')}")
    print(f"Duration: {result.get('duration_minutes')} minutes")
    print(f"Message: {result['message']}")
    print()
    
    # Check cooldown
    check = manager.check_cooldown("BTCUSDT")
    print(f"In Cooldown: {check['in_cooldown']}")
    print(f"Remaining: {check['remaining_minutes']} minutes")
    print(f"Status: {check['status']}")
    print()
    
    # ════════════════════════════════════════
    # SCENARIO 2: Double Loss (same coin)
    # ════════════════════════════════════════
    print("SCENARIO 2: Double Loss on BTCUSDT")
    print("-" * 70)
    
    # Simulate time pass
    import time
    time.sleep(1)
    
    trade2 = TradeHistory(
        trade_id="T002",
        symbol="BTCUSDT",
        timestamp=datetime.now() + timedelta(minutes=20),
        result="LOSS",
        pnl_percent=-1.3,
        setup_score=68
    )
    
    result2 = manager.on_trade_closed(trade2, current_time=trade2.timestamp)
    print(f"Level: {result2['level']}")
    print(f"Duration: {result2['duration_minutes']} minutes")
    print(f"Risk Multiplier: {result2['adjustments']['risk_multiplier']}")
    print(f"Message: {result2['message']}")
    print()
    
    # ════════════════════════════════════════
    # SCENARIO 3: Revenge Detection
    # ════════════════════════════════════════
    print("SCENARIO 3: Revenge Pattern Detection")
    print("-" * 70)
    
    # Try to trade immediately (revenge!)
    revenge_check = manager.detect_revenge_pattern(
        symbol="BTCUSDT",
        setup_score=52,  # Quality dropped!
        current_time=trade2.timestamp + timedelta(minutes=5)
    )
    
    print(f"Revenge Detected: {revenge_check['is_revenge']}")
    print(f"Score: {revenge_check['score']}")
    print(f"Signals: {revenge_check['signals']}")
    print(f"Reason: {revenge_check['reason']}")
    print()
    
    print("=" * 70)


if __name__ == "__main__":
    example_usage()