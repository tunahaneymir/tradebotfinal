"""
Overtrading Detector - Prevents excessive trading frequency

Monitors trading frequency and quality to prevent:
- Too many trades per day/hour
- Rapid succession trading
- Quality degradation from overtrading
- Coin-specific overtrading
- Churning (small wins/losses)
"""

from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class TradingLimits:
    """Trading frequency limits"""
    # Daily limits
    max_daily_trades: int = 10  # Start liberal (learning phase)
    max_coin_trades_daily: int = 3
    
    # Hourly limits
    max_trades_per_hour: int = 2
    
    # Time limits
    min_time_between_trades: int = 15  # minutes
    min_time_same_coin: int = 30  # minutes
    
    # Quality monitoring
    min_avg_setup_score: float = 55.0
    min_avg_pnl: float = 0.5  # % (avoid churning)


@dataclass
class TradeRecord:
    """Single trade record for tracking"""
    trade_id: str
    symbol: str
    timestamp: datetime
    setup_score: float
    pnl_percent: float
    result: str


class OvertradingDetector:
    """
    Overtrading Detection & Prevention System
    
    Monitors:
    - Trade frequency (daily, hourly, minute)
    - Coin-specific frequency
    - Quality degradation
    - Churning detection
    """
    
    def __init__(self, limits: Optional[TradingLimits] = None):
        self.limits = limits or TradingLimits()
        
        # Trade history
        self.trade_history: List[TradeRecord] = []
        
        # Coin-specific tracking
        self.coin_trade_counts: Dict[str, int] = {}
        self.coin_last_trade: Dict[str, datetime] = {}
        
        # Counters
        self.daily_trade_count = 0
        self.last_trade_time: Optional[datetime] = None
        
        # Phase tracking (learning vs production)
        self.phase: str = "learning"  # or "production"
    
    def check_overtrading(
        self,
        symbol: str,
        setup_score: float,
        current_time: Optional[datetime] = None
    ) -> Dict:
        """
        Check if new trade would constitute overtrading
        
        Returns:
            Dict with:
            - allowed: bool
            - signals: List[str]
            - reason: str
            - remaining_daily: int
            - next_available_time: datetime
        """
        
        if current_time is None:
            current_time = datetime.now()
        
        signals = []
        overtrading_score = 0
        
        # ════════════════════════════════════
        # CHECK 1: Daily Trade Count
        # ════════════════════════════════════
        today_trades = self._get_today_trades(current_time)
        
        if len(today_trades) >= self.limits.max_daily_trades:
            signals.append('daily_limit_reached')
            overtrading_score += 100  # Hard block
        
        # ════════════════════════════════════
        # CHECK 2: Hourly Frequency
        # ════════════════════════════════════
        recent_hour_trades = self._get_trades_in_last_hour(current_time)
        
        if len(recent_hour_trades) >= self.limits.max_trades_per_hour:
            signals.append('hourly_limit_reached')
            overtrading_score += 80
        
        # ════════════════════════════════════
        # CHECK 3: Rapid Succession
        # ════════════════════════════════════
        if self.last_trade_time:
            minutes_since = (current_time - self.last_trade_time).total_seconds() / 60
            
            if minutes_since < self.limits.min_time_between_trades:
                signals.append('rapid_succession')
                overtrading_score += 60
        
        # ════════════════════════════════════
        # CHECK 4: Coin-Specific Limit
        # ════════════════════════════════════
        coin_trades_today = self.coin_trade_counts.get(symbol, 0)
        
        if coin_trades_today >= self.limits.max_coin_trades_daily:
            signals.append('coin_limit_reached')
            overtrading_score += 90
        
        # ════════════════════════════════════
        # CHECK 5: Same Coin Timing
        # ════════════════════════════════════
        if symbol in self.coin_last_trade:
            coin_minutes_since = (
                current_time - self.coin_last_trade[symbol]
            ).total_seconds() / 60
            
            if coin_minutes_since < self.limits.min_time_same_coin:
                signals.append('same_coin_rapid')
                overtrading_score += 50
        
        # ════════════════════════════════════
        # CHECK 6: Quality Degradation
        # ════════════════════════════════════
        if len(today_trades) >= 3:
            recent_scores = [t.setup_score for t in today_trades[-3:]]
            avg_score = sum(recent_scores) / len(recent_scores)
            
            if avg_score < self.limits.min_avg_setup_score:
                signals.append('quality_declining')
                overtrading_score += 40
        
        # ════════════════════════════════════
        # CHECK 7: Churning Detection
        # ════════════════════════════════════
        if len(self.trade_history) >= 5:
            recent_pnl = [abs(t.pnl_percent) for t in self.trade_history[-5:]]
            avg_pnl = sum(recent_pnl) / len(recent_pnl)
            
            if avg_pnl < self.limits.min_avg_pnl:
                signals.append('churning')
                overtrading_score += 35
        
        # ════════════════════════════════════
        # DECISION
        # ════════════════════════════════════
        is_overtrading = overtrading_score >= 60
        
        # Calculate next available time
        next_available = self._calculate_next_available_time(
            signals, 
            current_time
        )
        
        remaining_daily = max(
            0, 
            self.limits.max_daily_trades - len(today_trades)
        )
        
        if is_overtrading:
            reason = f"Overtrading detected: {', '.join(signals)}"
            action = "BLOCK TRADE"
        else:
            reason = "Trading frequency acceptable"
            action = "ALLOW"
        
        return {
            'allowed': not is_overtrading,
            'is_overtrading': is_overtrading,
            'score': overtrading_score,
            'signals': signals,
            'reason': reason,
            'action': action,
            'remaining_daily': remaining_daily,
            'coin_trades_today': coin_trades_today,
            'next_available_time': next_available,
            'phase': self.phase
        }
    
    def on_trade_executed(
        self,
        trade: TradeRecord
    ):
        """Update tracking when trade executed"""
        
        # Add to history
        self.trade_history.append(trade)
        
        # Update counters
        self.daily_trade_count += 1
        self.last_trade_time = trade.timestamp
        
        # Update coin-specific
        self.coin_trade_counts[trade.symbol] = \
            self.coin_trade_counts.get(trade.symbol, 0) + 1
        self.coin_last_trade[trade.symbol] = trade.timestamp
    
    def reset_daily_counters(self):
        """Reset daily counters (call at 00:00 UTC)"""
        self.daily_trade_count = 0
        self.coin_trade_counts = {}
    
    def set_phase(self, phase: str):
        """
        Set trading phase (learning vs production)
        
        Adjusts limits based on phase:
        - Learning: More liberal (collect data)
        - Production: More strict (quality focus)
        """
        
        self.phase = phase
        
        if phase == "learning":
            # Liberal limits for learning
            self.limits.max_daily_trades = 10
            self.limits.max_coin_trades_daily = 3
            self.limits.min_time_between_trades = 10
            self.limits.min_avg_setup_score = 50.0
        
        elif phase == "production":
            # Strict limits for production
            self.limits.max_daily_trades = 5
            self.limits.max_coin_trades_daily = 2
            self.limits.min_time_between_trades = 15
            self.limits.min_avg_setup_score = 60.0
    
    def get_statistics(self) -> Dict:
        """Get trading statistics"""
        
        today_trades = self._get_today_trades(datetime.now())
        
        if not today_trades:
            return {
                'trades_today': 0,
                'avg_setup_score': 0,
                'avg_pnl': 0,
                'win_rate': 0
            }
        
        avg_score = sum(t.setup_score for t in today_trades) / len(today_trades)
        avg_pnl = sum(t.pnl_percent for t in today_trades) / len(today_trades)
        wins = sum(1 for t in today_trades if t.pnl_percent > 0)
        win_rate = wins / len(today_trades) if today_trades else 0
        
        return {
            'trades_today': len(today_trades),
            'avg_setup_score': avg_score,
            'avg_pnl': avg_pnl,
            'win_rate': win_rate,
            'phase': self.phase,
            'limits': {
                'max_daily': self.limits.max_daily_trades,
                'max_hourly': self.limits.max_trades_per_hour,
                'min_time_between': self.limits.min_time_between_trades
            }
        }
    
    # ════════════════════════════════════
    # INTERNAL METHODS
    # ════════════════════════════════════
    
    def _get_today_trades(self, current_time: datetime) -> List[TradeRecord]:
        """Get all trades from today"""
        today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        return [
            t for t in self.trade_history 
            if t.timestamp >= today_start
        ]
    
    def _get_trades_in_last_hour(self, current_time: datetime) -> List[TradeRecord]:
        """Get trades from last 60 minutes"""
        hour_ago = current_time - timedelta(hours=1)
        return [
            t for t in self.trade_history 
            if t.timestamp >= hour_ago
        ]
    
    def _calculate_next_available_time(
        self, 
        signals: List[str],
        current_time: datetime
    ) -> Optional[datetime]:
        """Calculate when next trade can be made"""
        
        if not signals:
            return current_time
        
        # Find longest wait time
        wait_times = []
        
        if 'rapid_succession' in signals and self.last_trade_time:
            wait = self.limits.min_time_between_trades
            next_time = self.last_trade_time + timedelta(minutes=wait)
            wait_times.append(next_time)
        
        if 'hourly_limit_reached' in signals:
            # Wait until oldest trade in last hour expires
            hour_ago = current_time - timedelta(hours=1)
            hour_trades = self._get_trades_in_last_hour(current_time)
            if hour_trades:
                oldest = min(t.timestamp for t in hour_trades)
                next_time = oldest + timedelta(hours=1, minutes=1)
                wait_times.append(next_time)
        
        if 'daily_limit_reached' in signals:
            # Wait until tomorrow
            tomorrow = current_time + timedelta(days=1)
            tomorrow_start = tomorrow.replace(hour=0, minute=0, second=0)
            wait_times.append(tomorrow_start)
        
        return max(wait_times) if wait_times else None


# ════════════════════════════════════════
# USAGE EXAMPLES
# ════════════════════════════════════════

def example_usage():
    """Example overtrading detection scenarios"""
    
    detector = OvertradingDetector()
    
    print("=" * 70)
    print("OVERTRADING DETECTOR EXAMPLES")
    print("=" * 70)
    print()
    
    # ════════════════════════════════════════
    # SCENARIO 1: Normal Trading (Allowed)
    # ════════════════════════════════════════
    print("SCENARIO 1: First Trade of the Day")
    print("-" * 70)
    
    result = detector.check_overtrading(
        symbol="BTCUSDT",
        setup_score=75.0,
        current_time=datetime.now()
    )
    
    print(f"Allowed: {result['allowed']}")
    print(f"Remaining Today: {result['remaining_daily']}")
    print(f"Phase: {result['phase']}")
    print(f"Action: {result['action']}")
    print()
    
    # Execute trade
    trade1 = TradeRecord(
        trade_id="T001",
        symbol="BTCUSDT",
        timestamp=datetime.now(),
        setup_score=75.0,
        pnl_percent=2.1,
        result="WIN"
    )
    detector.on_trade_executed(trade1)
    
    # ════════════════════════════════════════
    # SCENARIO 2: Rapid Succession (Blocked)
    # ════════════════════════════════════════
    print("SCENARIO 2: Rapid Trading (5 min later)")
    print("-" * 70)
    
    result2 = detector.check_overtrading(
        symbol="ETHUSDT",
        setup_score=72.0,
        current_time=datetime.now() + timedelta(minutes=5)
    )
    
    print(f"Allowed: {result2['allowed']}")
    print(f"Overtrading: {result2['is_overtrading']}")
    print(f"Score: {result2['score']}")
    print(f"Signals: {result2['signals']}")
    print(f"Reason: {result2['reason']}")
    if result2['next_available_time']:
        wait = (result2['next_available_time'] - datetime.now()).total_seconds() / 60
        print(f"Wait: {wait:.0f} minutes")
    print()
    
    # ════════════════════════════════════════
    # SCENARIO 3: Daily Limit Approaching
    # ════════════════════════════════════════
    print("SCENARIO 3: Multiple Trades (Daily Limit Check)")
    print("-" * 70)
    
    # Simulate 8 more trades
    for i in range(8):
        trade = TradeRecord(
            trade_id=f"T{i+2:03d}",
            symbol=f"COIN{i}",
            timestamp=datetime.now() + timedelta(minutes=20*i),
            setup_score=70.0 - i*2,  # Quality declining
            pnl_percent=1.0,
            result="WIN"
        )
        detector.on_trade_executed(trade)
    
    # Check 10th trade
    result3 = detector.check_overtrading(
        symbol="SOLUSDT",
        setup_score=68.0,
        current_time=datetime.now() + timedelta(minutes=200)
    )
    
    print(f"Trades Today: {result3['remaining_daily']} remaining")
    print(f"Allowed: {result3['allowed']}")
    print(f"Signals: {result3['signals']}")
    
    # Get statistics
    stats = detector.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total Today: {stats['trades_today']}")
    print(f"  Avg Setup Score: {stats['avg_setup_score']:.1f}")
    print(f"  Win Rate: {stats['win_rate']:.1%}")
    print()
    
    # ════════════════════════════════════════
    # SCENARIO 4: Phase Change (Production)
    # ════════════════════════════════════════
    print("SCENARIO 4: Switch to Production Phase")
    print("-" * 70)
    
    detector.set_phase("production")
    
    result4 = detector.check_overtrading(
        symbol="BTCUSDT",
        setup_score=75.0,
        current_time=datetime.now() + timedelta(minutes=220)
    )
    
    print(f"Phase: {result4['phase']}")
    print(f"New Limits:")
    print(f"  Max Daily: {detector.limits.max_daily_trades}")
    print(f"  Max Coin/Day: {detector.limits.max_coin_trades_daily}")
    print(f"  Min Time Between: {detector.limits.min_time_between_trades} min")
    print(f"\nAllowed: {result4['allowed']}")
    print(f"Reason: {result4['reason']}")
    print()
    
    # ════════════════════════════════════════
    # SCENARIO 5: Churning Detection
    # ════════════════════════════════════════
    print("SCENARIO 5: Churning Detection (Small Wins/Losses)")
    print("-" * 70)
    
    # Reset detector
    detector2 = OvertradingDetector()
    
    # Add 5 small trades
    small_trades = [0.3, -0.2, 0.4, -0.3, 0.5]  # All < 0.5%
    for i, pnl in enumerate(small_trades):
        trade = TradeRecord(
            trade_id=f"S{i+1:03d}",
            symbol="BTCUSDT",
            timestamp=datetime.now() + timedelta(minutes=20*i),
            setup_score=65.0,
            pnl_percent=pnl,
            result="WIN" if pnl > 0 else "LOSS"
        )
        detector2.on_trade_executed(trade)
    
    result5 = detector2.check_overtrading(
        symbol="BTCUSDT",
        setup_score=62.0,
        current_time=datetime.now() + timedelta(minutes=120)
    )
    
    print(f"Signals: {result5['signals']}")
    if 'churning' in result5['signals']:
        print("⚠️ Churning detected!")
        print("   Small wins/losses - not making progress")
        stats5 = detector2.get_statistics()
        print(f"   Avg PnL: {stats5['avg_pnl']:.2f}%")
    print()
    
    print("=" * 70)


if __name__ == "__main__":
    example_usage()