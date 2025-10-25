"""
Zone Memory - Parça 3
Based on: pa-strateji2 Parça 3

Zone Memory System:
- Tracks zone performance (wins/losses)
- Maintains trade history per zone
- Calculates zone statistics
- Blacklist management
- RL analysis and recommendations
"""

from __future__ import annotations
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ZoneTradeRecord:
    """Single trade record on a zone"""
    trade_id: str
    attempt_number: int  # 1, 2, 3 (first, re-entry 1, re-entry 2)
    entry_price: float
    stop_loss: float
    result: Literal["WIN", "LOSS", "STOP_LOSS"]
    pnl_percent: float
    entry_time: datetime
    exit_time: datetime
    choch_strength: float
    fib_level: str  # "0.705" or "0.618"
    reason: str  # Exit reason


@dataclass
class ZoneStatistics:
    """Statistical analysis of a zone"""
    total_attempts: int = 0
    wins: int = 0
    losses: int = 0
    stops: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    best_pnl: float = 0.0
    worst_pnl: float = 0.0
    avg_choch_strength: float = 0.0
    preferred_fib: Optional[str] = None  # "0.705" or "0.618"
    consecutive_losses: int = 0
    last_result: Optional[str] = None


@dataclass
class ZoneMemory:
    """Complete memory of a zone"""
    zone_id: str
    coin: str
    timeframe: str
    price_bottom: float
    price_top: float
    quality: float
    created: datetime
    
    # Trade history
    trades: List[ZoneTradeRecord] = field(default_factory=list)
    
    # Statistics
    statistics: ZoneStatistics = field(default_factory=ZoneStatistics)
    
    # RL Analysis
    rl_analysis: Dict = field(default_factory=dict)
    
    # Blacklist status
    blacklisted: bool = False
    blacklist_reason: Optional[str] = None
    blacklist_date: Optional[datetime] = None
    blacklist_type: Optional[Literal["PERMANENT", "TEMPORARY"]] = None
    blacklist_expires: Optional[datetime] = None


class ZoneMemoryManager:
    """
    Zone Memory Management System
    
    Tracks and analyzes zone performance:
    1. Records all trades on each zone
    2. Calculates statistics and metrics
    3. Provides RL analysis and recommendations
    4. Manages blacklist (temporary/permanent)
    5. Learns optimal parameters per zone
    
    Usage:
        manager = ZoneMemoryManager()
        
        # Record trade
        manager.record_trade(
            zone_id="BTC_1H_50000",
            trade_record=record
        )
        
        # Get zone memory
        memory = manager.get_zone_memory("BTC_1H_50000")
        
        # Check if blacklisted
        if manager.is_blacklisted("BTC_1H_50000"):
            # Skip this zone
        
        # Get RL recommendation
        recommendation = manager.get_rl_recommendation("BTC_1H_50000")
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize Zone Memory Manager
        
        Args:
            storage_path: Optional path to persist zone memories
        """
        self.storage_path = storage_path
        self.zones: Dict[str, ZoneMemory] = {}
        
        # Blacklist thresholds
        self.consecutive_loss_threshold = 3  # 3 losses → blacklist
        self.min_attempts_for_stats = 5  # Need 5+ attempts for statistics
        self.low_winrate_threshold = 0.30  # <30% → blacklist
        self.negative_pnl_threshold = -1.0  # Avg -1% → blacklist
        
        # Load existing memories if storage path provided
        if storage_path:
            self._load_from_storage()
    
    def create_zone_memory(
        self,
        zone_id: str,
        coin: str,
        timeframe: str,
        price_bottom: float,
        price_top: float,
        quality: float
    ) -> ZoneMemory:
        """
        Create new zone memory
        
        Args:
            zone_id: Unique zone ID
            coin: Coin symbol
            timeframe: Timeframe
            price_bottom: Zone bottom price
            price_top: Zone top price
            quality: Zone quality score
            
        Returns:
            ZoneMemory object
        """
        memory = ZoneMemory(
            zone_id=zone_id,
            coin=coin,
            timeframe=timeframe,
            price_bottom=price_bottom,
            price_top=price_top,
            quality=quality,
            created=datetime.now()
        )
        
        self.zones[zone_id] = memory
        return memory
    
    def record_trade(
        self,
        zone_id: str,
        trade_record: ZoneTradeRecord
    ) -> None:
        """
        Record a trade on a zone
        
        Args:
            zone_id: Zone ID
            trade_record: Trade record to add
        """
        if zone_id not in self.zones:
            raise ValueError(f"Zone {zone_id} not found in memory")
        
        memory = self.zones[zone_id]
        
        # Add trade to history
        memory.trades.append(trade_record)
        
        # Update statistics
        self._update_statistics(memory)
        
        # Check blacklist conditions
        self._check_blacklist(memory)
        
        # Update RL analysis
        self._update_rl_analysis(memory)
        
        # Save to storage if enabled
        if self.storage_path:
            self._save_to_storage()
    
    def get_zone_memory(self, zone_id: str) -> Optional[ZoneMemory]:
        """Get zone memory by ID"""
        return self.zones.get(zone_id)
    
    def is_blacklisted(self, zone_id: str) -> bool:
        """Check if zone is blacklisted"""
        memory = self.zones.get(zone_id)
        if not memory:
            return False
        
        # Check if blacklisted and not expired
        if memory.blacklisted:
            if memory.blacklist_type == "PERMANENT":
                return True
            elif memory.blacklist_type == "TEMPORARY":
                if memory.blacklist_expires and datetime.now() < memory.blacklist_expires:
                    return True
                else:
                    # Expired, remove blacklist
                    memory.blacklisted = False
                    return False
        
        return False
    
    def get_rl_recommendation(self, zone_id: str) -> Dict:
        """
        Get RL recommendation for a zone
        
        Returns:
            Dict with recommendation and reasoning
        """
        memory = self.zones.get(zone_id)
        if not memory:
            return {
                'recommended': False,
                'reason': 'Zone not in memory'
            }
        
        if memory.blacklisted:
            return {
                'recommended': False,
                'reason': f'Blacklisted: {memory.blacklist_reason}'
            }
        
        stats = memory.statistics
        
        # Not enough data
        if stats.total_attempts < 3:
            return {
                'recommended': True,
                'reason': 'Insufficient data, allow learning',
                'confidence': 0.3
            }
        
        # Good performance
        if stats.win_rate >= 0.6 and stats.avg_pnl > 0:
            return {
                'recommended': True,
                'reason': f'Good performance (WR: {stats.win_rate:.0%}, Avg PnL: {stats.avg_pnl:.1f}%)',
                'confidence': 0.8,
                'preferred_fib': stats.preferred_fib,
                'optimal_choch': stats.avg_choch_strength
            }
        
        # Mediocre performance
        if stats.win_rate >= 0.45:
            return {
                'recommended': True,
                'reason': f'Acceptable performance (WR: {stats.win_rate:.0%})',
                'confidence': 0.5,
                'preferred_fib': stats.preferred_fib
            }
        
        # Poor performance
        return {
            'recommended': False,
            'reason': f'Poor performance (WR: {stats.win_rate:.0%}, Avg PnL: {stats.avg_pnl:.1f}%)',
            'confidence': 0.2
        }
    
    def _update_statistics(self, memory: ZoneMemory) -> None:
        """Update zone statistics"""
        stats = memory.statistics
        trades = memory.trades
        
        if not trades:
            return
        
        # Basic counts
        stats.total_attempts = len(trades)
        stats.wins = sum(1 for t in trades if t.result == "WIN")
        stats.losses = sum(1 for t in trades if t.result == "LOSS")
        stats.stops = sum(1 for t in trades if t.result == "STOP_LOSS")
        
        # Win rate
        stats.win_rate = stats.wins / stats.total_attempts if stats.total_attempts > 0 else 0
        
        # PnL statistics
        pnls = [t.pnl_percent for t in trades]
        stats.avg_pnl = sum(pnls) / len(pnls) if pnls else 0
        stats.best_pnl = max(pnls) if pnls else 0
        stats.worst_pnl = min(pnls) if pnls else 0
        
        # ChoCH strength
        choch_strengths = [t.choch_strength for t in trades]
        stats.avg_choch_strength = sum(choch_strengths) / len(choch_strengths) if choch_strengths else 0
        
        # Preferred Fib level (which works better?)
        fib_705_trades = [t for t in trades if t.fib_level == "0.705"]
        fib_618_trades = [t for t in trades if t.fib_level == "0.618"]
        
        if fib_705_trades and fib_618_trades:
            wr_705 = sum(1 for t in fib_705_trades if t.result == "WIN") / len(fib_705_trades)
            wr_618 = sum(1 for t in fib_618_trades if t.result == "WIN") / len(fib_618_trades)
            stats.preferred_fib = "0.705" if wr_705 > wr_618 else "0.618"
        elif fib_705_trades:
            stats.preferred_fib = "0.705"
        elif fib_618_trades:
            stats.preferred_fib = "0.618"
        
        # Consecutive losses
        stats.consecutive_losses = 0
        for trade in reversed(trades):
            if trade.result in ["LOSS", "STOP_LOSS"]:
                stats.consecutive_losses += 1
            else:
                break
        
        # Last result
        stats.last_result = trades[-1].result if trades else None
    
    def _check_blacklist(self, memory: ZoneMemory) -> None:
        """Check if zone should be blacklisted"""
        stats = memory.statistics
        
        # Already blacklisted
        if memory.blacklisted:
            return
        
        # ═══════════════════════════════════════════════════════════
        # CONDITION 1: 3 Consecutive Losses
        # ═══════════════════════════════════════════════════════════
        if stats.consecutive_losses >= self.consecutive_loss_threshold:
            memory.blacklisted = True
            memory.blacklist_reason = f"{stats.consecutive_losses} consecutive losses"
            memory.blacklist_date = datetime.now()
            memory.blacklist_type = "PERMANENT"
            return
        
        # Need enough attempts for statistical blacklist
        if stats.total_attempts < self.min_attempts_for_stats:
            return
        
        # ═══════════════════════════════════════════════════════════
        # CONDITION 2: Very Low Win Rate
        # ═══════════════════════════════════════════════════════════
        if stats.win_rate < self.low_winrate_threshold:
            memory.blacklisted = True
            memory.blacklist_reason = f"Low win rate ({stats.win_rate:.0%})"
            memory.blacklist_date = datetime.now()
            memory.blacklist_type = "TEMPORARY"
            # Blacklist for 30 days
            from datetime import timedelta
            memory.blacklist_expires = datetime.now() + timedelta(days=30)
            return
        
        # ═══════════════════════════════════════════════════════════
        # CONDITION 3: Negative Average PnL
        # ═══════════════════════════════════════════════════════════
        if stats.avg_pnl < self.negative_pnl_threshold:
            memory.blacklisted = True
            memory.blacklist_reason = f"Negative avg PnL ({stats.avg_pnl:.1f}%)"
            memory.blacklist_date = datetime.now()
            memory.blacklist_type = "TEMPORARY"
            # Blacklist for 60 days
            from datetime import timedelta
            memory.blacklist_expires = datetime.now() + timedelta(days=60)
            return
    
    def _update_rl_analysis(self, memory: ZoneMemory) -> None:
        """Update RL analysis for zone"""
        stats = memory.statistics
        
        memory.rl_analysis = {
            'reliability_score': self._calculate_reliability_score(stats),
            're_entry_recommended': self._should_recommend_reentry(stats),
            'optimal_choch_strength': stats.avg_choch_strength if stats.total_attempts >= 3 else None,
            'optimal_fib_level': stats.preferred_fib,
            'notes': self._generate_notes(stats)
        }
    
    def _calculate_reliability_score(self, stats: ZoneStatistics) -> float:
        """Calculate reliability score (0-10)"""
        if stats.total_attempts == 0:
            return 5.0  # Neutral
        
        score = 5.0  # Base
        
        # Win rate contribution (±3 points)
        score += (stats.win_rate - 0.5) * 6
        
        # Avg PnL contribution (±2 points)
        score += stats.avg_pnl / 2
        
        return max(0, min(10, score))
    
    def _should_recommend_reentry(self, stats: ZoneStatistics) -> bool:
        """Should recommend re-entry on this zone?"""
        if stats.total_attempts < 2:
            return True  # Not enough data
        
        # Check re-entry specific performance
        reentry_trades = [t for t in stats.total_attempts if t.attempt_number > 1]
        if not reentry_trades:
            return True
        
        reentry_wins = sum(1 for t in reentry_trades if t.result == "WIN")
        reentry_wr = reentry_wins / len(reentry_trades)
        
        return reentry_wr >= 0.5
    
    def _generate_notes(self, stats: ZoneStatistics) -> str:
        """Generate RL notes about zone"""
        notes = []
        
        if stats.total_attempts >= 5:
            if stats.win_rate >= 0.7:
                notes.append("High win rate zone - prioritize")
            elif stats.win_rate <= 0.3:
                notes.append("Low win rate - consider avoiding")
        
        if stats.preferred_fib:
            notes.append(f"Prefers {stats.preferred_fib} entry")
        
        if stats.consecutive_losses >= 2:
            notes.append("Recent losing streak - caution")
        
        return "; ".join(notes) if notes else "Insufficient data"
    
    def _save_to_storage(self) -> None:
        """Save zone memories to file"""
        # Implementation for persistence (JSON/Database)
        pass
    
    def _load_from_storage(self) -> None:
        """Load zone memories from file"""
        # Implementation for loading (JSON/Database)
        pass


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ZONE MEMORY - TEST")
    print("="*60)
    
    # Create manager
    manager = ZoneMemoryManager()
    
    # Create zone memory
    zone_id = "BTC_1H_50000"
    memory = manager.create_zone_memory(
        zone_id=zone_id,
        coin="BTCUSDT",
        timeframe="1H",
        price_bottom=50000,
        price_top=50100,
        quality=8.0
    )
    
    print(f"\n📊 Created Zone Memory: {zone_id}")
    print(f"   Price Range: ${memory.price_bottom:,.0f} - ${memory.price_top:,.0f}")
    print(f"   Quality: {memory.quality}/10")
    
    # ═════════════════════════════════════════════════════════
    # Simulate trade history
    # ═════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SIMULATING TRADE HISTORY")
    print(f"{'='*60}\n")
    
    trades = [
        # Trade 1: STOP LOSS
        ZoneTradeRecord(
            trade_id="TRADE_001",
            attempt_number=1,
            entry_price=50050,
            stop_loss=49750,
            result="STOP_LOSS",
            pnl_percent=-0.6,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            choch_strength=0.65,
            fib_level="0.705",
            reason="Stop loss hit"
        ),
        # Trade 2: WIN (re-entry)
        ZoneTradeRecord(
            trade_id="TRADE_002",
            attempt_number=2,
            entry_price=50045,
            stop_loss=49750,
            result="WIN",
            pnl_percent=3.2,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            choch_strength=0.78,
            fib_level="0.618",
            reason="TP2 hit"
        ),
        # Trade 3: WIN
        ZoneTradeRecord(
            trade_id="TRADE_003",
            attempt_number=1,
            entry_price=50060,
            stop_loss=49750,
            result="WIN",
            pnl_percent=2.5,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            choch_strength=0.72,
            fib_level="0.705",
            reason="TP2 hit"
        ),
    ]
    
    for trade in trades:
        manager.record_trade(zone_id, trade)
        print(f"Trade #{trade.attempt_number}: {trade.result} ({trade.pnl_percent:+.1f}%)")
    
    # ═════════════════════════════════════════════════════════
    # Check statistics
    # ═════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("ZONE STATISTICS")
    print(f"{'='*60}\n")
    
    stats = memory.statistics
    
    print(f"Total Attempts: {stats.total_attempts}")
    print(f"Wins: {stats.wins}")
    print(f"Losses: {stats.losses}")
    print(f"Stops: {stats.stops}")
    print(f"Win Rate: {stats.win_rate:.0%}")
    print(f"Avg PnL: {stats.avg_pnl:+.2f}%")
    print(f"Best PnL: {stats.best_pnl:+.2f}%")
    print(f"Worst PnL: {stats.worst_pnl:+.2f}%")
    print(f"Avg ChoCH Strength: {stats.avg_choch_strength:.2f}")
    print(f"Preferred Fib: {stats.preferred_fib}")
    print(f"Consecutive Losses: {stats.consecutive_losses}")
    
    # ═════════════════════════════════════════════════════════
    # Check blacklist
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("BLACKLIST STATUS")
    print(f"{'='*60}\n")
    
    if manager.is_blacklisted(zone_id):
        print(f"❌ BLACKLISTED")
        print(f"   Reason: {memory.blacklist_reason}")
        print(f"   Type: {memory.blacklist_type}")
    else:
        print(f"✅ NOT BLACKLISTED")
    
    # ═════════════════════════════════════════════════════════
    # Get RL recommendation
    # ═════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("RL RECOMMENDATION")
    print(f"{'='*60}\n")
    
    recommendation = manager.get_rl_recommendation(zone_id)
    
    print(f"Recommended: {'✅ Yes' if recommendation['recommended'] else '❌ No'}")
    print(f"Reason: {recommendation['reason']}")
    if 'confidence' in recommendation:
        print(f"Confidence: {recommendation['confidence']:.0%}")
    if 'preferred_fib' in recommendation:
        print(f"Preferred Fib: {recommendation['preferred_fib']}")
    
    print(f"\nRL Analysis:")
    for key, value in memory.rl_analysis.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("✅ Zone Memory System working correctly!")
    print("="*60 + "\n")