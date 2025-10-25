"""
Performance Tracker - Track and analyze trading performance

Tracks:
- Overall performance (win rate, profit factor, Sharpe)
- Per-coin performance
- Per-timeframe performance
- Setup pattern performance
- Behavioral metrics
- Time-based performance

Used for:
- Performance monitoring
- Pattern identification
- Threshold optimization
- Strategy refinement
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


@dataclass
class TradeMetrics:
    """Metrics for a single trade"""
    trade_id: str
    timestamp: datetime
    symbol: str
    direction: str
    
    # Performance
    pnl_percent: float
    r_realized: float
    result: str  # "WIN" or "LOSS"
    
    # Setup quality
    setup_score: float
    zone_quality: float
    choch_strength: float
    
    # Duration
    duration_minutes: int
    
    # Behavioral
    fomo: bool = False
    revenge: bool = False
    overtrading: bool = False


class PerformanceTracker:
    """
    Comprehensive performance tracking system
    
    Tracks and analyzes:
    - Overall metrics
    - Coin-specific performance
    - Timeframe performance
    - Pattern performance
    - Behavioral patterns
    - Time-based patterns
    """
    
    def __init__(self):
        # Trade history
        self.trades: List[TradeMetrics] = []
        
        # Per-coin tracking
        self.coin_stats: Dict[str, List[TradeMetrics]] = defaultdict(list)
        
        # Pattern tracking (setup characteristics → performance)
        self.pattern_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Behavioral tracking
        self.behavioral_blocks = {
            'fomo': 0,
            'revenge': 0,
            'overtrading': 0
        }
        
        # Best/worst tracking
        self.best_trade: Optional[TradeMetrics] = None
        self.worst_trade: Optional[TradeMetrics] = None
        
        # Streak tracking
        self.current_streak: int = 0
        self.current_streak_type: str = ""  # "WIN" or "LOSS"
        self.max_win_streak: int = 0
        self.max_loss_streak: int = 0
    
    def add_trade(self, metrics: TradeMetrics):
        """Add trade to tracker"""
        
        # Add to history
        self.trades.append(metrics)
        
        # Update coin stats
        self.coin_stats[metrics.symbol].append(metrics)
        
        # Update pattern performance
        pattern_key = self._generate_pattern_key(metrics)
        self.pattern_performance[pattern_key].append(metrics.pnl_percent)
        
        # Update best/worst
        if self.best_trade is None or metrics.pnl_percent > self.best_trade.pnl_percent:
            self.best_trade = metrics
        if self.worst_trade is None or metrics.pnl_percent < self.worst_trade.pnl_percent:
            self.worst_trade = metrics
        
        # Update streak
        self._update_streak(metrics.result)
    
    def get_overall_stats(self) -> Dict:
        """Get overall performance statistics"""
        
        if not self.trades:
            return self._empty_stats()
        
        # Basic counts
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t.result == "WIN")
        losses = sum(1 for t in self.trades if t.result == "LOSS")
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        
        # PnL metrics
        total_pnl = sum(t.pnl_percent for t in self.trades)
        avg_pnl = total_pnl / total_trades
        
        # Win/Loss averages
        win_pnls = [t.pnl_percent for t in self.trades if t.result == "WIN"]
        loss_pnls = [t.pnl_percent for t in self.trades if t.result == "LOSS"]
        
        avg_win = statistics.mean(win_pnls) if win_pnls else 0.0
        avg_loss = statistics.mean(loss_pnls) if loss_pnls else 0.0
        
        # Profit factor
        total_wins = sum(win_pnls) if win_pnls else 0.0
        total_losses = abs(sum(loss_pnls)) if loss_pnls else 0.01
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        # Drawdown
        drawdown_info = self._calculate_drawdown()
        
        # Sharpe ratio
        sharpe = self._calculate_sharpe()
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pnl_percent': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': drawdown_info['max_dd'],
            'max_drawdown_duration': drawdown_info['max_dd_duration'],
            'current_drawdown': drawdown_info['current_dd'],
            'max_win_streak': self.max_win_streak,
            'max_loss_streak': self.max_loss_streak,
            'current_streak': f"{self.current_streak} {self.current_streak_type}" if self.current_streak > 0 else "0",
            'best_trade': f"{self.best_trade.symbol} {self.best_trade.pnl_percent:+.2f}%" if self.best_trade else "N/A",
            'worst_trade': f"{self.worst_trade.symbol} {self.worst_trade.pnl_percent:+.2f}%" if self.worst_trade else "N/A"
        }
    
    def get_coin_stats(self, symbol: Optional[str] = None) -> Dict:
        """
        Get per-coin statistics
        
        Args:
            symbol: Specific coin or None for all
        """
        
        if symbol:
            return self._calculate_coin_stats(symbol, self.coin_stats[symbol])
        
        # All coins
        result = {}
        for coin, trades in self.coin_stats.items():
            if len(trades) >= 5:  # Minimum 5 trades
                result[coin] = self._calculate_coin_stats(coin, trades)
        
        # Sort by win rate
        sorted_coins = sorted(
            result.items(),
            key=lambda x: x[1]['win_rate'],
            reverse=True
        )
        
        return dict(sorted_coins[:10])  # Top 10
    
    def get_pattern_performance(self, min_occurrences: int = 5) -> Dict:
        """
        Get pattern performance analysis
        
        Args:
            min_occurrences: Minimum pattern occurrences
        """
        
        result = {}
        
        for pattern, pnls in self.pattern_performance.items():
            if len(pnls) < min_occurrences:
                continue
            
            wins = sum(1 for p in pnls if p > 0)
            win_rate = wins / len(pnls)
            avg_pnl = statistics.mean(pnls)
            
            result[pattern] = {
                'occurrences': len(pnls),
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'status': self._categorize_pattern(win_rate)
            }
        
        # Sort by win rate
        sorted_patterns = sorted(
            result.items(),
            key=lambda x: x[1]['win_rate'],
            reverse=True
        )
        
        return dict(sorted_patterns)
    
    def get_recent_performance(self, n: int = 20) -> Dict:
        """Get performance of recent N trades"""
        
        if not self.trades:
            return self._empty_stats()
        
        recent = self.trades[-n:]
        
        wins = sum(1 for t in recent if t.result == "WIN")
        total = len(recent)
        win_rate = wins / total if total > 0 else 0.0
        
        avg_pnl = statistics.mean([t.pnl_percent for t in recent])
        avg_setup = statistics.mean([t.setup_score for t in recent])
        
        return {
            'trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'avg_setup_score': avg_setup
        }
    
    def get_behavioral_stats(self) -> Dict:
        """Get behavioral intervention statistics"""
        
        total_blocks = sum(self.behavioral_blocks.values())
        
        # Estimate savings (avg loss avoided)
        estimated_savings = {
            'fomo': self.behavioral_blocks['fomo'] * 1.5,  # Avg 1.5% loss avoided
            'revenge': self.behavioral_blocks['revenge'] * 1.8,  # Avg 1.8% loss
            'overtrading': self.behavioral_blocks['overtrading'] * 0.8  # Avg 0.8% loss
        }
        
        total_savings = sum(estimated_savings.values())
        
        return {
            'total_blocks': total_blocks,
            'fomo_blocks': self.behavioral_blocks['fomo'],
            'revenge_blocks': self.behavioral_blocks['revenge'],
            'overtrading_blocks': self.behavioral_blocks['overtrading'],
            'estimated_savings_percent': total_savings,
            'avg_savings_per_block': total_savings / total_blocks if total_blocks > 0 else 0.0
        }
    
    def register_block(self, block_type: str):
        """Register a behavioral block"""
        if block_type in self.behavioral_blocks:
            self.behavioral_blocks[block_type] += 1
    
    # ════════════════════════════════════
    # INTERNAL METHODS
    # ════════════════════════════════════
    
    def _generate_pattern_key(self, metrics: TradeMetrics) -> str:
        """Generate pattern key from trade metrics"""
        
        # Categorize values
        zone_cat = "high" if metrics.zone_quality >= 8 else "med" if metrics.zone_quality >= 6 else "low"
        choch_cat = "strong" if metrics.choch_strength >= 0.7 else "med" if metrics.choch_strength >= 0.5 else "weak"
        setup_cat = "excellent" if metrics.setup_score >= 80 else "good" if metrics.setup_score >= 65 else "acceptable"
        
        return f"{zone_cat}_zone_{choch_cat}_choch_{setup_cat}_setup"
    
    def _calculate_coin_stats(self, symbol: str, trades: List[TradeMetrics]) -> Dict:
        """Calculate statistics for a specific coin"""
        
        if not trades:
            return {}
        
        wins = sum(1 for t in trades if t.result == "WIN")
        total = len(trades)
        win_rate = wins / total
        
        total_pnl = sum(t.pnl_percent for t in trades)
        avg_pnl = total_pnl / total
        
        # Find best pattern
        patterns = [self._generate_pattern_key(t) for t in trades]
        best_pattern = max(set(patterns), key=patterns.count) if patterns else "N/A"
        
        return {
            'trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'best_pattern': best_pattern
        }
    
    def _calculate_drawdown(self) -> Dict:
        """Calculate drawdown metrics"""
        
        if not self.trades:
            return {'max_dd': 0.0, 'max_dd_duration': 0, 'current_dd': 0.0}
        
        # Calculate cumulative PnL
        cumulative = []
        total = 0.0
        for trade in self.trades:
            total += trade.pnl_percent
            cumulative.append(total)
        
        # Find max drawdown
        peak = cumulative[0]
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_duration = 0
        
        for i, value in enumerate(cumulative):
            if value > peak:
                peak = value
                current_dd_duration = 0
            else:
                dd = peak - value
                current_dd_duration += 1
                if dd > max_dd:
                    max_dd = dd
                    max_dd_duration = current_dd_duration
        
        # Current drawdown
        current_dd = peak - cumulative[-1] if cumulative else 0.0
        
        return {
            'max_dd': max_dd,
            'max_dd_duration': max_dd_duration,
            'current_dd': current_dd
        }
    
    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio"""
        
        if len(self.trades) < 2:
            return 0.0
        
        returns = [t.pnl_percent for t in self.trades]
        
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 1.0
        
        # Assuming 252 trading days
        sharpe = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0.0
        
        return sharpe
    
    def _update_streak(self, result: str):
        """Update win/loss streak"""
        
        if not self.current_streak_type or self.current_streak_type == result:
            # Continue streak
            self.current_streak += 1
            self.current_streak_type = result
        else:
            # Streak broken
            if self.current_streak_type == "WIN":
                self.max_win_streak = max(self.max_win_streak, self.current_streak)
            else:
                self.max_loss_streak = max(self.max_loss_streak, self.current_streak)
            
            # Start new streak
            self.current_streak = 1
            self.current_streak_type = result
    
    def _categorize_pattern(self, win_rate: float) -> str:
        """Categorize pattern performance"""
        if win_rate >= 0.75:
            return "EXCELLENT"
        elif win_rate >= 0.65:
            return "GOOD"
        elif win_rate >= 0.55:
            return "ACCEPTABLE"
        else:
            return "POOR"
    
    def _empty_stats(self) -> Dict:
        """Return empty stats structure"""
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'total_pnl_percent': 0.0,
            'avg_pnl': 0.0,
            'profit_factor': 0.0
        }
    
    def print_summary(self):
        """Print performance summary"""
        
        print("=" * 70)
        print("PERFORMANCE SUMMARY")
        print("=" * 70)
        
        # Overall stats
        overall = self.get_overall_stats()
        print("\nOVERALL STATISTICS:")
        print("-" * 70)
        print(f"Total Trades:      {overall['total_trades']}")
        print(f"Wins:              {overall['wins']} ({overall['win_rate']:.1%})")
        print(f"Losses:            {overall['losses']}")
        print(f"Total PnL:         {overall['total_pnl_percent']:+.2f}%")
        print(f"Avg PnL:           {overall['avg_pnl']:+.2f}%")
        print(f"Avg Win:           {overall['avg_win']:+.2f}%")
        print(f"Avg Loss:          {overall['avg_loss']:+.2f}%")
        print(f"Profit Factor:     {overall['profit_factor']:.2f}")
        print(f"Sharpe Ratio:      {overall['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:      {overall['max_drawdown']:.2f}%")
        print(f"Current Drawdown:  {overall['current_drawdown']:.2f}%")
        print(f"Current Streak:    {overall['current_streak']}")
        print(f"Best Trade:        {overall['best_trade']}")
        print(f"Worst Trade:       {overall['worst_trade']}")
        
        # Recent performance
        recent = self.get_recent_performance(20)
        print("\nRECENT PERFORMANCE (Last 20 trades):")
        print("-" * 70)
        print(f"Win Rate:          {recent['win_rate']:.1%}")
        print(f"Avg PnL:           {recent['avg_pnl']:+.2f}%")
        print(f"Avg Setup Score:   {recent['avg_setup_score']:.1f}")
        
        # Behavioral stats
        behavioral = self.get_behavioral_stats()
        print("\nBEHAVIORAL PROTECTION:")
        print("-" * 70)
        print(f"Total Blocks:      {behavioral['total_blocks']}")
        print(f"  FOMO:            {behavioral['fomo_blocks']}")
        print(f"  Revenge:         {behavioral['revenge_blocks']}")
        print(f"  Overtrading:     {behavioral['overtrading_blocks']}")
        print(f"Estimated Savings: {behavioral['estimated_savings_percent']:+.2f}%")
        
        # Top coins
        coin_stats = self.get_coin_stats()
        if coin_stats:
            print("\nTOP PERFORMING COINS:")
            print("-" * 70)
            for i, (coin, stats) in enumerate(list(coin_stats.items())[:5], 1):
                print(f"{i}. {coin:<10} WR: {stats['win_rate']:.1%}  "
                      f"PnL: {stats['total_pnl']:+.2f}%  "
                      f"Trades: {stats['trades']}")
        
        print("\n" + "=" * 70)


# ════════════════════════════════════════
# USAGE EXAMPLES
# ════════════════════════════════════════

def example_usage():
    """Example performance tracking"""
    
    tracker = PerformanceTracker()
    
    print("=" * 70)
    print("PERFORMANCE TRACKER EXAMPLES")
    print("=" * 70)
    print()
    
    # ════════════════════════════════════════
    # Add sample trades
    # ════════════════════════════════════════
    print("Adding sample trades...")
    
    # Simulate 50 trades
    import random
    
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"]
    
    for i in range(50):
        # Simulate varying performance
        is_win = random.random() < 0.65  # 65% win rate
        
        metrics = TradeMetrics(
            trade_id=f"T{i:03d}",
            timestamp=datetime.now() - timedelta(days=50-i),
            symbol=random.choice(symbols),
            direction="LONG" if random.random() < 0.6 else "SHORT",
            pnl_percent=random.uniform(1.5, 3.5) if is_win else random.uniform(-1.8, -0.8),
            r_realized=random.uniform(1.5, 2.0) if is_win else -1.0,
            result="WIN" if is_win else "LOSS",
            setup_score=random.uniform(70, 90) if is_win else random.uniform(50, 75),
            zone_quality=random.uniform(7, 10) if is_win else random.uniform(5, 8),
            choch_strength=random.uniform(0.6, 0.9) if is_win else random.uniform(0.4, 0.7),
            duration_minutes=random.randint(60, 300)
        )
        
        tracker.add_trade(metrics)
    
    # Add some behavioral blocks
    tracker.register_block('fomo')
    tracker.register_block('fomo')
    tracker.register_block('revenge')
    tracker.register_block('overtrading')
    
    print(f"Added {len(tracker.trades)} trades")
    print()
    
    # ════════════════════════════════════════
    # Print full summary
    # ════════════════════════════════════════
    tracker.print_summary()
    print()
    
    # ════════════════════════════════════════
    # Pattern analysis
    # ════════════════════════════════════════
    print("PATTERN PERFORMANCE:")
    print("-" * 70)
    
    patterns = tracker.get_pattern_performance(min_occurrences=3)
    for pattern, stats in list(patterns.items())[:5]:
        print(f"{pattern}:")
        print(f"  Occurrences: {stats['occurrences']}")
        print(f"  Win Rate: {stats['win_rate']:.1%}")
        print(f"  Avg PnL: {stats['avg_pnl']:+.2f}%")
        print(f"  Status: {stats['status']}")
        print()
    
    print("=" * 70)


if __name__ == "__main__":
    example_usage()