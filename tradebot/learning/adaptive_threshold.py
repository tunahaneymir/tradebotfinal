"""
Adaptive Threshold Optimizer - Learns optimal thresholds

Optimizes:
- Zone quality minimum
- ChoCH strength minimum  
- Setup score minimum
- Coin-specific thresholds

Based on:
- Historical performance
- Pattern analysis
- Win rate optimization
- Risk-reward analysis

Learning Phases:
- Phase 1 (0-3 months): Liberal thresholds, collect data
- Phase 2 (3-6 months): Begin optimization
- Phase 3 (6+ months): Coin-specific optimization
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class ThresholdConfig:
    """Current threshold configuration"""
    # Global thresholds
    min_zone_quality: float = 4.0
    min_choch_strength: float = 0.4
    min_setup_score: float = 40.0
    
    # Coin-specific overrides
    coin_thresholds: Dict[str, Dict] = None
    
    # Phase
    phase: str = "learning"  # learning, optimization, production
    months_trading: int = 0
    
    def __post_init__(self):
        if self.coin_thresholds is None:
            self.coin_thresholds = {}


@dataclass
class PerformanceData:
    """Performance data for threshold analysis"""
    threshold_value: float
    trades: int
    wins: int
    win_rate: float
    avg_pnl: float
    avg_rr: float


class AdaptiveThresholdOptimizer:
    """
    Adaptive Threshold Learning System
    
    Learns optimal thresholds by:
    1. Testing different threshold levels
    2. Analyzing performance at each level
    3. Finding optimal balance (quality vs frequency)
    4. Adjusting thresholds gradually
    5. Coin-specific optimization
    """
    
    def __init__(self):
        self.config = ThresholdConfig()
        
        # Performance tracking by threshold
        self.zone_quality_performance: Dict[int, List[float]] = defaultdict(list)
        self.choch_strength_performance: Dict[float, List[float]] = defaultdict(list)
        self.setup_score_performance: Dict[int, List[float]] = defaultdict(list)
        
        # Coin-specific tracking
        self.coin_performance: Dict[str, Dict] = defaultdict(dict)
        
        # Learning history
        self.optimization_history: List[Dict] = []
        
        # Start date
        self.start_date = datetime.now()
    
    def record_trade(
        self,
        zone_quality: float,
        choch_strength: float,
        setup_score: float,
        pnl_percent: float,
        symbol: str
    ):
        """
        Record trade for threshold learning
        
        Args:
            zone_quality: 0-10
            choch_strength: 0.0-1.0
            setup_score: 0-100
            pnl_percent: Trade PnL
            symbol: Coin symbol
        """
        
        # Record by threshold levels
        zone_bucket = int(zone_quality)
        self.zone_quality_performance[zone_bucket].append(pnl_percent)
        
        choch_bucket = round(choch_strength, 1)
        self.choch_strength_performance[choch_bucket].append(pnl_percent)
        
        setup_bucket = int(setup_score // 10) * 10  # Bucket by 10s
        self.setup_score_performance[setup_bucket].append(pnl_percent)
        
        # Record coin-specific
        if symbol not in self.coin_performance:
            self.coin_performance[symbol] = {
                'zone': defaultdict(list),
                'choch': defaultdict(list),
                'setup': defaultdict(list)
            }
        
        self.coin_performance[symbol]['zone'][zone_bucket].append(pnl_percent)
        self.coin_performance[symbol]['choch'][choch_bucket].append(pnl_percent)
        self.coin_performance[symbol]['setup'][setup_bucket].append(pnl_percent)
    
    def analyze_and_optimize(
        self,
        min_samples: int = 30
    ) -> Dict:
        """
        Analyze performance and optimize thresholds
        
        Args:
            min_samples: Minimum trades per threshold level
            
        Returns:
            Dict with optimization results
        """
        
        # Calculate months trading
        self.config.months_trading = (
            datetime.now() - self.start_date
        ).days // 30
        
        # Determine phase
        self._update_phase()
        
        # Analyze each threshold type
        zone_analysis = self._analyze_zone_quality(min_samples)
        choch_analysis = self._analyze_choch_strength(min_samples)
        setup_analysis = self._analyze_setup_score(min_samples)
        
        # Determine if optimization needed
        changes = []
        
        if zone_analysis['should_change']:
            old_value = self.config.min_zone_quality
            self.config.min_zone_quality = zone_analysis['recommended']
            changes.append(f"Zone quality: {old_value} → {zone_analysis['recommended']}")
        
        if choch_analysis['should_change']:
            old_value = self.config.min_choch_strength
            self.config.min_choch_strength = choch_analysis['recommended']
            changes.append(f"ChoCH strength: {old_value} → {choch_analysis['recommended']}")
        
        if setup_analysis['should_change']:
            old_value = self.config.min_setup_score
            self.config.min_setup_score = setup_analysis['recommended']
            changes.append(f"Setup score: {old_value} → {setup_analysis['recommended']}")
        
        # Coin-specific optimization (phase 3 only)
        coin_changes = []
        if self.config.phase == "production":
            coin_changes = self._optimize_coin_thresholds(min_samples)
        
        # Record optimization
        optimization_record = {
            'timestamp': datetime.now(),
            'phase': self.config.phase,
            'months_trading': self.config.months_trading,
            'changes': changes,
            'coin_changes': coin_changes,
            'zone_analysis': zone_analysis,
            'choch_analysis': choch_analysis,
            'setup_analysis': setup_analysis
        }
        
        self.optimization_history.append(optimization_record)
        
        return optimization_record
    
    def get_threshold_for_symbol(self, symbol: str) -> Dict:
        """
        Get thresholds for specific symbol
        
        Returns coin-specific if available, else global
        """
        
        if symbol in self.config.coin_thresholds:
            return self.config.coin_thresholds[symbol]
        
        return {
            'min_zone_quality': self.config.min_zone_quality,
            'min_choch_strength': self.config.min_choch_strength,
            'min_setup_score': self.config.min_setup_score,
            'source': 'global'
        }
    
    # ════════════════════════════════════
    # INTERNAL METHODS
    # ════════════════════════════════════
    
    def _update_phase(self):
        """Update trading phase based on months"""
        
        months = self.config.months_trading
        
        if months < 3:
            self.config.phase = "learning"
        elif months < 6:
            self.config.phase = "optimization"
        else:
            self.config.phase = "production"
    
    def _analyze_zone_quality(self, min_samples: int) -> Dict:
        """Analyze zone quality performance"""
        
        # Collect performance data
        performance_data = []
        
        for threshold in range(4, 11):  # 4-10
            trades = self.zone_quality_performance.get(threshold, [])
            
            if len(trades) < min_samples:
                continue
            
            wins = sum(1 for pnl in trades if pnl > 0)
            win_rate = wins / len(trades)
            avg_pnl = sum(trades) / len(trades)
            
            performance_data.append(PerformanceData(
                threshold_value=threshold,
                trades=len(trades),
                wins=wins,
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                avg_rr=0.0  # Not used here
            ))
        
        if not performance_data:
            return {'should_change': False, 'reason': 'Insufficient data'}
        
        # Find optimal threshold
        # Balance: win_rate * trades (quality vs frequency)
        scores = [
            (p.threshold_value, p.win_rate * (p.trades ** 0.5))
            for p in performance_data
        ]
        
        optimal_threshold, best_score = max(scores, key=lambda x: x[1])
        
        # Should we change?
        current = self.config.min_zone_quality
        should_change = abs(optimal_threshold - current) >= 1.0
        
        return {
            'should_change': should_change,
            'current': current,
            'recommended': float(optimal_threshold),
            'performance_data': performance_data,
            'reason': f"Optimal threshold: {optimal_threshold} (score: {best_score:.2f})"
        }
    
    def _analyze_choch_strength(self, min_samples: int) -> Dict:
        """Analyze ChoCH strength performance"""
        
        performance_data = []
        
        thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
        
        for threshold in thresholds:
            trades = self.choch_strength_performance.get(threshold, [])
            
            if len(trades) < min_samples:
                continue
            
            wins = sum(1 for pnl in trades if pnl > 0)
            win_rate = wins / len(trades)
            avg_pnl = sum(trades) / len(trades)
            
            performance_data.append(PerformanceData(
                threshold_value=threshold,
                trades=len(trades),
                wins=wins,
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                avg_rr=0.0
            ))
        
        if not performance_data:
            return {'should_change': False, 'reason': 'Insufficient data'}
        
        # Find optimal
        scores = [
            (p.threshold_value, p.win_rate * (p.trades ** 0.5))
            for p in performance_data
        ]
        
        optimal_threshold, _ = max(scores, key=lambda x: x[1])
        
        current = self.config.min_choch_strength
        should_change = abs(optimal_threshold - current) >= 0.1
        
        return {
            'should_change': should_change,
            'current': current,
            'recommended': optimal_threshold,
            'performance_data': performance_data
        }
    
    def _analyze_setup_score(self, min_samples: int) -> Dict:
        """Analyze setup score performance"""
        
        performance_data = []
        
        for threshold in range(40, 81, 10):  # 40, 50, 60, 70, 80
            trades = self.setup_score_performance.get(threshold, [])
            
            if len(trades) < min_samples:
                continue
            
            wins = sum(1 for pnl in trades if pnl > 0)
            win_rate = wins / len(trades)
            avg_pnl = sum(trades) / len(trades)
            
            performance_data.append(PerformanceData(
                threshold_value=threshold,
                trades=len(trades),
                wins=wins,
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                avg_rr=0.0
            ))
        
        if not performance_data:
            return {'should_change': False, 'reason': 'Insufficient data'}
        
        # Find optimal
        scores = [
            (p.threshold_value, p.win_rate * (p.trades ** 0.5))
            for p in performance_data
        ]
        
        optimal_threshold, _ = max(scores, key=lambda x: x[1])
        
        current = self.config.min_setup_score
        should_change = abs(optimal_threshold - current) >= 10.0
        
        return {
            'should_change': should_change,
            'current': current,
            'recommended': float(optimal_threshold),
            'performance_data': performance_data
        }
    
    def _optimize_coin_thresholds(self, min_samples: int) -> List[str]:
        """Optimize coin-specific thresholds (production phase only)"""
        
        changes = []
        
        for symbol, data in self.coin_performance.items():
            # Need enough data per coin
            total_trades = sum(len(trades) for trades in data['zone'].values())
            
            if total_trades < min_samples * 3:
                continue
            
            # Analyze zone quality for this coin
            best_zone = None
            best_score = 0.0
            
            for threshold, trades in data['zone'].items():
                if len(trades) < min_samples:
                    continue
                
                wins = sum(1 for pnl in trades if pnl > 0)
                win_rate = wins / len(trades)
                score = win_rate * (len(trades) ** 0.5)
                
                if score > best_score:
                    best_score = score
                    best_zone = threshold
            
            if best_zone and abs(best_zone - self.config.min_zone_quality) >= 1.0:
                self.config.coin_thresholds[symbol] = {
                    'min_zone_quality': float(best_zone),
                    'min_choch_strength': self.config.min_choch_strength,
                    'min_setup_score': self.config.min_setup_score
                }
                changes.append(f"{symbol}: zone quality → {best_zone}")
        
        return changes


# ════════════════════════════════════════
# USAGE EXAMPLE
# ════════════════════════════════════════

def example_usage():
    """Example threshold optimization"""
    
    optimizer = AdaptiveThresholdOptimizer()
    
    print("=" * 70)
    print("ADAPTIVE THRESHOLD OPTIMIZER")
    print("=" * 70)
    print()
    
    # Simulate 6 months of trading data
    import random
    
    print("Simulating 6 months of trading...")
    
    for month in range(6):
        for _ in range(50):  # 50 trades per month
            zone = random.randint(4, 10)
            choch = round(random.uniform(0.4, 0.9), 1)
            setup = random.randint(40, 90)
            
            # Higher thresholds → better performance
            base_win_prob = 0.5
            if zone >= 7:
                base_win_prob += 0.1
            if choch >= 0.6:
                base_win_prob += 0.1
            if setup >= 65:
                base_win_prob += 0.1
            
            is_win = random.random() < base_win_prob
            pnl = random.uniform(1.5, 3.0) if is_win else random.uniform(-1.8, -1.0)
            
            symbol = random.choice(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
            
            optimizer.record_trade(zone, choch, setup, pnl, symbol)
    
    print(f"Recorded {sum(len(v) for v in optimizer.zone_quality_performance.values())} trades")
    print()
    
    # Analyze and optimize
    print("Running optimization analysis...")
    print("-" * 70)
    result = optimizer.analyze_and_optimize(min_samples=20)
    
    print(f"Phase: {result['phase']}")
    print(f"Months Trading: {result['months_trading']}")
    print()
    
    if result['changes']:
        print("CHANGES APPLIED:")
        for change in result['changes']:
            print(f"  ✓ {change}")
    else:
        print("No changes recommended")
    
    print()
    print("CURRENT THRESHOLDS:")
    print(f"  Zone Quality: {optimizer.config.min_zone_quality}")
    print(f"  ChoCH Strength: {optimizer.config.min_choch_strength}")
    print(f"  Setup Score: {optimizer.config.min_setup_score}")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    example_usage()