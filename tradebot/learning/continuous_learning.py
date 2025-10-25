"""
Continuous Learning Manager - 24/7 Learning Loop

Manages:
- Experience collection
- Periodic training (every 4 hours)
- Performance analysis (daily)
- Threshold optimization (weekly)
- Model checkpointing

Learning Schedule:
- Every trade → Store experience
- Every 4 hours → Train RL model
- Every 24 hours → Analyze performance
- Every 7 days → Optimize thresholds
"""

from __future__ import annotations
from typing import Dict, Optional
from datetime import datetime, timedelta
import time


class ContinuousLearningManager:
    """
    24/7 Continuous Learning System
    
    Coordinates:
    - Experience buffer
    - RL agent training
    - Performance tracking
    - Threshold optimization
    """
    
    def __init__(
        self,
        experience_buffer,
        rl_agent,
        performance_tracker,
        threshold_optimizer
    ):
        self.experience_buffer = experience_buffer
        self.rl_agent = rl_agent
        self.performance_tracker = performance_tracker
        self.threshold_optimizer = threshold_optimizer
        
        # Schedule tracking
        self.last_training = datetime.now()
        self.last_analysis = datetime.now()
        self.last_optimization = datetime.now()
        
        # Training config
        self.training_interval_hours = 4
        self.analysis_interval_hours = 24
        self.optimization_interval_days = 7
        
        # Stats
        self.total_trainings = 0
        self.total_optimizations = 0
    
    def on_trade_completed(
        self,
        experience: 'Experience',
        trade_metrics: 'TradeMetrics'
    ):
        """
        Called when trade completes
        
        Args:
            experience: Experience object for RL
            trade_metrics: Metrics for performance tracking
        """
        
        # 1. Store experience
        self.experience_buffer.add(experience)
        
        # 2. Track performance
        self.performance_tracker.add_trade(trade_metrics)
        
        # 3. Record for threshold learning
        self.threshold_optimizer.record_trade(
            zone_quality=experience.state.get('zone_quality', 0),
            choch_strength=experience.state.get('choch_strength', 0),
            setup_score=experience.setup_score,
            pnl_percent=trade_metrics.pnl_percent,
            symbol=experience.symbol
        )
        
        # 4. Check if training needed
        self._check_and_train()
        
        # 5. Check if analysis needed
        self._check_and_analyze()
        
        # 6. Check if optimization needed
        self._check_and_optimize()
    
    def _check_and_train(self):
        """Check if training interval elapsed"""
        
        now = datetime.now()
        elapsed = (now - self.last_training).total_seconds() / 3600
        
        if elapsed >= self.training_interval_hours:
            print(f"\n{'='*70}")
            print(f"[TRAINING] {elapsed:.1f}h since last training")
            self._train_rl_agent()
            self.last_training = now
            print(f"{'='*70}\n")
    
    def _check_and_analyze(self):
        """Check if analysis interval elapsed"""
        
        now = datetime.now()
        elapsed = (now - self.last_analysis).total_seconds() / 3600
        
        if elapsed >= self.analysis_interval_hours:
            print(f"\n{'='*70}")
            print(f"[ANALYSIS] {elapsed:.1f}h since last analysis")
            self._analyze_performance()
            self.last_analysis = now
            print(f"{'='*70}\n")
    
    def _check_and_optimize(self):
        """Check if optimization interval elapsed"""
        
        now = datetime.now()
        elapsed = (now - self.last_optimization).days
        
        if elapsed >= self.optimization_interval_days:
            print(f"\n{'='*70}")
            print(f"[OPTIMIZATION] {elapsed} days since last optimization")
            self._optimize_thresholds()
            self.last_optimization = now
            print(f"{'='*70}\n")
    
    def _train_rl_agent(self):
        """Train RL agent with recent experiences"""
        
        # Get recent experiences (10K)
        recent = self.experience_buffer.get_recent(10000)
        
        if len(recent) < 1000:
            print("⚠️ Not enough experiences for training (min 1000)")
            return
        
        print(f"Training RL agent with {len(recent)} experiences...")
        
        # Train (implementation in rl_agent.py)
        try:
            training_result = self.rl_agent.train(
                experiences=recent,
                epochs=5,
                batch_size=128
            )
            
            self.total_trainings += 1
            
            print(f"✅ Training complete:")
            print(f"   Loss: {training_result.get('loss', 0):.4f}")
            print(f"   Total trainings: {self.total_trainings}")
            
            # Save checkpoint
            self.rl_agent.save_checkpoint(
                f"checkpoints/rl_agent_train_{self.total_trainings}.pt"
            )
            
    def _train_rl_agent(self):
        """Train RL agent with recent experiences"""
        
        # Get recent experiences (10K)
        recent = self.experience_buffer.get_recent(10000)
        
        if len(recent) < 1000:
            print("⚠️ Not enough experiences for training (min 1000)")
            return
        
        print(f"Training RL agent with {len(recent)} experiences...")
        
        # Train (implementation in rl_agent.py)
        try:
            training_result = self.rl_agent.train(
                experiences=recent,
                epochs=5,
                batch_size=128
            )
            
            self.total_trainings += 1
            
            print(f"✅ Training complete:")
            print(f"   Loss: {training_result.get('loss', 0):.4f}")
            print(f"   Total trainings: {self.total_trainings}")
            
            # Save checkpoint
            self.rl_agent.save_checkpoint(
                f"checkpoints/rl_agent_train_{self.total_trainings}.pt"
            )
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
    
    def _analyze_performance(self):
        """Daily performance analysis"""
        
        print("Analyzing performance...")
        
        # Print performance summary
        self.performance_tracker.print_summary()
        
        # Get insights
        recent = self.performance_tracker.get_recent_performance(50)
        
        # Check if performance declining
        if recent['win_rate'] < 0.55:
            print("\n⚠️ WARNING: Win rate below 55%")
            print("   Consider stricter thresholds or review strategy")
        
        # Check for improvement
        overall = self.performance_tracker.get_overall_stats()
        if recent['win_rate'] > overall['win_rate'] + 0.05:
            print("\n✅ IMPROVEMENT: Recent performance better than average")
    
    def _optimize_thresholds(self):
        """Weekly threshold optimization"""
        
        print("Optimizing thresholds...")
        
        result = self.threshold_optimizer.analyze_and_optimize(min_samples=30)
        
        if result['changes']:
            print("\n✅ Thresholds updated:")
            for change in result['changes']:
                print(f"   {change}")
        else:
            print("\n✓ No threshold changes needed")
        
        if result.get('coin_changes'):
            print("\n✅ Coin-specific updates:")
            for change in result['coin_changes']:
                print(f"   {change}")
        
        self.total_optimizations += 1
    
    def get_learning_status(self) -> Dict:
        """Get current learning system status"""
        
        now = datetime.now()
        
        return {
            'experience_buffer': {
                'size': len(self.experience_buffer),
                'capacity': self.experience_buffer.capacity,
                'usage_pct': len(self.experience_buffer) / self.experience_buffer.capacity * 100
            },
            'training': {
                'total_trainings': self.total_trainings,
                'last_training': self.last_training,
                'hours_since': (now - self.last_training).total_seconds() / 3600,
                'next_training_in': max(0, self.training_interval_hours - (now - self.last_training).total_seconds() / 3600)
            },
            'optimization': {
                'total_optimizations': self.total_optimizations,
                'last_optimization': self.last_optimization,
                'days_since': (now - self.last_optimization).days,
                'next_optimization_in': max(0, self.optimization_interval_days - (now - self.last_optimization).days)
            },
            'thresholds': {
                'zone_quality': self.threshold_optimizer.config.min_zone_quality,
                'choch_strength': self.threshold_optimizer.config.min_choch_strength,
                'setup_score': self.threshold_optimizer.config.min_setup_score,
                'phase': self.threshold_optimizer.config.phase
            }
        }
    
    def force_training(self):
        """Force immediate training"""
        print("[FORCE] Training RL agent...")
        self._train_rl_agent()
    
    def force_optimization(self):
        """Force immediate optimization"""
        print("[FORCE] Optimizing thresholds...")
        self._optimize_thresholds()
    
    def save_state(self, filepath: str = "learning_state.json"):
        """Save learning system state"""
        import json
        from pathlib import Path
        
        state = {
            'last_training': self.last_training.isoformat(),
            'last_analysis': self.last_analysis.isoformat(),
            'last_optimization': self.last_optimization.isoformat(),
            'total_trainings': self.total_trainings,
            'total_optimizations': self.total_optimizations,
            'thresholds': {
                'zone_quality': self.threshold_optimizer.config.min_zone_quality,
                'choch_strength': self.threshold_optimizer.config.min_choch_strength,
                'setup_score': self.threshold_optimizer.config.min_setup_score,
                'phase': self.threshold_optimizer.config.phase,
                'months_trading': self.threshold_optimizer.config.months_trading
            }
        }
        
        with Path(filepath).open('w') as f:
            json.dump(state, f, indent=2)
        
        print(f"✅ Learning state saved: {filepath}")
    
    def load_state(self, filepath: str = "learning_state.json") -> bool:
        """Load learning system state"""
        import json
        from pathlib import Path
        
        path = Path(filepath)
        if not path.exists():
            print(f"❌ State file not found: {filepath}")
            return False
        
        try:
            with path.open('r') as f:
                state = json.load(f)
            
            self.last_training = datetime.fromisoformat(state['last_training'])
            self.last_analysis = datetime.fromisoformat(state['last_analysis'])
            self.last_optimization = datetime.fromisoformat(state['last_optimization'])
            self.total_trainings = state['total_trainings']
            self.total_optimizations = state['total_optimizations']
            
            # Restore thresholds
            self.threshold_optimizer.config.min_zone_quality = state['thresholds']['zone_quality']
            self.threshold_optimizer.config.min_choch_strength = state['thresholds']['choch_strength']
            self.threshold_optimizer.config.min_setup_score = state['thresholds']['setup_score']
            self.threshold_optimizer.config.phase = state['thresholds']['phase']
            self.threshold_optimizer.config.months_trading = state['thresholds']['months_trading']
            
            print(f"✅ Learning state loaded: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading state: {e}")
            return False


# ════════════════════════════════════════
# USAGE EXAMPLE
# ════════════════════════════════════════

def example_usage():
    """Example continuous learning setup"""
    
    print("=" * 70)
    print("CONTINUOUS LEARNING MANAGER")
    print("=" * 70)
    print()
    
    # This would normally import real classes
    # For demo, we'll use mock objects
    
    class MockBuffer:
        def __init__(self):
            self.capacity = 100000
            self.buffer = []
        def add(self, exp):
            self.buffer.append(exp)
        def get_recent(self, n):
            return self.buffer[-n:]
        def __len__(self):
            return len(self.buffer)
    
    class MockAgent:
        def train(self, experiences, epochs, batch_size):
            return {'loss': 0.123}
        def save_checkpoint(self, path):
            print(f"   Saved checkpoint: {path}")
    
    class MockTracker:
        def add_trade(self, metrics):
            pass
        def print_summary(self):
            print("   Performance summary printed...")
        def get_recent_performance(self, n):
            return {'win_rate': 0.67}
        def get_overall_stats(self):
            return {'win_rate': 0.65}
    
    class MockOptimizer:
        def __init__(self):
            self.config = type('obj', (object,), {
                'min_zone_quality': 6.0,
                'min_choch_strength': 0.5,
                'min_setup_score': 50.0,
                'phase': 'optimization'
            })()
        def record_trade(self, **kwargs):
            pass
        def analyze_and_optimize(self, min_samples):
            return {
                'changes': ['Zone quality: 6.0 → 7.0'],
                'coin_changes': []
            }
    
    # Create manager
    manager = ContinuousLearningManager(
        experience_buffer=MockBuffer(),
        rl_agent=MockAgent(),
        performance_tracker=MockTracker(),
        threshold_optimizer=MockOptimizer()
    )
    
    print("Learning manager initialized")
    print()
    
    # Simulate some time passing
    manager.last_training = datetime.now() - timedelta(hours=5)
    manager.last_analysis = datetime.now() - timedelta(hours=25)
    manager.last_optimization = datetime.now() - timedelta(days=8)
    
    # Get status
    status = manager.get_learning_status()
    print("LEARNING STATUS:")
    print("-" * 70)
    print(f"Training:")
    print(f"  Total: {status['training']['total_trainings']}")
    print(f"  Hours since last: {status['training']['hours_since']:.1f}h")
    print(f"  Next in: {status['training']['next_training_in']:.1f}h")
    print()
    print(f"Optimization:")
    print(f"  Total: {status['optimization']['total_optimizations']}")
    print(f"  Days since last: {status['optimization']['days_since']}")
    print(f"  Next in: {status['optimization']['next_optimization_in']} days")
    print()
    print(f"Thresholds (Phase: {status['thresholds']['phase']}):")
    print(f"  Zone Quality: {status['thresholds']['zone_quality']}")
    print(f"  ChoCH Strength: {status['thresholds']['choch_strength']}")
    print(f"  Setup Score: {status['thresholds']['setup_score']}")
    print()
    
    # Trigger checks
    print("Checking schedules...")
    print("-" * 70)
    manager._check_and_train()
    manager._check_and_analyze()
    manager._check_and_optimize()
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    example_usage()