"""
Experience Buffer - Store trading experiences for RL learning

Stores trade experiences in a circular buffer for:
- RL Agent training
- Performance analysis
- Pattern recognition
- Threshold optimization

Buffer Structure:
- Capacity: 100,000 experiences
- Circular (FIFO when full)
- Efficient retrieval
- Batch sampling support
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from pathlib import Path
from collections import deque


@dataclass
class Experience:
    """Single trading experience (SARS + outcome)"""
    
    # Identity
    experience_id: str
    trade_id: str
    timestamp: datetime
    
    # State (before decision) - ~40 features
    state: Dict[str, Any] = field(default_factory=dict)
    
    # Action (RL decision)
    action: str = ""  # "ENTER_FULL", "ENTER_REDUCED", "SKIP", "WAIT"
    action_probs: Dict[str, float] = field(default_factory=dict)
    
    # Trade details (if entered)
    trade_details: Optional[Dict] = None
    
    # Next state (after trade)
    next_state: Dict[str, Any] = field(default_factory=dict)
    
    # Reward
    outcome_score: float = 0.0  # -200 to +200
    rl_reward: float = 0.0      # -1.0 to +1.0 (normalized)
    
    # Metadata
    symbol: str = ""
    setup_score: float = 0.0
    result: Optional[str] = None  # "WIN", "LOSS", "SKIPPED"
    pnl_percent: float = 0.0
    
    # Behavioral flags
    fomo_detected: bool = False
    revenge_detected: bool = False
    overtrading: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert datetime to string
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> Experience:
        """Create from dictionary"""
        # Convert timestamp back
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ExperienceBuffer:
    """
    Circular buffer for storing trading experiences
    
    Features:
    - Fixed capacity (100K default)
    - FIFO replacement when full
    - Efficient batch sampling
    - Persistence (save/load)
    - Statistics tracking
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        
        # Statistics
        self.total_added = 0
        self.wins = 0
        self.losses = 0
        self.skips = 0
        
        # Index for fast lookup
        self.experience_by_id: Dict[str, Experience] = {}
    
    def add(self, experience: Experience):
        """
        Add experience to buffer
        
        Args:
            experience: Experience object
        """
        
        # If buffer is full, oldest will be removed
        if len(self.buffer) >= self.capacity:
            oldest = self.buffer[0]
            if oldest.experience_id in self.experience_by_id:
                del self.experience_by_id[oldest.experience_id]
        
        # Add new experience
        self.buffer.append(experience)
        self.experience_by_id[experience.experience_id] = experience
        
        # Update statistics
        self.total_added += 1
        if experience.result == "WIN":
            self.wins += 1
        elif experience.result == "LOSS":
            self.losses += 1
        elif experience.result == "SKIPPED":
            self.skips += 1
    
    def get_recent(self, n: int = 1000) -> List[Experience]:
        """
        Get N most recent experiences
        
        Args:
            n: Number of experiences
            
        Returns:
            List of recent experiences
        """
        if n >= len(self.buffer):
            return list(self.buffer)
        return list(self.buffer)[-n:]
    
    def sample_batch(
        self, 
        batch_size: int = 128,
        filter_result: Optional[str] = None
    ) -> List[Experience]:
        """
        Sample random batch from buffer
        
        Args:
            batch_size: Batch size
            filter_result: Filter by result ("WIN", "LOSS", "SKIPPED")
            
        Returns:
            List of sampled experiences
        """
        import random
        
        # Filter if needed
        if filter_result:
            filtered = [exp for exp in self.buffer if exp.result == filter_result]
            population = filtered
        else:
            population = list(self.buffer)
        
        # Sample
        sample_size = min(batch_size, len(population))
        return random.sample(population, sample_size)
    
    def get_by_symbol(self, symbol: str, limit: int = 1000) -> List[Experience]:
        """Get experiences for specific symbol"""
        experiences = [exp for exp in self.buffer if exp.symbol == symbol]
        return experiences[-limit:] if len(experiences) > limit else experiences
    
    def get_statistics(self) -> Dict:
        """Get buffer statistics"""
        
        if not self.buffer:
            return {
                'total_experiences': 0,
                'wins': 0,
                'losses': 0,
                'skips': 0,
                'win_rate': 0.0,
                'avg_reward': 0.0
            }
        
        # Calculate stats
        total = len(self.buffer)
        wins = sum(1 for exp in self.buffer if exp.result == "WIN")
        losses = sum(1 for exp in self.buffer if exp.result == "LOSS")
        skips = sum(1 for exp in self.buffer if exp.result == "SKIPPED")
        
        trades = wins + losses
        win_rate = wins / trades if trades > 0 else 0.0
        
        avg_reward = sum(exp.rl_reward for exp in self.buffer) / total
        avg_outcome = sum(exp.outcome_score for exp in self.buffer) / total
        
        return {
            'total_experiences': total,
            'total_added_lifetime': self.total_added,
            'wins': wins,
            'losses': losses,
            'skips': skips,
            'trades': trades,
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'avg_outcome_score': avg_outcome,
            'buffer_usage': f"{total}/{self.capacity} ({total/self.capacity*100:.1f}%)"
        }
    
    def get_recent_performance(self, n: int = 100) -> Dict:
        """Get performance of recent N experiences"""
        
        recent = self.get_recent(n)
        
        if not recent:
            return {'count': 0}
        
        wins = sum(1 for exp in recent if exp.result == "WIN")
        losses = sum(1 for exp in recent if exp.result == "LOSS")
        trades = wins + losses
        
        return {
            'count': len(recent),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / trades if trades > 0 else 0.0,
            'avg_reward': sum(exp.rl_reward for exp in recent) / len(recent),
            'avg_setup_score': sum(exp.setup_score for exp in recent) / len(recent)
        }
    
    def save(self, filepath: str = "experience_buffer.json"):
        """
        Save buffer to disk
        
        Args:
            filepath: Path to save file
        """
        
        path = Path(filepath)
        
        # Convert to serializable format
        data = {
            'capacity': self.capacity,
            'total_added': self.total_added,
            'experiences': [exp.to_dict() for exp in self.buffer]
        }
        
        with path.open('w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Buffer saved: {len(self.buffer)} experiences → {filepath}")
    
    def load(self, filepath: str = "experience_buffer.json") -> bool:
        """
        Load buffer from disk
        
        Args:
            filepath: Path to load file
            
        Returns:
            bool: Success status
        """
        
        path = Path(filepath)
        
        if not path.exists():
            print(f"❌ File not found: {filepath}")
            return False
        
        try:
            with path.open('r') as f:
                data = json.load(f)
            
            # Restore buffer
            self.capacity = data.get('capacity', 100000)
            self.total_added = data.get('total_added', 0)
            
            self.buffer = deque(maxlen=self.capacity)
            self.experience_by_id = {}
            
            for exp_data in data['experiences']:
                exp = Experience.from_dict(exp_data)
                self.buffer.append(exp)
                self.experience_by_id[exp.experience_id] = exp
            
            # Recalculate stats
            self.wins = sum(1 for exp in self.buffer if exp.result == "WIN")
            self.losses = sum(1 for exp in self.buffer if exp.result == "LOSS")
            self.skips = sum(1 for exp in self.buffer if exp.result == "SKIPPED")
            
            print(f"✅ Buffer loaded: {len(self.buffer)} experiences from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading buffer: {e}")
            return False
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        self.experience_by_id.clear()
        self.wins = 0
        self.losses = 0
        self.skips = 0
        print("🗑️ Buffer cleared")
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"ExperienceBuffer("
            f"size={len(self.buffer)}/{self.capacity}, "
            f"win_rate={stats['win_rate']:.1%}, "
            f"avg_reward={stats['avg_reward']:+.3f})"
        )


# ════════════════════════════════════════
# USAGE EXAMPLES
# ════════════════════════════════════════

def example_usage():
    """Example experience buffer usage"""
    
    print("=" * 70)
    print("EXPERIENCE BUFFER EXAMPLES")
    print("=" * 70)
    print()
    
    # ════════════════════════════════════════
    # Create buffer
    # ════════════════════════════════════════
    buffer = ExperienceBuffer(capacity=10000)
    print(f"Buffer created: {buffer}")
    print()
    
    # ════════════════════════════════════════
    # Add experiences
    # ════════════════════════════════════════
    print("Adding experiences...")
    
    for i in range(100):
        exp = Experience(
            experience_id=f"EXP_{i:05d}",
            trade_id=f"T_{i:05d}",
            timestamp=datetime.now(),
            state={'zone_quality': 7.0 + i % 3, 'setup_score': 60 + i % 30},
            action="ENTER_FULL" if i % 2 == 0 else "SKIP",
            outcome_score=150 if i % 3 == 0 else -80,
            rl_reward=0.75 if i % 3 == 0 else -0.4,
            symbol="BTCUSDT",
            setup_score=60 + i % 30,
            result="WIN" if i % 3 == 0 else "LOSS",
            pnl_percent=2.0 if i % 3 == 0 else -1.5
        )
        buffer.add(exp)
    
    print(f"Added 100 experiences")
    print(f"Buffer: {buffer}")
    print()
    
    # ════════════════════════════════════════
    # Get statistics
    # ════════════════════════════════════════
    print("Statistics:")
    print("-" * 70)
    stats = buffer.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # ════════════════════════════════════════
    # Sample batch
    # ════════════════════════════════════════
    print("Sampling batch...")
    batch = buffer.sample_batch(batch_size=10)
    print(f"Sampled {len(batch)} experiences")
    print(f"First experience: {batch[0].experience_id}")
    print()
    
    # ════════════════════════════════════════
    # Recent performance
    # ════════════════════════════════════════
    print("Recent Performance (last 20):")
    print("-" * 70)
    recent_perf = buffer.get_recent_performance(n=20)
    print(f"  Win Rate: {recent_perf['win_rate']:.1%}")
    print(f"  Avg Reward: {recent_perf['avg_reward']:+.3f}")
    print(f"  Avg Setup Score: {recent_perf['avg_setup_score']:.1f}")
    print()
    
    # ════════════════════════════════════════
    # Save & Load
    # ════════════════════════════════════════
    print("Testing Save/Load:")
    print("-" * 70)
    buffer.save("test_buffer.json")
    
    # Create new buffer and load
    buffer2 = ExperienceBuffer()
    buffer2.load("test_buffer.json")
    print(f"Loaded buffer: {buffer2}")
    print()
    
    print("=" * 70)


if __name__ == "__main__":
    example_usage()