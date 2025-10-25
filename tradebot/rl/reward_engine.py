"""
Reward Engine for RL Agent - Outcome Score Calculation

Calculates reward/penalty based on:
- Trade outcome (win/loss)
- Setup quality
- Execution quality
- Behavioral flags (FOMO, revenge, etc)

Range: -200 to +200
"""

from __future__ import annotations
from typing import Dict, Optional, Literal
from dataclasses import dataclass


@dataclass
class TradeOutcome:
    """Trade sonucu detayları"""
    # Trade basics
    trade_id: str
    symbol: str
    direction: Literal["LONG", "SHORT"]
    
    # Performance
    pnl_percent: float  # PnL percentage (e.g., 2.5 for +2.5%)
    r_realized: float   # R multiple (e.g., 1.8 for 1.8R)
    
    # Setup quality
    setup_score: float  # 0-100 from setup scorer
    zone_quality: float  # 0-10
    choch_strength: float  # 0.0-1.0
    
    # Execution quality
    entry_quality: Literal["EXCELLENT", "GOOD", "ACCEPTABLE", "POOR"]  # Fib level
    perfect_timing: bool = False  # Entry within 1 candle of setup
    patience_shown: bool = True   # Waited for full setup
    
    # Risk management
    risk_discipline: bool = True  # Proper position sizing
    kademeli_exit: bool = False   # Used TP1/TP2/TP3
    
    # Behavioral flags (penalties)
    fomo_detected: bool = False
    revenge_detected: bool = False
    over_risk: bool = False
    overtrading: bool = False
    poor_timing: bool = False
    
    # Metadata
    duration_minutes: Optional[int] = None
    exit_reason: Optional[str] = None


class RewardEngine:
    """
    RL Reward Engine - Trade outcome'u reward score'a çevirir
    
    Philosopy:
    - Good decision + Good outcome → High reward
    - Good decision + Bad outcome → Low penalty (luck factor)
    - Bad decision + Good outcome → Low reward (don't repeat!)
    - Bad decision + Bad outcome → Heavy penalty
    """
    
    def __init__(self):
        # Reward ranges
        self.EXCELLENT_WIN_MIN = 150
        self.EXCELLENT_WIN_MAX = 200
        self.GOOD_WIN_MIN = 80
        self.GOOD_WIN_MAX = 150
        self.ACCEPTABLE_WIN_MIN = 30
        self.ACCEPTABLE_WIN_MAX = 80
        self.WEAK_WIN_MIN = 30
        self.WEAK_WIN_MAX = 60
        
        # Penalty ranges
        self.ACCEPTABLE_LOSS = -50  # Good setup, bad luck
        self.NORMAL_LOSS = -80
        self.WEAK_SETUP_LOSS = -120
        self.BAD_SETUP_LOSS = -180
        self.TERRIBLE_LOSS = -200
        
        # Bonus/penalty amounts
        self.PERFECT_EXECUTION_BONUS = 50
        self.PATIENCE_BONUS = 30
        self.RISK_DISCIPLINE_BONUS = 20
        self.KADEMELI_EXIT_BONUS = 15
        
        self.FOMO_PENALTY = 100
        self.REVENGE_PENALTY = 80
        self.OVER_RISK_PENALTY = 50
        self.OVERTRADING_PENALTY = 40
        self.POOR_TIMING_PENALTY = 30
    
    def calculate_outcome_score(self, outcome: TradeOutcome) -> Dict:
        """
        Trade outcome score hesapla
        
        Returns:
            Dict with:
            - total_score: -200 to +200
            - breakdown: Component scores
            - category: Outcome category
            - message: Explanation
        """
        
        # ════════════════════════════════════
        # 1. BASE REWARD (Win/Loss Category)
        # ════════════════════════════════════
        base_reward = self._calculate_base_reward(outcome)
        
        # ════════════════════════════════════
        # 2. BONUSES (Only for profitable trades)
        # ════════════════════════════════════
        bonuses = 0
        if outcome.pnl_percent > 0:
            bonuses = self._calculate_bonuses(outcome)
        
        # ════════════════════════════════════
        # 3. PENALTIES (Always apply)
        # ════════════════════════════════════
        penalties = self._calculate_penalties(outcome)
        
        # ════════════════════════════════════
        # 4. FINAL SCORE
        # ════════════════════════════════════
        final_score = base_reward + bonuses - penalties
        
        # Clip to range [-200, +200]
        final_score = max(-200, min(200, final_score))
        
        category = self._categorize_score(final_score)
        message = self._generate_message(outcome, final_score, category)
        
        return {
            'total_score': final_score,
            'breakdown': {
                'base': base_reward,
                'bonuses': bonuses,
                'penalties': penalties
            },
            'category': category,
            'message': message,
            'components': {
                'setup_score': outcome.setup_score,
                'pnl_percent': outcome.pnl_percent,
                'r_realized': outcome.r_realized
            }
        }
    
    def _calculate_base_reward(self, outcome: TradeOutcome) -> float:
        """Base reward based on setup quality and PnL"""
        
        setup_score = outcome.setup_score
        pnl = outcome.pnl_percent
        
        # ════════════════════════════════════
        # WIN SCENARIOS
        # ════════════════════════════════════
        if pnl > 0:
            # Excellent Win: Setup 80+, Profit 2%+
            if setup_score >= 80 and pnl >= 2.0:
                return 200
            
            # Good Win: Setup 65-79, Profit 1.5%+
            elif setup_score >= 65 and pnl >= 1.5:
                return 150
            
            # Acceptable Win: Setup 50-64, Profit 1%+
            elif setup_score >= 50 and pnl >= 1.0:
                return 100
            
            # Weak Win: Low setup or small profit
            else:
                return 50
        
        # ════════════════════════════════════
        # LOSS SCENARIOS
        # ════════════════════════════════════
        else:
            # Acceptable Loss: Excellent setup, bad luck
            if setup_score >= 80:
                return -50  # LOW PENALTY
            
            # Normal Loss: Good setup
            elif setup_score >= 65:
                return -80
            
            # Weak Setup Loss: Should have skipped
            elif setup_score >= 50:
                return -120  # MEDIUM PENALTY
            
            # Bad Setup Loss: Bad decision
            elif setup_score >= 40:
                return -180  # HEAVY PENALTY
            
            # Terrible: Should never trade sub-40
            else:
                return -200  # MAXIMUM PENALTY
    
    def _calculate_bonuses(self, outcome: TradeOutcome) -> float:
        """Calculate bonuses for good execution"""
        total = 0
        
        # Perfect execution bonus
        if outcome.perfect_timing and outcome.setup_score >= 85:
            total += self.PERFECT_EXECUTION_BONUS
        
        # Patience bonus
        if outcome.patience_shown:
            total += self.PATIENCE_BONUS
        
        # Risk discipline bonus
        if outcome.risk_discipline:
            total += self.RISK_DISCIPLINE_BONUS
        
        # Kademeli exit bonus
        if outcome.kademeli_exit:
            total += self.KADEMELI_EXIT_BONUS
        
        return total
    
    def _calculate_penalties(self, outcome: TradeOutcome) -> float:
        """Calculate penalties for bad behavior"""
        total = 0
        
        # FOMO penalty (heaviest!)
        if outcome.fomo_detected:
            total += self.FOMO_PENALTY
        
        # Revenge trading penalty
        if outcome.revenge_detected:
            total += self.REVENGE_PENALTY
        
        # Over-risk penalty
        if outcome.over_risk:
            total += self.OVER_RISK_PENALTY
        
        # Overtrading penalty
        if outcome.overtrading:
            total += self.OVERTRADING_PENALTY
        
        # Poor timing penalty
        if outcome.poor_timing:
            total += self.POOR_TIMING_PENALTY
        
        return total
    
    def _categorize_score(self, score: float) -> str:
        """Categorize outcome score"""
        if score >= 150:
            return "EXCELLENT"
        elif score >= 80:
            return "GOOD"
        elif score >= 30:
            return "ACCEPTABLE"
        elif score >= -30:
            return "NEUTRAL"
        elif score >= -80:
            return "ACCEPTABLE_LOSS"
        elif score >= -150:
            return "BAD"
        else:
            return "TERRIBLE"
    
    def _generate_message(
        self, 
        outcome: TradeOutcome, 
        score: float,
        category: str
    ) -> str:
        """Generate human-readable message"""
        
        if outcome.pnl_percent > 0:
            # Win scenarios
            if category == "EXCELLENT":
                return "Perfect trade! Excellent setup won big"
            elif category == "GOOD":
                return "Good trade! Setup worked well"
            elif outcome.fomo_detected:
                return "Won but violated rules - don't repeat!"
            else:
                return "Acceptable win"
        else:
            # Loss scenarios
            if category == "ACCEPTABLE_LOSS":
                return "Setup was perfect, bad luck"
            elif category == "BAD":
                return "Weak setup led to loss"
            elif outcome.revenge_detected:
                return "WORST CASE! Revenge + weak setup"
            else:
                return "Should not have taken this"


# ════════════════════════════════════════
# USAGE EXAMPLE
# ════════════════════════════════════════

def example_usage():
    """Example reward calculations"""
    
    engine = RewardEngine()
    
    # Example 1: Perfect trade
    perfect_trade = TradeOutcome(
        trade_id="T001",
        symbol="BTCUSDT",
        direction="LONG",
        pnl_percent=3.2,
        r_realized=1.8,
        setup_score=87,
        zone_quality=9,
        choch_strength=0.86,
        entry_quality="EXCELLENT",
        perfect_timing=True,
        patience_shown=True,
        risk_discipline=True,
        kademeli_exit=True
    )
    
    result = engine.calculate_outcome_score(perfect_trade)
    print(f"Perfect Trade Score: {result['total_score']}")
    print(f"Category: {result['category']}")
    print(f"Message: {result['message']}")
    print(f"Breakdown: {result['breakdown']}")
    print()
    
    # Example 2: FOMO win (low reward despite profit)
    fomo_trade = TradeOutcome(
        trade_id="T002",
        symbol="ETHUSDT",
        direction="LONG",
        pnl_percent=1.8,
        r_realized=1.2,
        setup_score=72,
        zone_quality=7,
        choch_strength=0.65,
        entry_quality="GOOD",
        patience_shown=False,
        fomo_detected=True,
        poor_timing=True
    )
    
    result = engine.calculate_outcome_score(fomo_trade)
    print(f"FOMO Trade Score: {result['total_score']}")
    print(f"Category: {result['category']}")
    print(f"Message: {result['message']}")
    print()
    
    # Example 3: Excellent setup, unlucky loss (low penalty)
    unlucky_loss = TradeOutcome(
        trade_id="T003",
        symbol="BTCUSDT",
        direction="LONG",
        pnl_percent=-1.5,
        r_realized=-1.0,
        setup_score=85,
        zone_quality=8,
        choch_strength=0.78,
        entry_quality="EXCELLENT",
        perfect_timing=True,
        patience_shown=True,
        risk_discipline=True
    )
    
    result = engine.calculate_outcome_score(unlucky_loss)
    print(f"Unlucky Loss Score: {result['total_score']}")
    print(f"Category: {result['category']}")
    print(f"Message: {result['message']}")
    print()
    
    # Example 4: Revenge trade disaster
    revenge_trade = TradeOutcome(
        trade_id="T004",
        symbol="BTCUSDT",
        direction="LONG",
        pnl_percent=-1.5,
        r_realized=-1.0,
        setup_score=48,
        zone_quality=5,
        choch_strength=0.42,
        entry_quality="POOR",
        patience_shown=False,
        revenge_detected=True,
        over_risk=True
    )
    
    result = engine.calculate_outcome_score(revenge_trade)
    print(f"Revenge Trade Score: {result['total_score']}")
    print(f"Category: {result['category']}")
    print(f"Message: {result['message']}")


    def calculate_rl_reward(
        self,
        outcome_score: float,
        dna_score: float = 50.0,
        emotion_stability: float = 0.5,
        param_gain: float = 0.0
    ) -> Dict:
        """
        Adaptive learning için RL reward hesapla
        
        Args:
            outcome_score: Base reward from calculate_outcome_score (-200 to +200)
            dna_score: DNA match score (0-100) - Future use
            emotion_stability: Emotional stability (0.0-1.0)
            param_gain: Parameter optimization gain (-1.0 to +1.0) - Future use
        
        Returns:
            Dict with:
            - rl_reward: Normalized reward (-1.0 to +1.0)
            - normalized_base: Base component
            - adaptive_components: DNA, emotion, param contributions
        """
        
        # ════════════════════════════════════
        # 1. Normalize base reward
        # ════════════════════════════════════
        normalized_base = outcome_score / 200.0
        
        # ════════════════════════════════════
        # 2. Adaptive learning components
        # ════════════════════════════════════
        dna_component = 0.05 * (dna_score / 100.0)      # Max 0.05
        emotion_component = 0.03 * emotion_stability    # Max 0.03
        param_component = 0.02 * param_gain             # Max 0.02
        
        # ════════════════════════════════════
        # 3. Weighted total
        # ════════════════════════════════════
        weighted_reward = (
            normalized_base
            + dna_component
            + emotion_component
            + param_component
        )
        
        # Clip to range [-1.0, +1.0]
        final_reward = max(-1.0, min(1.0, weighted_reward))
        
        # ════════════════════════════════════
        # 4. Log learning metrics
        # ════════════════════════════════════
        self._log_learning_metrics({
            'outcome_score': outcome_score,
            'normalized_base': normalized_base,
            'dna_score': dna_score,
            'dna_component': dna_component,
            'emotion_stability': emotion_stability,
            'emotion_component': emotion_component,
            'param_gain': param_gain,
            'param_component': param_component,
            'final_reward': final_reward
        })
        
        return {
            'rl_reward': final_reward,
            'normalized_base': normalized_base,
            'adaptive_components': {
                'dna': dna_component,
                'emotion': emotion_component,
                'param': param_component
            },
            'breakdown': {
                'base': f"{normalized_base:+.3f}",
                'dna': f"{dna_component:+.3f}",
                'emotion': f"{emotion_component:+.3f}",
                'param': f"{param_component:+.3f}",
                'total': f"{final_reward:+.3f}"
            }
        }
    
    def _log_learning_metrics(self, metrics: Dict):
        """Learning metriklerini logla"""
        print(f"[ADAPTIVE LEARNING] Reward calculation:")
        print(f"  Base: {metrics['normalized_base']:+.3f} (from {metrics['outcome_score']:+.0f})")
        print(f"  DNA:  {metrics['dna_component']:+.3f} (score: {metrics['dna_score']:.0f})")
        print(f"  Emotion: {metrics['emotion_component']:+.3f} (stability: {metrics['emotion_stability']:.2f})")
        print(f"  Param: {metrics['param_component']:+.3f} (gain: {metrics['param_gain']:+.2f})")
        print(f"  → Final RL Reward: {metrics['final_reward']:+.3f}")


# ════════════════════════════════════════
# USAGE EXAMPLE - EXTENDED
# ════════════════════════════════════════

def example_usage_with_rl_reward():
    """Extended examples with RL reward calculation"""
    
    engine = RewardEngine()
    
    print("=" * 70)
    print("REWARD ENGINE - WITH ADAPTIVE RL REWARD")
    print("=" * 70)
    print()
    
    # Example: Perfect trade with adaptive learning
    perfect_trade = TradeOutcome(
        trade_id="T001",
        symbol="BTCUSDT",
        direction="LONG",
        pnl_percent=3.2,
        r_realized=1.8,
        setup_score=87,
        zone_quality=9,
        choch_strength=0.86,
        entry_quality="EXCELLENT",
        perfect_timing=True,
        patience_shown=True,
        risk_discipline=True,
        kademeli_exit=True
    )
    
    # Step 1: Calculate outcome score
    outcome = engine.calculate_outcome_score(perfect_trade)
    print(f"PERFECT TRADE:")
    print(f"  Outcome Score: {outcome['total_score']}")
    print(f"  Category: {outcome['category']}")
    print(f"  Breakdown: Base={outcome['breakdown']['base']}, "
          f"Bonuses={outcome['breakdown']['bonuses']}, "
          f"Penalties={outcome['breakdown']['penalties']}")
    print()
    
    # Step 2: Calculate RL reward (with adaptive learning)
    rl_result = engine.calculate_rl_reward(
        outcome_score=outcome['total_score'],
        dna_score=75.0,  # Future: DNA matching system
        emotion_stability=0.85,  # High emotional stability
        param_gain=0.15  # Positive parameter optimization
    )
    print(f"  RL Reward: {rl_result['rl_reward']:.3f}")
    print(f"  Components: {rl_result['breakdown']}")
    print()
    print("-" * 70)
    print()


if __name__ == "__main__":
    example_usage()
    print()
    example_usage_with_rl_reward()