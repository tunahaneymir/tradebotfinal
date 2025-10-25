"""
Emotional State Manager - Tracks bot's psychological state

Simulates emotional states that affect trading decisions:
- Confidence (based on performance)
- Stress (based on losses/drawdown)
- Patience (based on waiting time/missed setups)

These states influence:
- Risk taking behavior
- FOMO susceptibility
- Quality thresholds
- Decision making
"""

from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class EmotionalState:
    """Bot's current emotional state"""
    # Core states (0.0 - 1.0)
    confidence: float = 0.5  # Based on win rate
    stress: float = 0.0      # Based on losses
    patience: float = 1.0    # Based on waiting time
    
    # Derived state
    stability: float = 0.5   # Overall emotional stability
    
    # Metadata
    last_updated: datetime = None


@dataclass
class PerformanceMetrics:
    """Performance data for emotional calculation"""
    recent_win_rate: float = 0.5  # Last 20 trades
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    current_drawdown: float = 0.0  # %
    weekly_dd: float = 0.0         # %
    avg_setup_score: float = 60.0


class EmotionalStateManager:
    """
    Emotional State Tracking System
    
    Tracks and updates bot's emotional state based on:
    - Trading performance (confidence)
    - Losses and drawdown (stress)
    - Time without trades (patience)
    
    States affect:
    - Risk taking (low confidence → reduced risk)
    - FOMO susceptibility (low patience → higher FOMO risk)
    - Quality thresholds (high stress → stricter standards)
    """
    
    def __init__(self):
        self.state = EmotionalState(last_updated=datetime.now())
        
        # Configuration
        self.CONFIDENCE_DECAY_RATE = 0.01  # per hour
        self.STRESS_DECAY_RATE = 0.1       # per hour
        self.PATIENCE_DECAY_RATE = 0.1     # per hour
        
        # Thresholds
        self.HIGH_STRESS_THRESHOLD = 0.7
        self.LOW_CONFIDENCE_THRESHOLD = 0.3
        self.LOW_PATIENCE_THRESHOLD = 0.3
        self.CRITICAL_STRESS = 0.8
    
    def update_from_trade(
        self,
        trade_result: str,  # "WIN" or "LOSS"
        pnl_percent: float,
        metrics: PerformanceMetrics
    ) -> Dict:
        """
        Update emotional state after trade
        
        Returns:
            Dict with updated states and actions
        """
        
        old_state = EmotionalState(
            confidence=self.state.confidence,
            stress=self.state.stress,
            patience=self.state.patience
        )
        
        # ════════════════════════════════════
        # UPDATE CONFIDENCE
        # ════════════════════════════════════
        if trade_result == "WIN":
            # Increase confidence
            self.state.confidence += 0.1
            self.state.confidence = min(1.0, self.state.confidence)
            
            # Decrease stress
            self.state.stress -= 0.15
            self.state.stress = max(0.0, self.state.stress)
            
            # Reset patience
            self.state.patience = 1.0
        
        else:  # LOSS
            # Decrease confidence
            self.state.confidence -= 0.1
            self.state.confidence = max(0.2, self.state.confidence)
            
            # Increase stress
            self.state.stress += 0.2
            self.state.stress = min(1.0, self.state.stress)
        
        # ════════════════════════════════════
        # APPLY PERFORMANCE ADJUSTMENTS
        # ════════════════════════════════════
        self._apply_performance_adjustments(metrics)
        
        # ════════════════════════════════════
        # CALCULATE STABILITY
        # ════════════════════════════════════
        self.state.stability = self._calculate_stability()
        
        # ════════════════════════════════════
        # CHECK FOR CRITICAL STATES
        # ════════════════════════════════════
        actions = self._check_critical_states()
        
        self.state.last_updated = datetime.now()
        
        return {
            'old_state': old_state,
            'new_state': EmotionalState(
                confidence=self.state.confidence,
                stress=self.state.stress,
                patience=self.state.patience,
                stability=self.state.stability
            ),
            'changes': {
                'confidence': self.state.confidence - old_state.confidence,
                'stress': self.state.stress - old_state.stress,
                'patience': self.state.patience - old_state.patience
            },
            'actions': actions
        }
    
    def update_from_time(
        self,
        hours_passed: float,
        missed_setups: int = 0
    ) -> Dict:
        """
        Update emotional state from time passing
        
        Args:
            hours_passed: Hours since last update
            missed_setups: Number of setups missed/skipped
        """
        
        # ════════════════════════════════════
        # NATURAL DECAY
        # ════════════════════════════════════
        
        # Stress decreases over time (recovery)
        self.state.stress -= self.STRESS_DECAY_RATE * hours_passed
        self.state.stress = max(0.0, self.state.stress)
        
        # Patience increases (rested)
        self.state.patience += self.PATIENCE_DECAY_RATE * hours_passed
        self.state.patience = min(1.0, self.state.patience)
        
        # ════════════════════════════════════
        # MISSED SETUP PENALTY
        # ════════════════════════════════════
        if missed_setups > 0:
            # Patience decreases (frustrated from missing)
            patience_penalty = 0.2 * missed_setups
            self.state.patience -= patience_penalty
            self.state.patience = max(0.0, self.state.patience)
        
        # ════════════════════════════════════
        # BOREDOM CHECK
        # ════════════════════════════════════
        if hours_passed > 6:
            # Long time without trading → patience drops
            self.state.patience -= 0.1
            self.state.patience = max(0.2, self.state.patience)
        
        # Update stability
        self.state.stability = self._calculate_stability()
        self.state.last_updated = datetime.now()
        
        return {
            'hours_passed': hours_passed,
            'missed_setups': missed_setups,
            'current_state': EmotionalState(
                confidence=self.state.confidence,
                stress=self.state.stress,
                patience=self.state.patience,
                stability=self.state.stability
            )
        }
    
    def get_risk_adjustment(self) -> float:
        """
        Get risk adjustment multiplier based on emotional state
        
        Returns:
            float: Risk multiplier (0.5 to 1.0)
        """
        
        # Base multiplier
        multiplier = 1.0
        
        # Low confidence → Reduce risk
        if self.state.confidence < self.LOW_CONFIDENCE_THRESHOLD:
            multiplier *= 0.7
        elif self.state.confidence < 0.5:
            multiplier *= 0.9
        
        # High stress → Reduce risk
        if self.state.stress > self.HIGH_STRESS_THRESHOLD:
            multiplier *= 0.7
        elif self.state.stress > 0.5:
            multiplier *= 0.85
        
        # Ensure minimum
        multiplier = max(0.5, multiplier)
        
        return multiplier
    
    def get_quality_adjustment(self) -> float:
        """
        Get quality threshold adjustment
        
        Returns:
            float: Points to add to quality threshold
        """
        
        adjustment = 0.0
        
        # High stress → Raise quality threshold
        if self.state.stress > self.HIGH_STRESS_THRESHOLD:
            adjustment += 2.0
        elif self.state.stress > 0.5:
            adjustment += 1.0
        
        # Low confidence → Raise quality threshold
        if self.state.confidence < self.LOW_CONFIDENCE_THRESHOLD:
            adjustment += 2.0
        
        return adjustment
    
    def should_take_break(self) -> Dict:
        """
        Check if bot should take a forced break
        
        Returns:
            Dict with break recommendation
        """
        
        if self.state.stress >= self.CRITICAL_STRESS:
            return {
                'should_break': True,
                'reason': 'Critical stress level',
                'duration_hours': 4,
                'stress_level': self.state.stress
            }
        
        if self.state.confidence < 0.2:
            return {
                'should_break': True,
                'reason': 'Extremely low confidence',
                'duration_hours': 2,
                'confidence': self.state.confidence
            }
        
        return {
            'should_break': False,
            'reason': 'Emotional state acceptable'
        }
    
    def get_state_summary(self) -> Dict:
        """Get current emotional state summary"""
        
        # Determine overall state
        if self.state.stability > 0.7:
            overall = "OPTIMAL"
            color = "🟢"
        elif self.state.stability > 0.5:
            overall = "GOOD"
            color = "🟡"
        elif self.state.stability > 0.3:
            overall = "CAUTION"
            color = "🟠"
        else:
            overall = "POOR"
            color = "🔴"
        
        return {
            'overall': overall,
            'color': color,
            'confidence': {
                'value': self.state.confidence,
                'level': self._categorize_value(self.state.confidence),
                'emoji': self._get_confidence_emoji(self.state.confidence)
            },
            'stress': {
                'value': self.state.stress,
                'level': self._categorize_stress(self.state.stress),
                'emoji': self._get_stress_emoji(self.state.stress)
            },
            'patience': {
                'value': self.state.patience,
                'level': self._categorize_value(self.state.patience),
                'emoji': self._get_patience_emoji(self.state.patience)
            },
            'stability': {
                'value': self.state.stability,
                'level': overall
            },
            'adjustments': {
                'risk_multiplier': self.get_risk_adjustment(),
                'quality_increase': self.get_quality_adjustment()
            },
            'break_needed': self.should_take_break()['should_break']
        }
    
    # ════════════════════════════════════
    # INTERNAL METHODS
    # ════════════════════════════════════
    
    def _apply_performance_adjustments(self, metrics: PerformanceMetrics):
        """Apply adjustments based on performance metrics"""
        
        # Win rate adjustment
        if metrics.recent_win_rate > 0.7:
            self.state.confidence = min(1.0, self.state.confidence + 0.05)
        elif metrics.recent_win_rate < 0.4:
            self.state.confidence = max(0.2, self.state.confidence - 0.05)
        
        # Streak adjustment
        if metrics.consecutive_wins >= 3:
            self.state.confidence = min(1.0, self.state.confidence + 0.05)
        if metrics.consecutive_losses >= 3:
            self.state.stress = min(1.0, self.state.stress + 0.1)
        
        # Drawdown adjustment
        if metrics.current_drawdown > 8.0:
            self.state.stress = min(1.0, self.state.stress + 0.2)
        if metrics.current_drawdown > 10.0:
            self.state.stress = min(1.0, self.state.stress + 0.3)
    
    def _calculate_stability(self) -> float:
        """
        Calculate overall emotional stability
        
        Weighted combination of states
        """
        
        # Weight factors
        confidence_weight = 0.4
        stress_weight = 0.4
        patience_weight = 0.2
        
        # Invert stress (high stress = low stability)
        stress_inverted = 1.0 - self.state.stress
        
        stability = (
            self.state.confidence * confidence_weight +
            stress_inverted * stress_weight +
            self.state.patience * patience_weight
        )
        
        return max(0.0, min(1.0, stability))
    
    def _check_critical_states(self) -> List[str]:
        """Check for critical emotional states"""
        
        actions = []
        
        if self.state.stress >= self.CRITICAL_STRESS:
            actions.append('FORCE_BREAK')
        
        if self.state.confidence < 0.2:
            actions.append('REDUCE_RISK_HEAVILY')
        
        if self.state.patience < 0.2:
            actions.append('FOMO_HIGH_RISK')
        
        if self.state.stress > 0.7:
            actions.append('INCREASE_QUALITY_THRESHOLD')
        
        return actions
    
    def _categorize_value(self, value: float) -> str:
        """Categorize a 0-1 value"""
        if value > 0.7:
            return "HIGH"
        elif value > 0.5:
            return "MEDIUM"
        elif value > 0.3:
            return "LOW"
        else:
            return "VERY LOW"
    
    def _categorize_stress(self, stress: float) -> str:
        """Categorize stress level"""
        if stress > 0.8:
            return "CRITICAL"
        elif stress > 0.6:
            return "HIGH"
        elif stress > 0.4:
            return "MEDIUM"
        elif stress > 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _get_confidence_emoji(self, confidence: float) -> str:
        if confidence > 0.7:
            return "💪"
        elif confidence > 0.5:
            return "👍"
        elif confidence > 0.3:
            return "😐"
        else:
            return "😰"
    
    def _get_stress_emoji(self, stress: float) -> str:
        if stress > 0.7:
            return "😱"
        elif stress > 0.5:
            return "😰"
        elif stress > 0.3:
            return "😟"
        else:
            return "😌"
    
    def _get_patience_emoji(self, patience: float) -> str:
        if patience > 0.7:
            return "🧘"
        elif patience > 0.5:
            return "😊"
        elif patience > 0.3:
            return "😤"
        else:
            return "🤬"


# ════════════════════════════════════════
# USAGE EXAMPLES
# ════════════════════════════════════════

def example_usage():
    """Example emotional state tracking"""
    
    manager = EmotionalStateManager()
    
    print("=" * 70)
    print("EMOTIONAL STATE MANAGER EXAMPLES")
    print("=" * 70)
    print()
    
    # ════════════════════════════════════════
    # SCENARIO 1: Initial State
    # ════════════════════════════════════════
    print("SCENARIO 1: Initial State")
    print("-" * 70)
    
    summary = manager.get_state_summary()
    print(f"Overall: {summary['color']} {summary['overall']}")
    print(f"Confidence: {summary['confidence']['emoji']} {summary['confidence']['value']:.2f} ({summary['confidence']['level']})")
    print(f"Stress: {summary['stress']['emoji']} {summary['stress']['value']:.2f} ({summary['stress']['level']})")
    print(f"Patience: {summary['patience']['emoji']} {summary['patience']['value']:.2f} ({summary['patience']['level']})")
    print(f"Stability: {summary['stability']['value']:.2f}")
    print(f"Risk Multiplier: {summary['adjustments']['risk_multiplier']:.2f}")
    print()
    
    # ════════════════════════════════════════
    # SCENARIO 2: After Winning Trade
    # ════════════════════════════════════════
    print("SCENARIO 2: After Winning Trade")
    print("-" * 70)
    
    metrics = PerformanceMetrics(
        recent_win_rate=0.65,
        consecutive_wins=1,
        consecutive_losses=0,
        current_drawdown=2.0
    )
    
    result = manager.update_from_trade("WIN", 2.3, metrics)
    print(f"Changes:")
    print(f"  Confidence: {result['changes']['confidence']:+.2f}")
    print(f"  Stress: {result['changes']['stress']:+.2f}")
    print(f"  Patience: {result['changes']['patience']:+.2f}")
    print()
    
    summary2 = manager.get_state_summary()
    print(f"New State: {summary2['color']} {summary2['overall']}")
    print(f"Confidence: {summary2['confidence']['value']:.2f}")
    print(f"Stress: {summary2['stress']['value']:.2f}")
    print()
    
    # ════════════════════════════════════════
    # SCENARIO 3: After 3 Consecutive Losses
    # ════════════════════════════════════════
    print("SCENARIO 3: After 3 Consecutive Losses")
    print("-" * 70)
    
    for i in range(3):
        metrics_loss = PerformanceMetrics(
            recent_win_rate=0.40,
            consecutive_wins=0,
            consecutive_losses=i+1,
            current_drawdown=5.0 + i*2
        )
        result_loss = manager.update_from_trade("LOSS", -1.5, metrics_loss)
    
    summary3 = manager.get_state_summary()
    print(f"State: {summary3['color']} {summary3['overall']}")
    print(f"Confidence: {summary3['confidence']['emoji']} {summary3['confidence']['value']:.2f}")
    print(f"Stress: {summary3['stress']['emoji']} {summary3['stress']['value']:.2f}")
    print(f"Risk Multiplier: {summary3['adjustments']['risk_multiplier']:.2f}")
    print(f"Quality Increase: +{summary3['adjustments']['quality_increase']:.0f}")
    
    break_check = manager.should_take_break()
    if break_check['should_break']:
        print(f"\n⚠️ BREAK RECOMMENDED!")
        print(f"Reason: {break_check['reason']}")
        print(f"Duration: {break_check['duration_hours']} hours")
    print()
    
    # ════════════════════════════════════════
    # SCENARIO 4: Time Passing (Recovery)
    # ════════════════════════════════════════
    print("SCENARIO 4: Time Passing (4 hours rest)")
    print("-" * 70)
    
    time_result = manager.update_from_time(hours_passed=4.0)
    summary4 = manager.get_state_summary()
    
    print(f"After Rest:")
    print(f"  Stress: {summary4['stress']['value']:.2f} (reduced)")
    print(f"  Patience: {summary4['patience']['value']:.2f} (recovered)")
    print(f"  Overall: {summary4['color']} {summary4['overall']}")
    print()
    
    print("=" * 70)


if __name__ == "__main__":
    example_usage()