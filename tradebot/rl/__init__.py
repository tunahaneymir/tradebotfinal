"""
RL Systems - Parça 4

This module contains RL architecture components:
- State Builder (40+ ML features aggregation)
- Setup Scorer (0-100 quality scoring)
- Gate System (4-gate validation)
- RL Agent (PPO - will be added later)

Usage:
    from rl import (
        StateBuilder,
        SetupScorer,
        GateSystem
    )
    
    # Build state for RL
    state_builder = StateBuilder()
    state = state_builder.build_state(
        trend=trend,
        zone=zone,
        choch=choch,
        ...
    )
    
    # Score setup quality
    scorer = SetupScorer(min_score=40)
    score = scorer.score_setup(trend, zone, choch, fib)
    
    # Validate through gates
    gate_system = GateSystem(config)
    result = gate_system.validate(entry_signal, score, account_state)
    
    if result.all_passed:
        # Send to RL agent for final decision
"""

# State Builder
from .state_builder import (
    StateBuilder,
    StateVector
)

# Setup Scorer
from .setup_scorer import (
    SetupScorer,
    SetupScore
)

# Gate System
from .gate_system import (
    GateSystem,
    GateResult,
    GateSystemResult
)

__all__ = [
    # ═══════════════════════════════════════════════════════════
    # State Builder
    # ═══════════════════════════════════════════════════════════
    'StateBuilder',
    'StateVector',
    
    # ═══════════════════════════════════════════════════════════
    # Setup Scorer
    # ═══════════════════════════════════════════════════════════
    'SetupScorer',
    'SetupScore',
    
    # ═══════════════════════════════════════════════════════════
    # Gate System
    # ═══════════════════════════════════════════════════════════
    'GateSystem',
    'GateResult',
    'GateSystemResult',
]

__version__ = '4.0.0'  # Parça 4 (RL Agent will be added in future)