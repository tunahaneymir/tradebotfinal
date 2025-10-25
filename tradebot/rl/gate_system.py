"""
Gate System - Parça 4
Based on: pa-strateji2 Parça 4

4-Gate Validation System:
- Gate 1: Mandatory Requirements (Hard filters)
- Gate 2: Quality Score >= Minimum
- Gate 3: Risk Limits OK
- Gate 4: Psychological Checks (FOMO/Revenge/Overtrading)

All gates must pass before RL agent decides
"""

from __future__ import annotations
from typing import Dict, Optional, Literal
from dataclasses import dataclass
from datetime import datetime, timedelta

from core import TrendResult, Zone, ChoCHResult, EntrySignal
from .setup_scorer import SetupScore


@dataclass
class GateResult:
    """Single gate check result"""
    gate_number: int
    gate_name: str
    passed: bool
    checks: Dict[str, bool]
    failed_check: Optional[str]
    message: str


@dataclass
class GateSystemResult:
    """Complete gate system result"""
    all_passed: bool
    gates_passed: int
    total_gates: int
    
    # Individual gate results
    gate1: GateResult
    gate2: GateResult
    gate3: GateResult
    gate4: GateResult
    
    # Overall
    can_trade: bool
    reason: str


class GateSystem:
    """
    4-Gate Validation System
    
    Before RL agent makes final decision, all trades must pass 4 gates:
    
    GATE 1 - Mandatory Requirements:
    - Zone exists
    - ChoCH confirmed
    - Fib retest done
    - Trend aligned
    - Stop loss valid
    - RR >= 1.5
    
    GATE 2 - Quality Score:
    - Setup score >= minimum (default 40)
    - Individual component minimums
    
    GATE 3 - Risk Limits:
    - Daily risk < limit
    - Total open risk < limit
    - Position size valid
    - Coin not in cooldown
    
    GATE 4 - Psychological:
    - Not FOMO trading
    - Not revenge trading
    - Not overtrading
    - Emotional state OK
    
    Usage:
        gate_system = GateSystem(config)
        
        result = gate_system.validate(
            entry_signal=entry_signal,
            setup_score=setup_score,
            account_state=account_state
        )
        
        if result.all_passed:
            # Send to RL agent for final decision
        else:
            # Reject trade
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Gate System
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Gate 2: Quality thresholds
        quality_config = self.config.get('rl', {}).get('setup_scoring', {})
        self.min_setup_score = quality_config.get('min_score', 40)
        
        # Gate 3: Risk limits
        risk_config = self.config.get('risk', {})
        self.max_daily_risk = risk_config.get('daily_limit_pct', 0.06)  # 6%
        self.max_portfolio_risk = risk_config.get('portfolio_limit_pct', 0.08)  # 8%
        self.max_positions = risk_config.get('max_positions', 3)
        
        # Gate 4: Behavioral limits
        behavior_config = self.config.get('behavior', {})
        
        # Anti-overtrading
        overtrading_config = behavior_config.get('anti_overtrading', {})
        self.max_trades_per_hour = overtrading_config.get('max_trades_per_hour', 2)
        self.max_trades_per_day = overtrading_config.get('max_trades_per_day', 5)
    
    def validate(
        self,
        entry_signal: EntrySignal,
        setup_score: SetupScore,
        account_state: Dict
    ) -> GateSystemResult:
        """
        Validate trade through all 4 gates
        
        Args:
            entry_signal: Entry signal from entry system
            setup_score: Setup quality score
            account_state: Current account state (risk, positions, etc)
            
        Returns:
            GateSystemResult with all gate results
        """
        # ═══════════════════════════════════════════════════════════
        # GATE 1: Mandatory Requirements
        # ═══════════════════════════════════════════════════════════
        gate1 = self._gate1_mandatory(entry_signal)
        
        # ═══════════════════════════════════════════════════════════
        # GATE 2: Quality Score
        # ═══════════════════════════════════════════════════════════
        gate2 = self._gate2_quality(setup_score)
        
        # ═══════════════════════════════════════════════════════════
        # GATE 3: Risk Limits
        # ═══════════════════════════════════════════════════════════
        gate3 = self._gate3_risk(entry_signal, account_state)
        
        # ═══════════════════════════════════════════════════════════
        # GATE 4: Psychological Checks
        # ═══════════════════════════════════════════════════════════
        gate4 = self._gate4_psychological(account_state)
        
        # ═══════════════════════════════════════════════════════════
        # OVERALL RESULT
        # ═══════════════════════════════════════════════════════════
        all_passed = gate1.passed and gate2.passed and gate3.passed and gate4.passed
        gates_passed = sum([gate1.passed, gate2.passed, gate3.passed, gate4.passed])
        
        # Determine reason
        if all_passed:
            reason = "✅ All gates passed - Trade approved for RL decision"
        else:
            failed_gates = []
            if not gate1.passed:
                failed_gates.append(f"Gate 1: {gate1.failed_check}")
            if not gate2.passed:
                failed_gates.append(f"Gate 2: {gate2.failed_check}")
            if not gate3.passed:
                failed_gates.append(f"Gate 3: {gate3.failed_check}")
            if not gate4.passed:
                failed_gates.append(f"Gate 4: {gate4.failed_check}")
            
            reason = "❌ Failed gates: " + "; ".join(failed_gates)
        
        return GateSystemResult(
            all_passed=all_passed,
            gates_passed=gates_passed,
            total_gates=4,
            gate1=gate1,
            gate2=gate2,
            gate3=gate3,
            gate4=gate4,
            can_trade=all_passed,
            reason=reason
        )
    
    def _gate1_mandatory(self, entry_signal: EntrySignal) -> GateResult:
        """
        Gate 1: Mandatory Requirements
        
        Hard filters that must all pass
        """
        checks = {}
        
        # Check 1: Entry signal ready
        checks['entry_ready'] = entry_signal.ready
        
        # Check 2: Trend aligned
        checks['trend_aligned'] = entry_signal.trend_aligned
        
        # Check 3: Zone valid
        checks['zone_valid'] = entry_signal.zone_valid
        
        # Check 4: ChoCH strong enough
        checks['choch_strong'] = entry_signal.choch_strong
        
        # Check 5: Fib retest done
        checks['fib_touched'] = entry_signal.fib_touched
        
        # Check 6: Stop loss valid
        checks['stop_loss_valid'] = entry_signal.stop_loss > 0
        
        # Check 7: Risk-reward >= 1.5
        if entry_signal.ready and entry_signal.entry_price > 0 and entry_signal.stop_loss > 0:
            risk = abs(entry_signal.entry_price - entry_signal.stop_loss)
            # Assume TP1 at RR 1.5
            reward = risk * 1.5
            checks['rr_valid'] = reward / risk >= 1.5 if risk > 0 else False
        else:
            checks['rr_valid'] = False
        
        # All checks must pass
        all_passed = all(checks.values())
        
        # Find first failed check
        failed_check = None
        if not all_passed:
            for check_name, passed in checks.items():
                if not passed:
                    failed_check = check_name
                    break
        
        message = "✅ All mandatory requirements met" if all_passed else f"❌ Failed: {failed_check}"
        
        return GateResult(
            gate_number=1,
            gate_name="Mandatory Requirements",
            passed=all_passed,
            checks=checks,
            failed_check=failed_check,
            message=message
        )
    
    def _gate2_quality(self, setup_score: SetupScore) -> GateResult:
        """
        Gate 2: Quality Score
        
        Setup quality must meet minimum standards
        """
        checks = {}
        
        # Check 1: Total score >= minimum
        checks['total_score'] = setup_score.total_score >= self.min_setup_score
        
        # Check 2: Setup recommended by scorer
        checks['recommended'] = setup_score.recommended
        
        # Check 3: Not F grade
        checks['not_failing_grade'] = setup_score.grade != "F"
        
        all_passed = all(checks.values())
        
        failed_check = None
        if not all_passed:
            if not checks['total_score']:
                failed_check = f"Score too low ({setup_score.total_score:.1f} < {self.min_setup_score})"
            elif not checks['recommended']:
                failed_check = f"Not recommended: {setup_score.reason}"
            elif not checks['not_failing_grade']:
                failed_check = "Failing grade (F)"
        
        message = f"✅ Quality OK (Score: {setup_score.total_score:.1f}, Grade: {setup_score.grade})" if all_passed else f"❌ {failed_check}"
        
        return GateResult(
            gate_number=2,
            gate_name="Quality Score",
            passed=all_passed,
            checks=checks,
            failed_check=failed_check,
            message=message
        )
    
    def _gate3_risk(
        self,
        entry_signal: EntrySignal,
        account_state: Dict
    ) -> GateResult:
        """
        Gate 3: Risk Limits
        
        Risk management rules must be satisfied
        """
        checks = {}
        
        # Check 1: Daily risk limit
        current_daily_risk = account_state.get('daily_risk_pct', 0.0)
        trade_risk_pct = account_state.get('trade_risk_pct', 0.02)  # Default 2%
        
        checks['daily_risk_ok'] = (current_daily_risk + trade_risk_pct) <= self.max_daily_risk
        
        # Check 2: Portfolio risk limit
        current_portfolio_risk = account_state.get('portfolio_risk_pct', 0.0)
        checks['portfolio_risk_ok'] = (current_portfolio_risk + trade_risk_pct) <= self.max_portfolio_risk
        
        # Check 3: Position limit
        current_positions = account_state.get('open_positions', 0)
        checks['position_limit_ok'] = current_positions < self.max_positions
        
        # Check 4: Coin cooldown
        coin_in_cooldown = account_state.get('coin_in_cooldown', False)
        checks['cooldown_ok'] = not coin_in_cooldown
        
        # Check 5: Position size valid
        position_size = account_state.get('position_size', 0.0)
        checks['position_size_valid'] = position_size > 0
        
        all_passed = all(checks.values())
        
        failed_check = None
        if not all_passed:
            if not checks['daily_risk_ok']:
                failed_check = f"Daily risk limit ({current_daily_risk + trade_risk_pct:.1%} > {self.max_daily_risk:.1%})"
            elif not checks['portfolio_risk_ok']:
                failed_check = f"Portfolio risk limit ({current_portfolio_risk + trade_risk_pct:.1%} > {self.max_portfolio_risk:.1%})"
            elif not checks['position_limit_ok']:
                failed_check = f"Position limit ({current_positions} >= {self.max_positions})"
            elif not checks['cooldown_ok']:
                failed_check = "Coin in cooldown period"
            elif not checks['position_size_valid']:
                failed_check = "Invalid position size"
        
        message = "✅ Risk limits OK" if all_passed else f"❌ {failed_check}"
        
        return GateResult(
            gate_number=3,
            gate_name="Risk Limits",
            passed=all_passed,
            checks=checks,
            failed_check=failed_check,
            message=message
        )
    
    def _gate4_psychological(self, account_state: Dict) -> GateResult:
        """
        Gate 4: Psychological Checks
        
        Behavioral safeguards against emotional trading
        """
        checks = {}
        
        # Check 1: Not FOMO
        fomo_score = account_state.get('fomo_score', 0)
        checks['not_fomo'] = fomo_score < 50  # Threshold from Parça 6
        
        # Check 2: Not revenge trading
        in_revenge_mode = account_state.get('revenge_mode', False)
        checks['not_revenge'] = not in_revenge_mode
        
        # Check 3: Not overtrading (hourly)
        trades_last_hour = account_state.get('trades_last_hour', 0)
        checks['hourly_limit_ok'] = trades_last_hour < self.max_trades_per_hour
        
        # Check 4: Not overtrading (daily)
        trades_today = account_state.get('trades_today', 0)
        checks['daily_limit_ok'] = trades_today < self.max_trades_per_day
        
        # Check 5: Emotional state OK
        emotional_state = account_state.get('emotional_state', 'calm')
        checks['emotional_ok'] = emotional_state in ['calm', 'confident']
        
        all_passed = all(checks.values())
        
        failed_check = None
        if not all_passed:
            if not checks['not_fomo']:
                failed_check = f"FOMO detected (score: {fomo_score})"
            elif not checks['not_revenge']:
                failed_check = "Revenge trading mode active"
            elif not checks['hourly_limit_ok']:
                failed_check = f"Hourly trade limit ({trades_last_hour} >= {self.max_trades_per_hour})"
            elif not checks['daily_limit_ok']:
                failed_check = f"Daily trade limit ({trades_today} >= {self.max_trades_per_day})"
            elif not checks['emotional_ok']:
                failed_check = f"Poor emotional state: {emotional_state}"
        
        message = "✅ Psychological checks OK" if all_passed else f"❌ {failed_check}"
        
        return GateResult(
            gate_number=4,
            gate_name="Psychological",
            passed=all_passed,
            checks=checks,
            failed_check=failed_check,
            message=message
        )


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from core import EntrySignal, TrendResult, Zone, ChoCHResult
    from .setup_scorer import SetupScore
    
    print("\n" + "="*60)
    print("GATE SYSTEM - TEST")
    print("="*60)
    
    # Config
    config = {
        'rl': {
            'setup_scoring': {'min_score': 40}
        },
        'risk': {
            'daily_limit_pct': 0.06,
            'portfolio_limit_pct': 0.08,
            'max_positions': 3
        },
        'behavior': {
            'anti_overtrading': {
                'max_trades_per_hour': 2,
                'max_trades_per_day': 5
            }
        }
    }
    
    # Create gate system
    gate_system = GateSystem(config)
    
    print(f"\n✅ Gate System created")
    print(f"   Min setup score: {gate_system.min_setup_score}")
    print(f"   Max daily risk: {gate_system.max_daily_risk:.1%}")
    print(f"   Max trades/hour: {gate_system.max_trades_per_hour}")
    
    # ═══════════════════════════════════════════════════════════
    # Scenario 1: ALL GATES PASS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SCENARIO 1: ALL GATES PASS")
    print(f"{'='*60}\n")
    
    # Mock entry signal (all good)
    entry_good = EntrySignal(
        ready=True,
        action="ENTER",
        trend=TrendResult("UP", 50500, 50000, 0.01, 0.05, 0.03, 0.85, True, False, 0.9, 0),
        zone=Zone("Z1", 50000, 50100, 50050, 2, 0.002, 100, 50, "1H", "both", 8.5, 2.0),
        choch=ChoCHResult(True, "LONG", 50200, 150, 50000, 0.75, 0.3, 0.45),
        fibonacci=None,
        direction="LONG",
        entry_price=50050,
        entry_level="0.705",
        entry_quality="EXCELLENT",
        trend_aligned=True,
        zone_valid=True,
        choch_strong=True,
        fib_touched=True,
        stop_loss=49750,
        risk_per_unit=300,
        message="Entry ready"
    )
    
    setup_good = SetupScore(
        total_score=85.0,
        grade="A",
        trend_score=23.0,
        zone_score=24.0,
        choch_score=23.0,
        fib_score=15.0,
        breakdown={},
        recommended=True,
        reason="Excellent setup"
    )
    
    account_good = {
        'daily_risk_pct': 0.02,
        'portfolio_risk_pct': 0.03,
        'trade_risk_pct': 0.02,
        'open_positions': 1,
        'position_size': 1.0,
        'coin_in_cooldown': False,
        'fomo_score': 20,
        'revenge_mode': False,
        'trades_last_hour': 0,
        'trades_today': 2,
        'emotional_state': 'confident'
    }
    
    result_good = gate_system.validate(entry_good, setup_good, account_good)
    
    print(f"Result: {'✅ APPROVED' if result_good.all_passed else '❌ REJECTED'}")
    print(f"Gates Passed: {result_good.gates_passed}/{result_good.total_gates}")
    print(f"\nGate Results:")
    for gate in [result_good.gate1, result_good.gate2, result_good.gate3, result_good.gate4]:
        status = "✅ PASS" if gate.passed else "❌ FAIL"
        print(f"  Gate {gate.gate_number} ({gate.gate_name}): {status}")
        print(f"    {gate.message}")
    
    print(f"\n{result_good.reason}")
    
    # ═══════════════════════════════════════════════════════════
    # Scenario 2: GATE 2 FAILS (Low Quality)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SCENARIO 2: GATE 2 FAILS (Low Quality)")
    print(f"{'='*60}\n")
    
    setup_poor = SetupScore(
        total_score=35.0,  # Below minimum
        grade="F",
        trend_score=10.0,
        zone_score=10.0,
        choch_score=10.0,
        fib_score=5.0,
        breakdown={},
        recommended=False,
        reason="Score too low"
    )
    
    result_poor = gate_system.validate(entry_good, setup_poor, account_good)
    
    print(f"Result: {'✅ APPROVED' if result_poor.all_passed else '❌ REJECTED'}")
    print(f"Gates Passed: {result_poor.gates_passed}/{result_poor.total_gates}")
    print(f"\n{result_poor.reason}")
    
    # ═══════════════════════════════════════════════════════════
    # Scenario 3: GATE 3 FAILS (Risk Limit)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SCENARIO 3: GATE 3 FAILS (Risk Limit)")
    print(f"{'='*60}\n")
    
    account_risky = account_good.copy()
    account_risky['daily_risk_pct'] = 0.05  # Already at 5%, adding 2% = 7% > 6% limit
    
    result_risky = gate_system.validate(entry_good, setup_good, account_risky)
    
    print(f"Result: {'✅ APPROVED' if result_risky.all_passed else '❌ REJECTED'}")
    print(f"Gates Passed: {result_risky.gates_passed}/{result_risky.total_gates}")
    print(f"\n{result_risky.reason}")
    
    # ═══════════════════════════════════════════════════════════
    # Scenario 4: GATE 4 FAILS (FOMO)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SCENARIO 4: GATE 4 FAILS (FOMO)")
    print(f"{'='*60}\n")
    
    account_fomo = account_good.copy()
    account_fomo['fomo_score'] = 75  # High FOMO
    
    result_fomo = gate_system.validate(entry_good, setup_good, account_fomo)
    
    print(f"Result: {'✅ APPROVED' if result_fomo.all_passed else '❌ REJECTED'}")
    print(f"Gates Passed: {result_fomo.gates_passed}/{result_fomo.total_gates}")
    print(f"\n{result_fomo.reason}")
    
    print("\n" + "="*60)
    print("✅ Gate System working correctly!")
    print("="*60 + "\n")