from __future__ import annotations
from typing import Optional
from .interfaces import EquityState, PADecisionPacket, RLDecision, RMMIntent, Side, RMMGuardResult
from .rmm_engine import RMMEngine
from .executor_demo import DemoExecutor


class PositionAgent:
    """PA + RL kararını alır, RMM ile pozisyon açar ve executor’a gönderir."""

    def __init__(self):
        self.rmm = RMMEngine()
        self.exec = DemoExecutor()

    def handle_decision(self, equity: EquityState, pa_packet: PADecisionPacket, rl_decision: RLDecision):
        candidate = next((c for c in pa_packet.candidates if c.id == rl_decision.candidate_id), None)
        if candidate is None:
            return None, RMMGuardResult(False, "no_candidate")

        side = "LONG" if rl_decision.decision == "BUY" else "SHORT"
        intent = RMMIntent(
            symbol=pa_packet.symbol,
            side=side,
            entry=candidate.entry,
            stop=candidate.stop,
            tp_list=candidate.tp_list,
            candidate_id=candidate.id,
            rl_confidence=rl_decision.confidence,
            re_entry=rl_decision.re_entry,
            risk_factor=rl_decision.risk_factor,
        )

        order, guard = self.rmm.build_order(equity, intent)
        if not guard.allowed or order is None:
            return None, guard

        trade_id = self.exec.place_order(order)
        return trade_id, guard

    def on_price(self, symbol: str, price: float):
        return self.exec.on_price(symbol, price)
