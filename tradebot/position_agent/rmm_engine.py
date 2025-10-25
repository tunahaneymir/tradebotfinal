from __future__ import annotations
from typing import Optional, Tuple
from .interfaces import (
    RMMConfig,
    EquityState,
    RMMIntent,
    RMMOrder,
    RMMGuardResult,
    round_qty,
)

class RMMError(Exception):
    """RMMEngine özel hata sınıfı."""

class RMMEngine:
    """Position Agent (RMM) — risk/qty/leverage hesaplama motoru."""

    def __init__(self, config: Optional[RMMConfig] = None):
        self.cfg = config or RMMConfig()

    def build_order(
        self,
        equity: EquityState,
        intent: RMMIntent,
    ) -> Tuple[Optional[RMMOrder], RMMGuardResult]:
        self._validate_intent(intent)

        risk_pct = self._resolve_risk_pct(intent)
        guard = self._check_risk_budgets(equity, risk_pct)
        if not guard.allowed:
            return None, guard

        stop_dist = abs(intent.entry - intent.stop)
        if stop_dist <= 0:
            return None, RMMGuardResult(False, "invalid_stop", {"stop": intent.stop})

        eff_stop_dist = self._apply_buffers(stop_dist, intent.entry)
        risk_usdt = equity.equity_usdt * risk_pct
        qty_coin, notional = self._size_position(intent.entry, eff_stop_dist, risk_usdt)

        lev_raw = max(1.0, notional / max(equity.equity_usdt, 1e-9))
        lev_capped = min(lev_raw, self.cfg.max_leverage)
        lev_safe = self._apply_liq_buffer_constraint(intent, eff_stop_dist, lev_capped)

        if lev_safe < lev_raw:
            scale = lev_safe / max(lev_raw, 1e-9)
            qty_coin *= scale
            notional = qty_coin * intent.entry

        qty_coin = round_qty(qty_coin)
        if qty_coin <= 0:
            return None, RMMGuardResult(False, "qty_underflow", {"qty_coin": qty_coin})

        order = RMMOrder(
            symbol=intent.symbol,
            side=intent.side,
            entry=intent.entry,
            stop=intent.stop,
            tp_list=intent.tp_list,
            qty_coin=qty_coin,
            notional_usdt=qty_coin * intent.entry,
            leverage=max(1.0, min(self.cfg.max_leverage, notional / max(equity.equity_usdt, 1e-9))),
            risk_pct_used=risk_pct,
            notes={"eff_stop_dist": eff_stop_dist, "liq_buffer_mult": self.cfg.liq_buffer_mult},
        )
        return order, RMMGuardResult(True, details={"risk_pct": risk_pct})

    # ---------------- yardımcı iç fonksiyonlar ----------------

    def _validate_intent(self, intent: RMMIntent):
        if intent.entry <= 0 or intent.stop <= 0:
            raise RMMError("entry/stop must be > 0")
        if intent.side not in ("LONG", "SHORT"):
            raise RMMError("side must be LONG or SHORT")

    def _resolve_risk_pct(self, intent: RMMIntent) -> float:
        risk_pct = self.cfg.base_risk_pct
        if intent.re_entry:
            risk_pct *= self.cfg.re_entry_risk_factor
        if self.cfg.enable_rl_risk_factor and intent.risk_factor is not None:
            rf = max(0.5, min(1.5, float(intent.risk_factor)))
            risk_pct *= rf
        return float(risk_pct)

    def _check_risk_budgets(self, equity: EquityState, risk_pct: float) -> RMMGuardResult:
        if (equity.daily_r_used + 1.0) > self.cfg.daily_r_budget:
            return RMMGuardResult(False, "daily_r_exhausted",
                                  {"daily_r_used": equity.daily_r_used, "daily_r_budget": self.cfg.daily_r_budget})
        if abs(equity.weekly_dd_r) >= self.cfg.weekly_dd_limit_r:
            return RMMGuardResult(False, "weekly_dd_limit_reached",
                                  {"weekly_dd_r": equity.weekly_dd_r, "limit": self.cfg.weekly_dd_limit_r})
        return RMMGuardResult(True)

    def _apply_buffers(self, stop_dist: float, entry: float) -> float:
        fee = (self.cfg.fee_bps_round_trip / 10000.0) * entry
        slip = (self.cfg.slippage_bps / 10000.0) * entry
        return stop_dist + fee + slip

    def _size_position(self, entry: float, eff_stop_dist: float, risk_usdt: float):
        if eff_stop_dist <= 0:
            raise RMMError("effective stop distance must be > 0")
        qty_coin = risk_usdt / eff_stop_dist
        notional = qty_coin * entry
        return qty_coin, notional

    def _apply_liq_buffer_constraint(self, intent: RMMIntent, eff_stop_dist: float, lev_capped: float) -> float:
        stop_distance = eff_stop_dist
        if stop_distance <= 0:
            return lev_capped
        numerator = intent.entry * 0.8
        denom = self.cfg.liq_buffer_mult * stop_distance
        if denom <= 0:
            return lev_capped
        lev_by_liq = numerator / denom
        return max(1.0, min(lev_capped, lev_by_liq))

    def on_trade_closed_update_r(
        self,
        equity: EquityState,
        r_realized: float,
        risk_pct_used: float,
        last_equity_usdt: Optional[float] = None,
    ):
        """Trade kapandığında R muhasebesini günceller."""
        equity.daily_r_used += 1.0
        if r_realized < 0:
            equity.weekly_dd_r += abs(r_realized)
