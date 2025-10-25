from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, TypedDict

Side = Literal["LONG", "SHORT"]
DecisionAction = Literal["BUY", "SELL", "SKIP"]
MillisTimestamp = int


class PartialExitConfig(TypedDict, total=False):
    tp1_ratio: float
    tp2_ratio: float
    tp3_ratio: float


class TrailingConfig(TypedDict, total=False):
    type: str
    mult: float


# ---------------- RL / PA arayüz tipleri ----------------

@dataclass(slots=True)
class RLPolicy:
    partial_exit: Optional[PartialExitConfig] = None
    breakeven_after: Optional[Literal["tp1", "tp2", "tp3"]] = None
    trailing: Optional[TrailingConfig] = None

    def __post_init__(self) -> None:
        if self.partial_exit:
            for key, ratio in self.partial_exit.items():
                if ratio < 0:
                    raise ValueError(f"Partial exit ratio for {key} must be non-negative.")
            if sum(self.partial_exit.values()) > 1.0:
                raise ValueError("Sum of partial exit ratios cannot exceed 1.0.")
        if self.trailing:
            mult = self.trailing.get("mult")
            if mult is not None and mult <= 0:
                raise ValueError("Trailing multiplier must be positive.")


@dataclass(slots=True)
class RLDecision:
    symbol: str
    decision: DecisionAction
    candidate_id: Optional[str] = None
    confidence: float = 0.5
    policy: Optional[RLPolicy] = None
    re_entry: bool = False
    risk_factor: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if self.risk_factor is not None and self.risk_factor <= 0:
            raise ValueError("risk_factor must be positive when provided.")


@dataclass(slots=True)
class PACandidate:
    id: str
    entry: float
    stop: float
    tp_list: List[float] = field(default_factory=list)
    rr: Optional[float] = None
    zone_quality: Optional[float] = None
    atr: Optional[float] = None
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.entry <= 0:
            raise ValueError("entry must be a positive number.")
        if self.stop <= 0:
            raise ValueError("stop must be a positive number.")
        if any(tp <= 0 for tp in self.tp_list):
            raise ValueError("All take-profit levels must be positive numbers.")
        if self.rr is not None and self.rr <= 0:
            raise ValueError("rr must be positive when provided.")
        # 🔧 zone_quality 0–10 aralığında
        if self.zone_quality is not None and not 0.0 <= self.zone_quality <= 10.0:
            raise ValueError("zone_quality must be between 0 and 10 when provided.")
        if self.atr is not None and self.atr <= 0:
            raise ValueError("atr must be positive when provided.")


@dataclass(slots=True)
class PADecisionPacket:
    symbol: str
    candidates: List[PACandidate]
    timestamp: MillisTimestamp = field(default_factory=lambda: int(time.time() * 1000))


# ---------------- RMM konfigürasyon & state ----------------

@dataclass(slots=True)
class RMMConfig:
    base_risk_pct: float = 0.03
    enable_rl_risk_factor: bool = False
    re_entry_risk_factor: float = 0.5
    max_leverage: float = 5.0
    daily_r_budget: float = 3.0
    weekly_dd_limit_r: float = 10.0
    fee_bps_round_trip: float = 6.0
    slippage_bps: float = 5.0
    liq_buffer_mult: float = 2.0


@dataclass(slots=True)
class EquityState:
    equity_usdt: float
    daily_r_used: float = 0.0
    weekly_dd_r: float = 0.0
    day_tag: Optional[str] = None
    week_tag: Optional[str] = None


# ---------------- RMM giriş / çıkış ----------------

@dataclass(slots=True)
class RMMIntent:
    symbol: str
    side: Side
    entry: float
    stop: float
    tp_list: List[float] = field(default_factory=list)
    candidate_id: Optional[str] = None
    rl_confidence: Optional[float] = None
    re_entry: bool = False
    risk_factor: Optional[float] = None


@dataclass(slots=True)
class RMMOrder:
    symbol: str
    side: Side
    entry: float
    stop: float
    tp_list: List[float]
    qty_coin: float
    notional_usdt: float
    leverage: float
    risk_pct_used: float
    r_value: float = 1.0
    notes: Dict = field(default_factory=dict)


@dataclass(slots=True)
class RMMGuardResult:
    allowed: bool
    reason: Optional[str] = None
    details: Dict = field(default_factory=dict)


@dataclass(slots=True)
class TradeResult:
    trade_id: str
    r_realized: float
    rr_realized: Optional[float] = None
    pnl_usdt: Optional[float] = None
    risk_pct: Optional[float] = None
    expectancy_like: Optional[float] = None
    closed_at: MillisTimestamp = field(default_factory=lambda: int(time.time() * 1000))


def round_qty(qty: float, step: float = 1e-4) -> float:
    """Round quantity down to the nearest step supported by the exchange."""
    if step <= 0:
        raise ValueError("step must be a positive number.")
    return math.floor(qty / step) * step
