from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .interfaces import RMMOrder, TradeResult


@dataclass
class OpenPosition:
    trade_id: str
    order: RMMOrder
    filled: bool = False
    entry_fill: Optional[float] = None
    remaining_qty: float = 0.0
    tp_list: List[float] = field(default_factory=list)
    closed: bool = False
    r_accum: float = 0.0


class DemoExecutor:
    """
    DemoExecutor (paper-trade simülasyonu)
    - TP1: %50 pozisyon, +0.6R
    - TP2: %30 pozisyon, +1.2R
    - TP3: %20 pozisyon, +1.8R
    - SL : -1.0R
    """

    def __init__(self):
        self.positions: Dict[str, OpenPosition] = {}

    def place_order(self, order: RMMOrder) -> str:
        trade_id = str(uuid.uuid4())[:8]
        pos = OpenPosition(trade_id=trade_id, order=order, filled=True,
                           entry_fill=order.entry, remaining_qty=order.qty_coin,
                           tp_list=list(order.tp_list))
        self.positions[trade_id] = pos
        return trade_id

    def on_price(self, symbol: str, price: float) -> List[TradeResult]:
        closed_results: List[TradeResult] = []

        for tid, pos in list(self.positions.items()):
            if pos.order.symbol != symbol or pos.closed:
                continue
            order = pos.order

            # === LONG ===
            if order.side == "LONG":
                if price <= order.stop:
                    res = TradeResult(trade_id=tid, r_realized=-1.0, risk_pct=order.risk_pct_used)
                    pos.closed = True
                    closed_results.append(res)
                    continue
                for _ in range(min(3, len(pos.tp_list))):
                    if price >= pos.tp_list[0]:
                        pos.r_accum += 0.6
                        pos.tp_list.pop(0)
                if not pos.tp_list:
                    res = TradeResult(trade_id=tid, r_realized=pos.r_accum, risk_pct=order.risk_pct_used)
                    pos.closed = True
                    closed_results.append(res)

            # === SHORT ===
            else:
                if price >= order.stop:
                    res = TradeResult(trade_id=tid, r_realized=-1.0, risk_pct=order.risk_pct_used)
                    pos.closed = True
                    closed_results.append(res)
                    continue
                for _ in range(min(3, len(pos.tp_list))):
                    if price <= pos.tp_list[0]:
                        pos.r_accum += 0.6
                        pos.tp_list.pop(0)
                if not pos.tp_list:
                    res = TradeResult(trade_id=tid, r_realized=pos.r_accum, risk_pct=order.risk_pct_used)
                    pos.closed = True
                    closed_results.append(res)

        for res in closed_results:
            self.positions.pop(res.trade_id, None)
        return closed_results

    def close_all(self) -> List[TradeResult]:
        out = []
        for tid, pos in list(self.positions.items()):
            if pos.closed:
                continue
            out.append(TradeResult(trade_id=tid, r_realized=pos.r_accum, risk_pct=pos.order.risk_pct_used))
            self.positions.pop(tid, None)
        return out
