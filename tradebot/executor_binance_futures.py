"""
Binance Futures Testnet Executor
Gerçek demo hesaba emir gönderir
"""

from __future__ import annotations
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from decimal import Decimal
from binance.client import Client
from binance.enums import *
from .interfaces import RMMOrder, TradeResult


@dataclass
class OpenPosition:
    """Açık pozisyon bilgisi"""
    trade_id: str
    order: RMMOrder
    binance_order_id: Optional[str] = None
    filled: bool = False
    entry_fill: Optional[float] = None
    remaining_qty: float = 0.0
    tp_orders: List[str] = field(default_factory=list)  # TP order ID'leri
    sl_order: Optional[str] = None  # SL order ID
    closed: bool = False
    r_accum: float = 0.0


class BinanceFuturesExecutor:
    """
    Binance Futures Testnet'e gerçek emir gönderen executor
    
    Features:
    - Market emirler
    - OCO (One-Cancels-Other) TP/SL
    - Position tracking
    - Risk management
    """
    
    def __init__(self, config: dict):
        """
        Initialize Binance Futures client
        
        Args:
            config: Exchange configuration
        """
        self.config = config
        
        # Initialize Binance client
        self.client = Client(
            api_key=config.get('api_key', ''),
            api_secret=config.get('api_secret', ''),
            testnet=config.get('testnet', True)
        )
        
        # Set to futures
        if config.get('testnet'):
            self.client.API_URL = 'https://testnet.binancefuture.com'
        
        # Open positions
        self.positions: Dict[str, OpenPosition] = {}
        
        # Set leverage if specified
        default_leverage = config.get('leverage', 5)
        print(f"✅ Binance Futures Executor initialized (Leverage: {default_leverage}x)")
    
    def place_order(self, order: RMMOrder) -> str:
        """
        Place order on Binance Futures Testnet
        
        Args:
            order: RMMOrder from position agent
            
        Returns:
            trade_id: Unique trade identifier
        """
        trade_id = str(uuid.uuid4())[:8]
        symbol = order.symbol
        
        try:
            # Set leverage for symbol
            leverage = self.config.get('leverage', 5)
            self.client.futures_change_leverage(
                symbol=symbol,
                leverage=leverage
            )
            
            # Set margin type
            margin_type = self.config.get('margin_type', 'ISOLATED')
            try:
                self.client.futures_change_margin_type(
                    symbol=symbol,
                    marginType=margin_type
                )
            except Exception:
                # Already set, ignore
                pass
            
            # Determine side
            side = SIDE_BUY if order.side == "LONG" else SIDE_SELL
            
            # Calculate quantity (round to symbol precision)
            quantity = self._round_quantity(symbol, order.qty_coin)
            
            print(f"\n🔄 Placing {order.side} order:")
            print(f"   Symbol: {symbol}")
            print(f"   Quantity: {quantity}")
            print(f"   Entry: ${order.entry:.2f}")
            print(f"   Stop Loss: ${order.stop:.2f}")
            print(f"   Take Profits: {len(order.tp_list)}")
            
            # Place MARKET order for entry
            entry_order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            
            print(f"   ✅ Entry order placed: {entry_order['orderId']}")
            
            # Wait a bit for fill
            time.sleep(0.5)
            
            # Get fill price
            fill_price = self._get_fill_price(entry_order['orderId'], symbol)
            
            # Create position record
            pos = OpenPosition(
                trade_id=trade_id,
                order=order,
                binance_order_id=entry_order['orderId'],
                filled=True,
                entry_fill=fill_price,
                remaining_qty=quantity
            )
            
            # Place Stop Loss order
            sl_side = SIDE_SELL if order.side == "LONG" else SIDE_BUY
            sl_order = self.client.futures_create_order(
                symbol=symbol,
                side=sl_side,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=self._round_price(symbol, order.stop),
                closePosition=True  # Close entire position
            )
            pos.sl_order = sl_order['orderId']
            print(f"   ✅ Stop Loss placed: ${order.stop:.2f}")
            
            # Place Take Profit orders (3 levels)
            tp_ratios = [0.5, 0.3, 0.2]  # 50%, 30%, 20%
            
            for i, (tp_price, ratio) in enumerate(zip(order.tp_list, tp_ratios)):
                tp_qty = self._round_quantity(symbol, quantity * ratio)
                
                tp_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=sl_side,
                    type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                    stopPrice=self._round_price(symbol, tp_price),
                    quantity=tp_qty
                )
                
                pos.tp_orders.append(tp_order['orderId'])
                print(f"   ✅ TP{i+1} placed: ${tp_price:.2f} ({ratio*100:.0f}%)")
            
            # Store position
            self.positions[trade_id] = pos
            
            print(f"✅ Order placed successfully! Trade ID: {trade_id}\n")
            
            return trade_id
            
        except Exception as e:
            print(f"❌ Error placing order: {e}")
            raise
    
    def on_price(self, symbol: str, price: float) -> List[TradeResult]:
        """
        Check positions for fills (TP/SL hit)
        
        Note: Binance will automatically execute TP/SL orders,
        so we just need to check order status periodically.
        """
        closed_results: List[TradeResult] = []
        
        for tid, pos in list(self.positions.items()):
            if pos.order.symbol != symbol or pos.closed:
                continue
            
            # Check if any orders filled
            updated = self._check_order_status(pos)
            
            if updated and pos.closed:
                closed_results.append(
                    TradeResult(
                        trade_id=tid,
                        r_realized=pos.r_accum,
                        risk_pct=pos.order.risk_pct_used
                    )
                )
        
        return closed_results
    
    def close_all(self) -> List[TradeResult]:
        """Close all open positions"""
        out = []
        
        for tid, pos in list(self.positions.items()):
            if pos.closed:
                continue
            
            try:
                # Cancel all open orders
                self.client.futures_cancel_all_open_orders(
                    symbol=pos.order.symbol
                )
                
                # Close position at market
                side = SIDE_SELL if pos.order.side == "LONG" else SIDE_BUY
                self.client.futures_create_order(
                    symbol=pos.order.symbol,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=pos.remaining_qty
                )
                
                out.append(
                    TradeResult(
                        trade_id=tid,
                        r_realized=pos.r_accum,
                        risk_pct=pos.order.risk_pct_used
                    )
                )
                
                self.positions.pop(tid, None)
                
            except Exception as e:
                print(f"❌ Error closing position {tid}: {e}")
        
        return out
    
    def _check_order_status(self, pos: OpenPosition) -> bool:
        """
        Check if any TP/SL orders filled
        Returns True if position updated
        """
        try:
            symbol = pos.order.symbol
            
            # Check TP orders
            for tp_id in list(pos.tp_orders):
                order = self.client.futures_get_order(
                    symbol=symbol,
                    orderId=tp_id
                )
                
                if order['status'] == 'FILLED':
                    # TP hit! Accumulate R
                    pos.r_accum += 0.6  # Each TP = +0.6R
                    pos.tp_orders.remove(tp_id)
                    print(f"✅ TP hit for trade {pos.trade_id}: +0.6R")
            
            # Check SL
            if pos.sl_order:
                sl_order = self.client.futures_get_order(
                    symbol=symbol,
                    orderId=pos.sl_order
                )
                
                if sl_order['status'] == 'FILLED':
                    # SL hit
                    pos.r_accum = -1.0
                    pos.closed = True
                    print(f"❌ Stop Loss hit for trade {pos.trade_id}: -1.0R")
                    return True
            
            # If all TPs filled, close position
            if not pos.tp_orders:
                pos.closed = True
                print(f"✅ All TPs hit for trade {pos.trade_id}: +{pos.r_accum:.1f}R")
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️  Error checking order status: {e}")
            return False
    
    def _get_fill_price(self, order_id: str, symbol: str) -> float:
        """Get fill price of an order"""
        try:
            order = self.client.futures_get_order(
                symbol=symbol,
                orderId=order_id
            )
            return float(order.get('avgPrice', 0))
        except Exception:
            return 0.0
    
    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to symbol's step size"""
        # Get symbol info
        info = self.client.futures_exchange_info()
        
        for s in info['symbols']:
            if s['symbol'] == symbol:
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        step_size = float(f['stepSize'])
                        precision = len(str(step_size).rstrip('0').split('.')[-1])
                        return round(quantity, precision)
        
        return round(quantity, 3)  # Default
    
    def _round_price(self, symbol: str, price: float) -> float:
        """Round price to symbol's tick size"""
        # Get symbol info
        info = self.client.futures_exchange_info()
        
        for s in info['symbols']:
            if s['symbol'] == symbol:
                for f in s['filters']:
                    if f['filterType'] == 'PRICE_FILTER':
                        tick_size = float(f['tickSize'])
                        precision = len(str(tick_size).rstrip('0').split('.')[-1])
                        return round(price, precision)
        
        return round(price, 2)  # Default
    
    def get_account_balance(self) -> float:
        """Get USDT balance"""
        try:
            account = self.client.futures_account()
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    return float(asset['walletBalance'])
            return 0.0
        except Exception as e:
            print(f"⚠️  Error getting balance: {e}")
            return 0.0
    
    def get_open_positions(self) -> List[dict]:
        """Get all open positions"""
        try:
            positions = self.client.futures_position_information()
            return [p for p in positions if float(p['positionAmt']) != 0]
        except Exception as e:
            print(f"⚠️  Error getting positions: {e}")
            return []


# Alias for compatibility
ExecutorDemo = BinanceFuturesExecutor
