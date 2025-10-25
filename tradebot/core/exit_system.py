"""
Exit System - Parça 2
Based on: pa-strateji2 Parça 2

Exit System:
- TP1 (50%): RR 1.5 (guaranteed profit)
- TP2 (30%): Liquidity/Zone target (RR 2.5+)
- TP3 (20%): Dynamic trailing until opposite ChoCH
- Breakeven management after TP1
- Trailing stop loss after TP2
- Stop loss management
"""

from __future__ import annotations
from typing import Optional, Dict, Literal, List
from dataclasses import dataclass
import numpy as np

from .choch_detector import ChoCHDetector, ChoCHResult


@dataclass
class TakeProfitLevel:
    """Take profit level information"""
    name: str  # "TP1", "TP2", "TP3"
    price: float
    size_ratio: float  # 0.5, 0.3, 0.2
    rr_ratio: float
    reason: str
    hit: bool = False


@dataclass
class ExitSignal:
    """Exit signal with all information"""
    # Signal type
    signal_type: Literal["TP1", "TP2", "TP3", "STOP_LOSS", "NONE"]
    
    # Action
    action: Optional[Literal["CLOSE_PARTIAL", "CLOSE_ALL", "UPDATE_SL", "NONE"]]
    close_ratio: float  # How much to close (0.0-1.0)
    
    # Price levels
    current_price: float
    triggered_tp: Optional[TakeProfitLevel]
    new_stop_loss: Optional[float]
    
    # Position state
    remaining_position: float  # Ratio remaining after action
    
    # Profit tracking
    realized_profit: float  # From this exit
    total_realized: float   # Total so far
    
    # Message
    message: str


@dataclass
class PositionState:
    """Active position state"""
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    current_stop_loss: float
    original_stop_loss: float
    
    # Position size
    original_size: float
    remaining_size: float
    
    # TP levels
    tp1: TakeProfitLevel
    tp2: TakeProfitLevel
    tp3: TakeProfitLevel
    
    # Tracking
    tp1_hit: bool = False
    tp2_hit: bool = False
    breakeven_moved: bool = False
    trailing_active: bool = False
    
    # Profit tracking
    total_realized_profit: float = 0.0


class ExitSystem:
    """
    Exit System Manager
    
    Manages all exit logic:
    1. TP1 (50%): RR 1.5 → Close + Move to breakeven
    2. TP2 (30%): Liquidity/Zone → Close + Start trailing
    3. TP3 (20%): Trail until opposite ChoCH
    4. Stop Loss: Risk management
    
    Usage:
        exit_sys = ExitSystem(config)
        
        # Initialize position
        position = exit_sys.initialize_position(
            direction="LONG",
            entry_price=50000,
            stop_loss=49500,
            position_size=1.0
        )
        
        # Check exits each candle
        signal = exit_sys.check_exit(
            position=position,
            current_high=51000,
            current_low=50800,
            current_close=50900,
            ...
        )
        
        if signal.action:
            # Execute exit!
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Exit System
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Exit configuration
        exit_config = config.get('exit', {}) if config else {}
        tp_config = exit_config.get('take_profit', {})
        
        # TP ratios and sizes
        self.tp1_size = tp_config.get('tp1', {}).get('ratio', 0.5)
        self.tp1_rr = tp_config.get('tp1', {}).get('rr', 1.5)
        
        self.tp2_size = tp_config.get('tp2', {}).get('ratio', 0.3)
        self.tp2_min_rr = tp_config.get('tp2', {}).get('rr_min', 2.5)
        
        self.tp3_size = tp_config.get('tp3', {}).get('ratio', 0.2)
        
        # Trailing configuration
        trailing_config = exit_config.get('trailing', {})
        self.trailing_enabled = trailing_config.get('enabled', True)
        self.trailing_buffer = trailing_config.get('buffer_pct', 0.005)  # 0.5%
        
        # Breakeven configuration
        be_config = exit_config.get('breakeven', {})
        self.breakeven_enabled = be_config.get('enabled', True)
        
        # ChoCH detector for TP3
        self.choch_detector = ChoCHDetector(config)
    
    def initialize_position(
        self,
        direction: Literal["LONG", "SHORT"],
        entry_price: float,
        stop_loss: float,
        position_size: float,
        tp2_target: Optional[float] = None
    ) -> PositionState:
        """
        Initialize position with calculated TP levels
        
        Args:
            direction: Trade direction
            entry_price: Entry price
            stop_loss: Initial stop loss
            position_size: Position size (in coins/contracts)
            tp2_target: Optional TP2 price (liquidity/zone)
            
        Returns:
            PositionState with all TP levels
        """
        risk = abs(entry_price - stop_loss)
        
        # ═══════════════════════════════════════════════════════════
        # TP1: RR 1.5 (Guaranteed)
        # ═══════════════════════════════════════════════════════════
        if direction == "LONG":
            tp1_price = entry_price + (risk * self.tp1_rr)
        else:
            tp1_price = entry_price - (risk * self.tp1_rr)
        
        tp1 = TakeProfitLevel(
            name="TP1",
            price=tp1_price,
            size_ratio=self.tp1_size,
            rr_ratio=self.tp1_rr,
            reason="Guaranteed profit"
        )
        
        # ═══════════════════════════════════════════════════════════
        # TP2: Liquidity/Zone (RR 2.5+)
        # ═══════════════════════════════════════════════════════════
        if tp2_target is None:
            # Default: RR 2.5
            if direction == "LONG":
                tp2_price = entry_price + (risk * self.tp2_min_rr)
            else:
                tp2_price = entry_price - (risk * self.tp2_min_rr)
            tp2_reason = f"RR {self.tp2_min_rr}"
        else:
            tp2_price = tp2_target
            tp2_rr = abs(tp2_price - entry_price) / risk
            tp2_reason = f"Liquidity/Zone (RR {tp2_rr:.1f})"
        
        tp2_rr = abs(tp2_price - entry_price) / risk
        
        tp2 = TakeProfitLevel(
            name="TP2",
            price=tp2_price,
            size_ratio=self.tp2_size,
            rr_ratio=tp2_rr,
            reason=tp2_reason
        )
        
        # ═══════════════════════════════════════════════════════════
        # TP3: Dynamic (ChoCH signal)
        # ═══════════════════════════════════════════════════════════
        tp3 = TakeProfitLevel(
            name="TP3",
            price=0.0,  # Dynamic
            size_ratio=self.tp3_size,
            rr_ratio=0.0,  # Dynamic
            reason="Trail until opposite ChoCH"
        )
        
        # Create position state
        return PositionState(
            direction=direction,
            entry_price=entry_price,
            current_stop_loss=stop_loss,
            original_stop_loss=stop_loss,
            original_size=position_size,
            remaining_size=position_size,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3
        )
    
    def check_exit(
        self,
        position: PositionState,
        current_high: float,
        current_low: float,
        current_close: float,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        open_prices: np.ndarray,
        volume: np.ndarray
    ) -> ExitSignal:
        """
        Check for exit signals
        
        Priority:
        1. Stop Loss (highest priority)
        2. TP1 (if not hit)
        3. TP2 (if TP1 hit, not TP2)
        4. TP3 / Trailing (if TP2 hit)
        
        Args:
            position: Current position state
            current_high: Current candle high
            current_low: Current candle low
            current_close: Current candle close
            high: Historical high prices
            low: Historical low prices
            close: Historical close prices
            open_prices: Historical open prices
            volume: Historical volume
            
        Returns:
            ExitSignal
        """
        # ═══════════════════════════════════════════════════════════
        # PRIORITY 1: CHECK STOP LOSS
        # ═══════════════════════════════════════════════════════════
        stop_hit = self._check_stop_loss(
            position.direction,
            position.current_stop_loss,
            current_high,
            current_low
        )
        
        if stop_hit:
            loss = self._calculate_profit(
                position.direction,
                position.entry_price,
                position.current_stop_loss,
                position.remaining_size
            )
            
            return ExitSignal(
                signal_type="STOP_LOSS",
                action="CLOSE_ALL",
                close_ratio=1.0,
                current_price=position.current_stop_loss,
                triggered_tp=None,
                new_stop_loss=None,
                remaining_position=0.0,
                realized_profit=loss,
                total_realized=position.total_realized_profit + loss,
                message=f"❌ STOP LOSS HIT @ ${position.current_stop_loss:,.2f}"
            )
        
        # ═══════════════════════════════════════════════════════════
        # PRIORITY 2: CHECK TP1
        # ═══════════════════════════════════════════════════════════
        if not position.tp1_hit:
            tp1_hit = self._check_tp_hit(
                position.direction,
                position.tp1.price,
                current_high,
                current_low
            )
            
            if tp1_hit:
                profit = self._calculate_profit(
                    position.direction,
                    position.entry_price,
                    position.tp1.price,
                    position.original_size * position.tp1.size_ratio
                )
                
                # Update position
                position.tp1_hit = True
                position.remaining_size = position.original_size * (1 - position.tp1.size_ratio)
                position.total_realized_profit += profit
                
                # Move SL to breakeven
                new_sl = None
                if self.breakeven_enabled and not position.breakeven_moved:
                    new_sl = position.entry_price
                    position.current_stop_loss = new_sl
                    position.breakeven_moved = True
                
                return ExitSignal(
                    signal_type="TP1",
                    action="CLOSE_PARTIAL",
                    close_ratio=position.tp1.size_ratio,
                    current_price=position.tp1.price,
                    triggered_tp=position.tp1,
                    new_stop_loss=new_sl,
                    remaining_position=1 - position.tp1.size_ratio,
                    realized_profit=profit,
                    total_realized=position.total_realized_profit,
                    message=f"🎯 TP1 HIT @ ${position.tp1.price:,.2f} | Profit: ${profit:,.2f} | SL → Breakeven"
                )
        
        # ═══════════════════════════════════════════════════════════
        # PRIORITY 3: CHECK TP2
        # ═══════════════════════════════════════════════════════════
        if position.tp1_hit and not position.tp2_hit:
            tp2_hit = self._check_tp_hit(
                position.direction,
                position.tp2.price,
                current_high,
                current_low
            )
            
            if tp2_hit:
                profit = self._calculate_profit(
                    position.direction,
                    position.entry_price,
                    position.tp2.price,
                    position.original_size * position.tp2.size_ratio
                )
                
                # Update position
                position.tp2_hit = True
                position.remaining_size = position.original_size * position.tp3.size_ratio
                position.total_realized_profit += profit
                
                # Start trailing
                new_sl = None
                if self.trailing_enabled:
                    new_sl = self._calculate_trailing_stop(
                        position.direction,
                        current_close,
                        position.tp1.price  # Trail from TP1 level
                    )
                    position.current_stop_loss = new_sl
                    position.trailing_active = True
                
                return ExitSignal(
                    signal_type="TP2",
                    action="CLOSE_PARTIAL",
                    close_ratio=position.tp2.size_ratio,
                    current_price=position.tp2.price,
                    triggered_tp=position.tp2,
                    new_stop_loss=new_sl,
                    remaining_position=position.tp3.size_ratio,
                    realized_profit=profit,
                    total_realized=position.total_realized_profit,
                    message=f"🎯🎯 TP2 HIT @ ${position.tp2.price:,.2f} | Profit: ${profit:,.2f} | Trailing Active"
                )
        
        # ═══════════════════════════════════════════════════════════
        # PRIORITY 4: CHECK TP3 / TRAILING
        # ═══════════════════════════════════════════════════════════
        if position.tp2_hit:
            # Check for opposite ChoCH (TP3 signal)
            opposite_direction = "SHORT" if position.direction == "LONG" else "LONG"
            
            choch = self.choch_detector.detect(
                high, low, close, open_prices, volume,
                direction=opposite_direction
            )
            
            if choch.detected and choch.strength >= 0.4:
                # Opposite ChoCH detected - Close final position
                profit = self._calculate_profit(
                    position.direction,
                    position.entry_price,
                    current_close,
                    position.remaining_size
                )
                
                return ExitSignal(
                    signal_type="TP3",
                    action="CLOSE_ALL",
                    close_ratio=position.tp3.size_ratio,
                    current_price=current_close,
                    triggered_tp=position.tp3,
                    new_stop_loss=None,
                    remaining_position=0.0,
                    realized_profit=profit,
                    total_realized=position.total_realized_profit + profit,
                    message=f"🎯🎯🎯 TP3 - Opposite ChoCH @ ${current_close:,.2f} | Final Profit: ${profit:,.2f}"
                )
            
            # Update trailing stop
            if position.trailing_active:
                new_trailing_sl = self._update_trailing_stop(
                    position.direction,
                    position.current_stop_loss,
                    current_high,
                    current_low,
                    current_close
                )
                
                if new_trailing_sl != position.current_stop_loss:
                    position.current_stop_loss = new_trailing_sl
                    
                    return ExitSignal(
                        signal_type="NONE",
                        action="UPDATE_SL",
                        close_ratio=0.0,
                        current_price=current_close,
                        triggered_tp=None,
                        new_stop_loss=new_trailing_sl,
                        remaining_position=position.tp3.size_ratio,
                        realized_profit=0.0,
                        total_realized=position.total_realized_profit,
                        message=f"📈 Trailing SL updated → ${new_trailing_sl:,.2f}"
                    )
        
        # No exit signal
        return ExitSignal(
            signal_type="NONE",
            action="NONE",
            close_ratio=0.0,
            current_price=current_close,
            triggered_tp=None,
            new_stop_loss=None,
            remaining_position=position.remaining_size / position.original_size,
            realized_profit=0.0,
            total_realized=position.total_realized_profit,
            message="Monitoring position..."
        )
    
    def _check_stop_loss(
        self,
        direction: Literal["LONG", "SHORT"],
        stop_loss: float,
        current_high: float,
        current_low: float
    ) -> bool:
        """Check if stop loss was hit"""
        if direction == "LONG":
            return current_low <= stop_loss
        else:
            return current_high >= stop_loss
    
    def _check_tp_hit(
        self,
        direction: Literal["LONG", "SHORT"],
        tp_price: float,
        current_high: float,
        current_low: float
    ) -> bool:
        """Check if TP was hit"""
        if direction == "LONG":
            return current_high >= tp_price
        else:
            return current_low <= tp_price
    
    def _calculate_profit(
        self,
        direction: Literal["LONG", "SHORT"],
        entry_price: float,
        exit_price: float,
        size: float
    ) -> float:
        """Calculate profit/loss"""
        if direction == "LONG":
            return (exit_price - entry_price) * size
        else:
            return (entry_price - exit_price) * size
    
    def _calculate_trailing_stop(
        self,
        direction: Literal["LONG", "SHORT"],
        current_price: float,
        min_level: float
    ) -> float:
        """Calculate initial trailing stop"""
        if direction == "LONG":
            # Trail below price, but not below min_level
            trailing = current_price * (1 - self.trailing_buffer)
            return max(trailing, min_level)
        else:
            # Trail above price, but not above min_level
            trailing = current_price * (1 + self.trailing_buffer)
            return min(trailing, min_level)
    
    def _update_trailing_stop(
        self,
        direction: Literal["LONG", "SHORT"],
        current_sl: float,
        current_high: float,
        current_low: float,
        current_close: float
    ) -> float:
        """Update trailing stop (only moves in favor)"""
        if direction == "LONG":
            # Trail upward as price rises
            new_sl = current_close * (1 - self.trailing_buffer)
            return max(new_sl, current_sl)  # Only move up
        else:
            # Trail downward as price falls
            new_sl = current_close * (1 + self.trailing_buffer)
            return min(new_sl, current_sl)  # Only move down


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    
    # Config
    config = {
        'exit': {
            'take_profit': {
                'tp1': {'ratio': 0.5, 'rr': 1.5},
                'tp2': {'ratio': 0.3, 'rr_min': 2.5},
                'tp3': {'ratio': 0.2, 'dynamic': True}
            },
            'trailing': {'enabled': True, 'buffer_pct': 0.005},
            'breakeven': {'enabled': True, 'after': 'tp1'}
        },
        'entry': {
            'choch': {'min_strength': 0.4}
        }
    }
    
    # Create exit system
    exit_sys = ExitSystem(config)
    
    print("\n" + "="*60)
    print("EXIT SYSTEM - COMPLETE FLOW TEST")
    print("="*60)
    
    # Initialize position
    position = exit_sys.initialize_position(
        direction="LONG",
        entry_price=50000,
        stop_loss=49500,
        position_size=1.0,
        tp2_target=51500
    )
    
    print(f"\n📊 Position Initialized:")
    print(f"   Entry: ${position.entry_price:,.2f}")
    print(f"   Stop Loss: ${position.current_stop_loss:,.2f}")
    print(f"   Size: {position.original_size} BTC")
    print(f"\n   TP Levels:")
    print(f"   ├─ TP1 ({position.tp1.size_ratio*100:.0f}%): ${position.tp1.price:,.2f} (RR {position.tp1.rr_ratio})")
    print(f"   ├─ TP2 ({position.tp2.size_ratio*100:.0f}%): ${position.tp2.price:,.2f} (RR {position.tp2.rr_ratio:.1f})")
    print(f"   └─ TP3 ({position.tp3.size_ratio*100:.0f}%): Dynamic (Trail until ChoCH)")
    
    # Simulate price movement
    prices = [
        50000,  # Entry
        50200,  # Moving up
        50500,  # Approaching TP1
        50750,  # TP1 HIT!
        51000,  # Continuing
        51500,  # TP2 HIT!
        51800,  # Trailing
        52200,  # More profit
        51900,  # Pullback (trailing protects)
    ]
    
    # Create full arrays
    n = len(prices)
    close = np.array(prices)
    high = close + 50
    low = close - 50
    open_prices = close + np.random.randn(n) * 20
    volume = np.random.rand(n) * 1000 + 500
    
    print(f"\n{'='*60}")
    print("PRICE MOVEMENT SIMULATION")
    print(f"{'='*60}\n")
    
    for i in range(1, len(prices)):
        signal = exit_sys.check_exit(
            position=position,
            current_high=high[i],
            current_low=low[i],
            current_close=close[i],
            high=high[:i+1],
            low=low[:i+1],
            close=close[:i+1],
            open_prices=open_prices[:i+1],
            volume=volume[:i+1]
        )
        
        print(f"Candle {i}: ${close[i]:,.2f}")
        print(f"  Signal: {signal.signal_type}")
        print(f"  Action: {signal.action}")
        print(f"  Message: {signal.message}")
        
        if signal.action in ["CLOSE_PARTIAL", "CLOSE_ALL"]:
            print(f"  💰 Realized Profit: ${signal.realized_profit:,.2f}")
            print(f"  📊 Total Realized: ${signal.total_realized:,.2f}")
            print(f"  📦 Remaining: {signal.remaining_position*100:.0f}%")
        
        if signal.new_stop_loss:
            print(f"  🛡️  New SL: ${signal.new_stop_loss:,.2f}")
        
        print()
        
        if signal.action == "CLOSE_ALL":
            print("🏁 Position fully closed!")
            break
    
    print(f"{'='*60}")
    print(f"✅ Exit System test complete!")
    print(f"{'='*60}\n")