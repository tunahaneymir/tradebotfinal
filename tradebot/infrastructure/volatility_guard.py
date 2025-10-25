"""
Volatility Guard
Based on: pa-strateji3 Parça 9

Features:
- Extreme move detection (>10% in 5min)
- New trade blocking (30min)
- Open position protection (tighten SL, closer TP)
- Auto-recovery
"""

from __future__ import annotations
from typing import List, Dict, Optional, Literal
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque


@dataclass
class VolatilityConfig:
    """Volatility guard konfigürasyonu"""
    extreme_move_threshold: float = 0.10  # %10
    lookback_minutes: int = 5
    block_duration_minutes: int = 30
    sl_tighten_factor: float = 0.5  # SL mesafesini yarıya düşür
    tp_closer_factor: float = 0.7  # TP'yi %70'e çek


@dataclass
class PricePoint:
    """Fiyat noktası"""
    timestamp: datetime
    price: float


@dataclass
class VolatilityEvent:
    """Volatilite olayı"""
    coin: str
    timestamp: datetime
    price_start: float
    price_peak: float
    move_pct: float
    direction: Literal["UP", "DOWN"]
    blocked_until: datetime


class VolatilityGuard:
    """
    Aşırı volatilite koruması
    
    Detection:
    - >10% fiyat hareketi 5 dakikada
    
    Actions:
    - Yeni trade'leri 30 dakika blokla
    - Açık pozisyonlar için:
      * Stop loss'u sıkılaştır (%50)
      * Take profit'i yakınlaştır (%70)
    
    Recovery:
    - 30 dakika sonra otomatik normal mode
    """
    
    def __init__(self, config: Optional[VolatilityConfig] = None):
        self.config = config or VolatilityConfig()
        
        # Price history (coin başına)
        self.price_history: Dict[str, deque] = {}
        
        # Active volatility events
        self.active_events: Dict[str, VolatilityEvent] = {}
        
        # Statistics
        self.total_events = 0
        self.blocked_trades = 0
        
        print(f"🛡️  Volatility Guard initialized")
        print(f"   Threshold: {self.config.extreme_move_threshold*100}% in {self.config.lookback_minutes}m")
        print(f"   Block duration: {self.config.block_duration_minutes}m")
    
    # ═══════════════════════════════════════════════════════════
    # PRICE MONITORING
    # ═══════════════════════════════════════════════════════════
    
    def update_price(self, coin: str, price: float, timestamp: Optional[datetime] = None):
        """
        Fiyat güncellemesi - volatilite kontrolü
        
        Args:
            coin: Coin sembolü
            price: Mevcut fiyat
            timestamp: Timestamp (None ise şimdi)
            
        Returns:
            True: Extreme volatility detected
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Price history'e ekle
        if coin not in self.price_history:
            self.price_history[coin] = deque(maxlen=100)  # Son 100 data point
        
        self.price_history[coin].append(PricePoint(timestamp, price))
        
        # Volatilite kontrolü
        return self._check_extreme_volatility(coin)
    
    def _check_extreme_volatility(self, coin: str) -> bool:
        """Extreme volatility var mı kontrol et"""
        history = self.price_history.get(coin)
        
        if not history or len(history) < 2:
            return False
        
        # Son N dakikalık data'yı al
        now = datetime.now()
        cutoff = now - timedelta(minutes=self.config.lookback_minutes)
        
        recent_points = [p for p in history if p.timestamp >= cutoff]
        
        if len(recent_points) < 2:
            return False
        
        # Min ve max fiyat bul
        prices = [p.price for p in recent_points]
        min_price = min(prices)
        max_price = max(prices)
        
        # Hareket yüzdesi
        move_pct = (max_price - min_price) / min_price
        
        # Threshold aşıldı mı?
        if move_pct >= self.config.extreme_move_threshold:
            # Yön belirle
            latest_price = recent_points[-1].price
            direction = "UP" if latest_price > min_price + (move_pct * min_price / 2) else "DOWN"
            
            # Event oluştur
            self._trigger_volatility_event(coin, min_price, max_price, move_pct, direction)
            return True
        
        return False
    
    def _trigger_volatility_event(self, coin: str, price_start: float, 
                                  price_peak: float, move_pct: float, direction: str):
        """Volatility event tetikle"""
        now = datetime.now()
        blocked_until = now + timedelta(minutes=self.config.block_duration_minutes)
        
        event = VolatilityEvent(
            coin=coin,
            timestamp=now,
            price_start=price_start,
            price_peak=price_peak,
            move_pct=move_pct,
            direction=direction,
            blocked_until=blocked_until
        )
        
        self.active_events[coin] = event
        self.total_events += 1
        
        print(f"\n{'='*60}")
        print(f"⚠️  🌊 EXTREME VOLATILITY DETECTED")
        print(f"{'='*60}")
        print(f"Coin: {coin}")
        print(f"Move: {move_pct*100:.1f}% in {self.config.lookback_minutes}m ({direction})")
        print(f"Price: ${price_start:,.2f} → ${price_peak:,.2f}")
        print(f"\nActions:")
        print(f"  ❌ New trades BLOCKED until {blocked_until.strftime('%H:%M:%S')}")
        print(f"  🔒 Open positions: SL tightened, TP closer")
        print(f"{'='*60}\n")
        
        # Telegram bildirim (implement edilecek)
        # self.telegram.send_critical(f"Extreme volatility: {coin} {move_pct*100:.1f}%")
    
    # ═══════════════════════════════════════════════════════════
    # TRADE CONTROLS
    # ═══════════════════════════════════════════════════════════
    
    def is_trading_blocked(self, coin: str) -> bool:
        """Coin için trading bloklu mu?"""
        if coin not in self.active_events:
            return False
        
        event = self.active_events[coin]
        
        # Süre doldu mu?
        if datetime.now() >= event.blocked_until:
            # Event'i temizle
            del self.active_events[coin]
            print(f"✅ Volatility block expired for {coin} - Trading resumed")
            return False
        
        return True
    
    def check_new_trade(self, coin: str) -> tuple[bool, Optional[str]]:
        """
        Yeni trade açılabilir mi kontrol et
        
        Returns:
            (allowed, reason)
        """
        if self.is_trading_blocked(coin):
            event = self.active_events[coin]
            remaining = (event.blocked_until - datetime.now()).total_seconds() / 60
            
            reason = f"Extreme volatility - blocked for {remaining:.0f}m more"
            self.blocked_trades += 1
            
            return False, reason
        
        return True, None
    
    def adjust_position_for_volatility(self, position: Dict) -> Dict:
        """
        Açık pozisyon için volatilite ayarlaması
        
        Args:
            position: Pozisyon bilgisi (entry, stop, tp1, tp2, tp3)
            
        Returns:
            Adjusted pozisyon
        """
        coin = position.get("coin")
        
        if not self.is_trading_blocked(coin):
            return position  # Volatility yok, değişiklik yapma
        
        event = self.active_events[coin]
        adjusted = position.copy()
        
        entry = position.get("entry", 0)
        stop = position.get("stop", 0)
        direction = position.get("direction", "LONG")
        
        # Stop loss sıkılaştır
        if direction == "LONG":
            stop_distance = entry - stop
            new_stop = entry - (stop_distance * self.config.sl_tighten_factor)
            adjusted["stop"] = new_stop
        else:  # SHORT
            stop_distance = stop - entry
            new_stop = entry + (stop_distance * self.config.sl_tighten_factor)
            adjusted["stop"] = new_stop
        
        # Take profit'leri yakınlaştır
        for tp_key in ["tp1", "tp2", "tp3"]:
            if tp_key in position and position[tp_key]:
                tp_orig = position[tp_key]
                
                if direction == "LONG":
                    tp_distance = tp_orig - entry
                    adjusted[tp_key] = entry + (tp_distance * self.config.tp_closer_factor)
                else:  # SHORT
                    tp_distance = entry - tp_orig
                    adjusted[tp_key] = entry - (tp_distance * self.config.tp_closer_factor)
        
        print(f"🔒 Position adjusted for volatility ({coin}):")
        print(f"   SL: ${stop:.2f} → ${adjusted['stop']:.2f} (tightened)")
        print(f"   TP1: ${position.get('tp1', 0):.2f} → ${adjusted.get('tp1', 0):.2f} (closer)")
        
        return adjusted
    
    # ═══════════════════════════════════════════════════════════
    # STATISTICS & MONITORING
    # ═══════════════════════════════════════════════════════════
    
    def get_active_events(self) -> List[Dict]:
        """Aktif volatility event'leri"""
        now = datetime.now()
        
        return [
            {
                "coin": coin,
                "move_pct": event.move_pct * 100,
                "direction": event.direction,
                "blocked_until": event.blocked_until.strftime("%H:%M:%S"),
                "remaining_minutes": (event.blocked_until - now).total_seconds() / 60
            }
            for coin, event in self.active_events.items()
        ]
    
    def get_statistics(self) -> Dict:
        """İstatistikler"""
        return {
            "total_events": self.total_events,
            "blocked_trades": self.blocked_trades,
            "active_blocks": len(self.active_events),
            "coins_affected": list(self.active_events.keys())
        }
    
    def get_coin_volatility(self, coin: str, minutes: int = 60) -> Optional[float]:
        """Son N dakikalık volatilite hesapla"""
        history = self.price_history.get(coin)
        
        if not history or len(history) < 2:
            return None
        
        # Son N dakikalık data
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent_points = [p for p in history if p.timestamp >= cutoff]
        
        if len(recent_points) < 2:
            return None
        
        # Volatility (price range / average)
        prices = [p.price for p in recent_points]
        price_range = max(prices) - min(prices)
        avg_price = sum(prices) / len(prices)
        
        return price_range / avg_price if avg_price > 0 else 0
    
    def clear_expired_events(self):
        """Süresi dolmuş event'leri temizle"""
        now = datetime.now()
        expired = [
            coin for coin, event in self.active_events.items()
            if now >= event.blocked_until
        ]
        
        for coin in expired:
            del self.active_events[coin]
            print(f"✅ Volatility block expired: {coin}")
    
    def force_clear_block(self, coin: str):
        """Bloğu zorla kaldır (manuel müdahale)"""
        if coin in self.active_events:
            del self.active_events[coin]
            print(f"🔓 Volatility block manually cleared: {coin}")
    
    def reset_statistics(self):
        """İstatistikleri sıfırla"""
        self.total_events = 0
        self.blocked_trades = 0
        print("🔄 Statistics reset")


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    
    # Volatility guard oluştur
    guard = VolatilityGuard()
    
    print("\n🧪 Testing Volatility Guard...\n")
    
    # Normal fiyat hareketleri (OK)
    print("📈 Normal price movements:")
    for i in range(10):
        price = 50000 + (i * 50)  # +$50 artış
        guard.update_price("BTCUSDT", price)
        time.sleep(0.1)
    
    allowed, reason = guard.check_new_trade("BTCUSDT")
    print(f"  Trade allowed: {allowed}\n")
    
    # Extreme volatility simülasyonu
    print("🌊 Simulating extreme volatility:")
    base_price = 50000
    
    # 5 dakikada %12 artış
    for i in range(5):
        spike_price = base_price * (1 + 0.12 * (i+1) / 5)
        detected = guard.update_price("BTCUSDT", spike_price)
        
        if detected:
            print(f"  ⚠️  Extreme move detected at ${spike_price:,.2f}")
        
        time.sleep(0.1)
    
    # Trade kontrolü
    print("\n🔍 Checking trade after volatility:")
    allowed, reason = guard.check_new_trade("BTCUSDT")
    print(f"  Trade allowed: {allowed}")
    if not allowed:
        print(f"  Reason: {reason}")
    
    # Position adjustment
    print("\n🔒 Adjusting open position:")
    position = {
        "coin": "BTCUSDT",
        "direction": "LONG",
        "entry": 50000,
        "stop": 49500,
        "tp1": 51000,
        "tp2": 51500,
        "tp3": 52000
    }
    
    adjusted = guard.adjust_position_for_volatility(position)
    print(f"  Original SL: ${position['stop']:,.2f}")
    print(f"  Adjusted SL: ${adjusted['stop']:,.2f}")
    print(f"  Original TP1: ${position['tp1']:,.2f}")
    print(f"  Adjusted TP1: ${adjusted['tp1']:,.2f}")
    
    # Active events
    print("\n📊 Active Events:")
    events = guard.get_active_events()
    for event in events:
        print(f"  {event['coin']}: {event['move_pct']:.1f}% {event['direction']}")
        print(f"    Blocked until: {event['blocked_until']} ({event['remaining_minutes']:.0f}m)")
    
    # Statistics
    print("\n📈 Statistics:")
    stats = guard.get_statistics()
    print(f"  Total events: {stats['total_events']}")
    print(f"  Blocked trades: {stats['blocked_trades']}")
    print(f"  Active blocks: {stats['active_blocks']}")
    
    # Volatility calculation
    vol = guard.get_coin_volatility("BTCUSDT", minutes=5)
    if vol:
        print(f"\n🌊 Current volatility (5m): {vol*100:.2f}%")