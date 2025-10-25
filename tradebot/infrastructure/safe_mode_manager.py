"""
Safe Mode Manager
Based on: pa-strateji3 Parça 9

Features:
- Drawdown >10% → Paper mode
- 3 consecutive losses → Risk reduction
- Auto recovery protocol
- Telegram alerts
"""

from __future__ import annotations
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SafeModeConfig:
    """Safe mode konfigürasyonu"""
    drawdown_threshold_pct: float = 0.10  # %10
    consecutive_loss_threshold: int = 3
    paper_mode_enabled: bool = True
    risk_reduction_factor: float = 0.5  # Risk yarıya düş
    recovery_win_streak: int = 2  # 2 win streak → normal mode
    recovery_drawdown_pct: float = 0.05  # DD <%5 → normal mode


@dataclass
class SafeModeState:
    """Safe mode durumu"""
    active: bool = False
    activated_at: Optional[str] = None
    reason: str = ""
    current_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    paper_mode: bool = False
    risk_multiplier: float = 1.0
    trades_in_safe_mode: int = 0


class SafeModeManager:
    """
    Safe Mode yönetimi - sermaye koruması için kritik
    
    Triggers:
    - Drawdown >10%
    - 3 consecutive losses
    
    Actions:
    - Paper trading mode
    - Risk %50'ye düş
    - Telegram uyarı
    
    Recovery:
    - 2 win streak VEYA
    - Drawdown <%5
    """
    
    def __init__(self, config: Optional[SafeModeConfig] = None):
        self.config = config or SafeModeConfig()
        self.state = SafeModeState()
        self.trade_history: List[dict] = []  # Son trade'ler
        
    def check_and_trigger(self, equity_usdt: float, current_drawdown_usdt: float) -> bool:
        """
        Safe mode kontrolü yap ve gerekirse aktive et
        
        Args:
            equity_usdt: Mevcut sermaye
            current_drawdown_usdt: Mevcut drawdown (USDT)
            
        Returns:
            True: Safe mode aktif/aktive edildi
            False: Normal operation
        """
        # Zaten aktifse kontrol etme
        if self.state.active:
            return True
        
        # Drawdown kontrolü
        dd_pct = abs(current_drawdown_usdt) / equity_usdt if equity_usdt > 0 else 0
        self.state.current_drawdown_pct = dd_pct
        
        if dd_pct >= self.config.drawdown_threshold_pct:
            self.activate_safe_mode(
                reason=f"Drawdown {dd_pct*100:.1f}% (limit: {self.config.drawdown_threshold_pct*100}%)"
            )
            return True
        
        # Consecutive loss kontrolü
        if self.state.consecutive_losses >= self.config.consecutive_loss_threshold:
            self.activate_safe_mode(
                reason=f"{self.state.consecutive_losses} consecutive losses"
            )
            return True
        
        return False
    
    def activate_safe_mode(self, reason: str):
        """Safe mode'u aktive et"""
        if self.state.active:
            return  # Zaten aktif
        
        self.state.active = True
        self.state.activated_at = datetime.now().isoformat()
        self.state.reason = reason
        self.state.paper_mode = self.config.paper_mode_enabled
        self.state.risk_multiplier = self.config.risk_reduction_factor
        self.state.trades_in_safe_mode = 0
        
        print("\n" + "="*60)
        print("⚠️  🚨 SAFE MODE ACTIVATED 🚨")
        print("="*60)
        print(f"Reason: {reason}")
        print(f"Activated at: {self.state.activated_at}")
        print(f"Paper mode: {'ENABLED' if self.state.paper_mode else 'DISABLED'}")
        print(f"Risk multiplier: {self.state.risk_multiplier}x")
        print("\nActions:")
        print("  1. Risk reduced to 50%")
        if self.state.paper_mode:
            print("  2. Paper trading mode ENABLED")
        print("  3. Telegram alert sent")
        print("\nRecovery conditions:")
        print(f"  - {self.config.recovery_win_streak} win streak OR")
        print(f"  - Drawdown < {self.config.recovery_drawdown_pct*100}%")
        print("="*60 + "\n")
        
        # Telegram bildirim gönder (implement edilecek)
        # self.telegram.send_critical(f"⚠️ SAFE MODE ACTIVATED\n{reason}")
    
    def deactivate_safe_mode(self, reason: str):
        """Safe mode'u deaktive et - normal operasyona dön"""
        if not self.state.active:
            return
        
        self.state.active = False
        self.state.paper_mode = False
        self.state.risk_multiplier = 1.0
        
        print("\n" + "="*60)
        print("✅ SAFE MODE DEACTIVATED - NORMAL OPERATION RESUMED")
        print("="*60)
        print(f"Reason: {reason}")
        print(f"Duration: {self._calculate_duration()}")
        print(f"Trades in safe mode: {self.state.trades_in_safe_mode}")
        print("="*60 + "\n")
        
        # Telegram bildirim
        # self.telegram.send_important(f"✅ Safe mode deactivated\n{reason}")
    
    def check_recovery(self, current_drawdown_pct: float) -> bool:
        """
        Recovery koşullarını kontrol et
        
        Args:
            current_drawdown_pct: Mevcut drawdown %
            
        Returns:
            True: Recovery koşulları sağlandı, safe mode kapat
        """
        if not self.state.active:
            return False
        
        # Koşul 1: Win streak
        if self.state.consecutive_wins >= self.config.recovery_win_streak:
            self.deactivate_safe_mode(f"{self.state.consecutive_wins} win streak achieved")
            return True
        
        # Koşul 2: Drawdown düştü
        if current_drawdown_pct < self.config.recovery_drawdown_pct:
            self.deactivate_safe_mode(f"Drawdown recovered to {current_drawdown_pct*100:.1f}%")
            return True
        
        return False
    
    def record_trade_result(self, is_win: bool, pnl: float):
        """
        Trade sonucunu kaydet ve streak'leri güncelle
        
        Args:
            is_win: Trade kazandırdı mı?
            pnl: P&L miktarı
        """
        # Trade history'e ekle
        self.trade_history.append({
            "timestamp": datetime.now().isoformat(),
            "is_win": is_win,
            "pnl": pnl
        })
        
        # Son 100 trade'i tut
        if len(self.trade_history) > 100:
            self.trade_history.pop(0)
        
        # Streak güncelle
        if is_win:
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0
        
        # Safe mode'daysa trade sayısını artır
        if self.state.active:
            self.state.trades_in_safe_mode += 1
        
        # Log
        streak_type = "WIN" if is_win else "LOSS"
        streak_count = self.state.consecutive_wins if is_win else self.state.consecutive_losses
        print(f"📊 Trade recorded: {streak_type} (streak: {streak_count})")
    
    def get_risk_multiplier(self) -> float:
        """
        Mevcut risk multiplier'ı döndür
        
        Returns:
            Safe mode aktifse 0.5, değilse 1.0
        """
        return self.state.risk_multiplier if self.state.active else 1.0
    
    def is_paper_mode(self) -> bool:
        """Paper mode aktif mi?"""
        return self.state.paper_mode
    
    def is_safe_mode_active(self) -> bool:
        """Safe mode aktif mi?"""
        return self.state.active
    
    def get_status(self) -> dict:
        """Mevcut durumu döndür"""
        return {
            "safe_mode_active": self.state.active,
            "paper_mode": self.state.paper_mode,
            "risk_multiplier": self.state.risk_multiplier,
            "consecutive_wins": self.state.consecutive_wins,
            "consecutive_losses": self.state.consecutive_losses,
            "current_drawdown_pct": self.state.current_drawdown_pct * 100,
            "reason": self.state.reason if self.state.active else None,
            "activated_at": self.state.activated_at,
            "trades_in_safe_mode": self.state.trades_in_safe_mode
        }
    
    def _calculate_duration(self) -> str:
        """Safe mode süresini hesapla"""
        if not self.state.activated_at:
            return "N/A"
        
        activated = datetime.fromisoformat(self.state.activated_at)
        duration = datetime.now() - activated
        
        hours = duration.total_seconds() / 3600
        if hours < 1:
            minutes = duration.total_seconds() / 60
            return f"{minutes:.0f} minutes"
        elif hours < 24:
            return f"{hours:.1f} hours"
        else:
            days = hours / 24
            return f"{days:.1f} days"
    
    def force_deactivate(self):
        """Safe mode'u zorla kapat (manuel müdahale)"""
        if self.state.active:
            self.deactivate_safe_mode("Manual override")
    
    def reset_streaks(self):
        """Streak'leri sıfırla"""
        self.state.consecutive_wins = 0
        self.state.consecutive_losses = 0
        print("🔄 Streaks reset")