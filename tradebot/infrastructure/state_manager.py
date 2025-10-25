"""
State Manager - Autosave & Recovery System
Based on: pa-strateji3 Parça 9
"""

from __future__ import annotations
import json
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class BotState:
    confidence: float = 0.5
    stress: float = 0.0
    patience: float = 0.5
    last_update: str = ""


@dataclass
class OpenPosition:
    trade_id: str
    coin: str
    direction: str
    entry: float
    position_size: float
    stop_loss: float
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    status: str = "ACTIVE"


@dataclass
class CooldownInfo:
    active: bool = False
    until: Optional[str] = None
    level: Optional[str] = None


class StateManager:
    """
    Bot state yönetimi
    
    Features:
    - 15 dakikalık otomatik checkpoint
    - Event-based acil kayıt
    - Cold-start recovery
    """

    def __init__(self, data_dir: str = "state"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.data_dir / "bot_state.json"
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        # State
        self.bot_state = BotState()
        self.open_positions: List[OpenPosition] = []
        self.cooldowns: Dict[str, CooldownInfo] = {}
        self.daily_trades = 0
        self.daily_risk_used = 0.0

        # Autosave
        self.autosave_enabled = False
        self.autosave_thread: Optional[threading.Thread] = None

        self.load_state()
        print(f"💾 State Manager initialized: {data_dir}")

    # ═══════════════════════════════════════════════════════════
    # AUTOSAVE
    # ═══════════════════════════════════════════════════════════

    def start_autosave(self):
        """15 dakikalık otomatik kayıt"""
        if not self.autosave_thread or not self.autosave_thread.is_alive():
            self.autosave_enabled = True
            self.autosave_thread = threading.Thread(target=self._autosave_loop, daemon=True)
            self.autosave_thread.start()
            print("✅ Autosave started (15 min interval)")

    def stop_autosave(self):
        """Autosave durdur"""
        self.autosave_enabled = False
        if self.autosave_thread:
            self.autosave_thread.join(timeout=2)
        print("⏸️  Autosave stopped")

    def _autosave_loop(self):
        """Autosave loop"""
        while self.autosave_enabled:
            time.sleep(900)  # 15 dakika
            if self.autosave_enabled:
                self.save_state(reason="15m_checkpoint")

    # ═══════════════════════════════════════════════════════════
    # SAVE & LOAD
    # ═══════════════════════════════════════════════════════════

    def save_state(self, reason: str = "manual"):
        """State'i kaydet"""
        state_dict = {
            "bot_state": asdict(self.bot_state),
            "open_positions": [asdict(p) for p in self.open_positions],
            "cooldowns": {k: asdict(v) for k, v in self.cooldowns.items()},
            "daily_trades": self.daily_trades,
            "daily_risk_used": self.daily_risk_used,
            "saved_at": datetime.now().isoformat()
        }

        try:
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state_dict, f, indent=2)
            temp_file.replace(self.state_file)
            print(f"💾 State saved: {reason}")
        except Exception as e:
            print(f"❌ Save error: {e}")

    def load_state(self) -> bool:
        """State'i yükle"""
        if not self.state_file.exists():
            print("ℹ️  No state file - fresh start")
            return False

        try:
            with open(self.state_file) as f:
                data = json.load(f)

            self.bot_state = BotState(**data.get("bot_state", {}))
            self.open_positions = [OpenPosition(**p) for p in data.get("open_positions", [])]
            self.cooldowns = {k: CooldownInfo(**v) for k, v in data.get("cooldowns", {}).items()}
            self.daily_trades = data.get("daily_trades", 0)
            self.daily_risk_used = data.get("daily_risk_used", 0.0)

            print(f"✅ State loaded - Open positions: {len(self.open_positions)}")
            return True
        except Exception as e:
            print(f"❌ Load error: {e}")
            return False

    # ═══════════════════════════════════════════════════════════
    # EVENT-BASED SAVE
    # ═══════════════════════════════════════════════════════════

    def event_save(self, event: str):
        """Event-based kayıt"""
        critical_events = ["new_trade_opened", "trade_closed", "error_detected", "stop_hit"]
        if event in critical_events:
            self.save_state(reason=f"event:{event}")

    # ═══════════════════════════════════════════════════════════
    # POSITION MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    def add_position(self, position: OpenPosition):
        """Pozisyon ekle"""
        self.open_positions.append(position)
        self.daily_trades += 1
        self.event_save("new_trade_opened")

    def close_position(self, trade_id: str):
        """Pozisyon kapat"""
        self.open_positions = [p for p in self.open_positions if p.trade_id != trade_id]
        self.event_save("trade_closed")

    def get_open_positions(self) -> List[OpenPosition]:
        """Açık pozisyonları döndür"""
        return self.open_positions

    # ═══════════════════════════════════════════════════════════
    # COOLDOWN MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    def add_cooldown(self, coin: str, duration_minutes: int, level: str = "soft"):
        """Cooldown ekle"""
        until = datetime.now() + timedelta(minutes=duration_minutes)
        self.cooldowns[coin] = CooldownInfo(active=True, until=until.isoformat(), level=level)
        self.event_save("cooldown_activated")

    def is_cooldown_active(self, coin: str) -> bool:
        """Cooldown aktif mi?"""
        if coin not in self.cooldowns:
            return False

        info = self.cooldowns[coin]
        if not info.active or not info.until:
            return False

        until_time = datetime.fromisoformat(info.until)
        if datetime.now() >= until_time:
            info.active = False
            return False

        return True

    # ═══════════════════════════════════════════════════════════
    # BACKUP
    # ═══════════════════════════════════════════════════════════

    def create_backup(self):
        """Backup oluştur"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"state_backup_{timestamp}.json"

        try:
            state_dict = {
                "bot_state": asdict(self.bot_state),
                "open_positions": [asdict(p) for p in self.open_positions],
                "cooldowns": {k: asdict(v) for k, v in self.cooldowns.items()},
                "daily_trades": self.daily_trades,
                "daily_risk_used": self.daily_risk_used,
                "backup_time": datetime.now().isoformat()
            }

            with open(backup_file, 'w') as f:
                json.dump(state_dict, f, indent=2)

            print(f"💾 Backup created: {backup_file.name}")
            self._cleanup_old_backups()
        except Exception as e:
            print(f"❌ Backup error: {e}")

    def _cleanup_old_backups(self, days: int = 7):
        """Eski backup'ları sil"""
        cutoff = datetime.now() - timedelta(days=days)
        for backup_file in self.backup_dir.glob("state_backup_*.json"):
            file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
            if file_time < cutoff:
                backup_file.unlink()
        return True

    # ═══════════════════════════════════════════════════════════
    # UTILITY
    # ═══════════════════════════════════════════════════════════

    def update_emotional_state(self, confidence: float, stress: float, patience: float):
        """Emotional state güncelle"""
        self.bot_state.confidence = confidence
        self.bot_state.stress = stress
        self.bot_state.patience = patience
        self.bot_state.last_update = datetime.now().isoformat()

    def get_stats(self) -> Dict:
        """İstatistikler"""
        return {
            "open_positions": len(self.open_positions),
            "daily_trades": self.daily_trades,
            "daily_risk_used": self.daily_risk_used,
            "active_cooldowns": sum(1 for c in self.cooldowns.values() if c.active),
            "confidence": self.bot_state.confidence,
            "stress": self.bot_state.stress,
            "patience": self.bot_state.patience
        }
