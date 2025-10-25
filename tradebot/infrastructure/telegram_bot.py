"""
Telegram Bot Integration
Based on: pa-strateji3 Parça 9

Features:
- Multi-level notifications (CRITICAL, IMPORTANT, SUMMARY, INFO)
- Bot commands (/durum, /pozisyonlar, etc.)
- Quiet hours (00:00-08:00)
- Message grouping
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from datetime import datetime, time
from dataclasses import dataclass
import asyncio


@dataclass
class TelegramConfig:
    """Telegram konfigürasyonu"""
    bot_token: str
    chat_id: str
    enabled: bool = True
    quiet_hours_start: time = time(0, 0)  # 00:00
    quiet_hours_end: time = time(8, 0)    # 08:00
    group_small_messages: bool = True
    max_message_length: int = 4096  # Telegram limit


class NotificationLevel:
    """Bildirim seviyeleri"""
    CRITICAL = "CRITICAL"     # Trade open/close, errors
    IMPORTANT = "IMPORTANT"   # FOMO blocked, gate rejected
    SUMMARY = "SUMMARY"       # Daily/weekly reports
    INFO = "INFO"             # Parameter changes


class TelegramBot:
    """
    Telegram bot interface
    
    Notification Levels:
    - CRITICAL: Trade events, system errors (her zaman gönderilir)
    - IMPORTANT: Behavioral blocks, zone detections
    - SUMMARY: Daily/weekly performance
    - INFO: Parameter updates, RL training
    
    Commands:
    - /durum: Mevcut durumu göster
    - /pozisyonlar: Açık pozisyonları listele
    - /bugun: Bugünkü performans
    - /hafta: Haftalık performans
    - /duraklat: Trading'i durdur
    - /devam: Trading'e devam et
    - /risk <pct>: Risk oranını ayarla
    - /export: CSV export
    """
    
    def __init__(self, config: TelegramConfig):
        self.config = config
        self.message_queue: List[Dict] = []
        self.last_summary_time: Optional[datetime] = None
        
        # Quiet hours kontrolü
        self.in_quiet_hours = False
        
        print(f"📱 Telegram bot initialized")
        print(f"   Enabled: {config.enabled}")
        print(f"   Quiet hours: {config.quiet_hours_start} - {config.quiet_hours_end}")
    
    # ═══════════════════════════════════════════════════════════
    # NOTIFICATION METHODS
    # ═══════════════════════════════════════════════════════════
    
    def send_critical(self, message: str, metadata: Optional[Dict] = None):
        """
        CRITICAL seviye bildirim
        
        Events:
        - Trade opened
        - Trade closed
        - Stop/TP hit
        - System error
        - Drawdown >5%
        - Daily limit hit
        """
        formatted = self._format_message("🚨 CRITICAL", message, metadata)
        self._send_now(formatted, force=True)  # Quiet hours'da bile gönder
    
    def send_important(self, message: str, metadata: Optional[Dict] = None):
        """
        IMPORTANT seviye bildirim
        
        Events:
        - New zone detected
        - ChoCH signal
        - FOMO blocked
        - Revenge blocked
        - Re-entry opportunity
        - Gate rejected
        """
        if self._is_quiet_hours():
            self._queue_message("⚠️  IMPORTANT", message, metadata)
        else:
            formatted = self._format_message("⚠️  IMPORTANT", message, metadata)
            self._send_now(formatted)
    
    def send_summary(self, message: str, metadata: Optional[Dict] = None):
        """
        SUMMARY seviye bildirim
        
        Events:
        - Daily end-of-day report
        - Weekly performance
        - Monthly report
        """
        formatted = self._format_message("📊 SUMMARY", message, metadata)
        self._send_now(formatted)
    
    def send_info(self, message: str, metadata: Optional[Dict] = None):
        """
        INFO seviye bildirim
        
        Events:
        - Parameter change
        - RL model update
        - Zone memory update
        """
        if self._is_quiet_hours():
            return  # Quiet hours'da INFO gönderme
        
        formatted = self._format_message("ℹ️  INFO", message, metadata)
        self._send_now(formatted)
    
    # ═══════════════════════════════════════════════════════════
    # TRADE NOTIFICATIONS
    # ═══════════════════════════════════════════════════════════
    
    def send_trade_opened(self, trade_data: Dict):
        """Trade açıldı bildirimi"""
        message = f"""
📈 TRADE OPENED

Coin: {trade_data.get('coin', 'N/A')}
Direction: {trade_data.get('direction', 'N/A')}
Entry: ${trade_data.get('entry', 0):,.2f}
Size: {trade_data.get('qty', 0):.4f}
Risk: {trade_data.get('risk_pct', 0):.2f}%
Stop Loss: ${trade_data.get('stop', 0):,.2f}

Setup Score: {trade_data.get('setup_score', 0):.0f}/100
Zone Quality: {trade_data.get('zone_quality', 0):.1f}/10
        """.strip()
        
        self.send_critical(message, metadata=trade_data)
    
    def send_trade_closed(self, trade_data: Dict):
        """Trade kapatıldı bildirimi"""
        pnl = trade_data.get('pnl_pct', 0)
        emoji = "✅" if pnl > 0 else "❌"
        
        message = f"""
{emoji} TRADE CLOSED

Coin: {trade_data.get('coin', 'N/A')}
Direction: {trade_data.get('direction', 'N/A')}
Exit Reason: {trade_data.get('exit_reason', 'N/A')}

Entry: ${trade_data.get('entry', 0):,.2f}
Exit: ${trade_data.get('exit', 0):,.2f}
Duration: {trade_data.get('duration_min', 0):.0f} min

P&L: {pnl:+.2f}% (${trade_data.get('pnl_usd', 0):+,.2f})
R-Multiple: {trade_data.get('r_multiple', 0):+.2f}R
        """.strip()
        
        self.send_critical(message, metadata=trade_data)
    
    def send_stop_hit(self, trade_data: Dict):
        """Stop loss hit bildirimi"""
        message = f"""
🛑 STOP LOSS HIT

Coin: {trade_data.get('coin', 'N/A')}
Entry: ${trade_data.get('entry', 0):,.2f}
Stop: ${trade_data.get('stop', 0):,.2f}
Loss: {trade_data.get('pnl_pct', 0):.2f}%

Cooldown: {trade_data.get('cooldown_minutes', 0)} minutes
        """.strip()
        
        self.send_critical(message, metadata=trade_data)
    
    def send_tp_hit(self, trade_data: Dict, tp_level: int):
        """Take profit hit bildirimi"""
        message = f"""
🎯 TP{tp_level} HIT

Coin: {trade_data.get('coin', 'N/A')}
Entry: ${trade_data.get('entry', 0):,.2f}
TP{tp_level}: ${trade_data.get(f'tp{tp_level}', 0):,.2f}
Profit: +{trade_data.get('pnl_pct', 0):.2f}%

Closed: {trade_data.get('closed_pct', 0):.0f}%
Remaining: {trade_data.get('remaining_pct', 0):.0f}%
        """.strip()
        
        self.send_critical(message, metadata=trade_data)
    
    # ═══════════════════════════════════════════════════════════
    # BEHAVIORAL NOTIFICATIONS
    # ═══════════════════════════════════════════════════════════
    
    def send_fomo_blocked(self, reason: str, details: Dict):
        """FOMO bloklama bildirimi"""
        message = f"""
🚫 FOMO TRADE BLOCKED

Reason: {reason}

Details:
- Setup complete: {details.get('setup_complete', False)}
- Price distance: {details.get('price_distance_pct', 0):.1f}%
- Time since last: {details.get('time_since_last', 0)} min
- FOMO score: {details.get('fomo_score', 0):.0f}/100

⚠️  Wait for proper setup!
        """.strip()
        
        self.send_important(message, metadata=details)
    
    def send_revenge_blocked(self, reason: str, details: Dict):
        """Revenge trade bloklama bildirimi"""
        message = f"""
🚫 REVENGE TRADE BLOCKED

Reason: {reason}

Details:
- Recent loss: {details.get('recent_loss_pct', 0):.2f}%
- Time since loss: {details.get('time_since_loss', 0)} min
- Consecutive losses: {details.get('consecutive_losses', 0)}
- Cooldown until: {details.get('cooldown_until', 'N/A')}

😌 Take a break, come back fresh!
        """.strip()
        
        self.send_important(message, metadata=details)
    
    def send_safe_mode_activated(self, reason: str, details: Dict):
        """Safe mode aktivasyon bildirimi"""
        message = f"""
⚠️  🚨 SAFE MODE ACTIVATED 🚨

Reason: {reason}

Current Status:
- Drawdown: {details.get('drawdown_pct', 0):.1f}%
- Consecutive losses: {details.get('consecutive_losses', 0)}
- Paper mode: {'ENABLED' if details.get('paper_mode') else 'DISABLED'}
- Risk: {details.get('risk_multiplier', 1) * 100:.0f}%

Actions Taken:
✓ Risk reduced to 50%
✓ Only best setups accepted
✓ Recovery mode active

Recovery Conditions:
- 2 win streak OR
- Drawdown < 5%
        """.strip()
        
        self.send_critical(message, metadata=details)
    
    # ═══════════════════════════════════════════════════════════
    # DAILY/WEEKLY SUMMARIES
    # ═══════════════════════════════════════════════════════════
    
    def send_daily_summary(self, stats: Dict):
        """Günlük özet rapor"""
        message = f"""
📊 DAILY SUMMARY - {stats.get('date', 'N/A')}

Performance:
- Total Trades: {stats.get('total_trades', 0)}
- Wins: {stats.get('wins', 0)} | Losses: {stats.get('losses', 0)}
- Win Rate: {stats.get('win_rate', 0):.1f}%

P&L:
- Total: {stats.get('total_pnl_pct', 0):+.2f}% (${stats.get('total_pnl_usd', 0):+,.2f})
- Best Trade: {stats.get('best_trade_pct', 0):+.2f}%
- Worst Trade: {stats.get('worst_trade_pct', 0):+.2f}%

Risk:
- Total Risk Used: {stats.get('risk_used_pct', 0):.1f}%
- Current DD: {stats.get('current_dd_pct', 0):.1f}%

Behavioral:
- FOMO Blocks: {stats.get('fomo_blocks', 0)}
- Revenge Blocks: {stats.get('revenge_blocks', 0)}
- Overtrade Blocks: {stats.get('overtrade_blocks', 0)}

Emotion:
- Confidence: {stats.get('confidence', 0):.2f}
- Stress: {stats.get('stress', 0):.2f}
- Patience: {stats.get('patience', 0):.2f}
        """.strip()
        
        self.send_summary(message, metadata=stats)
    
    def send_weekly_summary(self, stats: Dict):
        """Haftalık özet rapor"""
        message = f"""
📊 WEEKLY SUMMARY - Week {stats.get('week_number', 'N/A')}

Performance:
- Total Trades: {stats.get('total_trades', 0)}
- Win Rate: {stats.get('win_rate', 0):.1f}%
- Profit Factor: {stats.get('profit_factor', 0):.2f}
- Sharpe Ratio: {stats.get('sharpe', 0):.2f}

P&L:
- Total: {stats.get('total_pnl_pct', 0):+.2f}%
- Best Day: {stats.get('best_day_pct', 0):+.2f}%
- Worst Day: {stats.get('worst_day_pct', 0):+.2f}%

Top Coins:
{self._format_top_coins(stats.get('top_coins', []))}

Setup Quality:
- Avg Score: {stats.get('avg_setup_score', 0):.1f}/100
- Avg Zone Quality: {stats.get('avg_zone_quality', 0):.1f}/10

Targets:
{self._format_progress_bar('Win Rate', stats.get('win_rate', 0), 65)}
{self._format_progress_bar('Profit Factor', stats.get('profit_factor', 0) * 50, 100)}
        """.strip()
        
        self.send_summary(message, metadata=stats)
    
    # ═══════════════════════════════════════════════════════════
    # BOT COMMANDS
    # ═══════════════════════════════════════════════════════════
    
    def handle_command(self, command: str, args: List[str] = None) -> str:
        """
        Bot komutlarını işle
        
        Commands:
        - /durum: Mevcut durum
        - /pozisyonlar: Açık pozisyonlar
        - /bugun: Bugünkü performans
        - /hafta: Haftalık performans
        - /duraklat: Trading'i durdur
        - /devam: Trading'e devam et
        - /risk <pct>: Risk ayarla
        - /export: CSV export
        """
        args = args or []
        
        if command == "/durum":
            return self._cmd_status()
        elif command == "/pozisyonlar":
            return self._cmd_positions()
        elif command == "/bugun":
            return self._cmd_today()
        elif command == "/hafta":
            return self._cmd_week()
        elif command == "/duraklat":
            return self._cmd_pause()
        elif command == "/devam":
            return self._cmd_resume()
        elif command == "/risk" and args:
            return self._cmd_set_risk(args[0])
        elif command == "/export":
            return self._cmd_export()
        else:
            return self._cmd_help()
    
    def _cmd_status(self) -> str:
        """Mevcut durum komutu"""
        # Bu method'u gerçek bot state'i ile implement et
        return """
🤖 BOT STATUS

State: RUNNING ✅
Safe Mode: INACTIVE
Paper Mode: DISABLED

Open Positions: 2
Daily Trades: 3/5
Risk Used: 4.5%/6%

Emotion:
- Confidence: 0.72
- Stress: 0.15
- Patience: 0.85

Last Trade: 15 minutes ago
        """.strip()
    
    def _cmd_positions(self) -> str:
        """Açık pozisyonlar komutu"""
        return """
💼 OPEN POSITIONS (2)

1. BTCUSDT LONG
   Entry: $50,150
   Size: 0.05 BTC
   P&L: +2.3% (+$58)
   
2. ETHUSDT LONG
   Entry: $3,200
   Size: 1.5 ETH
   P&L: +1.1% (+$53)
   
Total P&L: +$111 (+1.8%)
        """.strip()
    
    def _cmd_today(self) -> str:
        """Bugünkü performans komutu"""
        return """
📊 TODAY'S PERFORMANCE

Trades: 3 (2W-1L)
Win Rate: 66.7%
P&L: +3.2% (+$320)

Best: +2.5% (BTCUSDT)
Worst: -1.0% (SOLUSDT)

FOMO Blocks: 1
Revenge Blocks: 0
        """.strip()
    
    def _cmd_week(self) -> str:
        """Haftalık performans komutu"""
        return """
📊 WEEKLY PERFORMANCE

Trades: 18 (12W-6L)
Win Rate: 66.7%
P&L: +12.5% (+$1,250)
Sharpe: 1.85

Best Day: +5.2% (Mon)
Worst Day: -2.1% (Thu)
        """.strip()
    
    def _cmd_pause(self) -> str:
        """Trading'i duraklat"""
        # Implement: Bot trading'i durdur
        return "⏸️  Trading PAUSED - No new trades will be opened"
    
    def _cmd_resume(self) -> str:
        """Trading'e devam et"""
        # Implement: Bot trading'i devam ettir
        return "▶️  Trading RESUMED - Normal operation"
    
    def _cmd_set_risk(self, risk_pct: str) -> str:
        """Risk oranını ayarla"""
        try:
            risk = float(risk_pct)
            if 0.5 <= risk <= 5.0:
                # Implement: Risk oranını değiştir
                return f"✅ Risk set to {risk}%"
            else:
                return "❌ Risk must be between 0.5% and 5%"
        except ValueError:
            return "❌ Invalid risk value"
    
    def _cmd_export(self) -> str:
        """CSV export"""
        # Implement: CSV oluştur ve gönder
        return "📁 Exporting trades to CSV..."
    
    def _cmd_help(self) -> str:
        """Komut listesi"""
        return """
📱 AVAILABLE COMMANDS

/durum - Mevcut durum
/pozisyonlar - Açık pozisyonlar
/bugun - Bugünkü performans
/hafta - Haftalık performans
/duraklat - Trading'i durdur
/devam - Trading'e devam et
/risk <pct> - Risk ayarla
/export - CSV export
        """.strip()
    
    # ═══════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════
    
    def _format_message(self, level: str, message: str, metadata: Optional[Dict] = None) -> str:
        """Mesajı formatla"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        formatted = f"[{timestamp}] {level}\n\n{message}"
        
        # Metadata ekle (debug için)
        if metadata and len(str(metadata)) < 200:
            formatted += f"\n\n_Metadata: {metadata}_"
        
        return formatted
    
    def _send_now(self, message: str, force: bool = False):
        """Mesajı hemen gönder"""
        if not self.config.enabled:
            return
        
        # Quiet hours kontrolü (force değilse)
        if not force and self._is_quiet_hours():
            self._queue_message("QUEUED", message, None)
            return
        
        try:
            # Telegram API call (implement edilecek)
            # telegram_api.send_message(self.config.chat_id, message)
            print(f"📤 Telegram message sent:\n{message}\n")
        except Exception as e:
            print(f"❌ Telegram send error: {e}")
    
    def _queue_message(self, level: str, message: str, metadata: Optional[Dict]):
        """Mesajı queue'ya ekle (quiet hours için)"""
        self.message_queue.append({
            "level": level,
            "message": message,
            "metadata": metadata,
            "timestamp": datetime.now()
        })
        
        # Queue çok büyüdüyse en eskiyi sil
        if len(self.message_queue) > 50:
            self.message_queue.pop(0)
    
    def flush_queue(self):
        """Queue'daki tüm mesajları gönder"""
        if not self.message_queue:
            return
        
        print(f"📤 Flushing {len(self.message_queue)} queued messages...")
        
        for item in self.message_queue:
            formatted = self._format_message(item["level"], item["message"], item["metadata"])
            self._send_now(formatted, force=True)
        
        self.message_queue.clear()
    
    def _is_quiet_hours(self) -> bool:
        """Quiet hours içinde miyiz?"""
        now = datetime.now().time()
        start = self.config.quiet_hours_start
        end = self.config.quiet_hours_end
        
        if start < end:
            return start <= now <= end
        else:  # Gece yarısını geçen durumlar
            return now >= start or now <= end
    
    def _format_top_coins(self, coins: List[Dict]) -> str:
        """Top coin listesini formatla"""
        if not coins:
            return "No data"
        
        lines = []
        for i, coin in enumerate(coins[:5], 1):
            name = coin.get('coin', 'N/A')
            wr = coin.get('win_rate', 0)
            pnl = coin.get('pnl_pct', 0)
            lines.append(f"{i}. {name}: {wr:.0f}% WR, {pnl:+.1f}%")
        
        return "\n".join(lines)
    
    def _format_progress_bar(self, label: str, value: float, target: float, length: int = 10) -> str:
        """Progress bar formatla"""
        progress = min(value / target, 1.0)
        filled = int(progress * length)
        bar = "█" * filled + "░" * (length - filled)
        
        return f"{label}: {bar} {value:.1f}/{target:.1f}"


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Telegram bot oluştur
    config = TelegramConfig(
        bot_token="YOUR_BOT_TOKEN",
        chat_id="YOUR_CHAT_ID",
        enabled=True
    )
    
    bot = TelegramBot(config)
    
    # Trade opened notification
    bot.send_trade_opened({
        "coin": "BTCUSDT",
        "direction": "LONG",
        "entry": 50150,
        "qty": 0.05,
        "risk_pct": 2.0,
        "stop": 49750,
        "setup_score": 85,
        "zone_quality": 8.5
    })
    
    # FOMO blocked
    bot.send_fomo_blocked(
        reason="Price chasing detected",
        details={
            "setup_complete": False,
            "price_distance_pct": 3.5,
            "time_since_last": 8,
            "fomo_score": 65
        }
    )
    
    # Daily summary
    bot.send_daily_summary({
        "date": "2025-01-15",
        "total_trades": 5,
        "wins": 3,
        "losses": 2,
        "win_rate": 60.0,
        "total_pnl_pct": 4.2,
        "total_pnl_usd": 420,
        "fomo_blocks": 2,
        "revenge_blocks": 1
    })