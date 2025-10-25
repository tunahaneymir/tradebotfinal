"""
CLI Dashboard with Rich
Based on: pa-strateji3 Parça 9

Features:
- Real-time status display
- Open positions table
- Daily P&L tracking
- Emotional state indicators
- Cooldown status
"""

from __future__ import annotations
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass
import time


# Rich import (pip install rich)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Rich not installed. Install with: pip install rich")


@dataclass
class DashboardData:
    """Dashboard için gerekli data"""
    # Bot status
    bot_running: bool = True
    safe_mode_active: bool = False
    paper_mode: bool = False
    
    # Positions
    open_positions: List[Dict] = None
    
    # Performance
    daily_trades: int = 0
    daily_wins: int = 0
    daily_losses: int = 0
    daily_pnl_pct: float = 0.0
    daily_pnl_usd: float = 0.0
    
    # Risk
    daily_risk_used: float = 0.0
    daily_risk_limit: float = 6.0
    portfolio_risk_pct: float = 0.0
    
    # Emotional state
    confidence: float = 0.5
    stress: float = 0.0
    patience: float = 0.5
    
    # Cooldowns
    cooldowns: Dict[str, Dict] = None
    
    # Behavioral
    fomo_blocks_today: int = 0
    revenge_blocks_today: int = 0
    
    def __post_init__(self):
        if self.open_positions is None:
            self.open_positions = []
        if self.cooldowns is None:
            self.cooldowns = {}


class CLIDashboard:
    """
    Terminal-based dashboard using Rich library
    
    Features:
    - Real-time updates
    - Color-coded status
    - Progress bars
    - Tables for positions
    - Emoji indicators
    """
    
    def __init__(self):
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required. Install with: pip install rich")
        
        self.console = Console()
        self.data = DashboardData()
        self.last_update = datetime.now()
    
    def update_data(self, data: DashboardData):
        """Dashboard data'sını güncelle"""
        self.data = data
        self.last_update = datetime.now()
    
    def render_static(self):
        """Statik dashboard göster (single render)"""
        self.console.clear()
        
        # Header
        self.console.print(self._create_header())
        self.console.print()
        
        # Main content
        layout = Layout()
        layout.split_column(
            Layout(self._create_status_panel(), name="status", size=8),
            Layout(self._create_positions_table(), name="positions", size=12),
            Layout(self._create_performance_panel(), name="performance", size=8),
        )
        
        self.console.print(layout)
        
        # Footer
        self.console.print(self._create_footer())
    
    def render_live(self, refresh_rate: float = 1.0):
        """Live dashboard (auto-refresh)"""
        with Live(self._generate_layout(), refresh_per_second=refresh_rate, console=self.console) as live:
            while True:
                live.update(self._generate_layout())
                time.sleep(1.0 / refresh_rate)
    
    def _generate_layout(self) -> Layout:
        """Layout oluştur"""
        layout = Layout()
        
        layout.split_column(
            Layout(self._create_header(), name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(self._create_footer(), name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["main"]["left"].split_column(
            Layout(self._create_status_panel(), name="status"),
            Layout(self._create_positions_table(), name="positions")
        )
        
        layout["main"]["right"].split_column(
            Layout(self._create_performance_panel(), name="performance"),
            Layout(self._create_emotional_panel(), name="emotional"),
            Layout(self._create_cooldowns_panel(), name="cooldowns")
        )
        
        return layout
    
    # ═══════════════════════════════════════════════════════════
    # PANEL CREATORS
    # ═══════════════════════════════════════════════════════════
    
    def _create_header(self) -> Panel:
        """Header panel"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        status_emoji = "🟢" if self.data.bot_running else "🔴"
        safe_mode_emoji = "⚠️ " if self.data.safe_mode_active else ""
        paper_emoji = "📄 PAPER MODE" if self.data.paper_mode else ""
        
        header_text = f"{status_emoji} PA + RL TRADING BOT  {safe_mode_emoji}{paper_emoji}\n{now}"
        
        return Panel(
            Text(header_text, justify="center", style="bold"),
            style="blue"
        )
    
    def _create_status_panel(self) -> Panel:
        """Bot status panel"""
        status = "🟢 RUNNING" if self.data.bot_running else "🔴 STOPPED"
        safe_mode = "⚠️  ACTIVE" if self.data.safe_mode_active else "✅ INACTIVE"
        
        content = f"""
[bold]Bot Status:[/bold] {status}
[bold]Safe Mode:[/bold] {safe_mode}
[bold]Paper Mode:[/bold] {'📄 YES' if self.data.paper_mode else '❌ NO'}

[bold]Open Positions:[/bold] {len(self.data.open_positions)}
[bold]Daily Trades:[/bold] {self.data.daily_trades}
        """.strip()
        
        return Panel(content, title="🤖 Status", border_style="green")
    
    def _create_positions_table(self) -> Table:
        """Open positions tablosu"""
        table = Table(title="💼 Open Positions", show_header=True, header_style="bold magenta")
        
        table.add_column("Coin", style="cyan", width=10)
        table.add_column("Dir", width=5)
        table.add_column("Entry", justify="right", width=10)
        table.add_column("Current", justify="right", width=10)
        table.add_column("P&L %", justify="right", width=10)
        table.add_column("P&L $", justify="right", width=10)
        
        if not self.data.open_positions:
            table.add_row("—", "—", "—", "—", "—", "—")
        else:
            for pos in self.data.open_positions:
                coin = pos.get("coin", "N/A")
                direction = pos.get("direction", "N/A")
                entry = pos.get("entry", 0)
                current = pos.get("current_price", entry)
                pnl_pct = pos.get("pnl_pct", 0)
                pnl_usd = pos.get("pnl_usd", 0)
                
                # Color coding
                pnl_color = "green" if pnl_pct > 0 else "red" if pnl_pct < 0 else "white"
                dir_color = "green" if direction == "LONG" else "red"
                
                table.add_row(
                    coin,
                    f"[{dir_color}]{direction}[/{dir_color}]",
                    f"${entry:,.2f}",
                    f"${current:,.2f}",
                    f"[{pnl_color}]{pnl_pct:+.2f}%[/{pnl_color}]",
                    f"[{pnl_color}]${pnl_usd:+,.2f}[/{pnl_color}]"
                )
        
        return table
    
    def _create_performance_panel(self) -> Panel:
        """Performance panel"""
        win_rate = (self.data.daily_wins / self.data.daily_trades * 100) if self.data.daily_trades > 0 else 0
        
        # Win rate color
        if win_rate >= 65:
            wr_color = "green"
        elif win_rate >= 55:
            wr_color = "yellow"
        else:
            wr_color = "red"
        
        # P&L color
        pnl_color = "green" if self.data.daily_pnl_pct > 0 else "red" if self.data.daily_pnl_pct < 0 else "white"
        
        content = f"""
[bold]Daily Performance:[/bold]
  Trades: {self.data.daily_trades} ({self.data.daily_wins}W-{self.data.daily_losses}L)
  Win Rate: [{wr_color}]{win_rate:.1f}%[/{wr_color}]
  P&L: [{pnl_color}]{self.data.daily_pnl_pct:+.2f}% (${self.data.daily_pnl_usd:+,.2f})[/{pnl_color}]

[bold]Risk Management:[/bold]
  Daily Risk: {self.data.daily_risk_used:.1f}% / {self.data.daily_risk_limit:.1f}%
  Portfolio Risk: {self.data.portfolio_risk_pct:.1f}%
  
[bold]Behavioral:[/bold]
  🚫 FOMO Blocks: {self.data.fomo_blocks_today}
  🚫 Revenge Blocks: {self.data.revenge_blocks_today}
        """.strip()
        
        return Panel(content, title="📊 Performance", border_style="yellow")
    
    def _create_emotional_panel(self) -> Panel:
        """Emotional state panel"""
        # Progress bars
        confidence_bar = self._create_progress_bar("Confidence", self.data.confidence, "green")
        stress_bar = self._create_progress_bar("Stress", self.data.stress, "red")
        patience_bar = self._create_progress_bar("Patience", self.data.patience, "blue")
        
        content = f"""
[bold]Emotional State:[/bold]

{confidence_bar}
{stress_bar}
{patience_bar}

[bold]Interpretation:[/bold]
{self._get_emotional_status()}
        """.strip()
        
        return Panel(content, title="🧠 Emotion", border_style="cyan")
    
    def _create_cooldowns_panel(self) -> Panel:
        """Cooldowns panel"""
        if not self.data.cooldowns:
            content = "[dim]No active cooldowns[/dim]"
        else:
            lines = ["[bold]Active Cooldowns:[/bold]\n"]
            for coin, info in self.data.cooldowns.items():
                if info.get("active"):
                    until = info.get("until", "N/A")
                    level = info.get("level", "N/A")
                    emoji = "⏸️" if level == "hard" else "⚠️"
                    lines.append(f"{emoji} {coin}: Until {until}")
            
            content = "\n".join(lines) if len(lines) > 1 else "[dim]No active cooldowns[/dim]"
        
        return Panel(content, title="⏰ Cooldowns", border_style="red")
    
    def _create_footer(self) -> Panel:
        """Footer panel"""
        last_update = self.last_update.strftime("%H:%M:%S")
        footer_text = f"Last Update: {last_update} | Press Ctrl+C to exit"
        
        return Panel(
            Text(footer_text, justify="center", style="dim"),
            style="blue"
        )
    
    # ═══════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════
    
    def _create_progress_bar(self, label: str, value: float, color: str) -> str:
        """Progress bar string oluştur"""
        value = max(0.0, min(1.0, value))  # Clamp 0-1
        filled = int(value * 20)
        bar = "█" * filled + "░" * (20 - filled)
        
        return f"{label}: [{color}]{bar}[/{color}] {value*100:.0f}%"
    
    def _get_emotional_status(self) -> str:
        """Emotional state yorumu"""
        if self.data.stress > 0.7:
            return "⚠️  [red]High stress - Risk reduced[/red]"
        elif self.data.confidence > 0.7 and self.data.stress < 0.3:
            return "✅ [green]Optimal state - Normal operation[/green]"
        elif self.data.patience < 0.3:
            return "⚠️  [yellow]Low patience - FOMO risk[/yellow]"
        else:
            return "ℹ️  [white]Normal state[/white]"
    
    # ═══════════════════════════════════════════════════════════
    # SIMPLE TEXT DASHBOARD (No Rich)
    # ═══════════════════════════════════════════════════════════
    
    def render_simple(self):
        """Simple text dashboard (Rich olmadan)"""
        print("\n" + "="*60)
        print("           PA + RL TRADING BOT DASHBOARD")
        print("="*60)
        
        # Status
        print("\n🤖 BOT STATUS:")
        print(f"  Running: {'YES' if self.data.bot_running else 'NO'}")
        print(f"  Safe Mode: {'ACTIVE' if self.data.safe_mode_active else 'INACTIVE'}")
        print(f"  Paper Mode: {'YES' if self.data.paper_mode else 'NO'}")
        
        # Positions
        print(f"\n💼 OPEN POSITIONS: {len(self.data.open_positions)}")
        if self.data.open_positions:
            for pos in self.data.open_positions:
                print(f"  {pos['coin']} {pos['direction']}: {pos.get('pnl_pct', 0):+.2f}%")
        
        # Performance
        win_rate = (self.data.daily_wins / self.data.daily_trades * 100) if self.data.daily_trades > 0 else 0
        print(f"\n📊 DAILY PERFORMANCE:")
        print(f"  Trades: {self.data.daily_trades} ({self.data.daily_wins}W-{self.data.daily_losses}L)")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  P&L: {self.data.daily_pnl_pct:+.2f}% (${self.data.daily_pnl_usd:+,.2f})")
        
        # Emotional
        print(f"\n🧠 EMOTIONAL STATE:")
        print(f"  Confidence: {self.data.confidence*100:.0f}%")
        print(f"  Stress: {self.data.stress*100:.0f}%")
        print(f"  Patience: {self.data.patience*100:.0f}%")
        
        print("\n" + "="*60)
        print(f"Last Update: {self.last_update.strftime('%H:%M:%S')}")
        print("="*60 + "\n")


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Dashboard oluştur
    dashboard = CLIDashboard()
    
    # Örnek data
    data = DashboardData(
        bot_running=True,
        safe_mode_active=False,
        paper_mode=False,
        open_positions=[
            {
                "coin": "BTCUSDT",
                "direction": "LONG",
                "entry": 50150,
                "current_price": 51200,
                "pnl_pct": 2.1,
                "pnl_usd": 105
            },
            {
                "coin": "ETHUSDT",
                "direction": "LONG",
                "entry": 3200,
                "current_price": 3250,
                "pnl_pct": 1.6,
                "pnl_usd": 80
            }
        ],
        daily_trades=5,
        daily_wins=3,
        daily_losses=2,
        daily_pnl_pct=4.2,
        daily_pnl_usd=420,
        daily_risk_used=4.5,
        daily_risk_limit=6.0,
        portfolio_risk_pct=3.8,
        confidence=0.72,
        stress=0.18,
        patience=0.85,
        cooldowns={
            "SOLUSDT": {
                "active": True,
                "until": "15:30",
                "level": "soft"
            }
        },
        fomo_blocks_today=2,
        revenge_blocks_today=1
    )
    
    dashboard.update_data(data)
    
    # Rich varsa live dashboard, yoksa simple
    if RICH_AVAILABLE:
        try:
            dashboard.render_static()
            # dashboard.render_live()  # Live mode için uncomment et
        except KeyboardInterrupt:
            print("\n👋 Dashboard closed")
    else:
        dashboard.render_simple()