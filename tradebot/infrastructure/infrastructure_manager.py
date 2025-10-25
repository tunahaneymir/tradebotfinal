"""
Infrastructure Manager
Parça 9'un tüm modüllerini tek yerden yönetir

Usage:
    from infrastructure import InfrastructureManager
    
    infra = InfrastructureManager(config)
    infra.start_all()
"""

from __future__ import annotations
from typing import Dict, Optional, Any
from pathlib import Path

# Import all infrastructure modules
from .state_manager import StateManager
from .safe_mode_manager import SafeModeManager
from .circuit_breaker import CircuitBreaker
from .telegram_bot import TelegramBot, TelegramConfig
from .database_layer import DatabaseLayer
from .cli_dashboard import CLIDashboard, DashboardData
from .indicator_learning_manager import IndicatorLearningManager
from .backup_manager import BackupManager
from .volatility_guard import VolatilityGuard


class InfrastructureManager:
    """
    Tüm infrastructure modüllerini yönetir (Parça 9)
    
    Features:
    - Tek başlatma/durdurma noktası
    - Merkezi konfigürasyon
    - State synchronization
    - Error handling
    
    Modules:
    1. StateManager - Autosave & recovery
    2. SafeModeManager - Drawdown protection
    3. CircuitBreaker - API error protection
    4. TelegramBot - Notifications & commands
    5. DatabaseLayer - Trade tracking
    6. CLIDashboard - Terminal UI
    7. IndicatorLearning - ZigZag++/Swing optimization
    8. BackupManager - Automatic backups
    9. VolatilityGuard - Extreme move protection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize infrastructure
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        
        # Modules (lazy initialization)
        self._state: Optional[StateManager] = None
        self._safe_mode: Optional[SafeModeManager] = None
        self._circuit: Optional[CircuitBreaker] = None
        self._telegram: Optional[TelegramBot] = None
        self._database: Optional[DatabaseLayer] = None
        self._dashboard: Optional[CLIDashboard] = None
        self._indicator_learning: Optional[IndicatorLearningManager] = None
        self._backup: Optional[BackupManager] = None
        self._volatility: Optional[VolatilityGuard] = None
        
        # State
        self.running = False
        
        print("🏗️  InfrastructureManager initialized")
    
    # ═══════════════════════════════════════════════════════════
    # LIFECYCLE MANAGEMENT
    # ═══════════════════════════════════════════════════════════
    
    def start_all(self):
        """Tüm infrastructure servislerini başlat"""
        if self.running:
            print("⚠️  Infrastructure already running")
            return
        
        print("\n" + "="*60)
        print("🚀 STARTING INFRASTRUCTURE")
        print("="*60)
        
        try:
            # Initialize modules
            self._initialize_modules()
            
            # Start services
            self.state.start_autosave()
            self.backup.start_auto_backup()
            
            self.running = True
            
            print("\n✅ Infrastructure started successfully!")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\n❌ Infrastructure start failed: {e}")
            raise
    
    def stop_all(self):
        """Tüm infrastructure servislerini durdur"""
        if not self.running:
            return
        
        print("\n" + "="*60)
        print("⏸️  STOPPING INFRASTRUCTURE")
        print("="*60)
        
        try:
            # Stop services
            if self._state:
                self._state.stop_autosave()
                self._state.save_state(reason="shutdown")
            
            if self._backup:
                self._backup.stop_auto_backup()
            
            # Close connections
            if self._database:
                self._database.close()
            
            self.running = False
            
            print("\n✅ Infrastructure stopped successfully!")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\n❌ Infrastructure stop failed: {e}")
    
    def _initialize_modules(self):
        """Tüm modülleri initialize et"""
        # 1. State Manager
        if not self._state:
            self._state = StateManager(
                data_dir=self.config.get('data_dir', 'state')
            )
        
        # 2. Safe Mode Manager
        if not self._safe_mode:
            self._safe_mode = SafeModeManager()
        
        # 3. Circuit Breaker
        if not self._circuit:
            self._circuit = CircuitBreaker(name="exchange_api")
        
        # 4. Telegram Bot
        if not self._telegram and self.config.get('telegram', {}).get('enabled'):
            telegram_config = TelegramConfig(
                bot_token=self.config['telegram']['bot_token'],
                chat_id=self.config['telegram']['chat_id'],
                enabled=True
            )
            self._telegram = TelegramBot(telegram_config)
        
        # 5. Database Layer
        if not self._database:
            self._database = DatabaseLayer(
                db_path=self.config.get('database_path', 'data/trades.db')
            )
        
        # 6. CLI Dashboard
        if not self._dashboard:
            self._dashboard = CLIDashboard()
        
        # 7. Indicator Learning
        if not self._indicator_learning:
            self._indicator_learning = IndicatorLearningManager()
        
        # 8. Backup Manager
        if not self._backup:
            self._backup = BackupManager(
                source_dirs=['state', 'data', 'models'],
                backup_root='backups'
            )
        
        # 9. Volatility Guard
        if not self._volatility:
            self._volatility = VolatilityGuard()
    
    # ═══════════════════════════════════════════════════════════
    # PROPERTIES (Lazy initialization)
    # ═══════════════════════════════════════════════════════════
    
    @property
    def state(self) -> StateManager:
        """State manager instance"""
        if self._state is None:
            self._state = StateManager()
        return self._state
    
    @property
    def safe_mode(self) -> SafeModeManager:
        """Safe mode manager instance"""
        if self._safe_mode is None:
            self._safe_mode = SafeModeManager()
        return self._safe_mode
    
    @property
    def circuit(self) -> CircuitBreaker:
        """Circuit breaker instance"""
        if self._circuit is None:
            self._circuit = CircuitBreaker(name="exchange_api")
        return self._circuit
    
    @property
    def telegram(self) -> Optional[TelegramBot]:
        """Telegram bot instance"""
        return self._telegram
    
    @property
    def database(self) -> DatabaseLayer:
        """Database layer instance"""
        if self._database is None:
            self._database = DatabaseLayer()
        return self._database
    
    @property
    def dashboard(self) -> CLIDashboard:
        """CLI dashboard instance"""
        if self._dashboard is None:
            self._dashboard = CLIDashboard()
        return self._dashboard
    
    @property
    def indicator_learning(self) -> IndicatorLearningManager:
        """Indicator learning manager instance"""
        if self._indicator_learning is None:
            self._indicator_learning = IndicatorLearningManager()
        return self._indicator_learning
    
    @property
    def backup(self) -> BackupManager:
        """Backup manager instance"""
        if self._backup is None:
            self._backup = BackupManager()
        return self._backup
    
    @property
    def volatility(self) -> VolatilityGuard:
        """Volatility guard instance"""
        if self._volatility is None:
            self._volatility = VolatilityGuard()
        return self._volatility
    
    # ═══════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Tüm modüllerin istatistikleri"""
        return {
            "running": self.running,
            "state": self.state.get_stats() if self._state else {},
            "safe_mode": self.safe_mode.get_status() if self._safe_mode else {},
            "circuit": self.circuit.get_stats() if self._circuit else {},
            "volatility": self.volatility.get_statistics() if self._volatility else {},
            "indicator_learning": self.indicator_learning.get_performance_summary() if self._indicator_learning else {}
        }
    
    def update_dashboard(self):
        """Dashboard'u güncelle"""
        if not self._dashboard:
            return
        
        # Dashboard data oluştur
        data = DashboardData(
            bot_running=self.running,
            safe_mode_active=self.safe_mode.is_safe_mode_active(),
            paper_mode=self.safe_mode.is_paper_mode(),
            open_positions=self.state.get_open_positions(),
            daily_trades=self.state.daily_trades,
            daily_risk_used=self.state.daily_risk_used,
            confidence=self.state.bot_state.confidence,
            stress=self.state.bot_state.stress,
            patience=self.state.bot_state.patience,
            cooldowns={k: v.__dict__ for k, v in self.state.cooldowns.items()}
        )
        
        self.dashboard.update_data(data)
        self.dashboard.render_static()
    
    def send_notification(self, level: str, message: str, metadata: Optional[Dict] = None):
        """Telegram notification gönder"""
        if not self._telegram:
            return
        
        if level == "critical":
            self.telegram.send_critical(message, metadata)
        elif level == "important":
            self.telegram.send_important(message, metadata)
        elif level == "summary":
            self.telegram.send_summary(message, metadata)
        else:
            self.telegram.send_info(message, metadata)
    
    def check_safe_mode(self, equity_usdt: float, drawdown_usdt: float) -> bool:
        """Safe mode kontrolü"""
        return self.safe_mode.check_and_trigger(equity_usdt, drawdown_usdt)
    
    def check_volatility(self, coin: str, price: float) -> bool:
        """Volatility kontrolü"""
        return self.volatility.update_price(coin, price)
    
    def is_trading_allowed(self, coin: str) -> tuple[bool, Optional[str]]:
        """Trading allowed mi?"""
        # Safe mode check
        if self.safe_mode.is_safe_mode_active():
            return False, "Safe mode active"
        
        # Volatility check
        allowed, reason = self.volatility.check_new_trade(coin)
        if not allowed:
            return False, reason
        
        # Cooldown check
        if self.state.is_cooldown_active(coin):
            return False, f"Cooldown active for {coin}"
        
        return True, None
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "data_dir": "state",
            "database_path": "data/trades.db",
            "telegram": {
                "enabled": False,
                "bot_token": "",
                "chat_id": ""
            }
        }


# Export
__all__ = [
    "InfrastructureManager",
    "StateManager",
    "SafeModeManager",
    "CircuitBreaker",
    "TelegramBot",
    "DatabaseLayer",
    "CLIDashboard",
    "IndicatorLearningManager",
    "BackupManager",
    "VolatilityGuard"
]