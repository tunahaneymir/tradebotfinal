"""
PA + RL Trading Bot - Main Entry Point

Usage:
    python main.py
    python main.py --config production.yaml
    python main.py --mode test
"""

import sys
import time
import signal
import argparse
from pathlib import Path
from typing import Optional

# Infrastructure
from infrastructure import InfrastructureManager

# Agents (will be implemented)
# from agents import CoinSelectionAgent, TradingAgent, RiskAgent

# Learning (will be implemented)
# from learning import ContinuousLearning


class TradingBot:
    """
    PA + RL Trading Bot
    
    Architecture:
    - Infrastructure (Parça 9): State, backups, monitoring
    - Core PA (Parça 1-2): Trend, zones, entry/exit
    - Adaptive (Parça 3): Re-entry, liquidity, zone memory
    - RL (Parça 4-7): Agent, rewards, behavioral guards
    - Learning (Parça 8): Continuous learning, optimization
    """
    
    def __init__(self, config_path: str = "config.yaml", mode: str = "live"):
        """
        Initialize trading bot
        
        Args:
            config_path: Path to configuration file
            mode: 'live', 'paper', or 'test'
        """
        self.mode = mode
        self.running = False
        
        print("\n" + "="*60)
        print(f"🤖 PA + RL TRADING BOT - {mode.upper()} MODE")
        print("="*60)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize infrastructure (Parça 9)
        print("\n📦 Initializing infrastructure...")
        self.infra = InfrastructureManager(self.config)
        
        # Initialize agents (TODO: implement)
        print("📦 Initializing agents...")
        self.coin_selector = None  # CoinSelectionAgent(self.config)
        self.trader = None         # TradingAgent(self.config, self.infra)
        self.risk_manager = None   # RiskAgent(self.config, self.infra)
        
        # Initialize learning system (TODO: implement)
        print("📦 Initializing learning system...")
        self.learning = None       # ContinuousLearning(self.infra)
        
        print("\n✅ Bot initialized successfully!")
        print("="*60 + "\n")
    
    def start(self):
        """Start the trading bot"""
        if self.running:
            print("⚠️  Bot already running")
            return
        
        try:
            # Start infrastructure
            self.infra.start_all()
            
            # Start learning (TODO)
            # if self.learning:
            #     self.learning.start_training_loop()
            
            self.running = True
            
            # Run main loop
            self._main_loop()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Keyboard interrupt received")
            self.stop()
        except Exception as e:
            print(f"\n\n❌ Fatal error: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the trading bot"""
        if not self.running:
            return
        
        print("\n" + "="*60)
        print("🛑 STOPPING BOT")
        print("="*60)
        
        try:
            # Stop learning (TODO)
            # if self.learning:
            #     self.learning.stop()
            
            # Stop infrastructure
            self.infra.stop_all()
            
            self.running = False
            
            print("\n✅ Bot stopped successfully!")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error during shutdown: {e}")
    
    def _main_loop(self):
        """Main trading loop"""
        print("\n" + "="*60)
        print("🔄 STARTING MAIN LOOP")
        print("="*60 + "\n")
        
        iteration = 0
        
        while self.running:
            try:
                iteration += 1
                
                # TODO: Implement actual trading logic
                # For now, just a placeholder
                
                print(f"[Iteration {iteration}] Waiting for trading logic implementation...")
                
                # Update dashboard
                self.infra.update_dashboard()
                
                # Sleep
                time.sleep(60)  # 1 minute
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                
                # Send error notification
                self.infra.send_notification("critical", f"Main loop error: {e}")
                
                # Continue after error
                time.sleep(5)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            print(f"⚠️  Config file not found: {config_path}")
            print("📝 Using default configuration")
            return self._default_config()
        
        try:
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuration loaded: {config_path}")
            return config
        except ImportError:
            print("⚠️  PyYAML not installed, using default config")
            return self._default_config()
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            print("📝 Using default configuration")
            return self._default_config()
    
    def _default_config(self) -> dict:
        """Default configuration"""
        return {
            "bot": {
                "name": "PA+RL Trading Bot",
                "version": "1.0.0"
            },
            "data_dir": "state",
            "database_path": "data/trades.db",
            "telegram": {
                "enabled": False,
                "bot_token": "",
                "chat_id": ""
            },
            "risk": {
                "per_trade": 0.02,
                "daily_limit": 0.06,
                "portfolio_limit": 0.08
            },
            "trading": {
                "timeframes": ["4H", "1H", "15M"],
                "max_positions": 3
            }
        }


def setup_signal_handlers(bot: TradingBot):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        print(f"\n\n⚠️  Signal {signum} received")
        bot.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="PA + RL Trading Bot")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--mode", default="live", choices=["live", "paper", "test"], help="Trading mode")
    args = parser.parse_args()
    
    # Create bot
    bot = TradingBot(config_path=args.config, mode=args.mode)
    
    # Setup signal handlers
    setup_signal_handlers(bot)
    
    # Start bot
    bot.start()


if __name__ == "__main__":
    main()