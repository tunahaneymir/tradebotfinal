"""
PPO Training Script
Train RL agent on trading environment
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from typing import Optional
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from rl.trading_env import TradingEnvironment
from rl.rl_agent_ppo import PPOAgent, PPOConfig, create_trading_ppo_config
from data.data_fetcher import BinanceDataFetcher
from strategies.pa_strategy import PAStrategy


class TrainingConfig:
    """Training configuration"""
    def __init__(self):
        # Data settings
        self.symbol = 'BTCUSDT'
        self.timeframe = '15m'
        self.train_days = 365  # 1 year of data
        self.test_days = 90    # 3 months for testing
        
        # Training settings
        self.total_episodes = 1000
        self.eval_interval = 50
        self.eval_episodes = 10
        self.save_interval = 100
        
        # Environment settings
        self.initial_balance = 10000
        self.max_steps_per_episode = 1000
        
        # Model settings
        self.model_dir = 'models/ppo'
        self.log_dir = 'logs/training'
        
        # Resume training
        self.resume_from = None  # Path to checkpoint
        
        # Tensorboard
        self.use_tensorboard = True


class Trainer:
    """
    PPO Training Manager
    """
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Setup directories
        self.model_dir = Path(config.model_dir)
        self.log_dir = Path(config.log_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load data
        self.logger.info("Loading data...")
        self.train_data, self.test_data = self._load_data()
        
        # Initialize PA strategy
        self.logger.info("Initializing PA strategy...")
        self.pa_strategy = PAStrategy()
        
        # Create environments
        self.logger.info("Creating training environment...")
        self.train_env = self._create_environment(self.train_data)
        
        self.logger.info("Creating evaluation environment...")
        self.eval_env = self._create_environment(self.test_data)
        
        # Create PPO agent
        self.logger.info("Creating PPO agent...")
        ppo_config = create_trading_ppo_config()
        self.agent = PPOAgent(ppo_config, model_dir=str(self.model_dir))
        
        # Resume from checkpoint if specified
        if config.resume_from:
            self.logger.info(f"Resuming from checkpoint: {config.resume_from}")
            self.agent.load(config.resume_from)
        
        # Tensorboard
        if config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
                self.logger.info("Tensorboard logging enabled")
            except ImportError:
                self.logger.warning("Tensorboard not available")
                self.writer = None
        else:
            self.writer = None
        
        # Training metrics
        self.training_metrics = []
        self.eval_metrics = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('Trainer')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        log_file = self.log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_data(self) -> tuple:
        """Load and prepare training/test data"""
        # Fetch data
        fetcher = BinanceDataFetcher()
        
        # Calculate dates
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.train_days + self.config.test_days)
        
        # Fetch
        df = fetcher.fetch_ohlcv(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            start_time=start_date,
            end_time=end_date
        )
        
        # Calculate indicators (EMA, RSI, ATR, etc.)
        df = self._add_indicators(df)
        
        # Split train/test
        split_idx = len(df) - int(self.config.test_days * 24 * 4)  # Assuming 15m data
        
        train_data = df.iloc[:split_idx].copy()
        test_data = df.iloc[split_idx:].copy()
        
        self.logger.info(f"Train data: {len(train_data)} rows")
        self.logger.info(f"Test data: {len(test_data)} rows")
        
        return train_data, test_data
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        # EMA
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Volume MA
        df['volume_ma'] = df['volume'].rolling(20).mean()
        
        # Drop NaN rows
        df = df.dropna()
        
        return df
    
    def _create_environment(self, data: pd.DataFrame) -> TradingEnvironment:
        """Create trading environment"""
        return TradingEnvironment(
            data=data,
            pa_detector=self.pa_strategy,
            initial_balance=self.config.initial_balance,
            max_steps=self.config.max_steps_per_episode
        )
    
    def train(self):
        """Main training loop"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING PPO TRAINING")
        self.logger.info("=" * 80)
        self.logger.info(f"Total episodes: {self.config.total_episodes}")
        self.logger.info(f"Evaluation interval: {self.config.eval_interval}")
        self.logger.info(f"Model directory: {self.model_dir}")
        self.logger.info("=" * 80)
        
        best_eval_reward = -float('inf')
        
        for episode in range(self.config.total_episodes):
            # Train one episode
            episode_metrics = self.agent.train_episode(
                self.train_env,
                max_steps=self.config.max_steps_per_episode
            )
            
            # Log episode metrics
            self._log_episode(episode, episode_metrics)
            
            # Tensorboard logging
            if self.writer:
                self.writer.add_scalar('Train/EpisodeReward', 
                                      episode_metrics['episode_reward'], episode)
                self.writer.add_scalar('Train/EpisodeLength', 
                                      episode_metrics['episode_length'], episode)
                
                if episode_metrics['update_metrics']:
                    metrics = episode_metrics['update_metrics']
                    self.writer.add_scalar('Train/PolicyLoss', 
                                          metrics['policy_loss'], episode)
                    self.writer.add_scalar('Train/ValueLoss', 
                                          metrics['value_loss'], episode)
                    self.writer.add_scalar('Train/Entropy', 
                                          metrics['entropy'], episode)
                    self.writer.add_scalar('Train/LearningRate', 
                                          metrics['learning_rate'], episode)
            
            # Evaluation
            if (episode + 1) % self.config.eval_interval == 0:
                eval_results = self._evaluate(episode)
                
                # Save best model
                if eval_results['mean_reward'] > best_eval_reward:
                    best_eval_reward = eval_results['mean_reward']
                    self.agent.save('ppo_agent_best.pt')
                    self.logger.info(f"✓ New best model saved! Reward: {best_eval_reward:.2f}")
            
            # Save checkpoint
            if (episode + 1) % self.config.save_interval == 0:
                checkpoint_name = f'ppo_agent_ep{episode+1}.pt'
                self.agent.save(checkpoint_name)
                self.logger.info(f"✓ Checkpoint saved: {checkpoint_name}")
        
        # Final save
        self.agent.save('ppo_agent_final.pt')
        self.logger.info("=" * 80)
        self.logger.info("TRAINING COMPLETED!")
        self.logger.info("=" * 80)
        
        # Save training metrics
        self._save_metrics()
        
        # Close tensorboard
        if self.writer:
            self.writer.close()
    
    def _log_episode(self, episode: int, metrics: dict):
        """Log episode metrics"""
        self.training_metrics.append({
            'episode': episode,
            'reward': metrics['episode_reward'],
            'length': metrics['episode_length'],
            'timestamp': datetime.now().isoformat()
        })
        
        # Console logging
        if episode % 10 == 0:
            recent_rewards = [m['reward'] for m in self.training_metrics[-10:]]
            mean_reward = np.mean(recent_rewards)
            
            self.logger.info(
                f"Episode {episode}/{self.config.total_episodes} | "
                f"Reward: {metrics['episode_reward']:.2f} | "
                f"Mean (last 10): {mean_reward:.2f} | "
                f"Length: {metrics['episode_length']} | "
                f"Total Steps: {self.agent.total_steps}"
            )
            
            if metrics['update_metrics']:
                um = metrics['update_metrics']
                self.logger.info(
                    f"  Policy Loss: {um['policy_loss']:.4f} | "
                    f"Value Loss: {um['value_loss']:.4f} | "
                    f"Entropy: {um['entropy']:.4f} | "
                    f"LR: {um['learning_rate']:.6f}"
                )
    
    def _evaluate(self, episode: int) -> dict:
        """Evaluate agent"""
        self.logger.info("-" * 80)
        self.logger.info(f"EVALUATION at episode {episode}")
        self.logger.info("-" * 80)
        
        eval_rewards = []
        eval_lengths = []
        eval_win_rates = []
        
        for eval_ep in range(self.config.eval_episodes):
            state = self.eval_env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Select action (deterministic for evaluation)
                action, _, _ = self.agent.select_action(
                    state,
                    deterministic=True,
                    explore=False
                )
                
                # Step
                next_state, reward, done, info = self.eval_env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
            
            # Get environment metrics
            env_metrics = self.eval_env.get_metrics()
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_win_rates.append(env_metrics['win_rate'])
            
            self.logger.info(
                f"  Eval {eval_ep+1}/{self.config.eval_episodes}: "
                f"Reward={episode_reward:.2f}, "
                f"Trades={env_metrics['total_trades']}, "
                f"WR={env_metrics['win_rate']:.2%}, "
                f"Return={env_metrics['total_return']:.2%}"
            )
        
        # Calculate statistics
        results = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'mean_win_rate': np.mean(eval_win_rates)
        }
        
        # Log summary
        self.logger.info("-" * 80)
        self.logger.info(
            f"Evaluation Summary: "
            f"Mean Reward={results['mean_reward']:.2f} ± {results['std_reward']:.2f}, "
            f"Mean WR={results['mean_win_rate']:.2%}"
        )
        self.logger.info("-" * 80)
        
        # Store results
        self.eval_metrics.append({
            'episode': episode,
            **results,
            'timestamp': datetime.now().isoformat()
        })
        
        # Tensorboard
        if self.writer:
            self.writer.add_scalar('Eval/MeanReward', results['mean_reward'], episode)
            self.writer.add_scalar('Eval/MeanWinRate', results['mean_win_rate'], episode)
        
        return results
    
    def _save_metrics(self):
        """Save training metrics to file"""
        metrics_file = self.log_dir / f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(metrics_file, 'w') as f:
            json.dump({
                'training_metrics': self.training_metrics,
                'eval_metrics': self.eval_metrics,
                'config': {
                    'symbol': self.config.symbol,
                    'timeframe': self.config.timeframe,
                    'total_episodes': self.config.total_episodes,
                    'initial_balance': self.config.initial_balance
                }
            }, f, indent=2)
        
        self.logger.info(f"Metrics saved to: {metrics_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train PPO agent for trading')
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='15m',
                       help='Timeframe (15m, 1h, 4h)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Total training episodes')
    parser.add_argument('--eval-interval', type=int, default=50,
                       help='Evaluation interval')
    parser.add_argument('--model-dir', type=str, default='models/ppo',
                       help='Model directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--no-tensorboard', action='store_true',
                       help='Disable tensorboard logging')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    config.symbol = args.symbol
    config.timeframe = args.timeframe
    config.total_episodes = args.episodes
    config.eval_interval = args.eval_interval
    config.model_dir = args.model_dir
    config.resume_from = args.resume
    config.use_tensorboard = not args.no_tensorboard
    
    # Create trainer
    trainer = Trainer(config)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        trainer.agent.save('ppo_agent_interrupted.pt')
        print("Model saved as: ppo_agent_interrupted.pt")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        trainer.agent.save('ppo_agent_error.pt')
        print("Model saved as: ppo_agent_error.pt")


if __name__ == '__main__':
    main()