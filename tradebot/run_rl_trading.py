"""
Example: Running RL Trading Bot
Complete example of training and using RL agent
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import logging
from datetime import datetime

# Import components
from data.data_fetcher import BinanceDataFetcher
from strategies.rl_strategy import RLStrategy, HybridRLPAStrategy
from strategies.pa_strategy import PAStrategy
from rl.train_ppo import Trainer, TrainingConfig


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RLTradingExample')


def example_1_train_from_scratch():
    """
    Example 1: Train RL agent from scratch
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Training RL Agent from Scratch")
    logger.info("=" * 80)
    
    # Create training config
    config = TrainingConfig()
    config.symbol = 'BTCUSDT'
    config.timeframe = '15m'
    config.total_episodes = 100  # Start with 100 episodes for testing
    config.eval_interval = 20
    config.save_interval = 50
    config.initial_balance = 10000
    
    # Create trainer
    trainer = Trainer(config)
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")
    logger.info(f"Best model saved at: {config.model_dir}/ppo_agent_best.pt")


def example_2_use_trained_agent():
    """
    Example 2: Use trained agent for live trading signals
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 2: Using Trained RL Agent")
    logger.info("=" * 80)
    
    # Load trained strategy
    strategy = RLStrategy(
        model_path='models/ppo/ppo_agent_best.pt',
        use_pa_validation=True,
        confidence_threshold=0.6
    )
    
    # Fetch recent data
    logger.info("Fetching market data...")
    fetcher = BinanceDataFetcher()
    data = fetcher.fetch_ohlcv(
        symbol='BTCUSDT',
        timeframe='15m',
        limit=500
    )
    
    # Add indicators
    data = add_indicators(data)
    
    # Generate signals
    logger.info("Generating signals...")
    for i in range(len(data) - 100, len(data)):
        window_data = data.iloc[:i+1]
        signal = strategy.generate_signals(window_data)
        
        if signal['action'] != 'HOLD':
            logger.info(
                f"Time: {data.index[i]} | "
                f"Action: {signal['action']} | "
                f"Confidence: {signal['confidence']:.2%} | "
                f"Setup Score: {signal['setup_score']:.0f}"
            )
    
    # Print statistics
    stats = strategy.get_statistics()
    logger.info("=" * 80)
    logger.info("Strategy Statistics:")
    for key, value in stats.items():
        if key != 'recent_actions':
            logger.info(f"  {key}: {value}")


def example_3_backtest_rl_strategy():
    """
    Example 3: Backtest RL strategy
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 3: Backtesting RL Strategy")
    logger.info("=" * 80)
    
    # Load strategy
    strategy = RLStrategy(
        model_path='models/ppo/ppo_agent_best.pt',
        use_pa_validation=True
    )
    
    # Fetch historical data
    logger.info("Fetching historical data...")
    fetcher = BinanceDataFetcher()
    data = fetcher.fetch_ohlcv(
        symbol='BTCUSDT',
        timeframe='15m',
        limit=5000  # ~50 days
    )
    
    # Add indicators
    data = add_indicators(data)
    
    # Backtest
    logger.info("Running backtest...")
    results = run_backtest(strategy, data)
    
    # Print results
    logger.info("=" * 80)
    logger.info("Backtest Results:")
    logger.info(f"  Total Trades: {results['total_trades']}")
    logger.info(f"  Wins: {results['wins']} | Losses: {results['losses']}")
    logger.info(f"  Win Rate: {results['win_rate']:.2%}")
    logger.info(f"  Total Return: {results['total_return']:.2%}")
    logger.info(f"  Profit Factor: {results['profit_factor']:.2f}")
    logger.info(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info("=" * 80)


def example_4_hybrid_strategy():
    """
    Example 4: Use Hybrid RL+PA Strategy
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 4: Hybrid RL+PA Strategy")
    logger.info("=" * 80)
    
    # Create hybrid strategy
    strategy = HybridRLPAStrategy(
        rl_model_path='models/ppo/ppo_agent_best.pt',
        rl_weight=0.6,  # 60% RL
        pa_weight=0.4   # 40% PA
    )
    
    # Fetch data
    fetcher = BinanceDataFetcher()
    data = fetcher.fetch_ohlcv('BTCUSDT', '15m', limit=500)
    data = add_indicators(data)
    
    # Generate signals
    logger.info("Generating hybrid signals...")
    signals = []
    
    for i in range(len(data) - 100, len(data)):
        window_data = data.iloc[:i+1]
        signal = strategy.generate_signals(window_data)
        
        if signal['action'] != 'HOLD':
            signals.append(signal)
            logger.info(
                f"Time: {data.index[i]} | "
                f"Action: {signal['action']} | "
                f"Combined Score: {signal['confidence']:.2%} | "
                f"RL Confidence: {signal['rl_signal']['confidence']:.2%} | "
                f"PA Score: {signal['pa_signal'].get('setup_score', 0):.0f}"
            )
    
    logger.info(f"Total signals generated: {len(signals)}")


def example_5_live_monitoring():
    """
    Example 5: Live monitoring and signal generation
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 5: Live Monitoring (Simulated)")
    logger.info("=" * 80)
    
    import time
    
    # Load strategy
    strategy = RLStrategy(
        model_path='models/ppo/ppo_agent_best.pt',
        use_pa_validation=True
    )
    
    # Fetch initial data
    fetcher = BinanceDataFetcher()
    
    logger.info("Starting live monitoring (Press Ctrl+C to stop)...")
    
    try:
        iteration = 0
        while iteration < 10:  # Run 10 iterations for demo
            iteration += 1
            
            # Fetch latest data
            data = fetcher.fetch_ohlcv('BTCUSDT', '15m', limit=500)
            data = add_indicators(data)
            
            # Generate signal
            signal = strategy.generate_signals(data)
            
            # Current price
            current_price = data['close'].iloc[-1]
            
            # Log
            logger.info(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"BTC: ${current_price:.2f} | "
                f"Action: {signal['action']} | "
                f"Confidence: {signal['confidence']:.2%}"
            )
            
            # If signal generated
            if signal['action'] == 'BUY':
                logger.info("🚀 BUY SIGNAL GENERATED!")
                logger.info(f"  Entry: ${signal['entry_price']:.2f}")
                logger.info(f"  Stop Loss: ${signal['stop_loss']:.2f}")
                logger.info(f"  Take Profit: ${signal['take_profit']:.2f}")
                logger.info(f"  Setup Score: {signal['setup_score']:.0f}")
            
            # Wait before next iteration (in real scenario, wait for new candle)
            time.sleep(5)
    
    except KeyboardInterrupt:
        logger.info("\nMonitoring stopped by user")


def example_6_compare_strategies():
    """
    Example 6: Compare RL vs PA vs Hybrid strategies
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 6: Strategy Comparison")
    logger.info("=" * 80)
    
    # Load all strategies
    rl_strategy = RLStrategy('models/ppo/ppo_agent_best.pt', use_pa_validation=False)
    rl_pa_strategy = RLStrategy('models/ppo/ppo_agent_best.pt', use_pa_validation=True)
    hybrid_strategy = HybridRLPAStrategy('models/ppo/ppo_agent_best.pt')
    pa_strategy = PAStrategy()
    
    # Fetch data
    fetcher = BinanceDataFetcher()
    data = fetcher.fetch_ohlcv('BTCUSDT', '15m', limit=3000)
    data = add_indicators(data)
    
    # Backtest all strategies
    logger.info("Backtesting RL Strategy (no PA validation)...")
    rl_results = run_backtest(rl_strategy, data)
    
    logger.info("Backtesting RL Strategy (with PA validation)...")
    rl_pa_results = run_backtest(rl_pa_strategy, data)
    
    logger.info("Backtesting Hybrid Strategy...")
    hybrid_results = run_backtest(hybrid_strategy, data)
    
    logger.info("Backtesting Pure PA Strategy...")
    pa_results = backtest_pa_strategy(pa_strategy, data)
    
    # Compare results
    logger.info("=" * 80)
    logger.info("STRATEGY COMPARISON RESULTS:")
    logger.info("=" * 80)
    
    strategies = {
        'RL Only': rl_results,
        'RL + PA Validation': rl_pa_results,
        'Hybrid (RL+PA)': hybrid_results,
        'Pure PA': pa_results
    }
    
    for name, results in strategies.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Win Rate: {results['win_rate']:.2%}")
        logger.info(f"  Total Return: {results['total_return']:.2%}")
        logger.info(f"  Profit Factor: {results['profit_factor']:.2f}")
        logger.info(f"  Max Drawdown: {results['max_drawdown']:.2%}")
        logger.info(f"  Total Trades: {results['total_trades']}")


# ═══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to dataframe"""
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
    
    # Drop NaN
    df = df.dropna()
    
    return df


def run_backtest(strategy, data: pd.DataFrame) -> dict:
    """Run backtest on strategy"""
    balance = 10000
    position = None
    trades = []
    
    for i in range(100, len(data)):
        window_data = data.iloc[:i+1]
        signal = strategy.generate_signals(window_data, position)
        
        current_price = data['close'].iloc[i]
        
        # Entry
        if signal['action'] == 'BUY' and position is None:
            position = {
                'entry_price': current_price,
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'entry_time': data.index[i],
                'side': 'LONG'
            }
        
        # Exit
        elif signal['action'] == 'SELL' and position is not None:
            pnl = (current_price - position['entry_price']) / position['entry_price']
            balance *= (1 + pnl * 0.02)  # 2% risk per trade
            
            trades.append({
                'entry': position['entry_price'],
                'exit': current_price,
                'pnl': pnl,
                'win': pnl > 0
            })
            
            position = None
        
        # Check SL/TP
        elif position is not None:
            if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                pnl = (current_price - position['entry_price']) / position['entry_price']
                balance *= (1 + pnl * 0.02)
                
                trades.append({
                    'entry': position['entry_price'],
                    'exit': current_price,
                    'pnl': pnl,
                    'win': pnl > 0
                })
                
                position = None
    
    # Calculate metrics
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_return': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
    
    wins = sum(1 for t in trades if t['win'])
    losses = len(trades) - wins
    total_profit = sum(t['pnl'] for t in trades if t['win'])
    total_loss = abs(sum(t['pnl'] for t in trades if not t['win']))
    
    return {
        'total_trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(trades),
        'total_return': (balance - 10000) / 10000,
        'profit_factor': total_profit / max(total_loss, 0.001),
        'max_drawdown': calculate_max_drawdown(trades),
        'sharpe_ratio': calculate_sharpe(trades)
    }


def backtest_pa_strategy(pa_strategy, data: pd.DataFrame) -> dict:
    """Backtest pure PA strategy"""
    balance = 10000
    position = None
    trades = []
    
    for i in range(100, len(data)):
        current_data = data.iloc[i]
        pa_signal = pa_strategy.analyze(current_data)
        
        current_price = current_data['close']
        
        # Entry on valid setup
        if pa_signal.get('setup_valid') and position is None:
            if pa_signal.get('setup_score', 0) >= 60:  # Minimum threshold
                position = {
                    'entry_price': current_price,
                    'stop_loss': pa_signal['stop_loss'],
                    'take_profit': pa_signal['take_profit'],
                    'entry_time': data.index[i],
                    'side': 'LONG'
                }
        
        # Check SL/TP
        elif position is not None:
            if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                pnl = (current_price - position['entry_price']) / position['entry_price']
                balance *= (1 + pnl * 0.02)
                
                trades.append({
                    'entry': position['entry_price'],
                    'exit': current_price,
                    'pnl': pnl,
                    'win': pnl > 0
                })
                
                position = None
    
    # Calculate metrics (same as run_backtest)
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_return': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
    
    wins = sum(1 for t in trades if t['win'])
    losses = len(trades) - wins
    total_profit = sum(t['pnl'] for t in trades if t['win'])
    total_loss = abs(sum(t['pnl'] for t in trades if not t['win']))
    
    return {
        'total_trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(trades),
        'total_return': (balance - 10000) / 10000,
        'profit_factor': total_profit / max(total_loss, 0.001),
        'max_drawdown': calculate_max_drawdown(trades),
        'sharpe_ratio': calculate_sharpe(trades)
    }


def calculate_max_drawdown(trades: list) -> float:
    """Calculate maximum drawdown"""
    cumulative = [0]
    for trade in trades:
        cumulative.append(cumulative[-1] + trade['pnl'])
    
    peak = cumulative[0]
    max_dd = 0
    
    for value in cumulative:
        if value > peak:
            peak = value
        dd = (peak - value) / max(peak, 0.001)
        if dd > max_dd:
            max_dd = dd
    
    return max_dd


def calculate_sharpe(trades: list) -> float:
    """Calculate Sharpe ratio"""
    if len(trades) < 2:
        return 0
    
    returns = [t['pnl'] for t in trades]
    mean_return = sum(returns) / len(returns)
    std_return = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5
    
    if std_return == 0:
        return 0
    
    return mean_return / std_return * (252 ** 0.5)  # Annualized


# ═══════════════════════════════════════════════════════
# MAIN MENU
# ═══════════════════════════════════════════════════════

def main():
    """Main menu"""
    print("\n" + "=" * 80)
    print("RL TRADING BOT - EXAMPLES")
    print("=" * 80)
    print("\nAvailable examples:")
    print("1. Train RL agent from scratch")
    print("2. Use trained agent for signals")
    print("3. Backtest RL strategy")
    print("4. Hybrid RL+PA strategy")
    print("5. Live monitoring (simulated)")
    print("6. Compare all strategies")
    print("0. Exit")
    print("=" * 80)
    
    choice = input("\nSelect example (0-6): ").strip()
    
    examples = {
        '1': example_1_train_from_scratch,
        '2': example_2_use_trained_agent,
        '3': example_3_backtest_rl_strategy,
        '4': example_4_hybrid_strategy,
        '5': example_5_live_monitoring,
        '6': example_6_compare_strategies
    }
    
    if choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            logger.error(f"Error running example: {e}")
            import traceback
            traceback.print_exc()
    elif choice == '0':
        print("Exiting...")
    else:
        print("Invalid choice!")
        main()


if __name__ == '__main__':
    main()
rl_agent_ppo import PPOAgent, create_trading_ppo_config
from rl.trading_env import TradingEnvironment
from rl.