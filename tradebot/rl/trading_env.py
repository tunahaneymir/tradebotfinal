"""
Trading Environment for RL Agent
Gym-compatible environment that integrates with PA strategy
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta


class TradingEnvironment(gym.Env):
    """
    Gym Environment for Crypto Trading with PA Strategy Integration
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        data: pd.DataFrame,
        pa_detector,  # PA detection module
        initial_balance: float = 10000,
        commission: float = 0.001,  # 0.1%
        max_position_size: float = 0.02,  # 2% risk per trade
        lookback_window: int = 100,
        max_steps: int = 1000,
        reward_scaling: float = 0.01
    ):
        super().__init__()
        
        self.data = data
        self.pa_detector = pa_detector
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.max_steps = max_steps
        self.reward_scaling = reward_scaling
        
        # State space: 50 features
        # [PA features (30) + Market features (10) + Portfolio features (10)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(50,),
            dtype=np.float32
        )
        
        # Action space: 3 actions
        # 0: HOLD, 1: BUY, 2: SELL
        self.action_space = spaces.Discrete(3)
        
        # Episode state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = 0
        self.position_size = 0
        self.stop_loss = 0
        self.take_profit = 0
        
        # Trade tracking
        self.trades = []
        self.current_trade = None
        
        # Performance tracking
        self.total_profit = 0
        self.total_loss = 0
        self.wins = 0
        self.losses = 0
        
        # Behavioral tracking
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.trades_today = 0
        self.daily_pnl = 0
        
        # Setup logger
        self.logger = logging.getLogger('TradingEnvironment')
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        # Reset to random point in data (for training)
        self.current_step = np.random.randint(
            self.lookback_window,
            len(self.data) - self.max_steps
        )
        
        # Reset portfolio
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.position_size = 0
        self.stop_loss = 0
        self.take_profit = 0
        
        # Reset tracking
        self.trades = []
        self.current_trade = None
        self.total_profit = 0
        self.total_loss = 0
        self.wins = 0
        self.losses = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.trades_today = 0
        self.daily_pnl = 0
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in environment
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
        
        Returns:
            observation, reward, done, info
        """
        # Get current market data
        current_data = self.data.iloc[self.current_step]
        current_price = current_data['close']
        
        # Initialize reward
        reward = 0
        info = {}
        
        # Execute action
        if action == 1:  # BUY
            reward = self._execute_buy(current_data)
            info['action'] = 'BUY'
            
        elif action == 2:  # SELL
            reward = self._execute_sell(current_data)
            info['action'] = 'SELL'
            
        else:  # HOLD
            reward = self._execute_hold(current_data)
            info['action'] = 'HOLD'
        
        # Check stop loss / take profit
        if self.position != 0:
            sl_tp_reward = self._check_sl_tp(current_data)
            if sl_tp_reward != 0:
                reward += sl_tp_reward
                info['sl_tp_triggered'] = True
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self._is_done()
        
        # Get next observation
        observation = self._get_observation()
        
        # Add info
        info.update({
            'balance': self.balance,
            'position': self.position,
            'total_trades': len(self.trades),
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.wins / max(len(self.trades), 1)
        })
        
        return observation, reward, done, info
    
    def _execute_buy(self, data: pd.Series) -> float:
        """Execute BUY action"""
        # Check if already in position
        if self.position != 0:
            return -10  # Penalty for trying to buy when already in position
        
        # Get PA signal
        pa_signal = self._get_pa_signal(data)
        
        # Check if setup is valid
        if pa_signal['setup_valid'] == False:
            return -20  # Penalty for invalid setup
        
        # Calculate position size
        setup_score = pa_signal['setup_score']
        zone_quality = pa_signal['zone_quality']
        
        # Risk calculation (from risk management system)
        base_risk = 0.02  # 2%
        risk_multiplier = self._calculate_risk_multiplier(pa_signal)
        position_risk = base_risk * risk_multiplier
        
        # Entry
        entry_price = data['close']
        stop_loss = pa_signal['stop_loss']
        take_profit = pa_signal['take_profit']
        
        # Position size calculation
        risk_amount = self.balance * position_risk
        price_risk = abs(entry_price - stop_loss) / entry_price
        position_size = risk_amount / price_risk if price_risk > 0 else 0
        
        if position_size > 0:
            # Execute trade
            self.position = 1
            self.entry_price = entry_price
            self.position_size = position_size
            self.stop_loss = stop_loss
            self.take_profit = take_profit
            
            # Record trade
            self.current_trade = {
                'entry_time': data.name,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'setup_score': setup_score,
                'zone_quality': zone_quality,
                'direction': 'LONG'
            }
            
            self.trades_today += 1
            
            # Reward for entering with good setup
            reward = setup_score * 0.1  # Scaled reward
            return reward
        
        return -5  # Penalty for failed entry
    
    def _execute_sell(self, data: pd.Series) -> float:
        """Execute SELL action (close position)"""
        if self.position == 0:
            return -10  # Penalty for trying to sell with no position
        
        # Calculate PnL
        exit_price = data['close']
        pnl_percent = (exit_price - self.entry_price) / self.entry_price * self.position
        pnl_amount = self.position_size * pnl_percent
        
        # Apply commission
        commission_cost = self.position_size * self.commission * 2  # Entry + Exit
        pnl_amount -= commission_cost
        
        # Update balance
        self.balance += pnl_amount
        
        # Close trade record
        if self.current_trade:
            self.current_trade.update({
                'exit_time': data.name,
                'exit_price': exit_price,
                'pnl_percent': pnl_percent * 100,
                'pnl_amount': pnl_amount,
                'exit_reason': 'MANUAL'
            })
            self.trades.append(self.current_trade)
        
        # Update statistics
        if pnl_amount > 0:
            self.wins += 1
            self.total_profit += pnl_amount
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.total_loss += abs(pnl_amount)
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        self.daily_pnl += pnl_amount
        
        # Reset position
        self.position = 0
        self.current_trade = None
        
        # Calculate reward (outcome score from Parça 6)
        reward = self._calculate_outcome_score(pnl_percent, self.current_trade)
        
        return reward
    
    def _execute_hold(self, data: pd.Series) -> float:
        """Execute HOLD action"""
        # Small penalty for holding to encourage action
        if self.position == 0:
            return -0.1
        
        # If in position, check unrealized PnL
        current_price = data['close']
        unrealized_pnl = (current_price - self.entry_price) / self.entry_price * self.position
        
        # Small reward for holding profitable position
        if unrealized_pnl > 0:
            return 0.5
        else:
            return -0.2
    
    def _check_sl_tp(self, data: pd.Series) -> float:
        """Check if stop loss or take profit hit"""
        current_price = data['close']
        
        # Check stop loss
        if self.position == 1 and current_price <= self.stop_loss:
            return self._execute_sell(data)
        
        # Check take profit
        if self.position == 1 and current_price >= self.take_profit:
            return self._execute_sell(data)
        
        return 0
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation (50 features)
        
        Features breakdown:
        - PA Features (30): Zone quality, ChoCH, Fibonacci, etc.
        - Market Features (10): Trend, volatility, volume, etc.
        - Portfolio Features (10): Position, PnL, win rate, etc.
        """
        # Get current and historical data
        current_idx = self.current_step
        start_idx = max(0, current_idx - self.lookback_window)
        window_data = self.data.iloc[start_idx:current_idx + 1]
        current_data = self.data.iloc[current_idx]
        
        # Get PA signal
        pa_signal = self._get_pa_signal(current_data)
        
        # ═══════════════════════════════════
        # PA FEATURES (30)
        # ═══════════════════════════════════
        pa_features = [
            pa_signal.get('zone_quality', 0) / 10,  # 0-1 normalized
            pa_signal.get('zone_distance', 0),
            pa_signal.get('zone_thickness', 0),
            pa_signal.get('zone_touch_count', 0) / 10,
            pa_signal.get('zone_recency', 0) / 30,
            
            pa_signal.get('choch_detected', 0),
            pa_signal.get('choch_strength', 0),
            pa_signal.get('choch_candles_ago', 0) / 20,
            pa_signal.get('choch_volume_ratio', 0),
            
            pa_signal.get('fib_level', 0),
            pa_signal.get('fib_distance', 0),
            pa_signal.get('fib_retest', 0),
            
            pa_signal.get('setup_score', 0) / 100,
            pa_signal.get('setup_valid', 0),
            pa_signal.get('gate1_pass', 0),
            pa_signal.get('gate2_pass', 0),
            pa_signal.get('gate3_pass', 0),
            pa_signal.get('gate4_pass', 0),
            
            pa_signal.get('liquidity_sweep', 0),
            pa_signal.get('volume_spike', 0),
            
            # Additional PA features
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # Placeholder for expansion
        ]
        
        # ═══════════════════════════════════
        # MARKET FEATURES (10)
        # ═══════════════════════════════════
        market_features = [
            self._normalize_price(current_data['close']),
            self._calculate_trend_strength(window_data),
            self._calculate_volatility(window_data),
            self._calculate_volume_ratio(window_data),
            current_data.get('rsi', 50) / 100,
            current_data.get('atr', 0) / current_data['close'],
            
            # EMA status
            1 if current_data['close'] > current_data.get('ema_20', current_data['close']) else 0,
            1 if current_data.get('ema_20', 0) > current_data.get('ema_50', 0) else 0,
            
            # Momentum
            (current_data['close'] - window_data['close'].iloc[0]) / window_data['close'].iloc[0],
            
            # Time features
            current_data.name.hour / 24 if hasattr(current_data.name, 'hour') else 0
        ]
        
        # ═══════════════════════════════════
        # PORTFOLIO FEATURES (10)
        # ═══════════════════════════════════
        portfolio_features = [
            self.position,  # 0, 1, or -1
            (self.balance - self.initial_balance) / self.initial_balance,  # Total return
            self.position_size / self.balance if self.balance > 0 else 0,
            
            # Unrealized PnL if in position
            ((current_data['close'] - self.entry_price) / self.entry_price * self.position) if self.position != 0 else 0,
            
            # Win rate
            self.wins / max(len(self.trades), 1),
            
            # Consecutive wins/losses
            self.consecutive_wins / 10,
            self.consecutive_losses / 10,
            
            # Daily metrics
            self.trades_today / 10,
            self.daily_pnl / self.initial_balance,
            
            # Time since last trade
            self._time_since_last_trade()
        ]
        
        # Combine all features
        observation = np.array(
            pa_features + market_features + portfolio_features,
            dtype=np.float32
        )
        
        # Clip to prevent extreme values
        observation = np.clip(observation, -10, 10)
        
        return observation
    
    def _get_pa_signal(self, data: pd.Series) -> Dict:
        """Get PA detection signal"""
        if self.pa_detector:
            return self.pa_detector.analyze(data)
        
        # Fallback: basic signal
        return {
            'setup_valid': False,
            'setup_score': 0,
            'zone_quality': 5,
            'choch_detected': 0,
            'choch_strength': 0,
            'stop_loss': data['close'] * 0.98,
            'take_profit': data['close'] * 1.04
        }
    
    def _calculate_risk_multiplier(self, pa_signal: Dict) -> float:
        """
        Calculate risk multiplier based on setup quality
        (From Parça 5: Risk Management)
        """
        multiplier = 1.0
        
        # Setup quality multiplier (0.5-1.5x)
        setup_score = pa_signal.get('setup_score', 50)
        if setup_score >= 80:
            multiplier *= 1.5
        elif setup_score >= 60:
            multiplier *= 1.2
        elif setup_score < 40:
            multiplier *= 0.5
        
        # Zone quality multiplier
        zone_quality = pa_signal.get('zone_quality', 5)
        if zone_quality >= 8:
            multiplier *= 1.2
        elif zone_quality <= 4:
            multiplier *= 0.7
        
        # Recent performance multiplier
        if len(self.trades) >= 5:
            recent_wr = sum(1 for t in self.trades[-5:] if t.get('pnl_amount', 0) > 0) / 5
            if recent_wr >= 0.8:
                multiplier *= 1.3
            elif recent_wr <= 0.3:
                multiplier *= 0.6
        
        # Consecutive losses multiplier
        if self.consecutive_losses >= 3:
            multiplier *= 0.5
        elif self.consecutive_losses >= 2:
            multiplier *= 0.7
        
        # Limit multiplier range
        multiplier = np.clip(multiplier, 0.5, 2.5)
        
        return multiplier
    
    def _calculate_outcome_score(self, pnl_percent: float, trade: Dict) -> float:
        """
        Calculate outcome score (reward)
        Based on Parça 6: Reward System
        """
        setup_score = trade.get('setup_score', 50) if trade else 50
        
        # Win scenarios
        if pnl_percent > 0:
            if pnl_percent >= 0.03:  # 3%+ (Excellent)
                return 200 * self.reward_scaling
            elif pnl_percent >= 0.02:  # 2-3% (Very Good)
                return 150 * self.reward_scaling
            elif pnl_percent >= 0.01:  # 1-2% (Good)
                return 100 * self.reward_scaling
            else:  # 0-1% (Small win)
                return 50 * self.reward_scaling
        
        # Loss scenarios
        else:
            if setup_score >= 70:  # Good setup, acceptable loss
                return -50 * self.reward_scaling
            elif setup_score >= 50:  # OK setup, small penalty
                return -100 * self.reward_scaling
            else:  # Bad setup, large penalty
                return -180 * self.reward_scaling
    
    def _is_done(self) -> bool:
        """Check if episode is finished"""
        # Max steps reached
        if self.current_step >= len(self.data) - 1:
            return True
        
        if self.current_step >= self.max_steps:
            return True
        
        # Balance too low (bankruptcy)
        if self.balance < self.initial_balance * 0.5:
            return True
        
        return False
    
    def _normalize_price(self, price: float) -> float:
        """Normalize price to 0-1 range"""
        window_data = self.data.iloc[max(0, self.current_step - 100):self.current_step + 1]
        price_min = window_data['close'].min()
        price_max = window_data['close'].max()
        
        if price_max > price_min:
            return (price - price_min) / (price_max - price_min)
        return 0.5
    
    def _calculate_trend_strength(self, window_data: pd.DataFrame) -> float:
        """Calculate trend strength (-1 to 1)"""
        if len(window_data) < 2:
            return 0
        
        returns = window_data['close'].pct_change().dropna()
        if len(returns) == 0:
            return 0
        
        trend = returns.mean() / (returns.std() + 1e-8)
        return np.clip(trend, -1, 1)
    
    def _calculate_volatility(self, window_data: pd.DataFrame) -> float:
        """Calculate volatility (0-1)"""
        if len(window_data) < 2:
            return 0
        
        returns = window_data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Normalize (typical crypto volatility: 0-10%)
        return np.clip(volatility / 0.1, 0, 1)
    
    def _calculate_volume_ratio(self, window_data: pd.DataFrame) -> float:
        """Calculate current volume vs average"""
        if 'volume' not in window_data.columns or len(window_data) < 10:
            return 1.0
        
        current_vol = window_data['volume'].iloc[-1]
        avg_vol = window_data['volume'].iloc[-10:].mean()
        
        if avg_vol > 0:
            return min(current_vol / avg_vol, 5.0)
        return 1.0
    
    def _time_since_last_trade(self) -> float:
        """Time since last trade (normalized)"""
        if len(self.trades) == 0:
            return 1.0
        
        # Placeholder: return normalized value
        # In real implementation, calculate actual time difference
        return min(self.current_step - len(self.trades) * 10, 100) / 100
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            current_data = self.data.iloc[self.current_step]
            
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step} | Time: {current_data.name}")
            print(f"Price: ${current_data['close']:.2f}")
            print(f"Balance: ${self.balance:.2f} ({(self.balance/self.initial_balance - 1)*100:+.2f}%)")
            print(f"Position: {['NONE', 'LONG', 'SHORT'][self.position]}")
            
            if self.position != 0:
                unrealized = (current_data['close'] - self.entry_price) / self.entry_price * 100
                print(f"Entry: ${self.entry_price:.2f} | Unrealized: {unrealized:+.2f}%")
            
            print(f"Trades: {len(self.trades)} | Wins: {self.wins} | Losses: {self.losses}")
            if len(self.trades) > 0:
                print(f"Win Rate: {self.wins/len(self.trades)*100:.1f}%")
            print(f"{'='*60}\n")
    
    def get_metrics(self) -> Dict:
        """Get environment metrics"""
        return {
            'total_trades': len(self.trades),
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.wins / max(len(self.trades), 1),
            'total_return': (self.balance - self.initial_balance) / self.initial_balance,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'profit_factor': self.total_profit / max(self.total_loss, 1),
            'balance': self.balance
        }


class VectorizedTradingEnvironment:
    """
    Vectorized environment for parallel training
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
    
    def reset(self):
        return np.array([env.reset() for env in self.envs])
    
    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        obs, rewards, dones, infos = zip(*results)
        return np.array(obs), np.array(rewards), np.array(dones), infos
    
    def close(self):
        for env in self.envs:
            env.close()