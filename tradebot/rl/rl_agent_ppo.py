"""
Production-Ready PPO Agent Implementation
Full implementation with all optimizations and features
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from collections import deque

from .networks import ActorCriticNetwork, LSTMActorCritic, ActorNetwork, CriticNetwork


@dataclass
class PPOConfig:
    """PPO Agent Configuration"""
    # Network architecture
    state_dim: int = 50
    action_dim: int = 3  # HOLD, BUY, SELL
    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    use_shared_network: bool = True
    use_lstm: bool = False
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    
    # PPO hyperparameters
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clip parameter
    value_loss_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Training parameters
    learning_rate: float = 3e-4
    adam_eps: float = 1e-5
    batch_size: int = 64
    n_epochs: int = 10  # PPO epochs per update
    n_steps: int = 2048  # Steps before update
    mini_batch_size: int = 64
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = 'cosine'  # 'linear', 'cosine', 'exponential'
    lr_decay_rate: float = 0.99
    
    # Advanced features
    use_gae: bool = True  # Use Generalized Advantage Estimation
    normalize_advantages: bool = True
    normalize_rewards: bool = True
    clip_value_loss: bool = True
    use_huber_loss: bool = False  # Use Huber loss for value
    
    # Exploration
    initial_exploration_rate: float = 1.0
    final_exploration_rate: float = 0.05
    exploration_decay_steps: int = 100000
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100


class ReplayBuffer:
    """
    Experience Replay Buffer for PPO
    Stores trajectories for off-policy learning
    """
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Dict[str, Any]):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch from buffer"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Stack experiences
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
        next_states = torch.stack([exp['next_state'] for exp in batch])
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.float32)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


class RolloutBuffer:
    """
    Rollout Buffer for storing trajectories during PPO training
    """
    def __init__(self, config: PPOConfig):
        self.config = config
        self.clear()
    
    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add transition to buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_returns_and_advantages(self, last_value: float):
        """
        Compute returns and advantages using GAE
        """
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        
        # Add last value
        values = torch.cat([values, torch.tensor([last_value])])
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        # Compute returns
        returns = advantages + values[:-1]
        
        self.returns = returns
        self.advantages = advantages
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data as tensors"""
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        
        return {
            'states': states,
            'actions': actions,
            'old_log_probs': log_probs,
            'returns': self.returns,
            'advantages': self.advantages,
            'values': torch.tensor(self.values, dtype=torch.float32)
        }
    
    def clear(self):
        """Clear buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.returns = None
        self.advantages = None
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    Full Production-Ready PPO Agent
    """
    def __init__(
        self,
        config: Optional[PPOConfig] = None,
        model_dir: str = 'models/ppo'
    ):
        self.config = config or PPOConfig()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Device
        self.device = torch.device(self.config.device)
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize networks
        self._build_networks()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_eps
        )
        
        # Learning rate scheduler
        self.lr_scheduler = self._build_lr_scheduler() if self.config.use_lr_scheduler else None
        
        # Buffers
        self.rollout_buffer = RolloutBuffer(self.config)
        self.replay_buffer = ReplayBuffer()
        
        # Tracking
        self.total_steps = 0
        self.episode_count = 0
        self.best_reward = -float('inf')
        
        # Statistics
        self.running_reward_mean = 0
        self.running_reward_std = 1
        
        # Hidden state for LSTM
        self.lstm_hidden = None
        
        self.logger.info("PPO Agent initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('PPOAgent')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _build_networks(self):
        """Build actor-critic networks"""
        if self.config.use_lstm:
            self.network = LSTMActorCritic(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                hidden_dim=self.config.lstm_hidden_dim,
                num_layers=self.config.lstm_num_layers
            ).to(self.device)
            self.logger.info("Using LSTM Actor-Critic network")
        
        elif self.config.use_shared_network:
            self.network = ActorCriticNetwork(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                shared_hidden_dims=self.config.hidden_dims[:2],
                actor_hidden_dims=self.config.hidden_dims[2:],
                critic_hidden_dims=self.config.hidden_dims[2:]
            ).to(self.device)
            self.logger.info("Using shared Actor-Critic network")
        
        else:
            self.actor = ActorNetwork(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                hidden_dims=self.config.hidden_dims
            ).to(self.device)
            
            self.critic = CriticNetwork(
                state_dim=self.config.state_dim,
                hidden_dims=self.config.hidden_dims
            ).to(self.device)
            
            self.network = nn.ModuleDict({
                'actor': self.actor,
                'critic': self.critic
            })
            self.logger.info("Using separate Actor and Critic networks")
    
    def _build_lr_scheduler(self):
        """Build learning rate scheduler"""
        if self.config.lr_scheduler_type == 'linear':
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=1000
            )
        elif self.config.lr_scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=1000,
                eta_min=1e-6
            )
        elif self.config.lr_scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.lr_decay_rate
            )
        return None
    
    def select_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        explore: bool = True
    ) -> Tuple[int, float, float]:
        """
        Select action using current policy
        
        Returns:
            action, log_prob, value
        """
        self.network.eval()
        
        with torch.no_grad():
            state = state.to(self.device)
            
            if self.config.use_lstm:
                action_logits, value, self.lstm_hidden = self.network(
                    state.unsqueeze(0), self.lstm_hidden
                )
            else:
                if self.config.use_shared_network:
                    action_logits, value = self.network(state.unsqueeze(0))
                else:
                    action_logits = self.actor(state.unsqueeze(0))
                    value = self.critic(state.unsqueeze(0))
            
            # Action distribution
            dist = Categorical(logits=action_logits)
            
            if deterministic:
                action = action_logits.argmax(dim=-1)
            else:
                # Exploration
                if explore and np.random.rand() < self._get_exploration_rate():
                    action = torch.randint(0, self.config.action_dim, (1,), device=self.device)
                else:
                    action = dist.sample()
            
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def _get_exploration_rate(self) -> float:
        """Get current exploration rate with decay"""
        progress = min(1.0, self.total_steps / self.config.exploration_decay_steps)
        return self.config.initial_exploration_rate - \
               (self.config.initial_exploration_rate - self.config.final_exploration_rate) * progress
    
    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store transition in rollout buffer"""
        # Normalize reward if enabled
        if self.config.normalize_rewards:
            reward = self._normalize_reward(reward)
        
        self.rollout_buffer.add(state, action, reward, value, log_prob, done)
        self.total_steps += 1
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics"""
        self.running_reward_mean = 0.99 * self.running_reward_mean + 0.01 * reward
        self.running_reward_std = 0.99 * self.running_reward_std + 0.01 * abs(reward - self.running_reward_mean)
        return (reward - self.running_reward_mean) / (self.running_reward_std + 1e-8)
    
    def update(self) -> Dict[str, float]:
        """
        Perform PPO update
        """
        # Compute returns and advantages
        if self.config.use_lstm:
            # Get last value with current hidden state
            with torch.no_grad():
                last_state = self.rollout_buffer.states[-1].unsqueeze(0).to(self.device)
                _, last_value, _ = self.network(last_state, self.lstm_hidden)
                last_value = last_value.item()
        else:
            with torch.no_grad():
                last_state = self.rollout_buffer.states[-1].unsqueeze(0).to(self.device)
                if self.config.use_shared_network:
                    _, last_value = self.network(last_state)
                else:
                    last_value = self.critic(last_state)
                last_value = last_value.item()
        
        self.rollout_buffer.compute_returns_and_advantages(last_value)
        
        # Get rollout data
        rollout_data = self.rollout_buffer.get()
        
        # Normalize advantages
        if self.config.normalize_advantages:
            advantages = rollout_data['advantages']
            rollout_data['advantages'] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        # PPO epochs
        for epoch in range(self.config.n_epochs):
            # Create mini-batches
            indices = np.random.permutation(len(self.rollout_buffer))
            
            for start_idx in range(0, len(self.rollout_buffer), self.config.mini_batch_size):
                end_idx = start_idx + self.config.mini_batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch
                batch_states = rollout_data['states'][batch_indices].to(self.device)
                batch_actions = rollout_data['actions'][batch_indices].to(self.device)
                batch_old_log_probs = rollout_data['old_log_probs'][batch_indices].to(self.device)
                batch_returns = rollout_data['returns'][batch_indices].to(self.device)
                batch_advantages = rollout_data['advantages'][batch_indices].to(self.device)
                batch_old_values = rollout_data['values'][batch_indices].to(self.device)
                
                # Forward pass
                if self.config.use_shared_network or self.config.use_lstm:
                    if self.config.use_lstm:
                        action_logits, values, _ = self.network(batch_states, None)
                    else:
                        action_logits, values = self.network(batch_states)
                else:
                    action_logits = self.actor(batch_states)
                    values = self.critic(batch_states)
                
                # Action distribution
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.config.clip_value_loss:
                    # Clipped value loss
                    values_clipped = batch_old_values + torch.clamp(
                        values - batch_old_values,
                        -self.config.clip_epsilon,
                        self.config.clip_epsilon
                    )
                    value_loss1 = F.mse_loss(values, batch_returns)
                    value_loss2 = F.mse_loss(values_clipped, batch_returns)
                    value_loss = torch.max(value_loss1, value_loss2)
                elif self.config.use_huber_loss:
                    value_loss = F.huber_loss(values, batch_returns)
                else:
                    value_loss = F.mse_loss(values, batch_returns)
                
                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss -
                    self.config.entropy_coef * entropy
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
        
        # Update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # Clear rollout buffer
        self.rollout_buffer.clear()
        
        # Reset LSTM hidden state
        if self.config.use_lstm:
            self.lstm_hidden = None
        
        # Return metrics
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'exploration_rate': self._get_exploration_rate()
        }
    
    def evaluate(
        self,
        env,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate agent performance
        """
        self.network.eval()
        
        episode_rewards = []
        episode_lengths = []
        win_count = 0
        
        for episode in range(n_episodes):
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Reset LSTM hidden state
            if self.config.use_lstm:
                self.lstm_hidden = None
            
            while not done:
                action, _, _ = self.select_action(
                    state,
                    deterministic=deterministic,
                    explore=False
                )
                
                next_state, reward, done, info = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if episode_reward > 0:
                win_count += 1
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'win_rate': win_count / n_episodes
        }
    
    def save(self, filename: str = 'ppo_agent.pt'):
        """Save agent"""
        save_path = self.model_dir / filename
        
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'best_reward': self.best_reward,
            'running_reward_mean': self.running_reward_mean,
            'running_reward_std': self.running_reward_std
        }
        
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Agent saved to {save_path}")
    
    def load(self, filename: str = 'ppo_agent.pt'):
        """Load agent"""
        load_path = self.model_dir / filename
        
        if not load_path.exists():
            self.logger.warning(f"Checkpoint not found: {load_path}")
            return False
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']
        self.best_reward = checkpoint['best_reward']
        self.running_reward_mean = checkpoint['running_reward_mean']
        self.running_reward_std = checkpoint['running_reward_std']
        
        if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        self.logger.info(f"Agent loaded from {load_path}")
        return True
    
    def train_episode(
        self,
        env,
        max_steps: int = 1000
    ) -> Dict[str, Any]:
        """
        Train for one episode
        """
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Reset LSTM hidden state
        if self.config.use_lstm:
            self.lstm_hidden = None
        
        update_metrics = None
        
        while not done and episode_length < max_steps:
            # Select action
            action, log_prob, value = self.select_action(state, explore=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            
            # Store transition
            self.store_transition(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            # Update if buffer is full
            if len(self.rollout_buffer) >= self.config.n_steps:
                update_metrics = self.update()
        
        # Final update if episode ended
        if len(self.rollout_buffer) > 0:
            update_metrics = self.update()
        
        self.episode_count += 1
        
        # Save best model
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.save('ppo_agent_best.pt')
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'update_metrics': update_metrics or {}
        }
    
    def train(
        self,
        env,
        n_episodes: int = 1000,
        eval_interval: int = 50,
        eval_episodes: int = 10
    ):
        """
        Full training loop
        """
        self.logger.info(f"Starting training for {n_episodes} episodes")
        
        episode_rewards = []
        
        for episode in range(n_episodes):
            # Train episode
            episode_metrics = self.train_episode(env)
            episode_rewards.append(episode_metrics['episode_reward'])
            
            # Logging
            if episode % self.config.log_interval == 0:
                recent_rewards = episode_rewards[-self.config.log_interval:]
                self.logger.info(
                    f"Episode {episode}/{n_episodes} | "
                    f"Reward: {episode_metrics['episode_reward']:.2f} | "
                    f"Mean Reward (last {self.config.log_interval}): {np.mean(recent_rewards):.2f} | "
                    f"Steps: {self.total_steps}"
                )
                
                if episode_metrics['update_metrics']:
                    metrics = episode_metrics['update_metrics']
                    self.logger.info(
                        f"  Policy Loss: {metrics['policy_loss']:.4f} | "
                        f"Value Loss: {metrics['value_loss']:.4f} | "
                        f"Entropy: {metrics['entropy']:.4f} | "
                        f"LR: {metrics['learning_rate']:.6f}"
                    )
            
            # Evaluation
            if episode % eval_interval == 0 and episode > 0:
                eval_metrics = self.evaluate(env, eval_episodes)
                self.logger.info(
                    f"Evaluation | "
                    f"Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f} | "
                    f"Win Rate: {eval_metrics['win_rate']:.2%}"
                )
            
            # Save checkpoint
            if episode % self.config.save_interval == 0 and episode > 0:
                self.save(f'ppo_agent_ep{episode}.pt')
        
        self.logger.info("Training completed!")
        self.save('ppo_agent_final.pt')


class PPOAgentEnsemble:
    """
    Ensemble of PPO agents for improved robustness
    """
    def __init__(
        self,
        n_agents: int = 3,
        config: Optional[PPOConfig] = None,
        model_dir: str = 'models/ppo_ensemble'
    ):
        self.n_agents = n_agents
        self.config = config or PPOConfig()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create agents
        self.agents = []
        for i in range(n_agents):
            agent_dir = self.model_dir / f'agent_{i}'
            agent = PPOAgent(self.config, str(agent_dir))
            self.agents.append(agent)
    
    def select_action(
        self,
        state: torch.Tensor,
        method: str = 'vote'  # 'vote', 'average', 'best'
    ) -> int:
        """
        Select action using ensemble
        """
        actions = []
        values = []
        
        for agent in self.agents:
            action, _, value = agent.select_action(state, deterministic=True, explore=False)
            actions.append(action)
            values.append(value)
        
        if method == 'vote':
            # Majority voting
            return np.bincount(actions).argmax()
        elif method == 'average':
            # Average Q-values (use values as proxy)
            action_values = {a: [] for a in range(self.config.action_dim)}
            for action, value in zip(actions, values):
                action_values[action].append(value)
            avg_values = {a: np.mean(vals) if vals else -float('inf') 
                         for a, vals in action_values.items()}
            return max(avg_values, key=avg_values.get)
        elif method == 'best':
            # Use agent with highest value
            best_idx = np.argmax(values)
            return actions[best_idx]
        
        return actions[0]
    
    def train(self, env, n_episodes: int = 1000):
        """Train all agents"""
        for i, agent in enumerate(self.agents):
            print(f"\nTraining Agent {i+1}/{self.n_agents}")
            agent.train(env, n_episodes)
    
    def save(self):
        """Save all agents"""
        for agent in self.agents:
            agent.save()
    
    def load(self):
        """Load all agents"""
        for agent in self.agents:
            agent.load()


# Utility functions
def create_trading_ppo_config() -> PPOConfig:
    """Create optimized config for trading"""
    return PPOConfig(
        state_dim=50,
        action_dim=3,
        hidden_dims=(256, 128, 64),
        use_shared_network=True,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        learning_rate=3e-4,
        batch_size=64,
        n_epochs=10,
        n_steps=2048,
        use_gae=True,
        normalize_advantages=True,
        normalize_rewards=True,
        entropy_coef=0.01
    )


def create_lstm_trading_config() -> PPOConfig:
    """Create LSTM config for sequential trading"""
    return PPOConfig(
        state_dim=50,
        action_dim=3,
        use_lstm=True,
        lstm_hidden_dim=128,
        lstm_num_layers=2,
        gamma=0.99,
        learning_rate=3e-4,
        n_steps=2048,
        normalize_rewards=True
    )