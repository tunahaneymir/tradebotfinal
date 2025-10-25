"""
Neural Network Architecture for PPO Agent
Actor-Critic networks with shared and separate architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class ActorNetwork(nn.Module):
    """
    Policy Network (Actor) - Outputs action probabilities
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        activation: str = 'relu',
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Build layers
        layers = []
        input_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self.activation)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layer - action logits
        self.action_head = nn.Linear(input_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns action logits
        """
        features = self.feature_extractor(state)
        action_logits = self.action_head(features)
        return action_logits
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities using softmax
        """
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)


class CriticNetwork(nn.Module):
    """
    Value Network (Critic) - Outputs state value
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        activation: str = 'relu',
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.state_dim = state_dim
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Build layers
        layers = []
        input_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self.activation)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layer - state value
        self.value_head = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns state value
        """
        features = self.feature_extractor(state)
        value = self.value_head(features)
        return value.squeeze(-1)


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic Network with shared feature extraction
    More efficient than separate networks
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        shared_hidden_dims: Tuple[int, ...] = (256, 128),
        actor_hidden_dims: Tuple[int, ...] = (64,),
        critic_hidden_dims: Tuple[int, ...] = (64,),
        activation: str = 'relu',
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Shared feature extractor
        shared_layers = []
        input_dim = state_dim
        
        for hidden_dim in shared_hidden_dims:
            shared_layers.append(nn.Linear(input_dim, hidden_dim))
            
            if use_batch_norm:
                shared_layers.append(nn.BatchNorm1d(hidden_dim))
            
            shared_layers.append(self.activation)
            
            if dropout_rate > 0:
                shared_layers.append(nn.Dropout(dropout_rate))
            
            input_dim = hidden_dim
        
        self.shared_features = nn.Sequential(*shared_layers)
        
        # Actor-specific layers
        actor_layers = []
        for hidden_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(input_dim, hidden_dim))
            actor_layers.append(self.activation)
            input_dim = hidden_dim
        
        self.actor_features = nn.Sequential(*actor_layers) if actor_layers else nn.Identity()
        self.action_head = nn.Linear(input_dim, action_dim)
        
        # Critic-specific layers
        critic_layers = []
        input_dim = shared_hidden_dims[-1]
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(input_dim, hidden_dim))
            critic_layers.append(self.activation)
            input_dim = hidden_dim
        
        self.critic_features = nn.Sequential(*critic_layers) if critic_layers else nn.Identity()
        self.value_head = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - returns action logits and state value
        """
        shared_features = self.shared_features(state)
        
        # Actor branch
        actor_features = self.actor_features(shared_features)
        action_logits = self.action_head(actor_features)
        
        # Critic branch
        critic_features = self.critic_features(shared_features)
        value = self.value_head(critic_features).squeeze(-1)
        
        return action_logits, value
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities
        """
        action_logits, _ = self.forward(state)
        return F.softmax(action_logits, dim=-1)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get state value
        """
        _, value = self.forward(state)
        return value


class LSTMActorCritic(nn.Module):
    """
    LSTM-based Actor-Critic for sequential decision making
    Useful for capturing temporal dependencies
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for feature extraction
        self.lstm = nn.LSTM(
            state_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with hidden state
        
        Args:
            state: (batch_size, seq_len, state_dim) or (batch_size, state_dim)
            hidden: Optional hidden state tuple (h, c)
        
        Returns:
            action_logits, value, new_hidden
        """
        # Handle 2D input (single timestep)
        if len(state.shape) == 2:
            state = state.unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward
        lstm_out, new_hidden = self.lstm(state, hidden)
        
        # Take last timestep output
        features = lstm_out[:, -1, :]
        
        # Actor and critic outputs
        action_logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        
        return action_logits, value, new_hidden
    
    def get_action_probs(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action probabilities
        """
        action_logits, _, new_hidden = self.forward(state, hidden)
        return F.softmax(action_logits, dim=-1), new_hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state
        """
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)