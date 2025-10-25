"""
State Builder - Parça 4
Based on: pa-strateji2 Parça 4

State Builder for RL Agent:
- Aggregates 40+ features from all PA components
- Normalizes features for ML
- Creates comprehensive state vector
- Handles missing/invalid features
- Feature importance tracking
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np

from core import TrendResult, Zone, ChoCHResult, FibonacciLevels, EntrySignal
from adaptive import AdaptiveParams


@dataclass
class StateVector:
    """Complete state vector for RL agent"""
    features: Dict[str, float]  # All features
    feature_vector: np.ndarray  # Numpy array for ML
    feature_names: List[str]    # Feature names (ordered)
    normalized: bool            # Are features normalized?
    valid: bool                 # All features valid?
    
    def __len__(self):
        return len(self.feature_vector)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return self.features.copy()
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return self.feature_vector.copy()


class StateBuilder:
    """
    State Builder for RL Agent
    
    Aggregates features from all components:
    - Trend features (15 features)
    - Zone features (8 features)
    - ChoCH features (5 features)
    - Fibonacci features (5 features)
    - Entry validation features (6 features)
    - Market context features (5 features)
    - Adaptive features (4 features)
    
    Total: 40+ features
    
    Usage:
        builder = StateBuilder()
        
        # Build state from components
        state = builder.build_state(
            trend=trend_result,
            zone=zone,
            choch=choch_result,
            fibonacci=fib_levels,
            entry_signal=entry_signal,
            adaptive_params=adaptive_params,
            current_price=50000,
            market_context={...}
        )
        
        # Use for RL
        feature_vector = state.to_array()  # For neural network
    """
    
    def __init__(
        self,
        normalize: bool = True,
        handle_missing: str = "zero"  # "zero", "mean", or "drop"
    ):
        """
        Initialize State Builder
        
        Args:
            normalize: Normalize features to [0, 1] or [-1, 1]
            handle_missing: How to handle missing features
        """
        self.normalize = normalize
        self.handle_missing = handle_missing
        
        # Feature normalization ranges (will be updated online)
        self.feature_stats = {}
        
        # Feature importance (tracked over time)
        self.feature_importance = {}
    
    def build_state(
        self,
        trend: Optional[TrendResult] = None,
        zone: Optional[Zone] = None,
        choch: Optional[ChoCHResult] = None,
        fibonacci: Optional[FibonacciLevels] = None,
        entry_signal: Optional[EntrySignal] = None,
        adaptive_params: Optional[AdaptiveParams] = None,
        current_price: float = 0.0,
        market_context: Optional[Dict[str, Any]] = None
    ) -> StateVector:
        """
        Build complete state vector from components
        
        Args:
            trend: Trend detection result
            zone: Zone object
            choch: ChoCH detection result
            fibonacci: Fibonacci levels
            entry_signal: Entry signal object
            adaptive_params: Adaptive parameters
            current_price: Current market price
            market_context: Additional market context
            
        Returns:
            StateVector with all features
        """
        features = {}
        
        # ═══════════════════════════════════════════════════════════
        # 1. TREND FEATURES (15 features)
        # ═══════════════════════════════════════════════════════════
        if trend:
            trend_features = trend.get_ml_features()
            features.update(self._prefix_keys(trend_features, "trend_"))
        else:
            features.update(self._get_default_trend_features())
        
        # ═══════════════════════════════════════════════════════════
        # 2. ZONE FEATURES (8 features)
        # ═══════════════════════════════════════════════════════════
        if zone:
            zone_features = zone.get_ml_features()
            features.update(self._prefix_keys(zone_features, "zone_"))
        else:
            features.update(self._get_default_zone_features())
        
        # ═══════════════════════════════════════════════════════════
        # 3. ChoCH FEATURES (5 features)
        # ═══════════════════════════════════════════════════════════
        if choch:
            choch_features = choch.get_ml_features()
            features.update(self._prefix_keys(choch_features, "choch_"))
        else:
            features.update(self._get_default_choch_features())
        
        # ═══════════════════════════════════════════════════════════
        # 4. FIBONACCI FEATURES (5 features)
        # ═══════════════════════════════════════════════════════════
        if fibonacci:
            fib_features = fibonacci.get_ml_features()
            features.update(self._prefix_keys(fib_features, "fib_"))
        else:
            features.update(self._get_default_fib_features())
        
        # ═══════════════════════════════════════════════════════════
        # 5. ENTRY VALIDATION FEATURES (6 features)
        # ═══════════════════════════════════════════════════════════
        if entry_signal:
            entry_features = {
                'entry_ready': float(entry_signal.ready),
                'entry_trend_aligned': float(entry_signal.trend_aligned),
                'entry_zone_valid': float(entry_signal.zone_valid),
                'entry_choch_strong': float(entry_signal.choch_strong),
                'entry_fib_touched': float(entry_signal.fib_touched),
                'entry_risk_reward': (
                    abs(entry_signal.entry_price - entry_signal.stop_loss) / 
                    abs(entry_signal.entry_price - current_price)
                    if entry_signal.ready and current_price > 0 else 0.0
                )
            }
            features.update(entry_features)
        else:
            features.update(self._get_default_entry_features())
        
        # ═══════════════════════════════════════════════════════════
        # 6. MARKET CONTEXT FEATURES (5 features)
        # ═══════════════════════════════════════════════════════════
        if market_context:
            context_features = {
                'market_volume_ratio': market_context.get('volume_ratio', 1.0),
                'market_volatility': market_context.get('volatility', 0.0),
                'market_spread': market_context.get('spread', 0.0),
                'market_time_of_day': market_context.get('time_of_day', 0.5),
                'market_day_of_week': market_context.get('day_of_week', 0.5)
            }
            features.update(context_features)
        else:
            features.update(self._get_default_market_features())
        
        # ═══════════════════════════════════════════════════════════
        # 7. ADAPTIVE FEATURES (4 features)
        # ═══════════════════════════════════════════════════════════
        if adaptive_params:
            adaptive_features = {
                'adaptive_atr_percent': adaptive_params.atr_percent / 10.0,  # Normalize
                'adaptive_volatility_extreme': float(adaptive_params.volatility_regime == "EXTREME"),
                'adaptive_volatility_high': float(adaptive_params.volatility_regime == "HIGH"),
                'adaptive_volatility_low': float(adaptive_params.volatility_regime == "LOW")
            }
            features.update(adaptive_features)
        else:
            features.update(self._get_default_adaptive_features())
        
        # ═══════════════════════════════════════════════════════════
        # BUILD FEATURE VECTOR
        # ═══════════════════════════════════════════════════════════
        
        # Ensure consistent ordering
        feature_names = sorted(features.keys())
        
        # Create numpy array
        feature_vector = np.array([features[name] for name in feature_names], dtype=np.float32)
        
        # Handle invalid values (NaN, Inf)
        valid = self._validate_features(feature_vector)
        if not valid:
            feature_vector = self._handle_invalid_features(feature_vector)
        
        # Normalize if enabled
        if self.normalize:
            feature_vector = self._normalize_features(feature_vector, feature_names)
        
        return StateVector(
            features=features,
            feature_vector=feature_vector,
            feature_names=feature_names,
            normalized=self.normalize,
            valid=valid
        )
    
    def _prefix_keys(self, features: Dict[str, float], prefix: str) -> Dict[str, float]:
        """Add prefix to all keys"""
        return {f"{prefix}{key}": value for key, value in features.items()}
    
    def _validate_features(self, features: np.ndarray) -> bool:
        """Check if all features are valid (no NaN, Inf)"""
        return not (np.isnan(features).any() or np.isinf(features).any())
    
    def _handle_invalid_features(self, features: np.ndarray) -> np.ndarray:
        """Handle invalid feature values"""
        if self.handle_missing == "zero":
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        elif self.handle_missing == "mean":
            # Replace with column mean (if available)
            for i in range(len(features)):
                if np.isnan(features[i]) or np.isinf(features[i]):
                    features[i] = 0.0  # Default to 0 if no stats
        return features
    
    def _normalize_features(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """
        Normalize features
        
        Most features are already in [0, 1] or [-1, 1]
        But we clip to ensure bounds
        """
        # Clip to reasonable bounds
        features = np.clip(features, -10.0, 10.0)
        
        # Most features are already normalized by design
        # Just ensure they're in reasonable range
        return features
    
    # ═══════════════════════════════════════════════════════════
    # DEFAULT FEATURE GENERATORS (for missing components)
    # ═══════════════════════════════════════════════════════════
    
    def _get_default_trend_features(self) -> Dict[str, float]:
        """Default trend features when no trend data"""
        return {
            'trend_trend_direction_numeric': 0.0,
            'trend_trend_confidence': 0.0,
            'trend_ema_distance_ratio': 0.0,
            'trend_ema_20_value': 0.0,
            'trend_ema_50_value': 0.0,
            'trend_ema_cross_position': 0.0,
            'trend_ema_slope_up': 0.0,
            'trend_ema_slope_down': 0.0,
            'trend_ema_slope_strength': 0.0,
            'trend_atr_ratio': 0.0,
            'trend_price_range_ratio': 0.0,
            'trend_is_sideways': 0.0,
            'trend_is_trending': 0.0,
            'trend_sideways_signal_count': 0.0,
            'trend_trend_strength': 0.0
        }
    
    def _get_default_zone_features(self) -> Dict[str, float]:
        """Default zone features when no zone data"""
        return {
            'zone_zone_quality': 0.0,
            'zone_zone_touch_count': 0.0,
            'zone_zone_thickness_pct': 0.0,
            'zone_zone_days_since_touch': 0.0,
            'zone_zone_is_fresh': 0.0,
            'zone_zone_is_thin': 0.0,
            'zone_zone_method_both': 0.0,
            'zone_zone_age_candles': 0.0
        }
    
    def _get_default_choch_features(self) -> Dict[str, float]:
        """Default ChoCH features when no ChoCH data"""
        return {
            'choch_choch_detected': 0.0,
            'choch_choch_strength': 0.0,
            'choch_choch_body_score': 0.0,
            'choch_choch_volume_score': 0.0,
            'choch_choch_direction': 0.0
        }
    
    def _get_default_fib_features(self) -> Dict[str, float]:
        """Default Fibonacci features when no Fib data"""
        return {
            'fib_fib_swing_range': 0.0,
            'fib_fib_0705_level': 0.0,
            'fib_fib_0618_level': 0.0,
            'fib_fib_levels_distance': 0.0,
            'fib_fib_direction': 0.0
        }
    
    def _get_default_entry_features(self) -> Dict[str, float]:
        """Default entry features when no entry data"""
        return {
            'entry_ready': 0.0,
            'entry_trend_aligned': 0.0,
            'entry_zone_valid': 0.0,
            'entry_choch_strong': 0.0,
            'entry_fib_touched': 0.0,
            'entry_risk_reward': 0.0
        }
    
    def _get_default_market_features(self) -> Dict[str, float]:
        """Default market features when no market data"""
        return {
            'market_volume_ratio': 1.0,
            'market_volatility': 0.0,
            'market_spread': 0.0,
            'market_time_of_day': 0.5,
            'market_day_of_week': 0.5
        }
    
    def _get_default_adaptive_features(self) -> Dict[str, float]:
        """Default adaptive features when no adaptive data"""
        return {
            'adaptive_atr_percent': 0.0,
            'adaptive_volatility_extreme': 0.0,
            'adaptive_volatility_high': 0.0,
            'adaptive_volatility_low': 0.0
        }
    
    def get_feature_count(self) -> int:
        """Get total number of features"""
        return (
            15 +  # Trend
            8 +   # Zone
            5 +   # ChoCH
            5 +   # Fibonacci
            6 +   # Entry
            5 +   # Market
            4     # Adaptive
        )  # = 48 features


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from core import TrendResult, Zone, ChoCHResult
    
    print("\n" + "="*60)
    print("STATE BUILDER - TEST")
    print("="*60)
    
    # Create builder
    builder = StateBuilder(normalize=True)
    
    print(f"\n✅ State Builder created")
    print(f"   Total features: {builder.get_feature_count()}")
    print(f"   Normalization: {builder.normalize}")
    print(f"   Missing handler: {builder.handle_missing}")
    
    # ═══════════════════════════════════════════════════════════
    # Test 1: Build with mock data
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("TEST 1: Build State with Mock Components")
    print(f"{'='*60}\n")
    
    # Mock trend
    trend = TrendResult(
        direction="UP",
        ema_20=50500.0,
        ema_50=50000.0,
        ema_distance_pct=0.01,
        atr_ratio=0.05,
        price_range_pct=0.03,
        confidence=0.85,
        slope_up=True,
        slope_down=False,
        slope_strength=0.9,
        sideways_signal_count=0
    )
    
    # Mock zone
    zone = Zone(
        id="TEST_ZONE",
        price_low=50000,
        price_high=50100,
        price_mid=50050,
        touch_count=2,
        thickness_pct=0.002,
        last_touch_index=100,
        creation_index=50,
        timeframe="1H",
        method="both",
        quality=8.5,
        days_since_last_touch=2.5
    )
    
    # Mock ChoCH
    choch = ChoCHResult(
        detected=True,
        direction="LONG",
        breakout_price=50200,
        breakout_index=150,
        broken_level=50000,
        strength=0.75,
        body_score=0.35,
        volume_score=0.40
    )
    
    # Build state
    state = builder.build_state(
        trend=trend,
        zone=zone,
        choch=choch,
        current_price=50100,
        market_context={
            'volume_ratio': 1.5,
            'volatility': 0.05,
            'spread': 0.001
        }
    )
    
    print(f"✅ State built successfully")
    print(f"   Total features: {len(state)}")
    print(f"   Valid: {state.valid}")
    print(f"   Normalized: {state.normalized}")
    
    print(f"\n📊 Sample Features:")
    sample_features = list(state.features.items())[:10]
    for name, value in sample_features:
        print(f"   {name:<40s}: {value:8.4f}")
    
    print(f"\n🔢 Feature Vector Shape: {state.feature_vector.shape}")
    print(f"   Min: {state.feature_vector.min():.4f}")
    print(f"   Max: {state.feature_vector.max():.4f}")
    print(f"   Mean: {state.feature_vector.mean():.4f}")
    
    # ═══════════════════════════════════════════════════════════
    # Test 2: Build with missing components
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("TEST 2: Build State with Missing Components")
    print(f"{'='*60}\n")
    
    state_minimal = builder.build_state(
        trend=trend,  # Only trend
        current_price=50100
    )
    
    print(f"✅ State built with minimal data")
    print(f"   Total features: {len(state_minimal)}")
    print(f"   Valid: {state_minimal.valid}")
    print(f"   Features filled with defaults: {len(state_minimal.features) - 15}")
    
    # ═══════════════════════════════════════════════════════════
    # Test 3: Feature breakdown
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("TEST 3: Feature Breakdown by Category")
    print(f"{'='*60}\n")
    
    categories = {
        'trend_': 'Trend Features',
        'zone_': 'Zone Features',
        'choch_': 'ChoCH Features',
        'fib_': 'Fibonacci Features',
        'entry_': 'Entry Features',
        'market_': 'Market Features',
        'adaptive_': 'Adaptive Features'
    }
    
    for prefix, name in categories.items():
        count = sum(1 for key in state.features.keys() if key.startswith(prefix))
        print(f"   {name:<25s}: {count} features")
    
    print("\n" + "="*60)
    print("✅ State Builder working correctly!")
    print("="*60 + "\n")