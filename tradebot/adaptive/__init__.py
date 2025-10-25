"""
Adaptive Systems - Parça 3

This module contains adaptive and learning systems:
- Re-entry Management (Stop loss sonrası 2. deneme)
- Adaptive Parameters (ATR-based parameter adjustment)
- Liquidity Detection (Wick sweep levels for TP2)
- Zone Memory (Performance tracking & blacklist)

Usage:
    from adaptive import (
        ReentryManager,
        AdaptiveParameterCalculator,
        LiquidityDetector,
        ZoneMemoryManager
    )
    
    # Initialize systems
    reentry_mgr = ReentryManager(config)
    adaptive_calc = AdaptiveParameterCalculator(config)
    liquidity_det = LiquidityDetector(config)
    zone_memory = ZoneMemoryManager()
    
    # Check re-entry eligibility
    eligibility = reentry_mgr.check_reentry_eligibility(...)
    
    # Calculate adaptive parameters
    params = adaptive_calc.calculate(high, low, close, timeframe="1H")
    
    # Detect liquidity levels
    levels = liquidity_det.detect_liquidity(...)
    
    # Track zone performance
    zone_memory.record_trade(zone_id, trade_record)
"""

# Re-entry Management
from .reentry_manager import (
    ReentryManager,
    ReentryEligibility,
    TradeHistory
)

# Adaptive Parameters
from .adaptive_parameters import (
    AdaptiveParameterCalculator,
    AdaptiveParams
)

# Liquidity Detection
from .liquidity_detector import (
    LiquidityDetector,
    LiquidityLevel
)

# Zone Memory
from .zone_memory import (
    ZoneMemoryManager,
    ZoneMemory,
    ZoneTradeRecord,
    ZoneStatistics
)

__all__ = [
    # ═══════════════════════════════════════════════════════════
    # Re-entry Management
    # ═══════════════════════════════════════════════════════════
    'ReentryManager',
    'ReentryEligibility',
    'TradeHistory',
    
    # ═══════════════════════════════════════════════════════════
    # Adaptive Parameters
    # ═══════════════════════════════════════════════════════════
    'AdaptiveParameterCalculator',
    'AdaptiveParams',
    
    # ═══════════════════════════════════════════════════════════
    # Liquidity Detection
    # ═══════════════════════════════════════════════════════════
    'LiquidityDetector',
    'LiquidityLevel',
    
    # ═══════════════════════════════════════════════════════════
    # Zone Memory
    # ═══════════════════════════════════════════════════════════
    'ZoneMemoryManager',
    'ZoneMemory',
    'ZoneTradeRecord',
    'ZoneStatistics',
]

__version__ = '3.0.0'  # Parça 3