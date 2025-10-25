"""
Core PA Detection & Entry/Exit Modules - Parça 1 & 2
Bu modül, Price Action (PA) tabanlı sistemin ana bileşenlerini dışa aktarır.

Parçalar:
1️⃣ Price Action Detection (trend & zone)
2️⃣ Entry/Exit System (ChoCH, Fibonacci, Exit)
"""

# ────────────────────────────────
# Parça 1: Price Action Detection
# ────────────────────────────────
from .trend_detector import TrendDetector, TrendResult
from .zone_detector import ZoneDetector, Zone, ZigZagPoint
from .zone_quality_scorer import ZoneQualityScorer

# ────────────────────────────────
# Parça 2: Entry & Exit System
# ────────────────────────────────
from .choch_detector import ChoCHDetector, ChoCHResult, SwingPoint
from .fibonacci_calculator import FibonacciCalculator, FibonacciLevels
from .entry_system import EntrySystem, EntrySignal
from .exit_system import ExitSystem, ExitSignal, PositionState, TakeProfitLevel

__all__ = [
    # Parça 1
    "TrendDetector", "TrendResult",
    "ZoneDetector", "Zone", "ZigZagPoint",
    "ZoneQualityScorer",
    
    # Parça 2
    "ChoCHDetector", "ChoCHResult", "SwingPoint",
    "FibonacciCalculator", "FibonacciLevels",
    "EntrySystem", "EntrySignal",
    "ExitSystem", "ExitSignal", "PositionState", "TakeProfitLevel",
]

__version__ = "2.0.0"

# Bu versiyon doğru! ✅
# core klasörü toplam 9 dosyadan oluşur:
#  - 8 ana modül (Parça 1 + Parça 2)
#  - 1 init.py
#
# Yapı:
# core/
# ├── __init__.py
# ├── trend_detector.py
# ├── zone_detector.py
# ├── zone_quality_scorer.py
# ├── choch_detector.py
# ├── fibonacci_calculator.py
# ├── entry_system.py
# └── exit_system.py
#
# Sonraki adım: Parça 3 (adaptive/) klasörünün eklenmesi
