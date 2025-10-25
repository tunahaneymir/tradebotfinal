"""
Indicator Learning Manager
Based on: pa-strateji3 Parça 9

Features:
- Dual confirmation (ZigZag++ + Swing HL)
- Parallel testing (3 methods)
- Coin-specific adaptive selection
- Performance tracking
"""

from __future__ import annotations
from typing import List, Dict, Literal, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


@dataclass
class ZoneDetectionMethod:
    """Zone detection yöntemi"""
    name: Literal["zigzag_only", "swing_only", "both"]
    total_zones: int = 0
    successful_zones: int = 0
    failed_zones: int = 0
    win_rate: float = 0.0
    avg_rr: float = 0.0
    avg_quality: float = 0.0


@dataclass
class CoinPreference:
    """Coin için tercih edilen method"""
    coin: str
    preferred_method: Literal["zigzag_only", "swing_only", "both"]
    confidence: float = 0.5  # 0-1 arası güven
    total_tests: int = 0
    last_update: str = ""
    
    # Method performance
    zigzag_wr: float = 0.0
    swing_wr: float = 0.0
    both_wr: float = 0.0


class IndicatorLearningManager:
    """
    Indicator Learning System
    
    Strategy: DUAL CONFIRMATION (ZigZag++ primary, Swing HL confirmation)
    
    Learning Phases:
    - Phase 1 (0-3m): DUAL (both) ile güvenli başlangıç, veri topla
    - Phase 2 (3-6m): 3 yöntemi paralel test et, coin-bazlı seçim
    - Phase 3 (6+m): Coin/timeframe bazlı otomatik en iyiyi kullan
    
    Implementation: ZoneDetectionManager sınıfı
    """
    
    def __init__(self, data_dir: str = "state"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.preferences_file = self.data_dir / "coin_preferences.json"
        self.performance_file = self.data_dir / "method_performance.json"
        
        # Learning phase
        self.phase: Literal[1, 2, 3] = 1
        self.phase_start_date: Optional[datetime] = None
        
        # Coin preferences (RL öğrenecek)
        self.coin_preferences: Dict[str, CoinPreference] = {}
        
        # Method performance tracking
        self.method_performance: Dict[str, ZoneDetectionMethod] = {
            "zigzag_only": ZoneDetectionMethod(name="zigzag_only"),
            "swing_only": ZoneDetectionMethod(name="swing_only"),
            "both": ZoneDetectionMethod(name="both")
        }
        
        # Load existing data
        self._load_state()
        
        print(f"🧠 Indicator Learning Manager initialized")
        print(f"   Phase: {self.phase}")
        print(f"   Coins tracked: {len(self.coin_preferences)}")
    
    # ═══════════════════════════════════════════════════════════
    # ZONE DETECTION
    # ═══════════════════════════════════════════════════════════
    
    def detect_zones(self, coin: str, ohlcv_data: Dict, timeframe: str = "4H") -> Dict:
        """
        Zone detection - phase'e göre farklı method kullan
        
        Args:
            coin: Coin sembolü
            ohlcv_data: OHLCV data
            timeframe: Zaman dilimi
            
        Returns:
            Phase 1: {method: "both", zones: [...]}
            Phase 2: {testing_mode: True, zigzag: [...], swing: [...], both: [...]}
            Phase 3: {method: "best", zones: [...]}
        """
        if self.phase == 1:
            # Phase 1: Her zaman "both" kullan
            return {
                "method": "both",
                "zones": self._detect_with_both(ohlcv_data, timeframe),
                "phase": 1
            }
        
        elif self.phase == 2:
            # Phase 2: Tüm yöntemleri paralel test et
            return {
                "testing_mode": True,
                "zigzag": self._detect_with_zigzag(ohlcv_data, timeframe),
                "swing": self._detect_with_swing(ohlcv_data, timeframe),
                "both": self._detect_with_both(ohlcv_data, timeframe),
                "phase": 2,
                "note": "All methods tested in parallel"
            }
        
        else:  # Phase 3
            # Phase 3: Coin için en iyi method'u kullan
            preferred_method = self._get_preferred_method(coin)
            
            if preferred_method == "zigzag_only":
                zones = self._detect_with_zigzag(ohlcv_data, timeframe)
            elif preferred_method == "swing_only":
                zones = self._detect_with_swing(ohlcv_data, timeframe)
            else:
                zones = self._detect_with_both(ohlcv_data, timeframe)
            
            return {
                "method": preferred_method,
                "zones": zones,
                "phase": 3,
                "confidence": self.coin_preferences.get(coin, CoinPreference(coin=coin, preferred_method="both")).confidence
            }
    
    def _detect_with_zigzag(self, ohlcv_data: Dict, timeframe: str) -> List[Dict]:
        """ZigZag++ only detection (implement edilecek)"""
        # Bu gerçek ZigZag++ implementasyonu ile değiştirilecek
        return [
            {"type": "zigzag", "price": 50000, "quality": 7.5}
        ]
    
    def _detect_with_swing(self, ohlcv_data: Dict, timeframe: str) -> List[Dict]:
        """Swing HL only detection (implement edilecek)"""
        # Bu gerçek Swing implementasyonu ile değiştirilecek
        return [
            {"type": "swing", "price": 50100, "quality": 6.8}
        ]
    
    def _detect_with_both(self, ohlcv_data: Dict, timeframe: str) -> List[Dict]:
        """ZigZag++ AND Swing (intersection) detection (implement edilecek)"""
        # ZigZag ve Swing'in kesişim noktaları
        zigzag_zones = self._detect_with_zigzag(ohlcv_data, timeframe)
        swing_zones = self._detect_with_swing(ohlcv_data, timeframe)
        
        # Intersection logic (basitleştirilmiş)
        return [
            {"type": "both", "price": 50050, "quality": 8.2}
        ]
    
    # ═══════════════════════════════════════════════════════════
    # LEARNING & PERFORMANCE TRACKING
    # ═══════════════════════════════════════════════════════════
    
    def record_zone_outcome(self, coin: str, method: str, zone_id: str, 
                           success: bool, rr: float, quality: float):
        """
        Zone sonucunu kaydet ve öğren
        
        Args:
            coin: Coin sembolü
            method: Kullanılan method ("zigzag_only", "swing_only", "both")
            zone_id: Zone ID
            success: Trade başarılı mı?
            rr: Risk-reward ratio
            quality: Zone quality score
        """
        # Method performance güncelle
        if method in self.method_performance:
            perf = self.method_performance[method]
            perf.total_zones += 1
            
            if success:
                perf.successful_zones += 1
            else:
                perf.failed_zones += 1
            
            # Win rate hesapla
            perf.win_rate = perf.successful_zones / perf.total_zones if perf.total_zones > 0 else 0
            
            # Averages güncelle
            perf.avg_rr = ((perf.avg_rr * (perf.total_zones - 1)) + rr) / perf.total_zones
            perf.avg_quality = ((perf.avg_quality * (perf.total_zones - 1)) + quality) / perf.total_zones
        
        # Coin preference güncelle (Phase 2 ve 3 için)
        if self.phase >= 2:
            self._update_coin_preference(coin, method, success)
        
        # Save state
        self._save_state()
        
        print(f"📊 Zone outcome recorded: {coin} ({method}) - {'WIN' if success else 'LOSS'}")
    
    def _update_coin_preference(self, coin: str, method: str, success: bool):
        """Coin için method tercihini güncelle"""
        if coin not in self.coin_preferences:
            self.coin_preferences[coin] = CoinPreference(
                coin=coin,
                preferred_method="both"
            )
        
        pref = self.coin_preferences[coin]
        pref.total_tests += 1
        pref.last_update = datetime.now().isoformat()
        
        # Method-specific win rate güncelle
        if method == "zigzag_only":
            pref.zigzag_wr = self.method_performance["zigzag_only"].win_rate
        elif method == "swing_only":
            pref.swing_wr = self.method_performance["swing_only"].win_rate
        elif method == "both":
            pref.both_wr = self.method_performance["both"].win_rate
        
        # En iyi method'u belirle (minimum 10 test sonrası)
        if pref.total_tests >= 10:
            best_method = max(
                [("zigzag_only", pref.zigzag_wr),
                 ("swing_only", pref.swing_wr),
                 ("both", pref.both_wr)],
                key=lambda x: x[1]
            )[0]
            
            pref.preferred_method = best_method
            pref.confidence = max(pref.zigzag_wr, pref.swing_wr, pref.both_wr)
            
            print(f"✅ {coin} preference updated: {best_method} (confidence: {pref.confidence:.2f})")
    
    def _get_preferred_method(self, coin: str) -> Literal["zigzag_only", "swing_only", "both"]:
        """Coin için tercih edilen method'u döndür"""
        if coin in self.coin_preferences:
            return self.coin_preferences[coin].preferred_method
        else:
            # Default: both (güvenli seçenek)
            return "both"
    
    # ═══════════════════════════════════════════════════════════
    # PHASE MANAGEMENT
    # ═══════════════════════════════════════════════════════════
    
    def advance_phase(self):
        """Bir sonraki phase'e geç"""
        if self.phase < 3:
            self.phase += 1
            self.phase_start_date = datetime.now()
            self._save_state()
            
            print(f"\n{'='*60}")
            print(f"🎓 PHASE {self.phase} ACTIVATED")
            print(f"{'='*60}")
            
            if self.phase == 2:
                print("Phase 2: Parallel testing mode")
                print("- All 3 methods will be tested")
                print("- Performance data will be collected")
                print("- Coin-specific preferences will be built")
            elif self.phase == 3:
                print("Phase 3: Adaptive mode")
                print("- Best method per coin will be used")
                print("- Automatic optimization")
                print(f"- {len(self.coin_preferences)} coins have preferences")
            
            print(f"{'='*60}\n")
    
    def check_phase_advancement(self, months_since_start: float) -> bool:
        """
        Phase ilerlemesi gerekli mi kontrol et
        
        Args:
            months_since_start: Başlangıçtan itibaren geçen ay sayısı
            
        Returns:
            True: Phase ilerledi
        """
        if self.phase == 1 and months_since_start >= 3:
            print("⏰ 3 months passed - Advancing to Phase 2")
            self.advance_phase()
            return True
        
        elif self.phase == 2 and months_since_start >= 6:
            print("⏰ 6 months passed - Advancing to Phase 3")
            self.advance_phase()
            return True
        
        return False
    
    # ═══════════════════════════════════════════════════════════
    # ANALYTICS & REPORTING
    # ═══════════════════════════════════════════════════════════
    
    def get_performance_summary(self) -> Dict:
        """Tüm method'ların performans özeti"""
        return {
            "phase": self.phase,
            "methods": {
                name: {
                    "total_zones": perf.total_zones,
                    "win_rate": perf.win_rate * 100,
                    "avg_rr": perf.avg_rr,
                    "avg_quality": perf.avg_quality
                }
                for name, perf in self.method_performance.items()
            },
            "best_method": self._get_best_global_method(),
            "coins_tracked": len(self.coin_preferences)
        }
    
    def get_coin_summary(self, coin: str) -> Optional[Dict]:
        """Belirli bir coin için özet"""
        if coin not in self.coin_preferences:
            return None
        
        pref = self.coin_preferences[coin]
        return {
            "coin": coin,
            "preferred_method": pref.preferred_method,
            "confidence": pref.confidence,
            "total_tests": pref.total_tests,
            "win_rates": {
                "zigzag": pref.zigzag_wr * 100,
                "swing": pref.swing_wr * 100,
                "both": pref.both_wr * 100
            },
            "last_update": pref.last_update
        }
    
    def _get_best_global_method(self) -> str:
        """Global olarak en iyi performans gösteren method"""
        best = max(
            self.method_performance.items(),
            key=lambda x: x[1].win_rate
        )
        return best[0]
    
    def print_report(self):
        """Detaylı rapor yazdır"""
        print("\n" + "="*60)
        print("     INDICATOR LEARNING PERFORMANCE REPORT")
        print("="*60)
        
        print(f"\nCurrent Phase: {self.phase}")
        
        print("\n📊 METHOD PERFORMANCE:")
        for name, perf in self.method_performance.items():
            print(f"\n  {name.upper().replace('_', ' ')}:")
            print(f"    Total Zones: {perf.total_zones}")
            print(f"    Win Rate: {perf.win_rate*100:.1f}%")
            print(f"    Avg R:R: {perf.avg_rr:.2f}")
            print(f"    Avg Quality: {perf.avg_quality:.1f}/10")
        
        print(f"\n🏆 Best Method: {self._get_best_global_method().upper()}")
        
        if self.coin_preferences:
            print(f"\n💰 TOP COINS ({min(5, len(self.coin_preferences))}):")
            sorted_coins = sorted(
                self.coin_preferences.items(),
                key=lambda x: x[1].confidence,
                reverse=True
            )[:5]
            
            for i, (coin, pref) in enumerate(sorted_coins, 1):
                print(f"  {i}. {coin}: {pref.preferred_method} (confidence: {pref.confidence:.2f})")
        
        print("\n" + "="*60 + "\n")
    
    # ═══════════════════════════════════════════════════════════
    # STATE PERSISTENCE
    # ═══════════════════════════════════════════════════════════
    
    def _save_state(self):
        """State'i kaydet"""
        # Coin preferences
        preferences_data = {
            coin: {
                "preferred_method": pref.preferred_method,
                "confidence": pref.confidence,
                "total_tests": pref.total_tests,
                "last_update": pref.last_update,
                "zigzag_wr": pref.zigzag_wr,
                "swing_wr": pref.swing_wr,
                "both_wr": pref.both_wr
            }
            for coin, pref in self.coin_preferences.items()
        }
        
        with open(self.preferences_file, 'w') as f:
            json.dump(preferences_data, f, indent=2)
        
        # Method performance
        performance_data = {
            "phase": self.phase,
            "phase_start_date": self.phase_start_date.isoformat() if self.phase_start_date else None,
            "methods": {
                name: {
                    "total_zones": perf.total_zones,
                    "successful_zones": perf.successful_zones,
                    "failed_zones": perf.failed_zones,
                    "win_rate": perf.win_rate,
                    "avg_rr": perf.avg_rr,
                    "avg_quality": perf.avg_quality
                }
                for name, perf in self.method_performance.items()
            }
        }
        
        with open(self.performance_file, 'w') as f:
            json.dump(performance_data, f, indent=2)
    
    def _load_state(self):
        """State'i yükle"""
        # Coin preferences
        if self.preferences_file.exists():
            with open(self.preferences_file) as f:
                preferences_data = json.load(f)
            
            self.coin_preferences = {
                coin: CoinPreference(
                    coin=coin,
                    preferred_method=data["preferred_method"],
                    confidence=data["confidence"],
                    total_tests=data["total_tests"],
                    last_update=data["last_update"],
                    zigzag_wr=data["zigzag_wr"],
                    swing_wr=data["swing_wr"],
                    both_wr=data["both_wr"]
                )
                for coin, data in preferences_data.items()
            }
        
        # Method performance
        if self.performance_file.exists():
            with open(self.performance_file) as f:
                performance_data = json.load(f)
            
            self.phase = performance_data.get("phase", 1)
            
            if performance_data.get("phase_start_date"):
                self.phase_start_date = datetime.fromisoformat(performance_data["phase_start_date"])
            
            for name, data in performance_data.get("methods", {}).items():
                if name in self.method_performance:
                    perf = self.method_performance[name]
                    perf.total_zones = data["total_zones"]
                    perf.successful_zones = data["successful_zones"]
                    perf.failed_zones = data["failed_zones"]
                    perf.win_rate = data["win_rate"]
                    perf.avg_rr = data["avg_rr"]
                    perf.avg_quality = data["avg_quality"]


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Manager oluştur
    manager = IndicatorLearningManager()
    
    # Phase 1: Zone detection (both method kullanılır)
    print("\n🔍 Phase 1 - Detecting zones for BTCUSDT...")
    result1 = manager.detect_zones("BTCUSDT", {}, "4H")
    print(f"Method: {result1['method']}")
    print(f"Zones found: {len(result1['zones'])}")
    
    # Zone sonucu kaydet
    manager.record_zone_outcome(
        coin="BTCUSDT",
        method="both",
        zone_id="zone_001",
        success=True,
        rr=2.5,
        quality=8.2
    )
    
    # Daha fazla test
    for i in range(20):
        success = i % 3 != 0  # %66 win rate simulate
        manager.record_zone_outcome(
            coin="BTCUSDT",
            method="both",
            zone_id=f"zone_{i:03d}",
            success=success,
            rr=2.0 if success else -1.0,
            quality=7.5
        )
    
    # Phase 2'ye geç
    print("\n⏭️  Advancing to Phase 2...")
    manager.advance_phase()
    
    # Phase 2: Parallel testing
    print("\n🔍 Phase 2 - Testing all methods for ETHUSDT...")
    result2 = manager.detect_zones("ETHUSDT", {}, "4H")
    print(f"Testing mode: {result2.get('testing_mode')}")
    print(f"Methods tested: {list(result2.keys())}")
    
    # Her method için sonuç kaydet
    for method in ["zigzag_only", "swing_only", "both"]:
        for i in range(10):
            # Simulate: both > zigzag > swing
            if method == "both":
                success = i < 7  # 70% WR
            elif method == "zigzag_only":
                success = i < 6  # 60% WR
            else:
                success = i < 5  # 50% WR
            
            manager.record_zone_outcome(
                coin="ETHUSDT",
                method=method,
                zone_id=f"{method}_{i}",
                success=success,
                rr=2.0,
                quality=7.0
            )
    
    # Phase 3'e geç
    print("\n⏭️  Advancing to Phase 3...")
    manager.advance_phase()
    
    # Phase 3: Adaptive selection
    print("\n🔍 Phase 3 - Using best method for ETHUSDT...")
    result3 = manager.detect_zones("ETHUSDT", {}, "4H")
    print(f"Method: {result3['method']}")
    print(f"Confidence: {result3['confidence']:.2f}")
    
    # Rapor
    manager.print_report()
    
    # Summary
    summary = manager.get_performance_summary()
    print("\n📊 Performance Summary:")
    print(f"Phase: {summary['phase']}")
    print(f"Best Method: {summary['best_method']}")
    print(f"Coins Tracked: {summary['coins_tracked']}")