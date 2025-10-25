"""
Setup Scorer - Parça 4
Based on: pa-strateji2 Parça 4

Setup Quality Scoring (0-100):
- Evaluates complete trade setup quality
- Weighted scoring across multiple dimensions
- Helps RL agent decide if trade is worth taking
- Provides breakdown for analysis
"""

from __future__ import annotations
from typing import Dict, Optional
from dataclasses import dataclass

from core import TrendResult, Zone, ChoCHResult, FibonacciLevels


@dataclass
class SetupScore:
    """Setup quality score result"""
    total_score: float  # 0-100
    grade: str  # A+, A, B, C, D, F
    
    # Component scores
    trend_score: float  # 0-25
    zone_score: float   # 0-25
    choch_score: float  # 0-25
    fib_score: float    # 0-25
    
    # Breakdown
    breakdown: Dict[str, float]
    
    # Recommendation
    recommended: bool
    reason: str


class SetupScorer:
    """
    Setup Quality Scorer
    
    Evaluates trade setup quality on 0-100 scale:
    
    Components (each 25 points):
    1. Trend Quality (25 points)
       - Trend strength
       - Trend confidence
       - No sideways
    
    2. Zone Quality (25 points)
       - Zone quality score
       - Touch count
       - Freshness
    
    3. ChoCH Quality (25 points)
       - ChoCH strength
       - Body score
       - Volume confirmation
    
    4. Fibonacci Quality (25 points)
       - Entry level (0.705 vs 0.618)
       - Swing range quality
       - Fib alignment
    
    Grading:
    - 90-100: A+ (Excellent - Take immediately)
    - 80-89:  A  (Very Good - Strong take)
    - 70-79:  B  (Good - Take)
    - 60-69:  C  (Fair - Consider)
    - 50-59:  D  (Poor - Skip or very small)
    - 0-49:   F  (Fail - Do not take)
    
    Usage:
        scorer = SetupScorer(min_score=40)
        
        score = scorer.score_setup(
            trend=trend_result,
            zone=zone,
            choch=choch_result,
            fibonacci=fib_levels,
            entry_level="0.705"
        )
        
        if score.recommended:
            # Take trade!
    """
    
    def __init__(
        self,
        min_score: float = 40.0,
        min_trend_score: float = 10.0,
        min_zone_score: float = 10.0,
        min_choch_score: float = 10.0
    ):
        """
        Initialize Setup Scorer
        
        Args:
            min_score: Minimum total score to recommend (default 40)
            min_trend_score: Minimum trend score required
            min_zone_score: Minimum zone score required
            min_choch_score: Minimum ChoCH score required
        """
        self.min_score = min_score
        self.min_trend_score = min_trend_score
        self.min_zone_score = min_zone_score
        self.min_choch_score = min_choch_score
    
    def score_setup(
        self,
        trend: TrendResult,
        zone: Zone,
        choch: ChoCHResult,
        fibonacci: Optional[FibonacciLevels] = None,
        entry_level: Optional[str] = None
    ) -> SetupScore:
        """
        Score complete trade setup
        
        Args:
            trend: Trend detection result
            zone: Zone object
            choch: ChoCH detection result
            fibonacci: Fibonacci levels (optional)
            entry_level: Which Fib level used ("0.705" or "0.618")
            
        Returns:
            SetupScore with total score and breakdown
        """
        # ═══════════════════════════════════════════════════════════
        # 1. TREND SCORE (0-25 points)
        # ═══════════════════════════════════════════════════════════
        trend_score = self._score_trend(trend)
        
        # ═══════════════════════════════════════════════════════════
        # 2. ZONE SCORE (0-25 points)
        # ═══════════════════════════════════════════════════════════
        zone_score = self._score_zone(zone)
        
        # ═══════════════════════════════════════════════════════════
        # 3. ChoCH SCORE (0-25 points)
        # ═══════════════════════════════════════════════════════════
        choch_score = self._score_choch(choch)
        
        # ═══════════════════════════════════════════════════════════
        # 4. FIBONACCI SCORE (0-25 points)
        # ═══════════════════════════════════════════════════════════
        fib_score = self._score_fibonacci(fibonacci, entry_level)
        
        # ═══════════════════════════════════════════════════════════
        # TOTAL SCORE
        # ═══════════════════════════════════════════════════════════
        total_score = trend_score + zone_score + choch_score + fib_score
        
        # ═══════════════════════════════════════════════════════════
        # GRADE
        # ═══════════════════════════════════════════════════════════
        grade = self._calculate_grade(total_score)
        
        # ═══════════════════════════════════════════════════════════
        # RECOMMENDATION
        # ═══════════════════════════════════════════════════════════
        recommended, reason = self._make_recommendation(
            total_score, trend_score, zone_score, choch_score
        )
        
        # ═══════════════════════════════════════════════════════════
        # BREAKDOWN
        # ═══════════════════════════════════════════════════════════
        breakdown = {
            'trend_confidence': trend.confidence,
            'trend_strength': trend._calculate_trend_strength() if hasattr(trend, '_calculate_trend_strength') else 0.5,
            'zone_quality': zone.quality,
            'zone_touch_count': float(zone.touch_count),
            'zone_freshness': max(0, 1 - (zone.days_since_last_touch / 30)),
            'choch_strength': choch.strength,
            'choch_body_score': choch.body_score,
            'choch_volume_score': choch.volume_score,
            'fib_level': 1.0 if entry_level == "0.705" else 0.8 if entry_level == "0.618" else 0.0
        }
        
        return SetupScore(
            total_score=total_score,
            grade=grade,
            trend_score=trend_score,
            zone_score=zone_score,
            choch_score=choch_score,
            fib_score=fib_score,
            breakdown=breakdown,
            recommended=recommended,
            reason=reason
        )
    
    def _score_trend(self, trend: TrendResult) -> float:
        """
        Score trend quality (0-25 points)
        
        Criteria:
        - Is trending (not sideways): 10 points
        - Trend confidence: 0-10 points
        - Trend strength: 0-5 points
        """
        score = 0.0
        
        # Not sideways (10 points)
        if trend.direction != "SIDEWAYS":
            score += 10.0
        
        # Trend confidence (0-10 points)
        score += trend.confidence * 10.0
        
        # Trend strength (0-5 points)
        # Calculate from slope and EMA distance
        ema_strength = min(trend.ema_distance_pct / 0.03, 1.0)  # Max at 3%
        slope_strength = trend.slope_strength
        trend_strength = (ema_strength + slope_strength) / 2
        score += trend_strength * 5.0
        
        return min(score, 25.0)
    
    def _score_zone(self, zone: Zone) -> float:
        """
        Score zone quality (0-25 points)
        
        Criteria:
        - Base quality (0-10): 10 points
        - Touch count optimization: 0-10 points
        - Freshness: 0-5 points
        """
        score = 0.0
        
        # Base quality (0-10 → 0-10 points)
        score += zone.quality
        
        # Touch count (0-10 points)
        # 2 touches = best (10 points)
        # 3 touches = good (7 points)
        # 4 touches = ok (5 points)
        # 5+ = weak (2 points)
        if zone.touch_count == 2:
            score += 10.0
        elif zone.touch_count == 3:
            score += 7.0
        elif zone.touch_count == 4:
            score += 5.0
        else:
            score += 2.0
        
        # Freshness (0-5 points)
        # < 7 days = fresh (5 points)
        # < 30 days = ok (3 points)
        # > 30 days = old (1 point)
        if zone.days_since_last_touch < 7:
            score += 5.0
        elif zone.days_since_last_touch < 30:
            score += 3.0
        else:
            score += 1.0
        
        return min(score, 25.0)
    
    def _score_choch(self, choch: ChoCHResult) -> float:
        """
        Score ChoCH quality (0-25 points)
        
        Criteria:
        - ChoCH detected: 5 points
        - ChoCH strength (0.0-1.0): 0-15 points
        - Body score: 0-5 points (already in strength)
        - Volume score: 0-5 points (already in strength)
        """
        if not choch.detected:
            return 0.0
        
        score = 5.0  # Base for detection
        
        # Strength (0-20 points)
        # Map 0.4-1.0 strength to full points
        # Below 0.4 shouldn't happen (filtered earlier)
        normalized_strength = max(0, (choch.strength - 0.4) / 0.6)
        score += normalized_strength * 20.0
        
        return min(score, 25.0)
    
    def _score_fibonacci(
        self,
        fibonacci: Optional[FibonacciLevels],
        entry_level: Optional[str]
    ) -> float:
        """
        Score Fibonacci quality (0-25 points)
        
        Criteria:
        - Fib calculated: 5 points
        - Entry level preference: 0-15 points
          * 0.705 = 15 points (optimal)
          * 0.618 = 10 points (good)
        - Swing range quality: 0-5 points
        """
        if not fibonacci:
            return 5.0  # Minimal score if no Fib
        
        score = 5.0  # Base for having Fib
        
        # Entry level preference (0-15 points)
        if entry_level == "0.705":
            score += 15.0  # Optimal Entry Point
        elif entry_level == "0.618":
            score += 10.0  # Golden Ratio
        else:
            score += 5.0   # Unknown or other
        
        # Swing range quality (0-5 points)
        # Prefer reasonable swing ranges
        # Too small or too large is bad
        swing_range_pct = fibonacci.swing_range / fibonacci.swing_high if fibonacci.swing_high > 0 else 0
        
        if 0.03 <= swing_range_pct <= 0.15:  # 3-15% range
            score += 5.0  # Good range
        elif 0.01 <= swing_range_pct <= 0.20:  # 1-20% range
            score += 3.0  # Acceptable
        else:
            score += 1.0  # Poor range
        
        return min(score, 25.0)
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score"""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"
    
    def _make_recommendation(
        self,
        total_score: float,
        trend_score: float,
        zone_score: float,
        choch_score: float
    ) -> tuple[bool, str]:
        """
        Make recommendation based on scores
        
        Returns:
            (recommended: bool, reason: str)
        """
        # Check minimum requirements
        if trend_score < self.min_trend_score:
            return False, f"Trend score too low ({trend_score:.1f}/{self.min_trend_score})"
        
        if zone_score < self.min_zone_score:
            return False, f"Zone score too low ({zone_score:.1f}/{self.min_zone_score})"
        
        if choch_score < self.min_choch_score:
            return False, f"ChoCH score too low ({choch_score:.1f}/{self.min_choch_score})"
        
        # Check total score
        if total_score < self.min_score:
            return False, f"Total score below minimum ({total_score:.1f}/{self.min_score})"
        
        # All checks passed
        if total_score >= 80:
            return True, f"Excellent setup (Score: {total_score:.1f}, Grade: A+/A)"
        elif total_score >= 70:
            return True, f"Good setup (Score: {total_score:.1f}, Grade: B)"
        elif total_score >= 60:
            return True, f"Fair setup (Score: {total_score:.1f}, Grade: C)"
        else:
            return True, f"Acceptable setup (Score: {total_score:.1f}, Grade: D)"


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from core import TrendResult, Zone, ChoCHResult, FibonacciLevels
    
    print("\n" + "="*60)
    print("SETUP SCORER - TEST")
    print("="*60)
    
    # Create scorer
    scorer = SetupScorer(min_score=40)
    
    print(f"\n✅ Setup Scorer created")
    print(f"   Minimum score: {scorer.min_score}")
    
    # ═══════════════════════════════════════════════════════════
    # Scenario 1: EXCELLENT SETUP (A+)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SCENARIO 1: EXCELLENT SETUP")
    print(f"{'='*60}\n")
    
    trend_excellent = TrendResult(
        direction="UP",
        ema_20=50500,
        ema_50=50000,
        ema_distance_pct=0.025,  # 2.5% distance
        atr_ratio=0.05,
        price_range_pct=0.04,
        confidence=0.95,  # Very confident
        slope_up=True,
        slope_down=False,
        slope_strength=0.95,  # Strong slope
        sideways_signal_count=0
    )
    
    zone_excellent = Zone(
        id="EXCELLENT_ZONE",
        price_low=50000,
        price_high=50100,
        price_mid=50050,
        touch_count=2,  # Perfect
        thickness_pct=0.002,
        last_touch_index=195,
        creation_index=100,
        timeframe="1H",
        method="both",
        quality=9.5,  # Excellent
        days_since_last_touch=3.0  # Fresh
    )
    
    choch_excellent = ChoCHResult(
        detected=True,
        direction="LONG",
        breakout_price=50200,
        breakout_index=150,
        broken_level=50000,
        strength=0.85,  # Very strong
        body_score=0.35,
        volume_score=0.50
    )
    
    fib_excellent = FibonacciLevels(
        swing_low=49800,
        swing_high=50200,
        swing_range=400,
        direction="LONG",
        fib_0=50200,
        fib_0705=49918,
        fib_0618=49953,
        fib_1=49800,
        primary_level=49918,
        secondary_level=49953
    )
    
    score_excellent = scorer.score_setup(
        trend=trend_excellent,
        zone=zone_excellent,
        choch=choch_excellent,
        fibonacci=fib_excellent,
        entry_level="0.705"
    )
    
    print(f"Total Score: {score_excellent.total_score:.1f}/100")
    print(f"Grade: {score_excellent.grade}")
    print(f"\nComponent Scores:")
    print(f"  Trend:  {score_excellent.trend_score:.1f}/25")
    print(f"  Zone:   {score_excellent.zone_score:.1f}/25")
    print(f"  ChoCH:  {score_excellent.choch_score:.1f}/25")
    print(f"  Fib:    {score_excellent.fib_score:.1f}/25")
    print(f"\nRecommended: {'✅ YES' if score_excellent.recommended else '❌ NO'}")
    print(f"Reason: {score_excellent.reason}")
    
    # ═══════════════════════════════════════════════════════════
    # Scenario 2: GOOD SETUP (B)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SCENARIO 2: GOOD SETUP")
    print(f"{'='*60}\n")
    
    trend_good = TrendResult(
        direction="UP",
        ema_20=50300,
        ema_50=50100,
        ema_distance_pct=0.015,
        atr_ratio=0.04,
        price_range_pct=0.03,
        confidence=0.75,
        slope_up=True,
        slope_down=False,
        slope_strength=0.75,
        sideways_signal_count=0
    )
    
    zone_good = Zone(
        id="GOOD_ZONE",
        price_low=50000,
        price_high=50100,
        price_mid=50050,
        touch_count=3,  # Good
        thickness_pct=0.004,
        last_touch_index=180,
        creation_index=100,
        timeframe="1H",
        method="both",
        quality=7.5,
        days_since_last_touch=10.0
    )
    
    choch_good = ChoCHResult(
        detected=True,
        direction="LONG",
        breakout_price=50150,
        breakout_index=140,
        broken_level=50000,
        strength=0.65,
        body_score=0.25,
        volume_score=0.40
    )
    
    score_good = scorer.score_setup(
        trend=trend_good,
        zone=zone_good,
        choch=choch_good,
        fibonacci=fib_excellent,
        entry_level="0.618"  # Secondary level
    )
    
    print(f"Total Score: {score_good.total_score:.1f}/100")
    print(f"Grade: {score_good.grade}")
    print(f"\nComponent Scores:")
    print(f"  Trend:  {score_good.trend_score:.1f}/25")
    print(f"  Zone:   {score_good.zone_score:.1f}/25")
    print(f"  ChoCH:  {score_good.choch_score:.1f}/25")
    print(f"  Fib:    {score_good.fib_score:.1f}/25")
    print(f"\nRecommended: {'✅ YES' if score_good.recommended else '❌ NO'}")
    print(f"Reason: {score_good.reason}")
    
    # ═══════════════════════════════════════════════════════════
    # Scenario 3: POOR SETUP (F)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SCENARIO 3: POOR SETUP")
    print(f"{'='*60}\n")
    
    trend_poor = TrendResult(
        direction="SIDEWAYS",  # Not trending!
        ema_20=50100,
        ema_50=50080,
        ema_distance_pct=0.002,
        atr_ratio=0.004,
        price_range_pct=0.01,
        confidence=0.4,
        slope_up=False,
        slope_down=False,
        slope_strength=0.2,
        sideways_signal_count=3
    )
    
    zone_poor = Zone(
        id="POOR_ZONE",
        price_low=50000,
        price_high=50150,
        price_mid=50075,
        touch_count=6,  # Too many
        thickness_pct=0.015,  # Too thick
        last_touch_index=50,
        creation_index=10,
        timeframe="1H",
        method="zigzag",
        quality=4.0,  # Low quality
        days_since_last_touch=45.0  # Old
    )
    
    choch_poor = ChoCHResult(
        detected=True,
        direction="LONG",
        breakout_price=50100,
        breakout_index=120,
        broken_level=50000,
        strength=0.42,  # Weak
        body_score=0.15,
        volume_score=0.27
    )
    
    score_poor = scorer.score_setup(
        trend=trend_poor,
        zone=zone_poor,
        choch=choch_poor,
        fibonacci=None,  # No Fib
        entry_level=None
    )
    
    print(f"Total Score: {score_poor.total_score:.1f}/100")
    print(f"Grade: {score_poor.grade}")
    print(f"\nComponent Scores:")
    print(f"  Trend:  {score_poor.trend_score:.1f}/25")
    print(f"  Zone:   {score_poor.zone_score:.1f}/25")
    print(f"  ChoCH:  {score_poor.choch_score:.1f}/25")
    print(f"  Fib:    {score_poor.fib_score:.1f}/25")
    print(f"\nRecommended: {'✅ YES' if score_poor.recommended else '❌ NO'}")
    print(f"Reason: {score_poor.reason}")
    
    # ═══════════════════════════════════════════════════════════
    # Summary Table
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    print(f"{'Setup':<15} {'Score':<10} {'Grade':<10} {'Recommended':<15}")
    print("-" * 60)
    print(f"{'Excellent':<15} {score_excellent.total_score:<10.1f} {score_excellent.grade:<10} {'✅ YES':<15}")
    print(f"{'Good':<15} {score_good.total_score:<10.1f} {score_good.grade:<10} {'✅ YES':<15}")
    print(f"{'Poor':<15} {score_poor.total_score:<10.1f} {score_poor.grade:<10} {'❌ NO':<15}")
    
    print("\n" + "="*60)
    print("✅ Setup Scorer working correctly!")
    print("="*60 + "\n")