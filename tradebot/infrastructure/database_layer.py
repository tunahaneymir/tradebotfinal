"""
SQLite Database Layer
Based on: pa-strateji3 Parça 9

Features:
- Trade history tracking
- Setup metrics storage
- Behavioral flags logging
- Performance analytics
- CSV export
"""

from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import csv


@dataclass
class TradeRecord:
    """Trade kayıt modeli"""
    # Trade details
    trade_id: str
    datetime: str
    coin: str
    timeframe: str
    direction: str  # LONG/SHORT
    entry: float
    exit: Optional[float]
    stop: float
    tp1: Optional[float]
    tp2: Optional[float]
    tp3: Optional[float]
    
    # Position details
    size: float
    risk_pct: float
    risk_usd: float
    
    # Results
    pnl_usd: Optional[float]
    pnl_pct: Optional[float]
    duration_min: Optional[int]
    exit_reason: Optional[str]
    
    # Setup metrics
    setup_score: float
    zone_quality: float
    choch_strength: float
    volume_ratio: float
    fib_level: Optional[float]
    market_regime: str
    outcome_score: Optional[float]
    
    # Behavioral flags
    fomo_detected: bool
    revenge_trade: bool
    reentry_attempt: bool
    cooldown_active: bool
    emotional_state: str
    risk_multiplier: float


class DatabaseLayer:
    """
    SQLite database yönetimi
    
    Tables:
    - trades: Trade kayıtları
    - daily_stats: Günlük istatistikler
    - zone_performance: Zone başarı oranları
    """
    
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_database()
        
        print(f"💾 Database initialized: {db_path}")
    
    def _initialize_database(self):
        """Database'i oluştur ve tabloları kur"""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Trades tablosu
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                datetime TEXT NOT NULL,
                coin TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry REAL NOT NULL,
                exit REAL,
                stop REAL NOT NULL,
                tp1 REAL,
                tp2 REAL,
                tp3 REAL,
                size REAL NOT NULL,
                risk_pct REAL NOT NULL,
                risk_usd REAL NOT NULL,
                pnl_usd REAL,
                pnl_pct REAL,
                duration_min INTEGER,
                exit_reason TEXT,
                setup_score REAL NOT NULL,
                zone_quality REAL NOT NULL,
                choch_strength REAL NOT NULL,
                volume_ratio REAL NOT NULL,
                fib_level REAL,
                market_regime TEXT NOT NULL,
                outcome_score REAL,
                fomo_detected INTEGER DEFAULT 0,
                revenge_trade INTEGER DEFAULT 0,
                reentry_attempt INTEGER DEFAULT 0,
                cooldown_active INTEGER DEFAULT 0,
                emotional_state TEXT,
                risk_multiplier REAL DEFAULT 1.0
            )
        """)
        
        # Index'ler
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_coin ON trades(coin)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_datetime ON trades(datetime)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_direction ON trades(direction)")
        
        # Daily stats tablosu
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                total_pnl_pct REAL DEFAULT 0,
                total_pnl_usd REAL DEFAULT 0,
                best_trade_pct REAL DEFAULT 0,
                worst_trade_pct REAL DEFAULT 0,
                fomo_blocks INTEGER DEFAULT 0,
                revenge_blocks INTEGER DEFAULT 0,
                overtrade_blocks INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0.5,
                avg_stress REAL DEFAULT 0,
                avg_patience REAL DEFAULT 0.5
            )
        """)
        
        self.conn.commit()
    
    # ═══════════════════════════════════════════════════════════
    # INSERT METHODS
    # ═══════════════════════════════════════════════════════════
    
    def insert_trade(self, trade: TradeRecord):
        """Yeni trade kaydet"""
        try:
            self.conn.execute("""
                INSERT INTO trades (
                    trade_id, datetime, coin, timeframe, direction,
                    entry, exit, stop, tp1, tp2, tp3,
                    size, risk_pct, risk_usd,
                    pnl_usd, pnl_pct, duration_min, exit_reason,
                    setup_score, zone_quality, choch_strength, volume_ratio,
                    fib_level, market_regime, outcome_score,
                    fomo_detected, revenge_trade, reentry_attempt,
                    cooldown_active, emotional_state, risk_multiplier
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.trade_id, trade.datetime, trade.coin, trade.timeframe, trade.direction,
                trade.entry, trade.exit, trade.stop, trade.tp1, trade.tp2, trade.tp3,
                trade.size, trade.risk_pct, trade.risk_usd,
                trade.pnl_usd, trade.pnl_pct, trade.duration_min, trade.exit_reason,
                trade.setup_score, trade.zone_quality, trade.choch_strength, trade.volume_ratio,
                trade.fib_level, trade.market_regime, trade.outcome_score,
                int(trade.fomo_detected), int(trade.revenge_trade), int(trade.reentry_attempt),
                int(trade.cooldown_active), trade.emotional_state, trade.risk_multiplier
            ))
            self.conn.commit()
            print(f"✅ Trade saved: {trade.trade_id}")
        except sqlite3.IntegrityError:
            print(f"⚠️  Trade already exists: {trade.trade_id}")
        except Exception as e:
            print(f"❌ Insert error: {e}")
    
    def update_trade_exit(self, trade_id: str, exit_price: float, pnl_usd: float, 
                          pnl_pct: float, duration_min: int, exit_reason: str):
        """Trade çıkışını güncelle"""
        try:
            self.conn.execute("""
                UPDATE trades 
                SET exit = ?, pnl_usd = ?, pnl_pct = ?, duration_min = ?, exit_reason = ?
                WHERE trade_id = ?
            """, (exit_price, pnl_usd, pnl_pct, duration_min, exit_reason, trade_id))
            self.conn.commit()
            print(f"✅ Trade exit updated: {trade_id}")
        except Exception as e:
            print(f"❌ Update error: {e}")
    
    # ═══════════════════════════════════════════════════════════
    # QUERY METHODS
    # ═══════════════════════════════════════════════════════════
    
    def get_trade(self, trade_id: str) -> Optional[Dict]:
        """Trade'i ID ile çek"""
        cursor = self.conn.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Son N trade'i çek"""
        cursor = self.conn.execute("""
            SELECT * FROM trades 
            ORDER BY datetime DESC 
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_daily_trades(self, date: str) -> List[Dict]:
        """Belirli bir günün trade'lerini çek"""
        cursor = self.conn.execute("""
            SELECT * FROM trades 
            WHERE DATE(datetime) = ?
            ORDER BY datetime DESC
        """, (date,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_coin_trades(self, coin: str, limit: int = 50) -> List[Dict]:
        """Belirli bir coin'in trade'lerini çek"""
        cursor = self.conn.execute("""
            SELECT * FROM trades 
            WHERE coin = ?
            ORDER BY datetime DESC
            LIMIT ?
        """, (coin, limit))
        return [dict(row) for row in cursor.fetchall()]
    
    # ═══════════════════════════════════════════════════════════
    # ANALYTICS
    # ═══════════════════════════════════════════════════════════
    
    def get_daily_stats(self, date: str) -> Dict:
        """Günlük istatistikler"""
        trades = self.get_daily_trades(date)
        
        if not trades:
            return {"date": date, "total_trades": 0}
        
        wins = [t for t in trades if t["pnl_pct"] and t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] and t["pnl_pct"] < 0]
        
        return {
            "date": date,
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "total_pnl_pct": sum(t["pnl_pct"] or 0 for t in trades),
            "total_pnl_usd": sum(t["pnl_usd"] or 0 for t in trades),
            "best_trade_pct": max((t["pnl_pct"] for t in trades if t["pnl_pct"]), default=0),
            "worst_trade_pct": min((t["pnl_pct"] for t in trades if t["pnl_pct"]), default=0),
            "avg_setup_score": sum(t["setup_score"] for t in trades) / len(trades),
            "avg_zone_quality": sum(t["zone_quality"] for t in trades) / len(trades)
        }
    
    def get_weekly_stats(self, start_date: str, end_date: str) -> Dict:
        """Haftalık istatistikler"""
        cursor = self.conn.execute("""
            SELECT * FROM trades 
            WHERE DATE(datetime) BETWEEN ? AND ?
            ORDER BY datetime
        """, (start_date, end_date))
        
        trades = [dict(row) for row in cursor.fetchall()]
        
        if not trades:
            return {"start_date": start_date, "end_date": end_date, "total_trades": 0}
        
        wins = [t for t in trades if t["pnl_pct"] and t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] and t["pnl_pct"] < 0]
        
        # Profit factor
        gross_profit = sum(t["pnl_usd"] for t in wins)
        gross_loss = abs(sum(t["pnl_usd"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            "start_date": start_date,
            "end_date": end_date,
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "total_pnl_pct": sum(t["pnl_pct"] or 0 for t in trades),
            "total_pnl_usd": sum(t["pnl_usd"] or 0 for t in trades),
            "profit_factor": profit_factor,
            "avg_win": sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
        }
    
    def get_coin_performance(self, coin: str) -> Dict:
        """Coin bazlı performans"""
        trades = self.get_coin_trades(coin, limit=999)
        
        if not trades:
            return {"coin": coin, "total_trades": 0}
        
        wins = [t for t in trades if t["pnl_pct"] and t["pnl_pct"] > 0]
        
        return {
            "coin": coin,
            "total_trades": len(trades),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "total_pnl_pct": sum(t["pnl_pct"] or 0 for t in trades),
            "avg_setup_score": sum(t["setup_score"] for t in trades) / len(trades),
            "avg_zone_quality": sum(t["zone_quality"] for t in trades) / len(trades)
        }
    
    def get_setup_quality_analysis(self) -> Dict:
        """Setup quality ile win rate ilişkisi"""
        cursor = self.conn.execute("""
            SELECT 
                CASE 
                    WHEN setup_score >= 80 THEN 'Excellent (80+)'
                    WHEN setup_score >= 65 THEN 'Good (65-79)'
                    WHEN setup_score >= 50 THEN 'Medium (50-64)'
                    ELSE 'Weak (<50)'
                END as quality_range,
                COUNT(*) as total,
                SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                AVG(pnl_pct) as avg_pnl_pct
            FROM trades
            WHERE pnl_pct IS NOT NULL
            GROUP BY quality_range
            ORDER BY MIN(setup_score) DESC
        """)
        
        results = {}
        for row in cursor.fetchall():
            row_dict = dict(row)
            total = row_dict["total"]
            wins = row_dict["wins"]
            
            results[row_dict["quality_range"]] = {
                "total_trades": total,
                "wins": wins,
                "win_rate": wins / total * 100 if total > 0 else 0,
                "avg_pnl_pct": row_dict["avg_pnl_pct"] or 0
            }
        
        return results
    
    def get_zone_quality_analysis(self) -> Dict:
        """Zone quality ile win rate ilişkisi"""
        cursor = self.conn.execute("""
            SELECT 
                CASE 
                    WHEN zone_quality >= 8 THEN 'Excellent (8+)'
                    WHEN zone_quality >= 6 THEN 'Good (6-7)'
                    WHEN zone_quality >= 4 THEN 'Medium (4-5)'
                    ELSE 'Weak (<4)'
                END as quality_range,
                COUNT(*) as total,
                SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                AVG(pnl_pct) as avg_pnl_pct
            FROM trades
            WHERE pnl_pct IS NOT NULL
            GROUP BY quality_range
            ORDER BY MIN(zone_quality) DESC
        """)
        
        results = {}
        for row in cursor.fetchall():
            row_dict = dict(row)
            total = row_dict["total"]
            wins = row_dict["wins"]
            
            results[row_dict["quality_range"]] = {
                "total_trades": total,
                "wins": wins,
                "win_rate": wins / total * 100 if total > 0 else 0,
                "avg_pnl_pct": row_dict["avg_pnl_pct"] or 0
            }
        
        return results
    
    def get_behavioral_stats(self) -> Dict:
        """Behavioral flag istatistikleri"""
        cursor = self.conn.execute("""
            SELECT 
                SUM(fomo_detected) as fomo_blocks,
                SUM(revenge_trade) as revenge_blocks,
                SUM(reentry_attempt) as reentry_trades,
                AVG(CASE WHEN fomo_detected = 0 THEN pnl_pct END) as avg_pnl_normal,
                AVG(CASE WHEN fomo_detected = 1 THEN pnl_pct END) as avg_pnl_fomo
            FROM trades
            WHERE pnl_pct IS NOT NULL
        """)
        
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    # ═══════════════════════════════════════════════════════════
    # EXPORT METHODS
    # ═══════════════════════════════════════════════════════════
    
    def export_to_csv(self, output_path: str, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> bool:
        """Trade'leri CSV'ye export et"""
        try:
            query = "SELECT * FROM trades"
            params = []
            
            if start_date and end_date:
                query += " WHERE DATE(datetime) BETWEEN ? AND ?"
                params = [start_date, end_date]
            
            query += " ORDER BY datetime DESC"
            
            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                print("⚠️  No trades to export")
                return False
            
            # CSV yaz
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows([dict(row) for row in rows])
            
            print(f"✅ Exported {len(rows)} trades to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Export error: {e}")
            return False
    
    def get_summary_report(self) -> str:
        """Özet rapor oluştur"""
        # Son 30 günün trade'leri
        cursor = self.conn.execute("""
            SELECT * FROM trades 
            WHERE datetime >= DATE('now', '-30 days')
            ORDER BY datetime DESC
        """)
        
        trades = [dict(row) for row in cursor.fetchall()]
        
        if not trades:
            return "No trades in the last 30 days"
        
        wins = [t for t in trades if t["pnl_pct"] and t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] and t["pnl_pct"] < 0]
        
        report = f"""
═══════════════════════════════════════════════════════════
                    PERFORMANCE SUMMARY (30 DAYS)
═══════════════════════════════════════════════════════════

OVERVIEW:
  Total Trades: {len(trades)}
  Wins: {len(wins)} | Losses: {len(losses)}
  Win Rate: {len(wins)/len(trades)*100:.1f}%
  
P&L:
  Total: {sum(t['pnl_pct'] or 0 for t in trades):+.2f}% (${sum(t['pnl_usd'] or 0 for t in trades):+,.2f})
  Best Trade: {max((t['pnl_pct'] for t in trades if t['pnl_pct']), default=0):+.2f}%
  Worst Trade: {min((t['pnl_pct'] for t in trades if t['pnl_pct']), default=0):+.2f}%
  Avg Win: {sum(t['pnl_pct'] for t in wins)/len(wins) if wins else 0:.2f}%
  Avg Loss: {sum(t['pnl_pct'] for t in losses)/len(losses) if losses else 0:.2f}%

SETUP QUALITY:
  Avg Setup Score: {sum(t['setup_score'] for t in trades)/len(trades):.1f}/100
  Avg Zone Quality: {sum(t['zone_quality'] for t in trades)/len(trades):.1f}/10
  Avg ChoCH Strength: {sum(t['choch_strength'] for t in trades)/len(trades):.2f}

BEHAVIORAL:
  FOMO Detected: {sum(t['fomo_detected'] for t in trades)}
  Revenge Trades: {sum(t['revenge_trade'] for t in trades)}
  Re-entry Attempts: {sum(t['reentry_attempt'] for t in trades)}

TOP COINS:
""".strip()
        
        # Top coins by trade count
        coin_counts = {}
        for t in trades:
            coin = t['coin']
            if coin not in coin_counts:
                coin_counts[coin] = 0
            coin_counts[coin] += 1
        
        top_coins = sorted(coin_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (coin, count) in enumerate(top_coins, 1):
            coin_trades = [t for t in trades if t['coin'] == coin]
            coin_wins = [t for t in coin_trades if t['pnl_pct'] and t['pnl_pct'] > 0]
            coin_wr = len(coin_wins) / len(coin_trades) * 100 if coin_trades else 0
            report += f"\n  {i}. {coin}: {count} trades ({coin_wr:.0f}% WR)"
        
        report += "\n═══════════════════════════════════════════════════════════"
        
        return report
    
    # ═══════════════════════════════════════════════════════════
    # CLEANUP & MAINTENANCE
    # ═══════════════════════════════════════════════════════════
    
    def cleanup_old_trades(self, days: int = 90):
        """Eski trade'leri sil (retention policy)"""
        cursor = self.conn.execute("""
            DELETE FROM trades 
            WHERE datetime < DATE('now', '-' || ? || ' days')
        """, (days,))
        
        deleted = cursor.rowcount
        self.conn.commit()
        
        print(f"🗑️  Deleted {deleted} trades older than {days} days")
        return deleted
    
    def vacuum(self):
        """Database optimize et"""
        self.conn.execute("VACUUM")
        print("🔧 Database vacuumed")
    
    def close(self):
        """Database bağlantısını kapat"""
        if self.conn:
            self.conn.close()
            print("💾 Database connection closed")


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Database oluştur
    db = DatabaseLayer("data/trades.db")
    
    # Örnek trade ekle
    trade = TradeRecord(
        trade_id="T001",
        datetime=datetime.now().isoformat(),
        coin="BTCUSDT",
        timeframe="4H",
        direction="LONG",
        entry=50150,
        exit=51500,
        stop=49750,
        tp1=50500,
        tp2=51000,
        tp3=51500,
        size=0.05,
        risk_pct=2.0,
        risk_usd=100,
        pnl_usd=67.5,
        pnl_pct=2.7,
        duration_min=180,
        exit_reason="TP3_HIT",
        setup_score=85,
        zone_quality=8.5,
        choch_strength=0.78,
        volume_ratio=1.5,
        fib_level=0.705,
        market_regime="UPTREND",
        outcome_score=150,
        fomo_detected=False,
        revenge_trade=False,
        reentry_attempt=False,
        cooldown_active=False,
        emotional_state="CONFIDENT",
        risk_multiplier=1.0
    )
    
    db.insert_trade(trade)
    
    # Son trade'leri çek
    recent = db.get_recent_trades(5)
    print(f"\n📊 Recent trades: {len(recent)}")
    
    # Setup quality analysis
    setup_analysis = db.get_setup_quality_analysis()
    print("\n📈 Setup Quality Analysis:")
    for quality, stats in setup_analysis.items():
        print(f"  {quality}: {stats['win_rate']:.1f}% WR")
    
    # Summary report
    print(db.get_summary_report())
    
    # CSV export
    db.export_to_csv("exports/trades_export.csv")
    
    db.close()