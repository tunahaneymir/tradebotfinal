"""
Backup Manager
Based on: pa-strateji3 Parça 9

Features:
- Hourly backups (state files)
- Daily backups (full system)
- Model checkpoints
- Retention policy (7 days)
- Compression support
"""

from __future__ import annotations
import shutil
import gzip
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import threading
import time
import json


class BackupManager:
    """
    Otomatik backup sistemi
    
    Backup Types:
    - Hourly: state files (bot_state, zone_memory)
    - Daily: Full system (state + database + models)
    - Model: RL model checkpoints (weekly)
    
    Retention:
    - Hourly: 24 hours
    - Daily: 7 days
    - Model: 4 weeks
    """
    
    def __init__(
        self,
        source_dirs: List[str] = None,
        backup_root: str = "backups",
        enable_compression: bool = True
    ):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        # Source directories to backup
        self.source_dirs = source_dirs or ["state", "data", "models"]
        
        # Backup directories
        self.hourly_dir = self.backup_root / "hourly"
        self.daily_dir = self.backup_root / "daily"
        self.model_dir = self.backup_root / "models"
        
        for dir in [self.hourly_dir, self.daily_dir, self.model_dir]:
            dir.mkdir(exist_ok=True)
        
        # Settings
        self.enable_compression = enable_compression
        
        # Auto-backup threads
        self.auto_backup_enabled = False
        self.hourly_thread: Optional[threading.Thread] = None
        self.daily_thread: Optional[threading.Thread] = None
        
        print(f"💾 Backup Manager initialized")
        print(f"   Backup root: {backup_root}")
        print(f"   Compression: {'Enabled' if enable_compression else 'Disabled'}")
    
    # ═══════════════════════════════════════════════════════════
    # BACKUP CREATION
    # ═══════════════════════════════════════════════════════════
    
    def create_hourly_backup(self) -> Optional[Path]:
        """Saatlik backup oluştur (state files)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H0000")
        backup_name = f"hourly_{timestamp}"
        backup_path = self.hourly_dir / backup_name
        
        try:
            backup_path.mkdir(exist_ok=True)
            
            # State files'ı kopyala
            state_dir = Path("state")
            if state_dir.exists():
                for file in state_dir.glob("*.json"):
                    dest = backup_path / file.name
                    shutil.copy2(file, dest)
            
            # Compression
            if self.enable_compression:
                self._compress_directory(backup_path)
            
            print(f"✅ Hourly backup created: {backup_name}")
            
            # Cleanup old hourly backups (>24h)
            self._cleanup_old_backups(self.hourly_dir, hours=24)
            
            return backup_path
            
        except Exception as e:
            print(f"❌ Hourly backup error: {e}")
            return None
    
    def create_daily_backup(self) -> Optional[Path]:
        """Günlük backup oluştur (full system)"""
        timestamp = datetime.now().strftime("%Y%m%d")
        backup_name = f"daily_{timestamp}"
        backup_path = self.daily_dir / backup_name
        
        try:
            backup_path.mkdir(exist_ok=True)
            
            # Tüm source directory'leri kopyala
            for source_name in self.source_dirs:
                source_dir = Path(source_name)
                
                if not source_dir.exists():
                    continue
                
                dest_dir = backup_path / source_name
                shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
            
            # Metadata ekle
            metadata = {
                "backup_type": "daily",
                "timestamp": datetime.now().isoformat(),
                "source_dirs": self.source_dirs,
                "compressed": self.enable_compression
            }
            
            with open(backup_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Compression
            if self.enable_compression:
                self._compress_directory(backup_path)
            
            print(f"✅ Daily backup created: {backup_name}")
            
            # Cleanup old daily backups (>7 days)
            self._cleanup_old_backups(self.daily_dir, days=7)
            
            return backup_path
            
        except Exception as e:
            print(f"❌ Daily backup error: {e}")
            return None
    
    def create_model_checkpoint(self, model_name: str = "rl_model") -> Optional[Path]:
        """Model checkpoint oluştur"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{model_name}_{timestamp}"
        backup_path = self.model_dir / backup_name
        
        try:
            backup_path.mkdir(exist_ok=True)
            
            # Model files'ı kopyala
            model_source = Path("models")
            if model_source.exists():
                for file in model_source.glob(f"{model_name}.*"):
                    dest = backup_path / file.name
                    shutil.copy2(file, dest)
            
            # Metadata
            metadata = {
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "checkpoint_type": "weekly"
            }
            
            with open(backup_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Model checkpoint created: {backup_name}")
            
            # Cleanup old model backups (>4 weeks)
            self._cleanup_old_backups(self.model_dir, days=28)
            
            return backup_path
            
        except Exception as e:
            print(f"❌ Model checkpoint error: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════
    # RESTORE
    # ═══════════════════════════════════════════════════════════
    
    def restore_from_backup(self, backup_path: Path) -> bool:
        """Backup'tan restore et"""
        try:
            if not backup_path.exists():
                print(f"❌ Backup not found: {backup_path}")
                return False
            
            # Decompress if needed
            if backup_path.suffix == '.gz':
                backup_path = self._decompress_directory(backup_path)
            
            # Metadata oku
            metadata_file = backup_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                print(f"📋 Backup info: {metadata.get('backup_type')} - {metadata.get('timestamp')}")
            
            # Restore
            for source_name in self.source_dirs:
                source_backup = backup_path / source_name
                
                if not source_backup.exists():
                    continue
                
                dest_dir = Path(source_name)
                
                # Backup current state
                if dest_dir.exists():
                    temp_backup = dest_dir.parent / f"{dest_dir.name}_temp_backup"
                    shutil.move(str(dest_dir), str(temp_backup))
                
                # Restore
                shutil.copytree(source_backup, dest_dir)
                
                # Remove temp backup
                if temp_backup.exists():
                    shutil.rmtree(temp_backup)
            
            print(f"✅ Restored from backup: {backup_path.name}")
            return True
            
        except Exception as e:
            print(f"❌ Restore error: {e}")
            return False
    
    def list_backups(self, backup_type: str = "all") -> List[Dict]:
        """Mevcut backup'ları listele"""
        backups = []
        
        dirs_to_check = []
        if backup_type in ["hourly", "all"]:
            dirs_to_check.append(("hourly", self.hourly_dir))
        if backup_type in ["daily", "all"]:
            dirs_to_check.append(("daily", self.daily_dir))
        if backup_type in ["model", "all"]:
            dirs_to_check.append(("model", self.model_dir))
        
        for btype, bdir in dirs_to_check:
            for item in sorted(bdir.iterdir(), reverse=True):
                if item.is_dir() or item.suffix == '.gz':
                    # Metadata oku (varsa)
                    metadata_file = item / "metadata.json" if item.is_dir() else None
                    
                    backup_info = {
                        "type": btype,
                        "name": item.name,
                        "path": str(item),
                        "size_mb": self._get_size_mb(item),
                        "created": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    }
                    
                    if metadata_file and metadata_file.exists():
                        with open(metadata_file) as f:
                            backup_info["metadata"] = json.load(f)
                    
                    backups.append(backup_info)
        
        return backups
    
    # ═══════════════════════════════════════════════════════════
    # AUTO-BACKUP (Threading)
    # ═══════════════════════════════════════════════════════════
    
    def start_auto_backup(self):
        """Otomatik backup'ı başlat"""
        if self.auto_backup_enabled:
            print("⚠️  Auto-backup already running")
            return
        
        self.auto_backup_enabled = True
        
        # Hourly backup thread
        self.hourly_thread = threading.Thread(target=self._hourly_backup_loop, daemon=True)
        self.hourly_thread.start()
        
        # Daily backup thread
        self.daily_thread = threading.Thread(target=self._daily_backup_loop, daemon=True)
        self.daily_thread.start()
        
        print("✅ Auto-backup started")
        print("   Hourly: Every hour")
        print("   Daily: Every day at 00:00")
    
    def stop_auto_backup(self):
        """Otomatik backup'ı durdur"""
        self.auto_backup_enabled = False
        
        if self.hourly_thread:
            self.hourly_thread.join(timeout=2)
        if self.daily_thread:
            self.daily_thread.join(timeout=2)
        
        print("⏸️  Auto-backup stopped")
    
    def _hourly_backup_loop(self):
        """Saatlik backup loop"""
        while self.auto_backup_enabled:
            # Her saat başında çalış
            now = datetime.now()
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            wait_seconds = (next_hour - now).total_seconds()
            
            time.sleep(wait_seconds)
            
            if self.auto_backup_enabled:
                self.create_hourly_backup()
    
    def _daily_backup_loop(self):
        """Günlük backup loop"""
        while self.auto_backup_enabled:
            # Her gün 00:00'da çalış
            now = datetime.now()
            next_day = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            wait_seconds = (next_day - now).total_seconds()
            
            time.sleep(wait_seconds)
            
            if self.auto_backup_enabled:
                self.create_daily_backup()
    
    # ═══════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════
    
    def _compress_directory(self, dir_path: Path) -> Path:
        """Directory'i compress et"""
        archive_path = dir_path.with_suffix('.tar.gz')
        shutil.make_archive(str(dir_path), 'gztar', dir_path.parent, dir_path.name)
        
        # Original directory'i sil
        if archive_path.exists():
            shutil.rmtree(dir_path)
        
        return archive_path
    
    def _decompress_directory(self, archive_path: Path) -> Path:
        """Archive'i decompress et"""
        extract_path = archive_path.parent / archive_path.stem.replace('.tar', '')
        shutil.unpack_archive(str(archive_path), str(extract_path))
        return extract_path
    
    def _cleanup_old_backups(self, backup_dir: Path, days: int = None, hours: int = None):
        """Eski backup'ları temizle"""
        if days:
            cutoff = datetime.now() - timedelta(days=days)
        elif hours:
            cutoff = datetime.now() - timedelta(hours=hours)
        else:
            return
        
        deleted_count = 0
        
        for item in backup_dir.iterdir():
            # Dosya/klasör oluşturma zamanı
            created = datetime.fromtimestamp(item.stat().st_mtime)
            
            if created < cutoff:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                
                deleted_count += 1
        
        if deleted_count > 0:
            print(f"🗑️  Cleaned up {deleted_count} old backups from {backup_dir.name}")
    
    def _get_size_mb(self, path: Path) -> float:
        """Directory/file boyutunu MB olarak hesapla"""
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)
        
        total = 0
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        
        return total / (1024 * 1024)
    
    def get_storage_summary(self) -> Dict:
        """Backup storage özeti"""
        return {
            "hourly": {
                "count": len(list(self.hourly_dir.iterdir())),
                "total_size_mb": self._get_size_mb(self.hourly_dir)
            },
            "daily": {
                "count": len(list(self.daily_dir.iterdir())),
                "total_size_mb": self._get_size_mb(self.daily_dir)
            },
            "model": {
                "count": len(list(self.model_dir.iterdir())),
                "total_size_mb": self._get_size_mb(self.model_dir)
            }
        }


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Backup manager oluştur
    manager = BackupManager(
        source_dirs=["state", "data", "models"],
        backup_root="backups",
        enable_compression=True
    )
    
    # Manual backups
    print("\n📦 Creating manual backups...")
    manager.create_hourly_backup()
    manager.create_daily_backup()
    manager.create_model_checkpoint("rl_model")
    
    # List backups
    print("\n📋 Listing backups...")
    backups = manager.list_backups("all")
    for backup in backups[:5]:  # İlk 5
        print(f"  {backup['type']}: {backup['name']} ({backup['size_mb']:.1f} MB)")
    
    # Storage summary
    print("\n💾 Storage Summary:")
    summary = manager.get_storage_summary()
    for btype, info in summary.items():
        print(f"  {btype}: {info['count']} backups ({info['total_size_mb']:.1f} MB)")
    
    # Auto-backup başlat (örnek)
    # manager.start_auto_backup()
    # time.sleep(3600)  # 1 saat bekle
    # manager.stop_auto_backup()