"""
Circuit Breaker Pattern
Based on: pa-strateji3 Parça 9

Features:
- 5 consecutive API errors → OPEN circuit
- Exponential backoff retry
- Auto recovery after cooldown
"""

from __future__ import annotations
import time
from typing import Callable, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass


class CircuitState(Enum):
    """Circuit durumu"""
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Circuit broken, requests blocked
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker konfigürasyonu"""
    failure_threshold: int = 5  # Kaç hata sonra circuit açılsın
    recovery_timeout: int = 60  # Kaç saniye sonra HALF_OPEN'a geç
    success_threshold: int = 2  # Kaç başarılı request sonra CLOSED'a dön
    max_retries: int = 3  # Maksimum retry sayısı
    backoff_multiplier: float = 2.0  # Exponential backoff çarpanı


class CircuitBreaker:
    """
    Circuit Breaker Pattern implementasyonu
    
    States:
    - CLOSED: Normal operation, tüm istekler geçer
    - OPEN: Circuit broken, tüm istekler bloke
    - HALF_OPEN: Test mode, sınırlı istekler geçer
    
    Flow:
    CLOSED --[5 failures]--> OPEN --[60s timeout]--> HALF_OPEN --[2 success]--> CLOSED
    """
    
    def __init__(self, name: str = "default", config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        
        # Timing
        self.last_failure_time: Optional[datetime] = None
        self.opened_at: Optional[datetime] = None
        
        # Stats
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Function'ı circuit breaker üzerinden çağır
        
        Args:
            func: Çağrılacak function
            *args: Function argümanları
            **kwargs: Function keyword argümanları
            
        Returns:
            Function'ın return değeri
            
        Raises:
            Exception: Circuit OPEN veya function hata verdi
        """
        self.total_calls += 1
        
        # Circuit OPEN mı kontrol et
        if self.state == CircuitState.OPEN:
            # Recovery timeout geçti mi?
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise Exception(f"Circuit breaker OPEN for {self.name}")
        
        # Function'ı çağır
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure(e)
            raise
    
    def call_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Exponential backoff ile retry
        
        Args:
            func: Çağrılacak function
            *args, **kwargs: Function argümanları
            
        Returns:
            Function'ın return değeri
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                return self.call(func, *args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                # Son attempt değilse bekle
                if attempt < self.config.max_retries - 1:
                    wait_time = self._calculate_backoff(attempt)
                    print(f"⏳ Retry {attempt + 1}/{self.config.max_retries} "
                          f"after {wait_time:.1f}s: {str(e)[:50]}")
                    time.sleep(wait_time)
        
        # Tüm retry'ler başarısız
        raise last_exception
    
    def _on_success(self):
        """Başarılı call sonrası"""
        self.total_successes += 1
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            # Yeterli success var mı?
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
    
    def _on_failure(self, exception: Exception):
        """Başarısız call sonrası"""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        # Log
        print(f"❌ Circuit breaker failure ({self.failure_count}/{self.config.failure_threshold}): "
              f"{str(exception)[:100]}")
        
        # Threshold aşıldı mı?
        if self.failure_count >= self.config.failure_threshold:
            self._transition_to_open()
        
        # HALF_OPEN'daysa tekrar OPEN'a dön
        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
    
    def _transition_to_open(self):
        """Circuit'i OPEN durumuna geçir"""
        if self.state == CircuitState.OPEN:
            return
        
        self.state = CircuitState.OPEN
        self.opened_at = datetime.now()
        self.success_count = 0
        
        print(f"\n{'='*60}")
        print(f"🔴 CIRCUIT BREAKER OPENED: {self.name}")
        print(f"{'='*60}")
        print(f"Reason: {self.failure_count} consecutive failures")
        print(f"Recovery timeout: {self.config.recovery_timeout}s")
        print(f"All requests will be blocked until recovery")
        print(f"{'='*60}\n")
        
        # Telegram bildirim (implement edilecek)
        # self.telegram.send_critical(f"Circuit breaker OPEN: {self.name}")
    
    def _transition_to_half_open(self):
        """Circuit'i HALF_OPEN durumuna geçir"""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        
        print(f"🟡 Circuit breaker HALF_OPEN: {self.name} - Testing recovery...")
    
    def _transition_to_closed(self):
        """Circuit'i CLOSED durumuna geçir"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.opened_at = None
        
        print(f"🟢 Circuit breaker CLOSED: {self.name} - Normal operation resumed")
        
        # Telegram bildirim
        # self.telegram.send_important(f"Circuit breaker recovered: {self.name}")
    
    def _should_attempt_reset(self) -> bool:
        """Recovery timeout doldu mu?"""
        if self.opened_at is None:
            return False
        
        elapsed = (datetime.now() - self.opened_at).total_seconds()
        return elapsed >= self.config.recovery_timeout
    
    def _calculate_backoff(self, attempt: int) -> float:
        """
        Exponential backoff hesapla
        
        Args:
            attempt: Kaçıncı deneme (0-indexed)
            
        Returns:
            Bekleme süresi (saniye)
        """
        return min(30, self.config.backoff_multiplier ** attempt)
    
    def get_state(self) -> str:
        """Mevcut state'i döndür"""
        return self.state.value
    
    def is_open(self) -> bool:
        """Circuit OPEN mı?"""
        return self.state == CircuitState.OPEN
    
    def is_closed(self) -> bool:
        """Circuit CLOSED mı?"""
        return self.state == CircuitState.CLOSED
    
    def get_stats(self) -> dict:
        """İstatistikleri döndür"""
        success_rate = (self.total_successes / self.total_calls * 100) if self.total_calls > 0 else 0
        
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.total_calls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": f"{success_rate:.1f}%",
            "current_failure_count": self.failure_count,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None
        }
    
    def reset(self):
        """Circuit'i sıfırla (manuel müdahale)"""
        print(f"🔄 Circuit breaker reset: {self.name}")
        self._transition_to_closed()
    
    def force_open(self):
        """Circuit'i zorla aç (bakım için)"""
        print(f"⚠️  Circuit breaker forced OPEN: {self.name}")
        self._transition_to_open()


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

class ExchangeAPI:
    """Örnek exchange API wrapper"""
    
    def __init__(self):
        self.circuit = CircuitBreaker(name="exchange_api")
        self.request_count = 0
    
    def get_ticker(self, symbol: str) -> dict:
        """Ticker bilgisi al (circuit breaker korumalı)"""
        def _fetch():
            self.request_count += 1
            
            # Simulate error
            if self.request_count % 10 == 0:
                raise Exception("API rate limit exceeded")
            
            return {"symbol": symbol, "price": 50000}
        
        return self.circuit.call_with_retry(_fetch)
    
    def place_order(self, symbol: str, side: str, qty: float) -> dict:
        """Order yerleştir (circuit breaker korumalı)"""
        def _place():
            # Simulate order placement
            return {"order_id": "12345", "status": "filled"}
        
        return self.circuit.call(_place)


if __name__ == "__main__":
    # Test
    api = ExchangeAPI()
    
    for i in range(20):
        try:
            result = api.get_ticker("BTCUSDT")
            print(f"✅ Request {i+1}: {result}")
        except Exception as e:
            print(f"❌ Request {i+1} failed: {e}")
        
        time.sleep(0.5)