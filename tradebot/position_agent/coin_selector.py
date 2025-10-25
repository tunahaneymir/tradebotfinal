"""Command-line tool for selecting Binance USDT perpetual futures pairs.
The original implementation shipped as part of the trading agents was written as a
single script.  This refactor keeps the behaviour but introduces a couple of
quality-of-life improvements:
* The heavy lifting lives inside :class:`CoinSelector`, which makes it easier to
  import the logic from other modules or tests.
* HTTP requests reuse a :class:`requests.Session`, dramatically reducing the
  number of TCP handshakes required when computing the ATR for dozens of pairs.
* Logs are routed through :mod:`logging`, allowing the caller to choose between
  quiet and verbose output without touching the code.
* The CLI now relies on :mod:`argparse`.  You can still run the script without
  arguments to get the familiar interactive menu, but non-interactive usage is
  now trivial (``python usdt_perp_selector.py rebuild``).
Despite the changes, the defaults stay close to the original: the same
volatility, spread and volume filters are applied and results are still stored
in ``usdt_perp_coin_pool.json``.
"""
from __future__ import annotations
import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import requests
# ==============================================================================
# Configuration data classes
# ==============================================================================
@dataclass(frozen=True)
class SelectorConfig:
    """Configuration values used while building the coin pool."""
    pool_file: Path = Path("usdt_perp_coin_pool.json")
    min_volume: float = 80_000_000  # USD
    max_spread: float = 0.15  # percent
    min_volatility: float = 1.5  # percent
    max_volatility: float = 20.0  # percent
    top_coin_limit: int = 20
    update_interval_hours: int = 24
    atr_period: int = 14
    kline_interval: str = "1d"
    rate_limit_delay: float = 0.05  # seconds
    base_url: str = "https://fapi.binance.com"
    def to_filters_dict(self) -> Dict[str, Any]:
        """Return a dictionary suitable for JSON serialisation."""
        return {
            "min_volume": self.min_volume,
            "max_spread": self.max_spread,
            "min_volatility": self.min_volatility,
            "max_volatility": self.max_volatility,
            "limit": self.top_coin_limit,
            "atr_period": self.atr_period,
            "kline_interval": self.kline_interval,
        }
@dataclass
class CoinCandidate:
    """Holds the interim values collected before storing a coin in the pool."""
    symbol: str
    volume: float
    spread: float
    bid: float
    ask: float
    atr: Optional[float] = field(default=None)
DEFAULT_CONFIG = SelectorConfig()
# ==============================================================================
# Coin selector implementation
# ==============================================================================
class CoinSelector:
    """Builds a USDT perpetual futures coin pool using Binance public data."""
    def __init__(self, config: SelectorConfig, session: Optional[requests.Session] = None,
                 logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.logger = logger or logging.getLogger(__name__)
    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _fetch_json(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Fetch JSON from the Binance REST API."""
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:  # pragma: no cover - network
            self.logger.error("API request failed for %s: %s", endpoint, exc)
            return None
    def _fetch_futures_tickers(self) -> List[Dict[str, Any]]:
        data = self._fetch_json("fapi/v1/ticker/bookTicker")
        return data or []
    def _fetch_24h_volume(self) -> List[Dict[str, Any]]:
        data = self._fetch_json("fapi/v1/ticker/24hr")
        return data or []
    def _fetch_klines(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        params = {
            "symbol": symbol,
            "interval": self.config.kline_interval,
            "limit": limit + 1,  # +1 needed for prev_close in ATR calculation
        }
        data = self._fetch_json("fapi/v1/klines", params=params)
        if not data:
            return []
        klines: List[Dict[str, float]] = []
        for kline in data:
            try:
                klines.append(
                    {
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[5]),
                    }
                )
            except (IndexError, ValueError) as exc:
                self.logger.debug("Skipping malformed kline entry for %s: %s", symbol, exc)
        return klines
    # ------------------------------------------------------------------
    # Calculation helpers
    # ------------------------------------------------------------------
    def _calculate_atr_percentage(self, symbol: str) -> Optional[float]:
        klines = self._fetch_klines(symbol, self.config.atr_period)
        if len(klines) < self.config.atr_period + 1:
            return None
        tr_values: List[float] = []
        for idx in range(1, len(klines)):
            high = klines[idx]["high"]
            low = klines[idx]["low"]
            prev_close = klines[idx - 1]["close"]
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(prev_close - low),
            )
            tr_values.append(tr)
        atr = sum(tr_values) / len(tr_values)
        current_price = klines[-1]["close"]
        if not current_price:
            return None
        return (atr / current_price) * 100
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_coin_pool(self) -> List[Dict[str, Any]]:
        """Build the coin pool applying the configured filters."""
        self.logger.info("Fetching market data from Binance")
        tickers = self._fetch_futures_tickers()
        volumes_24h = self._fetch_24h_volume()
        if not tickers or not volumes_24h:
            self.logger.error("Could not retrieve ticker or volume data; aborting")
            return []
        volume_map = {entry.get("symbol"): float(entry.get("quoteVolume", 0.0))
                      for entry in volumes_24h}
        candidates: List[CoinCandidate] = []
        for ticker in tickers:
            symbol = ticker.get("symbol", "")
            if not symbol.endswith("USDT"):
                continue
            try:
                bid = float(ticker.get("bidPrice", 0.0))
                ask = float(ticker.get("askPrice", 0.0))
            except (TypeError, ValueError):
                continue
            if not bid or not ask:
                continue
            volume = volume_map.get(symbol, 0.0)
            if volume < self.config.min_volume:
                continue
            spread = abs(ask - bid) / bid * 100
            if spread > self.config.max_spread:
                continue
            candidates.append(
                CoinCandidate(
                    symbol=symbol,
                    volume=volume,
                    spread=spread,
                    bid=bid,
                    ask=ask,
                )
            )
        self.logger.info("Volume + spread filter retained %s coins", len(candidates))
        accepted: List[Dict[str, Any]] = []
        rejected_low = rejected_high = rejected_no_data = 0
        for index, coin in enumerate(candidates, start=1):
            self.logger.debug("(%s/%s) Calculating ATR for %s", index, len(candidates), coin.symbol)
            atr = self._calculate_atr_percentage(coin.symbol)
            if atr is None:
                rejected_no_data += 1
                continue
            if atr < self.config.min_volatility:
                rejected_low += 1
                continue
            if atr > self.config.max_volatility:
                rejected_high += 1
                continue
            coin.atr = round(atr, 2)
            accepted.append(
                {
                    "symbol": coin.symbol,
                    "volume": coin.volume,
                    "spread": coin.spread,
                    "bid": coin.bid,
                    "ask": coin.ask,
                    "atr": coin.atr,
                    "market": "USDT-M Perpetual",
                }
            )
            if self.config.rate_limit_delay > 0:
                time.sleep(self.config.rate_limit_delay)
        self.logger.info("ATR filter accepted %s coins", len(accepted))
        self.logger.debug(
            "ATR filter stats – rejected low: %s, rejected high: %s, no data: %s",
            rejected_low,
            rejected_high,
            rejected_no_data,
        )
        accepted.sort(key=lambda coin: coin["volume"], reverse=True)
        top_coins = accepted[: self.config.top_coin_limit]
        if not top_coins:
            self.logger.warning("No coins matched all filters")
            return []
        self._print_summary(top_coins)
        self._store_pool(top_coins)
        return top_coins
    def load_or_update_pool(self, force: bool = False) -> List[Dict[str, Any]]:
        """Load the cached pool or rebuild it if necessary."""
        pool_file = self.config.pool_file
        if not force and pool_file.exists():
            try:
                with pool_file.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                self.logger.warning("Failed to read %s: %s", pool_file, exc)
            else:
                timestamp_str = data.get("timestamp")
                if timestamp_str:
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        self.logger.debug("Unexpected timestamp format: %s", timestamp_str)
                    else:
                        age = datetime.now() - timestamp
                        if age < timedelta(hours=self.config.update_interval_hours):
                            pool = data.get("pool", [])
                            self.logger.info(
                                "Loaded cached pool (%s coins, updated %s ago)",
                                len(pool),
                                self._format_timedelta(age),
                            )
                            return pool
        self.logger.info("Cache is missing or stale – rebuilding pool")
        return self.build_coin_pool()
    def auto_update_loop(self) -> None:
        """Run an infinite loop that refreshes the pool every interval."""
        delay_seconds = self.config.update_interval_hours * 3600
        self.logger.info(
            "Starting auto-update loop (interval: %s hours)", self.config.update_interval_hours
        )
        try:
            while True:
                self.build_coin_pool()
                self.logger.info("Sleeping for %s hours", self.config.update_interval_hours)
                time.sleep(delay_seconds)
        except KeyboardInterrupt:
            self.logger.info("Auto-update loop interrupted by user")
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _store_pool(self, pool: Iterable[Dict[str, Any]]) -> None:
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filters": self.config.to_filters_dict(),
            "pool": list(pool),
        }
        try:
            with self.config.pool_file.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=4)
        except OSError as exc:
            self.logger.error("Failed to write %s: %s", self.config.pool_file, exc)
        else:
            self.logger.info("Saved pool to %s", self.config.pool_file)
    def _print_summary(self, pool: List[Dict[str, Any]]) -> None:
        header = f"{'#':<3} {'Symbol':<12} {'Market':<20} {'Volume':<15} {'Spread':<10} {'ATR %':<10}"
        separator = "=" * len(header)
        self.logger.info(separator)
        self.logger.info("Selected coins (USDT-M Perpetual Futures):")
        self.logger.info(separator)
        self.logger.info(header)
        self.logger.info("-" * len(header))
        for idx, coin in enumerate(pool, start=1):
            self.logger.info(
                f"{idx:<3} {coin['symbol']:<12} {coin.get('market', 'USDT-PERP'):<20} "
                f"${coin['volume']:>13,.0f} {coin['spread']:>8.4f}% {coin['atr']:>8.2f}%"
            )
        self.logger.info(separator)
    @staticmethod
    def _format_timedelta(delta: timedelta) -> str:
        total_seconds = int(delta.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        parts = []
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if seconds and not parts:
            parts.append(f"{seconds}s")
        return " ".join(parts) or "0s"
# ==============================================================================
# Command-line interface
# ==============================================================================
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="USDT perpetual coin selector")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("load", "rebuild", "auto"),
        help="Operation to perform. If omitted the interactive menu is shown.",
    )
    parser.add_argument(
        "--pool-file",
        type=Path,
        default=DEFAULT_CONFIG.pool_file,
        help="Where to store the generated coin pool JSON.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_CONFIG.base_url,
        help="Binance Futures REST API base URL.",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=DEFAULT_CONFIG.min_volume,
        help="Minimum 24h quote volume (USD).",
    )
    parser.add_argument(
        "--max-spread",
        type=float,
        default=DEFAULT_CONFIG.max_spread,
        help="Maximum allowed bid/ask spread in percent.",
    )
    parser.add_argument(
        "--min-volatility",
        type=float,
        default=DEFAULT_CONFIG.min_volatility,
        help="Minimum ATR percentage.",
    )
    parser.add_argument(
        "--max-volatility",
        type=float,
        default=DEFAULT_CONFIG.max_volatility,
        help="Maximum ATR percentage.",
    )
    parser.add_argument(
        "--top-coin-limit",
        type=int,
        default=DEFAULT_CONFIG.top_coin_limit,
        help="Number of coins to keep after sorting by volume.",
    )
    parser.add_argument(
        "--atr-period",
        type=int,
        default=DEFAULT_CONFIG.atr_period,
        help="Number of candles to use for ATR calculation.",
    )
    parser.add_argument(
        "--kline-interval",
        default=DEFAULT_CONFIG.kline_interval,
        help="Kline interval (e.g. 1d, 4h).",
    )
    parser.add_argument(
        "--update-interval-hours",
        type=int,
        default=DEFAULT_CONFIG.update_interval_hours,
        help="How often the auto-update loop refreshes the pool.",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=DEFAULT_CONFIG.rate_limit_delay,
        help="Sleep duration between ATR requests.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuilding the pool when using the 'load' command.",
    )
    return parser
def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
def create_selector_from_args(args: argparse.Namespace) -> CoinSelector:
    config = SelectorConfig(
        pool_file=args.pool_file,
        min_volume=args.min_volume,
        max_spread=args.max_spread,
        min_volatility=args.min_volatility,
        max_volatility=args.max_volatility,
        top_coin_limit=args.top_coin_limit,
        update_interval_hours=args.update_interval_hours,
        atr_period=args.atr_period,
        kline_interval=args.kline_interval,
        rate_limit_delay=args.rate_limit_delay,
        base_url=args.base_url,
    )
    return CoinSelector(config=config)
def interactive_menu(selector: CoinSelector, force: bool) -> None:
    print("\n" + "=" * 60)
    print("USDT-PERP Coin Selector")
    print("   with Volatility Filter (ATR)")
    print("=" * 60 + "\n")
    print("Seçenekler:")
    print("1. Mevcut havuzu yükle (varsa)")
    print("2. Zorla yeniden oluştur")
    print("3. Otomatik güncelleme modu")
    choice = input("\nSeçiminiz (1/2/3): ").strip()
    if choice == "1":
        selector.load_or_update_pool(force=force)
    elif choice == "2":
        selector.build_coin_pool()
    elif choice == "3":
        selector.auto_update_loop()
    else:
        print("Geçersiz seçim! Varsayılan: Mevcut havuzu yükle")
        selector.load_or_update_pool(force=force)
def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    selector = create_selector_from_args(args)
    command = args.command
    if command == "load":
        selector.load_or_update_pool(force=args.force)
    elif command == "rebuild":
        selector.build_coin_pool()
    elif command == "auto":
        selector.auto_update_loop()
    else:
        interactive_menu(selector, force=args.force)
if __name__ == "__main__":
    main()