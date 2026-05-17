"""
EMA200/EMA100 macro bias computation across BTC/ETH/SOL/XRP.

Loads cached 4h parquets, extends to "now" via OKX (Binance is geo-blocked),
computes EMA50/100/200 on 4h, resamples to daily and recomputes there.

Output: storage/ema200_data/macro_bias.parquet with columns
(symbol, ts_ms, ts, close,
 ema50_4h, ema100_4h, ema200_4h, macro_bias_4h_50, macro_bias_4h_100, macro_bias_4h_200,
 ema50_d,  ema100_d,  ema200_d,  macro_bias_d_50,  macro_bias_d_100,  macro_bias_d_200)

macro_bias_*_N: +1 if close > emaN, -1 if close < emaN, 0 if |close-emaN|/emaN <= 0.005

Read-only on production code; this is a tooling-only script.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

LOG = logging.getLogger("ema200_macro")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO = Path("/home/opc/crypto-trading-bot")
PARQUET_DIR = REPO / "storage" / "candle_cache"
OUT_DIR = REPO / "storage" / "ema200_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ("BTC", "ETH", "SOL", "XRP")
CHOP_BAND = 0.005  # 0.5%


def _ema(series: pd.Series, length: int) -> pd.Series:
    """Standard pandas EMA (adjust=False) — matches TA-Lib / TradingView."""
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def _classify_bias(close: pd.Series, ema: pd.Series, band: float = CHOP_BAND) -> pd.Series:
    diff = (close - ema) / ema
    bias = pd.Series(0, index=close.index, dtype="int8")
    bias[diff > band] = 1
    bias[diff < -band] = -1
    return bias


def _load_4h(symbol: str) -> pd.DataFrame:
    pq = PARQUET_DIR / f"{symbol}_USDT_4h.parquet"
    df = pd.read_parquet(pq)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        else:
            df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_index()
    df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    df = df[~df.index.duplicated(keep="last")]
    return df[["open", "high", "low", "close", "volume"]]


def _extend_via_okx(symbol: str, df_existing: pd.DataFrame) -> pd.DataFrame:
    """Fetch any 4h candles after df_existing.index.max() from OKX and append."""
    try:
        import ccxt
    except ImportError:
        LOG.warning("ccxt not available; skipping extension for %s", symbol)
        return df_existing
    ex = ccxt.okx()
    pair = f"{symbol}/USDT"
    last_ts = df_existing.index.max()
    if last_ts is None or pd.isna(last_ts):
        since_ms = int((time.time() - 60 * 86400) * 1000)
    else:
        since_ms = int(last_ts.timestamp() * 1000) + 1
    now_ms = int(time.time() * 1000)
    if since_ms >= now_ms - 4 * 3600 * 1000:
        return df_existing  # nothing meaningful to add

    rows: list = []
    cursor = since_ms
    iters = 0
    while cursor < now_ms and iters < 30:
        try:
            ohlcv = ex.fetch_ohlcv(pair, "4h", since=cursor, limit=300)
        except Exception as e:  # noqa: BLE001
            LOG.warning("OKX fetch failed for %s: %s", pair, e)
            break
        if not ohlcv:
            break
        rows.extend(ohlcv)
        last = ohlcv[-1][0]
        if last <= cursor:
            break
        cursor = last + 1
        iters += 1
        time.sleep(0.25)

    if not rows:
        return df_existing
    new = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    new = new.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    new.index = pd.to_datetime(new["timestamp"], unit="ms", utc=True)
    new = new[["open", "high", "low", "close", "volume"]]
    new = new[new.index > df_existing.index.max()]
    if new.empty:
        return df_existing
    LOG.info("[%s] extended %d 4h candles via OKX (latest %s)", symbol, len(new), new.index.max())
    out = pd.concat([df_existing, new]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _build_daily(df_4h: pd.DataFrame) -> pd.DataFrame:
    daily = df_4h.resample("1D", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    daily = daily.dropna(subset=["close"])
    return daily


def _attach_emas_4h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema50_4h"] = _ema(df["close"], 50)
    df["ema100_4h"] = _ema(df["close"], 100)
    df["ema200_4h"] = _ema(df["close"], 200)
    df["macro_bias_4h_50"] = _classify_bias(df["close"], df["ema50_4h"])
    df["macro_bias_4h_100"] = _classify_bias(df["close"], df["ema100_4h"])
    df["macro_bias_4h_200"] = _classify_bias(df["close"], df["ema200_4h"])
    return df


def _attach_emas_daily(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["ema50_d"] = _ema(daily["close"], 50)
    daily["ema100_d"] = _ema(daily["close"], 100)
    daily["ema200_d"] = _ema(daily["close"], 200)
    daily["macro_bias_d_50"] = _classify_bias(daily["close"], daily["ema50_d"])
    daily["macro_bias_d_100"] = _classify_bias(daily["close"], daily["ema100_d"])
    daily["macro_bias_d_200"] = _classify_bias(daily["close"], daily["ema200_d"])
    return daily[["close", "ema50_d", "ema100_d", "ema200_d",
                  "macro_bias_d_50", "macro_bias_d_100", "macro_bias_d_200"]]


def _merge_daily_onto_4h(df_4h: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Forward-merge: at each 4h ts, look up the *most recent CLOSED* daily bar.

    Daily index is right-labelled (00:00 UTC of next day = close of prior day).
    For 4h ts t, the macro_bias_d should reflect the daily close at the most
    recent UTC midnight <= t. Use merge_asof.
    """
    daily_lookup = daily.copy()
    daily_lookup.index.name = "daily_ts"
    daily_lookup = daily_lookup.reset_index()
    df_4h = df_4h.copy()
    df_4h.index.name = "ts"
    df_4h = df_4h.reset_index()
    merged = pd.merge_asof(
        df_4h.sort_values("ts"),
        daily_lookup.sort_values("daily_ts").rename(columns={"close": "close_d"}),
        left_on="ts",
        right_on="daily_ts",
        direction="backward",
    )
    merged = merged.set_index("ts")
    return merged


def process_symbol(symbol: str) -> pd.DataFrame:
    df = _load_4h(symbol)
    LOG.info("[%s] loaded %d 4h candles (%s -> %s)", symbol, len(df), df.index.min(), df.index.max())
    df = _extend_via_okx(symbol, df)
    df = _attach_emas_4h(df)

    daily_raw = _build_daily(df)
    daily = _attach_emas_daily(daily_raw)
    LOG.info("[%s] daily bars=%d (%s -> %s)", symbol, len(daily), daily.index.min(), daily.index.max())

    merged = _merge_daily_onto_4h(df, daily)
    merged["symbol"] = symbol
    merged["ts_ms"] = (merged.index.astype("int64") // 1_000_000).astype("int64")

    keep = [
        "symbol", "ts_ms", "close",
        "ema50_4h", "ema100_4h", "ema200_4h",
        "macro_bias_4h_50", "macro_bias_4h_100", "macro_bias_4h_200",
        "close_d", "ema50_d", "ema100_d", "ema200_d",
        "macro_bias_d_50", "macro_bias_d_100", "macro_bias_d_200",
    ]
    keep = [c for c in keep if c in merged.columns]
    out = merged[keep].copy()
    out.index.name = "ts"
    return out


def main() -> int:
    frames: list[pd.DataFrame] = []
    for sym in SYMBOLS:
        try:
            frames.append(process_symbol(sym))
        except Exception as e:  # noqa: BLE001
            LOG.error("[%s] processing failed: %s", sym, e)
    if not frames:
        LOG.error("No frames produced; aborting")
        return 1

    big = pd.concat(frames).sort_values(["symbol", "ts_ms"]).reset_index()
    out_path = OUT_DIR / "macro_bias.parquet"
    big.to_parquet(out_path, index=False)
    LOG.info("Wrote %s rows to %s", len(big), out_path)

    # quick health summary
    for sym in SYMBOLS:
        sub = big[big["symbol"] == sym]
        if sub.empty:
            continue
        latest = sub.iloc[-1]
        LOG.info(
            "[%s] latest ts=%s close=%.2f ema200_d=%.2f bias_d_200=%+d ema100_d=%.2f bias_d_100=%+d",
            sym, latest["ts"], latest["close"],
            latest.get("ema200_d", float("nan")), int(latest.get("macro_bias_d_200", 0) or 0),
            latest.get("ema100_d", float("nan")), int(latest.get("macro_bias_d_100", 0) or 0),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
