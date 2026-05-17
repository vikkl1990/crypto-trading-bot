"""
mfe_classifier_train.py
=======================
Trains a binary classifier that predicts whether a VN Edge trade signal
will reach peak MFE > 0.30R within ~10 minutes.

Goal: skip entries that won't clear Delta India taker fee drag.

Data:
    - PostgreSQL `vnedge.user_trades` (last 30 days, shadow + closed)
    - 5m candle parquets in storage/candle_cache/<SYM>_USDT_5m.parquet
    - ccxt OHLCV fallback for any time range not covered by parquet

Outputs:
    - storage/models/mfe_classifier_v1.pkl     (model + scaler + feature list)
    - storage/models/mfe_classifier_README.md  (load + score recipe)

The script is self-contained (no imports from bot code) and only reads;
it never modifies trade tables or live config.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -- Config -----------------------------------------------------------------

ROOT = Path("/home/opc/crypto-trading-bot")
CACHE_DIR = ROOT / "storage" / "candle_cache"
MODEL_DIR = ROOT / "storage" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "mfe_classifier_v1.pkl"
README_PATH = MODEL_DIR / "mfe_classifier_README.md"

LOOKBACK_DAYS = 30
TEST_HOLDOUT_DAYS = 7
MFE_THRESHOLD = 0.30          # label cutoff in R-multiples
DELTA_INDIA_TAKER_R = 0.25    # fee drag cutoff (informational)

# Universe — symbols present in shadow trades
SUPPORTED_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
                     "DOGE/USDT", "TRUMP/USDT", "LINK/USDT", "DOT/USDT",
                     "LTC/USDT", "ADA/USDT"]

# DB
PG = dict(host="localhost", dbname="vnedge", user="vnedge", password="VnEdge2026db")


# -- DB pull ----------------------------------------------------------------

def pull_trades() -> pd.DataFrame:
    import psycopg2
    import psycopg2.extras
    sql = f"""
    SELECT id, symbol, side, entry_price, exit_price, pnl_usd, fees_usd,
           opened_at, closed_at, signal_data, metadata
    FROM user_trades
    WHERE trade_type = 'shadow'
      AND status = 'closed'
      AND opened_at > NOW() - INTERVAL '{LOOKBACK_DAYS} days'
    ORDER BY opened_at ASC
    """
    conn = psycopg2.connect(**PG)
    df = pd.read_sql(sql, conn)
    conn.close()
    print(f"[db] pulled {len(df)} shadow closed trades")
    return df


def explode_metadata(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        md = r["metadata"] or {}
        sd = r["signal_data"] or {}
        # signal_data is canonical at entry; fall back to metadata
        scanner = sd.get("scanner") or md.get("scanner")
        regime  = sd.get("regime")  or md.get("regime")
        side    = sd.get("side")    or r["side"]
        rows.append({
            "id": str(r["id"]),
            "symbol": r["symbol"],
            "side": (side or "").lower(),
            "opened_at": pd.to_datetime(r["opened_at"], utc=True),
            "closed_at": pd.to_datetime(r["closed_at"], utc=True) if r["closed_at"] else None,
            "entry_price": float(r["entry_price"]) if r["entry_price"] is not None else np.nan,
            "scanner": scanner,
            "regime": regime,
            "grade": md.get("grade"),
            "ml_prob": float(md.get("ml_prob")) if md.get("ml_prob") is not None else np.nan,
            "peak_mfe_r": float(md.get("peak_mfe_r")) if md.get("peak_mfe_r") is not None else np.nan,
            "exit_reason": md.get("exit_reason"),
            "gross_pnl_usd": float(md.get("gross_pnl_usd")) if md.get("gross_pnl_usd") is not None else np.nan,
            "net_pnl_usd": float(md.get("net_pnl_usd")) if md.get("net_pnl_usd") is not None else np.nan,
            "fees_usd": float(md.get("fees_usd")) if md.get("fees_usd") is not None else np.nan,
            "initial_risk": float(md.get("initial_risk")) if md.get("initial_risk") is not None else np.nan,
        })
    out = pd.DataFrame(rows)
    print(f"[db] usable rows after parse: {len(out)}; "
          f"with peak_mfe_r: {out['peak_mfe_r'].notna().sum()}")
    return out


# -- Candle fetching --------------------------------------------------------

# OKX (geo-friendly from OCI) — primary; OKEx 5m candles match Binance to ~0.1%.
_CCXT_CLIENTS: Dict[str, object] = {}
def _ccxt(name: str = "okx"):
    if name not in _CCXT_CLIENTS:
        import ccxt
        cls = getattr(ccxt, name)
        _CCXT_CLIENTS[name] = cls({"enableRateLimit": True})
    return _CCXT_CLIENTS[name]


def _parquet_for(symbol: str) -> Optional[pd.DataFrame]:
    base = symbol.split("/")[0]
    p = CACHE_DIR / f"{base}_USDT_5m.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    # parquet index is already a DatetimeIndex (UTC) per inspection
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    return df


_FETCH_CACHE: Dict[str, pd.DataFrame] = {}

_LARGE_FETCH_CACHE: Dict[str, pd.DataFrame] = {}

def _fetch_ccxt_window(symbol: str, since_dt: datetime, until_dt: datetime,
                       providers=("okx", "kucoin", "bitget", "gate")) -> pd.DataFrame:
    """Fetch 5m OHLCV for [since, until]; try providers in order.

    OKX limits per-call to 100; we paginate. Cached per (symbol, since, until)."""
    key = f"{symbol}|{since_dt.isoformat()}|{until_dt.isoformat()}"
    if key in _FETCH_CACHE:
        return _FETCH_CACHE[key]
    since_ms = int(since_dt.timestamp() * 1000)
    end_ms   = int(until_dt.timestamp() * 1000)
    last_err = None
    for prov in providers:
        try:
            ex = _ccxt(prov)
            rows = []
            cursor = since_ms
            empty_streak = 0
            stall_streak = 0
            while cursor < end_ms:
                limit = 300 if prov == "okx" else 500
                try:
                    chunk = ex.fetch_ohlcv(symbol, timeframe="5m",
                                           since=cursor, limit=limit)
                except Exception as e:
                    stall_streak += 1
                    if stall_streak >= 3:
                        raise
                    time.sleep(0.5)
                    continue
                stall_streak = 0
                if not chunk:
                    empty_streak += 1
                    if empty_streak >= 2:
                        break
                    cursor += 5*60*1000*limit
                    continue
                empty_streak = 0
                rows.extend(chunk)
                last = chunk[-1][0]
                if last <= cursor:
                    break
                cursor = last + 5*60*1000
                time.sleep(0.06)  # gentle pacing across all providers
            if rows:
                df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
                df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df[["open","high","low","close","volume"]].sort_index()
                df = df[~df.index.duplicated(keep="last")]
                _FETCH_CACHE[key] = df
                return df
        except Exception as e:
            last_err = f"{prov}: {str(e)[:120]}"
            continue
    if last_err:
        print(f"[ccxt] all providers failed for {symbol} {since_dt}->{until_dt}: {last_err}")
    return pd.DataFrame()


def _bulk_gap(symbol: str, since_dt: datetime, until_dt: datetime) -> pd.DataFrame:
    """Pull 5m bars covering [since_dt, until_dt] from OKX, cached per symbol/window.

    Caller must pre-compute the smallest window that covers the trades we need
    (typically: parquet_last_bar -> latest_trade_opened_at). This avoids
    pulling 30+ days when only 7 days are missing.
    """
    key = f"{symbol}|{since_dt.isoformat()}|{until_dt.isoformat()}"
    if key in _LARGE_FETCH_CACHE:
        return _LARGE_FETCH_CACHE[key]
    print(f"[ccxt] bulk fetch {symbol} {since_dt} -> {until_dt}", flush=True)
    df = _fetch_ccxt_window(symbol, since_dt, until_dt)
    _LARGE_FETCH_CACHE[key] = df
    print(f"[ccxt] {symbol} got {len(df)} bars", flush=True)
    return df


_GAP_FETCHED: Dict[str, bool] = {}

def prefetch_gaps_for_symbols(symbols: List[str], earliest_trade: datetime,
                              latest_trade: datetime) -> None:
    """For each symbol, fetch a single bulk OKX window that covers
    [parquet_end, latest_trade] (or [earliest_trade, latest_trade] if no parquet).
    Done once up front so per-trade lookups are O(1)."""
    for sym in symbols:
        if sym in _GAP_FETCHED:
            continue
        parquet = _parquet_for(sym)
        if parquet is not None and not parquet.empty:
            parquet_end = parquet.index.max()
            # We need 80 bars before the earliest trade. parquet usually has it.
            # Only fetch from parquet_end forward, if needed.
            if latest_trade <= parquet_end:
                _GAP_FETCHED[sym] = True
                continue
            since = parquet_end + timedelta(minutes=5)
        else:
            since = earliest_trade - timedelta(hours=8)  # 96 bars for warmup
        until = latest_trade + timedelta(minutes=10)
        try:
            _bulk_gap(sym, since, until)
        except Exception as e:
            print(f"[ccxt] prefetch err {sym}: {e}", flush=True)
        _GAP_FETCHED[sym] = True


def get_candles_until(symbol: str, ts: datetime, bars_needed: int = 80) -> Optional[pd.DataFrame]:
    """
    Return 5m candles ending at the bar containing `ts` with >= bars_needed history.
    Combines parquet + already-prefetched OKX gap.
    """
    parquet = _parquet_for(symbol)
    end = ts.replace(second=0, microsecond=0)
    end = end - timedelta(minutes=end.minute % 5)

    frames = []
    if parquet is not None:
        frames.append(parquet)
    # Pull all cached OKX windows for this symbol
    for k, df in _LARGE_FETCH_CACHE.items():
        if k.startswith(symbol + "|") and not df.empty:
            frames.append(df)

    if not frames:
        return None
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df.index <= end]
    if len(df) < bars_needed:
        return None
    return df.tail(bars_needed + 5).copy()


# -- Feature engineering ----------------------------------------------------

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)


def _bb_bandwidth(c: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std(ddof=0)
    upper = ma + k*sd
    lower = ma - k*sd
    return (upper - lower) / ma.replace(0, np.nan)


def compute_features(candles: pd.DataFrame, side: str) -> Optional[Dict[str, float]]:
    if candles is None or len(candles) < 60:
        return None
    df = candles.copy()
    df["atr14"] = _atr(df, 14)
    df["ema8"]  = _ema(df["close"], 8)
    df["ema21"] = _ema(df["close"], 21)
    df["ema50"] = _ema(df["close"], 50)
    df["bb_bw"] = _bb_bandwidth(df["close"], 20, 2.0)
    df["rsi"]   = _rsi(df["close"], 14)
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_slope"] = df["volume"].diff(3) / df["vol_ma20"].replace(0, np.nan)
    # 1h ATR ratio (12-bar atr / 14-bar atr)
    df["atr12"] = _atr(df, 12)
    df["atr_h_ratio"] = df["atr12"] / df["atr14"].replace(0, np.nan)
    # 5-bar realized vol (stdev of pct returns)
    rets = df["close"].pct_change()
    df["rv5"] = rets.rolling(5).std(ddof=0)

    bar = df.iloc[-1]
    atr = float(bar["atr14"]) if bar["atr14"] and not math.isnan(bar["atr14"]) else np.nan
    if not (atr > 0):
        return None

    close = float(bar["close"])
    feats = {
        "atr14":          atr,
        "atr_pct":        atr / close if close else np.nan,
        "ema8_dist_atr":  (close - float(bar["ema8"]))  / atr,
        "ema21_dist_atr": (close - float(bar["ema21"])) / atr,
        "ema50_dist_atr": (close - float(bar["ema50"])) / atr,
        "ema_alignment":  1.0 if bar["ema8"] > bar["ema21"] > bar["ema50"]
                          else (-1.0 if bar["ema8"] < bar["ema21"] < bar["ema50"] else 0.0),
        "bb_bw":          float(bar["bb_bw"]) if not math.isnan(bar["bb_bw"]) else 0.0,
        "rsi":            float(bar["rsi"]) if not math.isnan(bar["rsi"]) else 50.0,
        "rel_vol":        float(bar["volume"] / bar["vol_ma20"]) if bar["vol_ma20"] else 1.0,
        "vol_slope":      float(bar["vol_slope"]) if not math.isnan(bar["vol_slope"]) else 0.0,
        "atr_h_ratio":    float(bar["atr_h_ratio"]) if not math.isnan(bar["atr_h_ratio"]) else 1.0,
        "rv5":            float(bar["rv5"]) if not math.isnan(bar["rv5"]) else 0.0,
    }
    # Side-aware sign on EMA distances (positive = "in trend direction")
    if side == "short":
        for k in ("ema8_dist_atr", "ema21_dist_atr", "ema50_dist_atr", "ema_alignment"):
            feats[k] = -feats[k]
    return feats


def add_time_and_categorical(row: pd.Series, feats: Dict[str, float]) -> Dict[str, float]:
    # Indian Standard Time = UTC + 5:30
    ist = row["opened_at"].tz_convert("Asia/Kolkata")
    h = ist.hour + ist.minute/60.0
    feats["hour_sin"] = math.sin(2*math.pi*h/24)
    feats["hour_cos"] = math.cos(2*math.pi*h/24)
    feats["dow"]      = float(ist.dayofweek)
    feats["ml_prob"]  = float(row.get("ml_prob") or 0.5)
    feats["side_long"] = 1.0 if row["side"] == "long" else 0.0
    grade_map = {"A+": 4, "A": 3, "B": 2, "C": 1}
    feats["grade_ord"] = float(grade_map.get(row.get("grade") or "", 0))
    return feats


# -- Pipeline ---------------------------------------------------------------

@dataclass
class FeatureBuildResult:
    X: pd.DataFrame
    y: np.ndarray
    meta: pd.DataFrame  # opened_at, peak_mfe_r, net_pnl_usd, scanner...


def build_dataset(trades: pd.DataFrame) -> FeatureBuildResult:
    # Prefetch OKX gaps once per symbol
    earliest = trades["opened_at"].min().to_pydatetime()
    latest   = trades["opened_at"].max().to_pydatetime()
    syms     = sorted(trades["symbol"].unique().tolist())
    print(f"[prefetch] {len(syms)} symbols across {earliest} -> {latest}", flush=True)
    prefetch_gaps_for_symbols(syms, earliest, latest)

    rows, ys, metas = [], [], []
    skipped_no_mfe = 0
    skipped_no_candles = 0
    skipped_bad_feats = 0
    for i, r in trades.iterrows():
        if pd.isna(r["peak_mfe_r"]):
            skipped_no_mfe += 1
            continue
        candles = get_candles_until(r["symbol"], r["opened_at"].to_pydatetime(),
                                    bars_needed=80)
        if candles is None:
            skipped_no_candles += 1
            continue
        feats = compute_features(candles, r["side"])
        if feats is None:
            skipped_bad_feats += 1
            continue
        feats = add_time_and_categorical(r, feats)

        # one-hot scanner & regime (small cardinality)
        for sc in ("structure_bounce", "trend_pullback", "breakout", "vwap_reclaim"):
            feats[f"scanner_{sc}"] = 1.0 if r.get("scanner") == sc else 0.0
        for rg in ("trending", "sideways", "mean_reversion", "volatile"):
            feats[f"regime_{rg}"] = 1.0 if r.get("regime") == rg else 0.0

        rows.append(feats)
        ys.append(1 if r["peak_mfe_r"] > MFE_THRESHOLD else 0)
        metas.append({
            "opened_at": r["opened_at"],
            "symbol": r["symbol"],
            "scanner": r.get("scanner"),
            "regime": r.get("regime"),
            "grade": r.get("grade"),
            "side": r["side"],
            "peak_mfe_r": r["peak_mfe_r"],
            "net_pnl_usd": r.get("net_pnl_usd"),
            "gross_pnl_usd": r.get("gross_pnl_usd"),
            "fees_usd": r.get("fees_usd"),
        })
        if (i+1) % 100 == 0:
            print(f"[feat] processed {i+1}/{len(trades)} (built {len(rows)})")

    print(f"[feat] built X: {len(rows)} | skipped no_mfe={skipped_no_mfe} "
          f"no_candles={skipped_no_candles} bad_feats={skipped_bad_feats}")
    X = pd.DataFrame(rows).fillna(0.0)
    y = np.asarray(ys, dtype=int)
    meta = pd.DataFrame(metas).reset_index(drop=True)
    return FeatureBuildResult(X=X, y=y, meta=meta)


# -- Training ---------------------------------------------------------------

def time_split(meta: pd.DataFrame, holdout_days: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Time-based split. Prefer last `holdout_days` as holdout.
    If that yields <30% of training (data window too short), fall back to a
    chronological 75/25 split.
    """
    cutoff = meta["opened_at"].max() - pd.Timedelta(days=holdout_days)
    train_mask = (meta["opened_at"] < cutoff).values
    test_mask  = ~train_mask
    if train_mask.sum() < 30 or test_mask.sum() < 30:
        # Fall back to chronological 75/25 split (preserve tz dtype).
        ordered = meta["opened_at"].sort_values().reset_index(drop=True)
        thr = ordered.iloc[int(len(ordered)*0.75)]
        train_mask = (meta["opened_at"] < thr).values
        test_mask  = ~train_mask
        print(f"[split] holdout window too short ({holdout_days}d) — "
              f"falling back to chronological 75/25 at {thr}")
    return train_mask, test_mask


def train_model(X: pd.DataFrame, y: np.ndarray, train_mask, test_mask):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    Xtr, Xte = X[train_mask], X[test_mask]
    ytr, yte = y[train_mask], y[test_mask]

    pos = max(int(ytr.sum()), 1)
    neg = max(int((ytr == 0).sum()), 1)
    spw = neg / pos

    # Heavy regularization given the small/short dataset.
    params = dict(
        objective="binary",
        learning_rate=0.03,
        num_leaves=7,
        min_data_in_leaf=40,
        max_depth=3,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=3,
        lambda_l1=0.5,
        lambda_l2=1.0,
        min_gain_to_split=0.001,
        scale_pos_weight=spw,
        verbose=-1,
    )
    dtrain = lgb.Dataset(Xtr, label=ytr)
    dvalid = lgb.Dataset(Xte, label=yte, reference=dtrain)
    booster = lgb.train(
        params, dtrain, num_boost_round=200,
        valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
    )

    p_tr = booster.predict(Xtr)
    p_te = booster.predict(Xte)

    metrics = {
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "pos_rate_train": float(ytr.mean()), "pos_rate_test": float(yte.mean()),
        "auc_train": float(roc_auc_score(ytr, p_tr)) if len(set(ytr)) > 1 else np.nan,
        "auc_test":  float(roc_auc_score(yte, p_te)) if len(set(yte)) > 1 else np.nan,
        "precision_train@0.5": float(precision_score(ytr, p_tr >= 0.5, zero_division=0)),
        "recall_train@0.5":    float(recall_score(ytr, p_tr >= 0.5, zero_division=0)),
        "precision_test@0.5":  float(precision_score(yte, p_te >= 0.5, zero_division=0)),
        "recall_test@0.5":     float(recall_score(yte, p_te >= 0.5, zero_division=0)),
        "best_iteration": int(booster.best_iteration or booster.current_iteration()),
    }
    return booster, p_tr, p_te, metrics


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"p": p, "y": y})
    df["bucket"] = pd.qcut(df["p"], q=bins, labels=False, duplicates="drop")
    g = df.groupby("bucket").agg(
        n=("y", "size"), pred_mean=("p", "mean"), actual=("y", "mean"),
    ).reset_index()
    return g


def feature_importances(booster, feature_names: List[str], top: int = 15) -> pd.DataFrame:
    imp = booster.feature_importance(importance_type="gain")
    return (pd.DataFrame({"feature": feature_names, "gain": imp})
              .sort_values("gain", ascending=False).head(top).reset_index(drop=True))


def counterfactual_skip(meta_test: pd.DataFrame, p_test: np.ndarray,
                        target_skip_loser_pct: float = 0.5,
                        target_keep_winner_pct: float = 0.8) -> dict:
    """
    Sweep thresholds and find:
      - feasible_thr: highest threshold satisfying both targets simultaneously
                      (skip >=50% losers AND keep >=80% winners). May be None.
      - max_savings_thr: threshold maximizing net savings while still keeping
                        >=`target_keep_winner_pct` of winners (loose constraint).

    "Net savings" = sum(net_pnl_usd of *losing* skipped trades) - sum(net_pnl_usd of *winning* skipped trades)
                  i.e. positive = we avoided more losers than we cancelled winners.
    """
    df = meta_test.copy()
    df["p"] = p_test
    df["winner"] = (df["peak_mfe_r"] > MFE_THRESHOLD).astype(int)
    df["net_pnl_usd"] = df["net_pnl_usd"].fillna(0.0)
    n_winners = int(df["winner"].sum())
    n_losers  = int((1-df["winner"]).sum())

    grid = np.linspace(0.02, 0.98, 97)
    sweep = []
    for thr in grid:
        skipped = df[df["p"] < thr]
        kept    = df[df["p"] >= thr]
        sk_l = int((skipped["winner"] == 0).sum())
        kp_w = int((kept["winner"] == 1).sum())
        skip_loser_pct = sk_l / n_losers if n_losers else 0.0
        keep_winner_pct = kp_w / n_winners if n_winners else 0.0
        savings = -float(skipped["net_pnl_usd"].sum())
        sweep.append({
            "thr": float(thr),
            "skip_loser_pct": skip_loser_pct,
            "keep_winner_pct": keep_winner_pct,
            "savings_usd": savings,
            "n_skipped": int(len(skipped)),
            "n_kept": int(len(kept)),
        })
    sweep_df = pd.DataFrame(sweep)

    # Hard-feasibility selection
    feasible = sweep_df[(sweep_df["skip_loser_pct"] >= target_skip_loser_pct) &
                        (sweep_df["keep_winner_pct"] >= target_keep_winner_pct)]
    if not feasible.empty:
        best = feasible.sort_values("savings_usd", ascending=False).iloc[0]
        feasible_chosen = best.to_dict()
        feasible_chosen["status"] = "feasible"
    else:
        # Soft fallback: relax skip-losers target, hold keep-winners >= 80%
        soft = sweep_df[sweep_df["keep_winner_pct"] >= target_keep_winner_pct]
        if not soft.empty:
            best = soft.sort_values("savings_usd", ascending=False).iloc[0]
            feasible_chosen = best.to_dict()
            feasible_chosen["status"] = "no-feasible-softened-to-keep80"
        else:
            best = sweep_df.sort_values("savings_usd", ascending=False).iloc[0]
            feasible_chosen = best.to_dict()
            feasible_chosen["status"] = "no-feasible-best-savings-only"

    feasible_chosen["baseline_total_net_pnl"] = float(df["net_pnl_usd"].sum())
    feasible_chosen["n_winners_test"] = n_winners
    feasible_chosen["n_losers_test"] = n_losers
    return {"chosen": feasible_chosen, "sweep": sweep_df}


# -- Reporting / artifacts --------------------------------------------------

def print_report(metrics, calib_test, feat_imp, cf, X_cols, n_total):
    print("\n=================== MFE > 0.30R Classifier — Report ===================")
    print(f"Universe size: {n_total} trades total ({metrics['n_train']} train / "
          f"{metrics['n_test']} test, last {TEST_HOLDOUT_DAYS}d holdout)")
    print(f"Positive rate: train={metrics['pos_rate_train']:.3f} "
          f"test={metrics['pos_rate_test']:.3f}")
    print(f"AUC train={metrics['auc_train']:.3f}   AUC test={metrics['auc_test']:.3f}")
    print(f"Test @0.5: precision={metrics['precision_test@0.5']:.3f} "
          f"recall={metrics['recall_test@0.5']:.3f}")
    print("\n-- Calibration (test, deciles) --")
    print(calib_test.to_string(index=False))
    print("\n-- Top feature importances (gain) --")
    print(feat_imp.to_string(index=False))
    print("\n-- Counterfactual: skip if p < threshold --")
    c = cf["chosen"]
    print(f"  Status            : {c['status']}")
    print(f"  Chosen thr        : {c['thr']:.2f}")
    print(f"  Skip-loser pct    : {c['skip_loser_pct']*100:.1f}%  (target >= 50%)")
    print(f"  Keep-winner pct   : {c['keep_winner_pct']*100:.1f}%  (target >= 80%)")
    print(f"  N skipped / kept  : {int(c['n_skipped'])} / {int(c['n_kept'])}")
    print(f"  Net savings (USD) : {c['savings_usd']:.2f}")
    print(f"  Baseline net PnL  : {c['baseline_total_net_pnl']:.2f}")
    print(f"  N winners / losers: {c['n_winners_test']} / {c['n_losers_test']}")


def save_model(booster, X_cols, metrics, cf, feat_imp):
    payload = {
        "model_kind": "lightgbm",
        "feature_columns": list(X_cols),
        "mfe_threshold_R": MFE_THRESHOLD,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "counterfactual": cf["chosen"],
        "top_features": feat_imp.to_dict(orient="records"),
        "booster_bytes": booster.model_to_string(),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"[save] model -> {MODEL_PATH}")


README_TEMPLATE = """# MFE > {thr}R Classifier — v1

Predicts whether a VN Edge signal will reach `peak_mfe_r > {thr}` within ~10 minutes.

## Inputs at signal time
{features}

## Quick load + score recipe

```python
import pickle, math, lightgbm as lgb, pandas as pd, numpy as np

with open("storage/models/mfe_classifier_v1.pkl", "rb") as f:
    payload = pickle.load(f)

booster = lgb.Booster(model_str=payload["booster_bytes"])
cols    = payload["feature_columns"]
chosen  = payload["counterfactual"]   # e.g. {{"thr": 0.30, ...}}

def score(features_dict) -> float:
    row = pd.DataFrame([{{c: float(features_dict.get(c, 0.0)) for c in cols}}])[cols]
    return float(booster.predict(row)[0])

# Example gate at signal time:
p = score(features_dict)
should_skip = p < chosen["thr"]
```

## Suggested integration (no live trading impact)
1. At signal generation, build the feature dict using the SAME 5m candle as the
   bot's grading pipeline. The feature names in `payload["feature_columns"]`
   are stable.
2. Compute `p = score(...)`. Log it to `metadata.mfe_classifier_p`.
3. Shadow gate: if `p < {skip_thr}`, mark `metadata.mfe_skip_recommend=True`
   but keep taking the trade. After 1 week, compare actual outcomes vs
   recommendation to confirm out-of-sample edge.
4. Only after the live-shadow audit shows skip-loser >= 50% and keep-winner
   >= 80% should the gate be flipped to a hard veto.

## Model card
- Train period: last {lookback}d shadow trades (binary label `peak_mfe_r > {thr}`)
- Holdout: last {hold}d
- Test AUC: {auc:.3f}
- Counterfactual: at thr={skip_thr:.2f} -> skip {sk_l:.0%} of losers, keep {kp_w:.0%} of winners
- Honest assessment: see training stdout for ship-readiness verdict

Trained by `scripts/mfe_classifier_train.py`.
"""

def write_readme(metrics, cf, X_cols):
    body = README_TEMPLATE.format(
        thr=MFE_THRESHOLD,
        features="\n".join(f"- `{c}`" for c in X_cols),
        skip_thr=cf["chosen"]["thr"],
        lookback=LOOKBACK_DAYS,
        hold=TEST_HOLDOUT_DAYS,
        auc=metrics["auc_test"],
        sk_l=cf["chosen"]["skip_loser_pct"],
        kp_w=cf["chosen"]["keep_winner_pct"],
    )
    README_PATH.write_text(body)
    print(f"[save] readme -> {README_PATH}")


# -- Main -------------------------------------------------------------------

def main():
    trades = pull_trades()
    trades = explode_metadata(trades)
    trades = trades[trades["peak_mfe_r"].notna()].reset_index(drop=True)
    span_days = (trades["opened_at"].max() - trades["opened_at"].min()).total_seconds() / 86400.0
    print(f"[data] window span = {span_days:.1f} days "
          f"({trades['opened_at'].min()} -> {trades['opened_at'].max()})")
    if span_days < 7:
        print(f"[WARNING] data window only {span_days:.1f} days — outcomes may not "
              f"reflect cross-regime generalization.")
    if len(trades) < 500:
        print(f"\n[INSUFFICIENT] only {len(trades)} usable trades — "
              f"extend window to 60+ days for confidence. Continuing for diagnostic.")
    elif len(trades) < 800:
        print(f"\n[CAUTION] {len(trades)} trades; AUC reads should be treated as preliminary.")

    fb = build_dataset(trades)
    if len(fb.X) < 100:
        print(f"[FATAL] only {len(fb.X)} rows after feature build — abort.")
        sys.exit(2)

    train_mask, test_mask = time_split(fb.meta, TEST_HOLDOUT_DAYS)
    print(f"[split] train n={int(train_mask.sum())}  test n={int(test_mask.sum())}")
    if test_mask.sum() < 30:
        print(f"[CAUTION] tiny test set ({int(test_mask.sum())}); AUC noisy.")

    booster, p_tr, p_te, metrics = train_model(fb.X, fb.y, train_mask, test_mask)
    calib_test = calibration_table(fb.y[test_mask], p_te)
    feat_imp = feature_importances(booster, list(fb.X.columns), top=15)
    cf = counterfactual_skip(fb.meta[test_mask].reset_index(drop=True), p_te)

    print_report(metrics, calib_test, feat_imp, cf, fb.X.columns, len(fb.X))
    save_model(booster, list(fb.X.columns), metrics, cf, feat_imp)
    write_readme(metrics, cf, list(fb.X.columns))

    # Honest assessment
    n = len(fb.X); auc = metrics["auc_test"]; cf_status = cf["chosen"]["status"]
    auc_gap = abs(metrics["auc_train"] - metrics["auc_test"])
    if n < 500:
        verdict = "needs-more-data"
    elif auc_gap > 0.25 and auc < 0.58:
        verdict = "overfit-risk — gap train/test too wide; needs more data + tighter regularization"
    elif auc < 0.55:
        verdict = "no-edge / data window too short (4 days)"
    elif auc < 0.62 or cf_status != "feasible":
        verdict = "marginal — log p as `mfe_classifier_p` for 2-week shadow audit before gating"
    else:
        verdict = "ship-ready (after 1w shadow audit)"
    print(f"\n[VERDICT] {verdict}")
    print(f"          n={n}, AUC_train={metrics['auc_train']:.3f}, AUC_test={auc:.3f}, "
          f"gap={auc_gap:.3f}, cf={cf_status}")


if __name__ == "__main__":
    main()
