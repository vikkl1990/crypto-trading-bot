"""W/F STUDY: Fractional Kelly sizing vs Patch H matrix vs Fixed.

Hypothesis: Replace the bot's empirical Patch H 27-cell sizing matrix
(execution/user_real_manager.py:GRADE_BY_REGIME_SIZE_MULT) with one Kelly
formula:

    f* = (p * b - q) / b      (full Kelly)
    f  = alpha * f*           (fractional Kelly, alpha in [0.25, 0.5])

where p = ml_probability, b = RR = (TP-entry)/(entry-SL), q = 1-p.

This is a SIZING study (not selection). The trade SET is the same across
sizing modes; only NOTIONAL differs. We're validating whether Kelly produces
better RISK-ADJUSTED returns (Sharpe, Sortino, max DD, profit factor) than
Patch H matrix.

Setup detection (rebuild structure_bounce minimally):
  - 5m bar shows S/R rejection wick (wick >= 0.55 * ATR)
  - Inside structure zone: price within 1 ATR of recent 20-bar swing
  - Volume confirmation: rel_vol >= 1.0
  - LONG at swing low rejection, SHORT at swing high rejection
  - Entry at signal bar close. SL = swing extreme +/- 0.3 * ATR.
  - Hard time stop 30 min (6 bars).

Per-signal pricing:
  - p (ml_prob proxy) = 0.5 + 0.05 * confluence_count + 0.1 * wick_atr,
    clamped to [0.4, 0.85]
  - b = RR
  - f_star = (p*b - (1-p)) / b
  - f = alpha * f_star, capped to [0.01, 0.25] of bankroll
  - Skip if f_star < 0 (negative EV)

Sizing modes:
  - "patch_h": notional = 400 * GRADE_BY_REGIME_MULT
  - "fixed":   notional = 400 (control)
  - "kelly":   notional = bankroll * f (compounded)

Grade lookup (for patch_h mode): grade derived from p:
  p >= 0.75 -> A+, >= 0.65 -> A, >= 0.55 -> B, else C (skipped)

Regime lookup (for patch_h mode): ATR-percentile based:
  pctile >= 80 -> high_volatility, <= 20 -> sideways, else trending_up.

Special metrics: Sharpe, Sortino, max_dd, profit_factor per cell.
PASS criteria: standard + max_dd < 25%.
"""
from __future__ import annotations
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.wf_harness import (
    Strategy, Trade, WalkForwardEngine, FEE_VARIANTS, QUARTERS,
    IS_QUARTERS, OOS_QUARTERS, PASS_OOS_GAP_MAX, PASS_Q4_EV_MIN, PASS_IS_EV_MIN,
)

BAR_SEC = 300
TIME_STOP_SEC = 1800
TIME_STOP_BARS = TIME_STOP_SEC // BAR_SEC  # 6 bars
SL_ATR_BUFFER = 0.3
SWING_WINDOW = 20
ATR_PERIOD = 14
BASE_NOTIONAL = 400.0
MAX_DD_THRESHOLD = 0.25  # NEW: reject configs with max DD > 25%
H4_SEC = 14400
H4_ATR_PERIOD = 14
EXPANSION_MIN_RATIO = 1.3  # 4h phase filter — locks in positive edge baseline

# Patch H Grade x Regime matrix (from user_real_manager.py:1264)
PATCH_H_MULT: Dict[Tuple[str, str], float] = {
    ("trending_up", "A+"): 1.00, ("trending_up", "A"): 0.80, ("trending_up", "B"): 0.60,
    ("trending_down", "A+"): 0.30, ("trending_down", "A"): 0.50, ("trending_down", "B"): 0.60,
    ("breakout", "A+"): 0.50, ("breakout", "A"): 0.60, ("breakout", "B"): 0.70,
    ("high_volatility", "A+"): 0.50, ("high_volatility", "A"): 0.80, ("high_volatility", "B"): 1.20,
    ("sideways", "A+"): 0.25,
    ("sideways", "A"): 0.80, ("sideways", "B"): 0.60,
    ("mean_reversion", "A+"): 0.40, ("mean_reversion", "A"): 0.50, ("mean_reversion", "B"): 0.50,
}
PATCH_H_DEFAULT = 0.80


# ─────────────────────────────────────────────────────────────────────
# Numpy ATR
# ─────────────────────────────────────────────────────────────────────
def add_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    n = len(high)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        a = high[i] - low[i]
        b = abs(high[i] - close[i - 1])
        c = abs(low[i] - close[i - 1])
        v = a
        if b > v:
            v = b
        if c > v:
            v = c
        tr[i] = v
    atr = np.full(n, np.nan, dtype=np.float64)
    csum = np.cumsum(tr)
    for i in range(period - 1, n):
        if i == period - 1:
            atr[i] = csum[i] / period
        else:
            atr[i] = (csum[i] - csum[i - period]) / period
    return atr


def _walk_forward_exit_np(high_a: np.ndarray, low_a: np.ndarray,
                          close_a: np.ndarray, entry_idx: int, side: str,
                          entry: float, sl: float, tp: float,
                          max_bars: int) -> Tuple[int, float, str]:
    n = len(close_a)
    end = min(entry_idx + 1 + max_bars, n)
    for j in range(entry_idx + 1, end):
        if side == "long":
            if low_a[j] <= sl:
                return (j, sl, "sl_hit")
            if high_a[j] >= tp:
                return (j, tp, "tp_hit")
        else:
            if high_a[j] >= sl:
                return (j, sl, "sl_hit")
            if low_a[j] <= tp:
                return (j, tp, "tp_hit")
        if (j - entry_idx) >= max_bars:
            return (j, close_a[j], "time_stop")
    last_idx = min(entry_idx + max_bars, n - 1)
    return (last_idx, close_a[last_idx], "forced_end")


def _load_4h(symbol: str) -> pd.DataFrame:
    p = ROOT / "storage" / "candle_cache" / f"{symbol}_USDT_4h.parquet"
    df4 = pd.read_parquet(p)
    if "datetime" in df4.columns:
        df4 = df4.set_index("datetime")
    if df4.index.tz is None:
        df4.index = df4.index.tz_localize("UTC")
    df4 = df4.sort_index()
    for c in ("open", "high", "low", "close"):
        df4[c] = df4[c].astype(float)
    return df4


def precompute_h4_expansion_ratio(
    df5_index: pd.DatetimeIndex,
    high5: np.ndarray, low5: np.ndarray,
    df4: pd.DataFrame,
) -> np.ndarray:
    """4h expansion ratio per 5m bar — copy of proven sb_phase_filter impl."""
    n = len(df5_index)
    out = np.full(n, np.nan, dtype=np.float64)

    h4 = df4["high"].to_numpy(dtype=np.float64)
    l4 = df4["low"].to_numpy(dtype=np.float64)
    c4 = df4["close"].to_numpy(dtype=np.float64)
    atr4 = add_atr_np(h4, l4, c4, H4_ATR_PERIOD)
    atr4_prev = np.full(len(atr4), np.nan, dtype=np.float64)
    atr4_prev[1:] = atr4[:-1]

    ts5_us = pd.DatetimeIndex(df5_index).as_unit("us").asi8
    h4_open_us = pd.DatetimeIndex(df4.index).as_unit("us").asi8
    H4_US = H4_SEC * 1_000_000

    idx_in_h4 = np.searchsorted(h4_open_us, ts5_us, side="right") - 1

    cur_bin = -1
    run_hi = -np.inf
    run_lo = np.inf
    for i in range(n):
        k = idx_in_h4[i]
        if k < 0 or k >= len(h4_open_us):
            continue
        if k != cur_bin:
            cur_bin = k
            run_hi = high5[i]
            run_lo = low5[i]
        else:
            if high5[i] > run_hi:
                run_hi = high5[i]
            if low5[i] < run_lo:
                run_lo = low5[i]
        ref_atr = atr4_prev[k]
        if not np.isfinite(ref_atr) or ref_atr <= 0:
            continue
        elapsed_us = ts5_us[i] - h4_open_us[k] + 5 * 60 * 1_000_000
        progress = elapsed_us / H4_US
        if progress <= 0 or progress > 1.0001:
            continue
        expected = progress * ref_atr
        if expected <= 0:
            continue
        cur_range = run_hi - run_lo
        out[i] = cur_range / expected
    return out


def _grade_from_p(p: float) -> Optional[str]:
    """Convert ml_prob proxy into grade label. C grade = skip."""
    if p >= 0.75:
        return "A+"
    elif p >= 0.65:
        return "A"
    elif p >= 0.55:
        return "B"
    return None  # C grade — skip


def _regime_from_atr_pctile(atr_val: float, p20: float, p80: float) -> str:
    """Simple ATR-based regime classifier."""
    if atr_val >= p80:
        return "high_volatility"
    elif atr_val <= p20:
        return "sideways"
    return "trending_up"


# ─────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────
class KellySizingStrategy(Strategy):
    """Kelly sizing A/B vs Patch H matrix vs Fixed.

    Trade SET is determined ONLY by setup detection (independent of params).
    Params control: sizing_mode (and alpha, bankroll, tp_rr).

    To avoid recomputing the same trade set 36x, we cache "candidate signals"
    per symbol and re-price them per param combo.
    """
    name = "kelly_sizing"

    # Cache: {symbol: candidate_signals_array}
    _signal_cache: Dict[str, Dict[str, Any]] = {}
    # 4h frames cached for phase filter
    _h4_cache: Dict[str, pd.DataFrame] = {}

    def param_grid(self):
        for sizing_mode in ["patch_h", "fixed", "kelly"]:
            for alpha in [0.25, 0.4, 0.5]:
                for bankroll_initial in [1000, 5000]:
                    for tp_rr in [1.5, 2.0]:
                        # alpha only matters for kelly; collapse non-kelly
                        # cells to alpha=0.4 representative to save compute
                        if sizing_mode != "kelly" and alpha != 0.4:
                            continue
                        yield {
                            "sizing_mode": sizing_mode,
                            "alpha": alpha,
                            "bankroll_initial": bankroll_initial,
                            "tp_rr": tp_rr,
                        }

    def _build_signals(self, df: pd.DataFrame, tp_rr: float) -> List[Dict[str, Any]]:
        """Detect all structure_bounce candidates. Tp_rr changes TP, so we
        need separate caches per tp_rr."""
        symbol = df.attrs.get("symbol", "BTC")
        cache_key = f"{symbol}_{tp_rr}"
        if cache_key in self._signal_cache:
            return self._signal_cache[cache_key]["signals"]

        n = len(df)
        if n < 100:
            self._signal_cache[cache_key] = {"signals": []}
            return []

        open_a = df["open"].to_numpy(dtype=np.float64)
        high_a = df["high"].to_numpy(dtype=np.float64)
        low_a = df["low"].to_numpy(dtype=np.float64)
        close_a = df["close"].to_numpy(dtype=np.float64)
        vol_a = df["volume"].to_numpy(dtype=np.float64)
        atr_a = add_atr_np(high_a, low_a, close_a, ATR_PERIOD)
        ts_a = df.index.to_numpy()

        # rel_vol
        v_ser = pd.Series(vol_a)
        rel_vol = (v_ser / v_ser.rolling(20, min_periods=10).mean()).to_numpy()

        # Swing high/low over PRIOR 20 bars
        h_ser = pd.Series(high_a)
        l_ser = pd.Series(low_a)
        swing_high = h_ser.rolling(SWING_WINDOW).max().shift(1).to_numpy()
        swing_low = l_ser.rolling(SWING_WINDOW).min().shift(1).to_numpy()

        # Wicks
        upper_wick = high_a - np.maximum(open_a, close_a)
        lower_wick = np.minimum(open_a, close_a) - low_a

        # ATR percentiles for regime classifier
        atr_finite = atr_a[np.isfinite(atr_a)]
        atr_p20 = float(np.percentile(atr_finite, 20)) if len(atr_finite) else 0.0
        atr_p80 = float(np.percentile(atr_finite, 80)) if len(atr_finite) else 0.0

        # 4h phase filter — proven from sb_phase_filter_study to lock in positive
        # baseline edge. Without this the SB rebuild is unprofitable noise and
        # sizing comparisons become meaningless.
        if symbol not in self._h4_cache:
            self._h4_cache[symbol] = _load_4h(symbol)
        df4 = self._h4_cache[symbol]
        expansion_ratio = precompute_h4_expansion_ratio(
            df.index, high_a, low_a, df4
        )

        # Tighter setup than baseline SB: wick >= 0.55 ATR, vol >= 1.0 (spec)
        # PLUS phase filter expansion >= 1.3 (proven from sb_phase_filter)
        wick_atr_min = 0.55
        vol_min = 1.0

        signals: List[Dict[str, Any]] = []
        open_until = -1
        end_i = n - (TIME_STOP_BARS + 5)
        start_i = max(SWING_WINDOW + ATR_PERIOD + 5, 25)

        for i in range(start_i, end_i):
            if i <= open_until:
                continue
            atr = atr_a[i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            rv = rel_vol[i]
            if not np.isfinite(rv) or rv < vol_min:
                continue

            # Phase gate — only fire during 4h EXPANSION (proven baseline)
            er = expansion_ratio[i]
            if not np.isfinite(er) or er < EXPANSION_MIN_RATIO:
                continue

            sh = swing_high[i]; sl_ref = swing_low[i]
            if not (np.isfinite(sh) and np.isfinite(sl_ref)):
                continue

            o = open_a[i]; h = high_a[i]; l = low_a[i]; c = close_a[i]
            uw = upper_wick[i]; lw = lower_wick[i]

            side = None
            entry = sl = tp = 0.0
            wick_atr = 0.0

            # LONG: rejection at swing low
            if lw >= wick_atr_min * atr and l <= sl_ref + atr and l >= sl_ref - atr:
                mid = (h + l) / 2.0
                if c > mid:
                    side = "long"
                    entry = c
                    sl = l - SL_ATR_BUFFER * atr
                    if sl >= entry:
                        continue
                    tp = entry + tp_rr * (entry - sl)
                    wick_atr = lw / atr
            # SHORT: rejection at swing high
            elif uw >= wick_atr_min * atr and h >= sh - atr and h <= sh + atr:
                mid = (h + l) / 2.0
                if c < mid:
                    side = "short"
                    entry = c
                    sl = h + SL_ATR_BUFFER * atr
                    if sl <= entry:
                        continue
                    tp = entry - tp_rr * (sl - entry)
                    wick_atr = uw / atr

            if side is None:
                continue
            if (side == "long" and tp <= entry) or (side == "short" and tp >= entry):
                continue

            # Confluence count: a simple proxy — wick strength tier + vol tier
            # 0..3 (wick: 0.55<x<=0.7=0, 0.7<x<=1.0=1, x>1.0=2; vol: rv>=1.5 +1)
            confluence = 0
            if wick_atr > 0.7:
                confluence += 1
            if wick_atr > 1.0:
                confluence += 1
            if rv >= 1.5:
                confluence += 1

            # ML prob proxy
            p = 0.5 + 0.05 * confluence + 0.1 * wick_atr
            p = max(0.4, min(0.85, p))

            # RR (always positive — same as tp_rr but be explicit)
            if side == "long":
                b = (tp - entry) / (entry - sl)
            else:
                b = (entry - tp) / (sl - entry)

            # Walk-forward exit
            exit_idx, exit_price, reason = _walk_forward_exit_np(
                high_a, low_a, close_a, i, side, entry, sl, tp, TIME_STOP_BARS
            )

            # Per-unit P&L: if side=long, pct = (exit-entry)/entry; for short reverse
            if side == "long":
                pct_pnl = (exit_price - entry) / entry
            else:
                pct_pnl = (entry - exit_price) / entry

            # Regime + grade tagging
            grade = _grade_from_p(p)
            regime = _regime_from_atr_pctile(atr, atr_p20, atr_p80)

            signals.append({
                "entry_idx": i,
                "exit_idx": exit_idx,
                "side": side,
                "entry_price": float(entry),
                "exit_price": float(exit_price),
                "entry_ts": pd.Timestamp(ts_a[i]),
                "exit_ts": pd.Timestamp(ts_a[exit_idx]),
                "exit_reason": reason,
                "holding_sec": int((exit_idx - i) * BAR_SEC),
                "p": float(p),
                "b": float(b),
                "pct_pnl": float(pct_pnl),
                "grade": grade,
                "regime": regime,
                "confluence": int(confluence),
                "wick_atr": float(wick_atr),
            })
            open_until = exit_idx

        self._signal_cache[cache_key] = {"signals": signals}
        return signals

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        symbol = df.attrs.get("symbol", "BTC")
        sizing_mode = str(params["sizing_mode"])
        alpha = float(params["alpha"])
        bankroll_initial = float(params["bankroll_initial"])
        tp_rr = float(params["tp_rr"])

        signals = self._build_signals(df, tp_rr)
        if not signals:
            return []

        trades: List[Trade] = []
        bankroll = bankroll_initial

        for s in signals:
            p = s["p"]
            b = s["b"]
            pct_pnl = s["pct_pnl"]
            grade = s["grade"]
            regime = s["regime"]

            # Determine notional based on sizing_mode
            notional = 0.0
            if sizing_mode == "fixed":
                notional = BASE_NOTIONAL
            elif sizing_mode == "patch_h":
                if grade is None:
                    continue  # C grade — skip in patch_h mode
                mult = PATCH_H_MULT.get((regime, grade), PATCH_H_DEFAULT)
                notional = BASE_NOTIONAL * mult
            elif sizing_mode == "kelly":
                # f_star = (p*b - (1-p)) / b
                if b <= 0:
                    continue
                f_star = (p * b - (1.0 - p)) / b
                if f_star <= 0:
                    continue  # negative EV — skip
                f = alpha * f_star
                f = max(0.01, min(0.25, f))  # clamp
                notional = bankroll * f
                if notional <= 0:
                    continue
            else:
                continue

            # Build the Trade
            trade = Trade(
                symbol=symbol,
                side=s["side"],
                entry_price=s["entry_price"],
                exit_price=s["exit_price"],
                notional_usd=float(notional),
                holding_sec=s["holding_sec"],
                entry_ts=s["entry_ts"],
                exit_ts=s["exit_ts"],
                exit_reason=s["exit_reason"],
                extra={
                    "p": p, "b": b, "grade": grade, "regime": regime,
                    "sizing_mode": sizing_mode, "f": notional / bankroll if bankroll > 0 else 0,
                    "bankroll_pre": bankroll,
                },
            )

            # Realized PnL — use gross (no fees) for bankroll updates
            realized = pct_pnl * notional
            if sizing_mode == "kelly":
                bankroll = max(1.0, bankroll + realized)  # don't go to 0

            trades.append(trade)

        return trades


# ─────────────────────────────────────────────────────────────────────
# Custom engine — adds Sharpe, Sortino, max_dd, PF metrics
# ─────────────────────────────────────────────────────────────────────
class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol, tf):
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df

    def aggregate_extra(self, trades: List[Trade], variant: str) -> Dict[str, float]:
        """Compute Sharpe, Sortino, max_dd, PF for the trade list."""
        if not trades:
            return {"sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0, "profit_factor": 0.0}

        # Net P&L per trade
        nets = np.array([self.apply_fees(t, variant) for t in trades])
        if len(nets) < 2:
            return {"sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0, "profit_factor": 0.0}

        mean = float(nets.mean())
        std = float(nets.std(ddof=1))
        sharpe = mean / std if std > 0 else 0.0

        downside = nets[nets < 0]
        d_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        sortino = mean / d_std if d_std > 0 else 0.0

        # Max DD on cumulative bankroll evolution
        # Use trades sorted by entry_ts for the equity curve
        ordered = sorted(zip(trades, nets), key=lambda x: x[0].entry_ts)
        equity = [0.0]
        for t, n in ordered:
            equity.append(equity[-1] + n)
        equity_arr = np.array(equity)
        # Need a base bankroll to compute % DD. Use bankroll_initial heuristic:
        # find max bankroll_pre across trades (kelly mode varies, fixed=400).
        b_pre = trades[0].extra.get("bankroll_pre", 1000.0) if trades[0].extra else 1000.0
        if b_pre <= 0:
            b_pre = 1000.0
        equity_arr = equity_arr + b_pre
        peak = np.maximum.accumulate(equity_arr)
        dd = (peak - equity_arr) / peak
        max_dd = float(dd.max()) if len(dd) > 0 else 0.0

        pos = nets[nets > 0]
        neg = nets[nets < 0]
        pf = float(pos.sum() / abs(neg.sum())) if neg.sum() != 0 else float("inf")
        if not np.isfinite(pf):
            pf = 999.0

        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_dd": max_dd,
            "profit_factor": pf,
        }

    def aggregate(self, trades, variant):
        result = super().aggregate(trades, variant)
        # Attach extra metrics
        extras = self.aggregate_extra(trades, variant)
        # max_dd > threshold => downgrade to KILL
        if result.verdict == "PASS" and extras["max_dd"] >= MAX_DD_THRESHOLD:
            result.verdict = "KILL_MAX_DD_TOO_HIGH"
        # Stash the extras in the result via the gap_pct slot proxy — but we
        # need a clean place. Use a side-channel dict keyed by id(result).
        _RESULT_EXTRAS[id(result)] = extras
        return result


# Side-channel storage for extra metrics (since CellResult dataclass is fixed)
_RESULT_EXTRAS: Dict[int, Dict[str, float]] = {}


# ─────────────────────────────────────────────────────────────────────
# Main runner with custom report
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== KELLY SIZING — W/F STUDY ===\n")
    print("A/B comparison: patch_h vs fixed vs kelly (alpha=0.25/0.4/0.5)")
    print("Symbols: BTC, ETH, SOL, XRP × 5m × 3 fee variants\n")

    engine = _PatchedEngine(
        study=KellySizingStrategy(),
        symbols=["BTC", "ETH", "SOL", "XRP"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "kelly_sizing",
        verbose=False,
    )

    # We need to capture extras alongside cells. Patch run() lightly by
    # wrapping it to record extras after.
    results = engine.run()

    cells = list(results["cells"].items())

    # Recompute extras by re-running aggregate per cell — but we already have
    # them in _RESULT_EXTRAS keyed by the CellResult python id. Since the
    # original CellResult instances are no longer in memory, instead let's
    # re-walk and recompute using stored asdict info combined with a re-sim.
    # Simpler: recompute extras inline here from per-cell trade replay.
    # To do this without re-simulating, we add extras to the JSON as a
    # second pass during the engine run. Patched by overriding run() — but
    # run() already serialized via asdict. Cleanest fix: walk cells again
    # and recompute the trades for each (cached signals make this cheap).

    # Re-build per-cell extras (cached signals make this cheap)
    print("\nComputing risk-adjusted metrics per cell...")
    extras_by_key: Dict[str, Dict[str, float]] = {}
    cache_keyed = {}
    seen_cells: Dict[str, List[Trade]] = {}

    for symbol in engine.symbols:
        for tf in engine.timeframes:
            try:
                df = engine.load_candles(symbol, tf)
            except FileNotFoundError:
                continue
            for params in engine.study.param_grid():
                cell_id = engine.study.cell_id(params)
                trades = engine.study.simulate(df, params)
                for variant in engine.fee_variants:
                    extras = engine.aggregate_extra(trades, variant)
                    key = f"{symbol}|{tf}|{cell_id}|{variant}"
                    extras_by_key[key] = extras

    # Merge extras into JSON-saved cells
    enriched_cells: Dict[str, Any] = {}
    for k, v in cells:
        enriched = dict(v)
        enriched.update(extras_by_key.get(k, {"sharpe": 0.0, "sortino": 0.0,
                                              "max_dd": 0.0, "profit_factor": 0.0}))
        # Re-evaluate verdict with max_dd gate
        if enriched["verdict"] == "PASS" and enriched.get("max_dd", 0.0) >= MAX_DD_THRESHOLD:
            enriched["verdict"] = "KILL_MAX_DD_TOO_HIGH"
        enriched_cells[k] = enriched

    # Also persist enriched JSON
    out_dir = ROOT / "storage" / "wf_studies" / "kelly_sizing"
    enriched_payload = dict(results)
    enriched_payload["cells"] = enriched_cells
    (out_dir / "walkforward_enriched.json").write_text(
        json.dumps(enriched_payload, indent=2, default=str)
    )

    # ─── A/B summary by sizing_mode ─────────────────────────────────
    def parse_cell_key(k: str):
        """Parse cell key. cell format is k1=v1_k2=v2_... but values can
        contain underscores (e.g. patch_h). Parse by known keys instead."""
        sym, tf, cell, variant = k.split("|", 3)
        kv: Dict[str, str] = {}
        # Known param keys (longest-first to avoid partial matches)
        known = ["sizing_mode", "bankroll_initial", "alpha", "tp_rr"]
        # Find each key's position then read until next key
        positions: List[Tuple[int, str]] = []
        for k_name in known:
            idx = cell.find(k_name + "=")
            if idx >= 0:
                positions.append((idx, k_name))
        positions.sort()
        for i, (start, k_name) in enumerate(positions):
            val_start = start + len(k_name) + 1
            val_end = positions[i + 1][0] - 1 if i + 1 < len(positions) else len(cell)
            kv[k_name] = cell[val_start:val_end]
        return sym, tf, cell, variant, kv

    # We compare at FIXED bankroll_initial=5000 to neutralize bankroll-vs-notional
    # confounding (fixed/patch_h notional doesn't depend on bankroll, so DD% is
    # purely a function of bankroll size — apples-to-apples requires one bankroll).
    # Also restrict to tp_rr=2.0 for the headline (matches live config).
    by_mode: Dict[str, Dict[str, List[float]]] = {
        "patch_h": {"is_ev": [], "sharpe": [], "pf": [], "max_dd": [],
                    "n_pass": [0], "n_total": [0], "sortino": []},
        "fixed":   {"is_ev": [], "sharpe": [], "pf": [], "max_dd": [],
                    "n_pass": [0], "n_total": [0], "sortino": []},
        "kelly":   {"is_ev": [], "sharpe": [], "pf": [], "max_dd": [],
                    "n_pass": [0], "n_total": [0], "sortino": []},
    }
    by_mode_alpha: Dict[str, Dict[str, List[float]]] = {
        "kelly_a0.25": {"is_ev": [], "sharpe": [], "pf": [], "max_dd": [],
                        "n_pass": [0], "n_total": [0], "sortino": []},
        "kelly_a0.4":  {"is_ev": [], "sharpe": [], "pf": [], "max_dd": [],
                        "n_pass": [0], "n_total": [0], "sortino": []},
        "kelly_a0.5":  {"is_ev": [], "sharpe": [], "pf": [], "max_dd": [],
                        "n_pass": [0], "n_total": [0], "sortino": []},
    }
    by_pair_mode: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for k, v in enriched_cells.items():
        sym, tf, cell, variant, kv = parse_cell_key(k)
        if variant != "A_full_taker":
            continue
        # Headline normalized comparison: bankroll=5000 only (so all modes
        # see the same bankroll-relative DD). Use both tp_rr values.
        if kv.get("bankroll_initial") != "5000":
            continue
        mode = kv.get("sizing_mode", "?")
        if mode not in by_mode:
            continue
        by_mode[mode]["is_ev"].append(v["is_ev"])
        by_mode[mode]["sharpe"].append(v["sharpe"])
        by_mode[mode]["sortino"].append(v.get("sortino", 0.0))
        pf = v["profit_factor"]
        if np.isfinite(pf) and pf < 100:
            by_mode[mode]["pf"].append(pf)
        by_mode[mode]["max_dd"].append(v["max_dd"])
        by_mode[mode]["n_total"][0] += 1
        if v["verdict"] == "PASS":
            by_mode[mode]["n_pass"][0] += 1

        if mode == "kelly":  # bankroll=5000 already filtered above
            ak = f"kelly_a{kv.get('alpha', '0.4')}"
            if ak in by_mode_alpha:
                by_mode_alpha[ak]["is_ev"].append(v["is_ev"])
                by_mode_alpha[ak]["sharpe"].append(v["sharpe"])
                by_mode_alpha[ak]["sortino"].append(v.get("sortino", 0.0))
                if np.isfinite(pf) and pf < 100:
                    by_mode_alpha[ak]["pf"].append(pf)
                by_mode_alpha[ak]["max_dd"].append(v["max_dd"])
                by_mode_alpha[ak]["n_total"][0] += 1
                if v["verdict"] == "PASS":
                    by_mode_alpha[ak]["n_pass"][0] += 1

        # Per-pair tracking
        by_pair_mode.setdefault(sym, {}).setdefault(mode, {
            "is_ev": [], "sharpe": [], "pf": [], "max_dd": [], "n_pass": 0, "n_total": 0
        })
        bpm = by_pair_mode[sym][mode]
        bpm["is_ev"].append(v["is_ev"])
        bpm["sharpe"].append(v["sharpe"])
        if np.isfinite(pf) and pf < 100:
            bpm["pf"].append(pf)
        bpm["max_dd"].append(v["max_dd"])
        bpm["n_total"] += 1
        if v["verdict"] == "PASS":
            bpm["n_pass"] += 1

    def _avg(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    # ─── Verdict distribution ────────────────────────────────────────
    verdicts: Dict[str, int] = {}
    for v in enriched_cells.values():
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1

    print(f"\n=== DONE ({results['wall_sec']}s) ===")
    print(f"Cells run: {results['cells_run']}  PASS (raw): {results['cells_pass']}")
    print(f"Verdict distribution (with max_dd gate):")
    for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {n}")

    print(f"\n=== A/B BY SIZING_MODE (variant A_full_taker, all pairs) ===")
    print(f"{'Mode':<10} {'cells':<7} {'PASS':<6} {'IS_EV':<10} {'Sharpe':<8} {'PF':<7} {'MaxDD':<7}")
    for mode, m in by_mode.items():
        nt = m["n_total"][0]; np_ = m["n_pass"][0]
        print(f"{mode:<10} {nt:<7} {np_:<6} ${_avg(m['is_ev']):<9.3f} "
              f"{_avg(m['sharpe']):<8.3f} {_avg(m['pf']):<7.2f} {_avg(m['max_dd'])*100:<7.1f}%")

    print(f"\n=== KELLY ALPHA COMPARISON ===")
    print(f"{'Alpha':<14} {'cells':<7} {'PASS':<6} {'IS_EV':<10} {'Sharpe':<8} {'PF':<7} {'MaxDD':<7}")
    for ak, m in by_mode_alpha.items():
        nt = m["n_total"][0]; np_ = m["n_pass"][0]
        print(f"{ak:<14} {nt:<7} {np_:<6} ${_avg(m['is_ev']):<9.3f} "
              f"{_avg(m['sharpe']):<8.3f} {_avg(m['pf']):<7.2f} {_avg(m['max_dd'])*100:<7.1f}%")

    print(f"\n=== PER-PAIR x MODE (variant A) ===")
    for sym in sorted(by_pair_mode):
        print(f"\n  {sym}:")
        print(f"    {'Mode':<10} {'cells':<7} {'PASS':<6} {'IS_EV':<10} {'Sharpe':<8} {'PF':<7} {'MaxDD':<7}")
        for mode in ["patch_h", "fixed", "kelly"]:
            if mode not in by_pair_mode[sym]:
                continue
            m = by_pair_mode[sym][mode]
            print(f"    {mode:<10} {m['n_total']:<7} {m['n_pass']:<6} "
                  f"${_avg(m['is_ev']):<9.3f} {_avg(m['sharpe']):<8.3f} "
                  f"{_avg(m['pf']):<7.2f} {_avg(m['max_dd'])*100:<7.1f}%")

    # Top 10 PASS cells overall
    pass_cells = [(k, v) for k, v in enriched_cells.items() if v["verdict"] == "PASS"]
    if pass_cells:
        pass_cells.sort(key=lambda kv: kv[1]["q4_ev"], reverse=True)
        print(f"\n=== TOP 10 PASS CELLS by Q4 EV ===")
        for k, v in pass_cells[:10]:
            sym, tf, cell, variant = k.split("|", 3)
            print(f"  {sym} {variant} {cell}")
            print(f"    IS_EV=${v['is_ev']:+.3f} Q4_EV=${v['q4_ev']:+.3f} "
                  f"Sharpe={v['sharpe']:.3f} PF={v['profit_factor']:.2f} "
                  f"DD={v['max_dd']*100:.1f}%")

    # ─── Build markdown report ───────────────────────────────────────
    REPORT_PATH = Path("/tmp/agent_kelly_sizing_report.md")
    md_lines: List[str] = []

    def _fmt_dd(x: float) -> str:
        return f"{x*100:.1f}%"

    md_lines.extend([
        f"# Kelly Sizing W/F Study — Report",
        f"",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Symbols: BTC, ETH, SOL, XRP — 5m — 3 fee variants",
        f"Cells: {results['cells_run']} per pair × 3 fee variants",
        f"Wall: {results['wall_sec']}s",
        f"",
        f"## Hypothesis",
        f"Replace 27-cell Patch H matrix (regime × grade) with one Kelly formula: "
        f"f = alpha × (p×b - q) / b. SIZING study — same trade SET, different notionals.",
        f"",
        f"## Headline A/B (variant A_full_taker, all pairs averaged)",
        f"",
        f"| Sizing mode | cells | PASS | Avg IS_EV | Sharpe | Profit Factor | Max DD |",
        f"|---|---|---|---|---|---|---|",
    ])
    for mode, m in by_mode.items():
        md_lines.append(
            f"| {mode} | {m['n_total'][0]} | {m['n_pass'][0]} | "
            f"${_avg(m['is_ev']):.3f} | {_avg(m['sharpe']):.3f} | "
            f"{_avg(m['pf']):.2f} | {_fmt_dd(_avg(m['max_dd']))} |"
        )

    md_lines.extend([
        f"",
        f"## Kelly alpha sweep",
        f"",
        f"| Alpha | cells | PASS | Avg IS_EV | Sharpe | Profit Factor | Max DD |",
        f"|---|---|---|---|---|---|---|",
    ])
    for ak, m in by_mode_alpha.items():
        md_lines.append(
            f"| {ak} | {m['n_total'][0]} | {m['n_pass'][0]} | "
            f"${_avg(m['is_ev']):.3f} | {_avg(m['sharpe']):.3f} | "
            f"{_avg(m['pf']):.2f} | {_fmt_dd(_avg(m['max_dd']))} |"
        )

    md_lines.extend([
        f"",
        f"## Per-pair best sizing mode",
        f"",
        f"| Symbol | Best mode | IS_EV | Sharpe | PF | DD | PASS/total |",
        f"|---|---|---|---|---|---|---|",
    ])
    for sym in sorted(by_pair_mode):
        # Best by Sharpe (proxy for risk-adjusted)
        ranked = []
        for mode in ["patch_h", "fixed", "kelly"]:
            if mode not in by_pair_mode[sym]:
                continue
            m = by_pair_mode[sym][mode]
            ranked.append((mode, _avg(m["sharpe"]), _avg(m["is_ev"]),
                          _avg(m["pf"]), _avg(m["max_dd"]),
                          m["n_pass"], m["n_total"]))
        ranked.sort(key=lambda x: x[1], reverse=True)
        if ranked:
            best = ranked[0]
            md_lines.append(
                f"| {sym} | {best[0]} | ${best[2]:.3f} | {best[1]:.3f} | "
                f"{best[3]:.2f} | {_fmt_dd(best[4])} | {best[5]}/{best[6]} |"
            )

    md_lines.extend([
        f"",
        f"## Verdict distribution (with max_dd<25% gate)",
        f"",
        f"| Verdict | Count |",
        f"|---|---|",
    ])
    for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        md_lines.append(f"| {v} | {n} |")

    if pass_cells:
        md_lines.extend([
            f"",
            f"## Top 10 PASS cells by Q4 EV",
            f"",
            f"| symbol | cell | variant | IS_EV | Q4_EV | Sharpe | PF | MaxDD |",
            f"|---|---|---|---|---|---|---|---|",
        ])
        for k, v in pass_cells[:10]:
            sym, tf, cell, variant = k.split("|", 3)
            md_lines.append(
                f"| {sym} | {cell} | {variant} | ${v['is_ev']:.3f} | ${v['q4_ev']:.3f} | "
                f"{v['sharpe']:.3f} | {v['profit_factor']:.2f} | {_fmt_dd(v['max_dd'])} |"
            )

    # Recommendation logic
    md_lines.extend([
        f"",
        f"## Recommendation",
        f"",
    ])

    kelly_sharpe = _avg(by_mode["kelly"]["sharpe"])
    kelly_dd = _avg(by_mode["kelly"]["max_dd"])
    kelly_pf = _avg(by_mode["kelly"]["pf"])
    kelly_pass = by_mode["kelly"]["n_pass"][0]
    patch_sharpe = _avg(by_mode["patch_h"]["sharpe"])
    patch_dd = _avg(by_mode["patch_h"]["max_dd"])
    patch_pf = _avg(by_mode["patch_h"]["pf"])
    patch_pass = by_mode["patch_h"]["n_pass"][0]
    fixed_sharpe = _avg(by_mode["fixed"]["sharpe"])
    fixed_pass = by_mode["fixed"]["n_pass"][0]

    rec_lines = []
    rec_lines.append(f"- Kelly avg Sharpe: {kelly_sharpe:.3f}, max DD: {_fmt_dd(kelly_dd)}, PF: {kelly_pf:.2f}, PASS: {kelly_pass}")
    rec_lines.append(f"- Patch H avg Sharpe: {patch_sharpe:.3f}, max DD: {_fmt_dd(patch_dd)}, PF: {patch_pf:.2f}, PASS: {patch_pass}")
    rec_lines.append(f"- Fixed avg Sharpe: {fixed_sharpe:.3f}, PASS: {fixed_pass}")

    if kelly_dd >= 0.40:
        verdict = ("**SHIP NOTHING / STAY ON PATCH H**: Kelly max DD too high "
                   f"({_fmt_dd(kelly_dd)} ≥ 40%) — model probabilities not "
                   "calibrated enough for Kelly. Tighten alpha cap or fix "
                   "ml_prob calibration first.")
    elif kelly_sharpe > patch_sharpe and kelly_dd <= patch_dd:
        verdict = ("**SHIP KELLY AS PATCH Q**: Kelly beats Patch H on both "
                   "Sharpe AND max DD. Replace 27-cell matrix with single "
                   "f = alpha*(p*b-q)/b formula.")
    elif kelly_sharpe > patch_sharpe:
        verdict = ("**HOLD — Kelly beats Patch H on Sharpe but at higher DD. "
                   "Consider reducing alpha cap below 0.25 or shipping Kelly "
                   "with tighter per-trade notional cap.")
    elif patch_sharpe > kelly_sharpe and patch_sharpe > fixed_sharpe:
        verdict = ("**KEEP PATCH H**: Patch H matrix wins on risk-adjusted "
                   "returns. The empirical 27-cell tuning encodes information "
                   "Kelly's analytic formula misses.")
    else:
        verdict = ("**STAY ON FIXED**: Neither Kelly nor Patch H beats fixed "
                   "on risk-adjusted returns. The signal set may not have "
                   "enough edge for sizing to matter.")

    rec_lines.append("")
    rec_lines.append(verdict)

    md_lines.extend(rec_lines)

    REPORT_PATH.write_text("\n".join(md_lines) + "\n")
    print(f"\nReport: {REPORT_PATH}")
    print(f"Enriched JSON: {out_dir / 'walkforward_enriched.json'}")
