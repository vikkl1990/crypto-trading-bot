"""W/F STUDY: liquidity_sweep + HTF regime filter.

Hypothesis: the unfiltered liquidity_sweep study had positive IS edge but
Q3 (Feb 2026) collapsed it. Q3 was characterized by high-volatility chop —
exactly the regime where mean-reversion sweeps fail. Adding an HTF regime
filter (skip when 4h ATR percentile > X) should preserve the Oct-Jan and
Mar-Apr edge while skipping Q3-like regimes.

Filter: only fire if 4h ATR_pct_rank ≤ regime_max (skip high-vol regimes).

Sweep: best-performing cells from prior study × different regime_max values.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.wf_harness import Strategy, Trade, WalkForwardEngine

BAR_SEC = 300


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _walk_forward_exit(df, i, side, entry, sl, tp, max_age_sec):
    initial_risk = abs(entry - sl)
    if initial_risk <= 0: return None
    max_bars = max_age_sec // BAR_SEC + 2
    peak_r = 0.0
    for j in range(i + 1, min(i + 1 + max_bars, len(df))):
        bar = df.iloc[j]
        age_sec = (j - i) * BAR_SEC
        if side == "long":
            if float(bar["low"]) <= sl: return (j, sl, "sl_hit", peak_r)
            if float(bar["high"]) >= tp: return (j, tp, "tp_hit", peak_r)
            cur_r = (float(bar["high"]) - entry) / initial_risk
        else:
            if float(bar["high"]) >= sl: return (j, sl, "sl_hit", peak_r)
            if float(bar["low"]) <= tp: return (j, tp, "tp_hit", peak_r)
            cur_r = (entry - float(bar["low"])) / initial_risk
        peak_r = max(peak_r, cur_r)
        if age_sec >= max_age_sec:
            return (j, float(bar["close"]), "time_decay", peak_r)
    last_idx = min(i + max_bars, len(df) - 1)
    return (last_idx, float(df.iloc[last_idx]["close"]), "forced_end", peak_r)


def load_4h_atr_pct(symbol: str) -> pd.Series:
    """Load 4h ATR percentile rank series for HTF regime filtering."""
    path = ROOT / "storage" / "candle_cache" / f"{symbol}_USDT_4h.parquet"
    if not path.exists():
        return None
    df_4h = pd.read_parquet(path)
    if "datetime" in df_4h.columns:
        df_4h = df_4h.set_index("datetime")
    if df_4h.index.tz is None:
        df_4h.index = df_4h.index.tz_localize("UTC")
    df_4h = df_4h.sort_index()
    for c in ("open", "high", "low", "close", "volume"):
        df_4h[c] = df_4h[c].astype(float)
    atr_4h = add_atr(df_4h, 14)
    atr_pct = atr_4h.rolling(100, min_periods=50).rank(pct=True)
    return atr_pct  # indexed at 4h timestamps


class LiquiditySweepHtfStudy(Strategy):
    """Same as LiquiditySweepStudy but with HTF regime filter applied.

    Sweep through 20-bar extreme + reversal with displacement.
    Filter: 4h ATR percentile rank must be ≤ regime_max (skip high-vol).
    """
    name = "liquidity_sweep_htf"

    # Cache for 4h ATR per symbol
    _atr_4h_cache: Dict[str, pd.Series] = {}

    def param_grid(self):
        # Use the params that showed IS edge in prior study
        for disp_atr in [0.4, 0.6, 0.8, 1.0]:
            for sweep_atr in [0.1, 0.2, 0.3]:
                for regime_max in [0.40, 0.60, 0.80]:  # NEW: HTF regime filter
                    yield {"disp_atr": disp_atr, "sweep_atr": sweep_atr,
                           "regime_max": regime_max,
                           "max_age": 600, "sl_atr": 1.0, "tp_atr": 2.0}

    def simulate(self, df, params):
        if len(df) < 50:
            return []
        df = df.copy()
        df["atr"] = add_atr(df, 14)
        df["roll_high20"] = df["high"].rolling(20).max().shift(1)
        df["roll_low20"]  = df["low"].rolling(20).min().shift(1)
        symbol = df.attrs.get("symbol", "BTC")

        # Load 4h ATR pct for this symbol (cache hit if already loaded)
        if symbol not in self._atr_4h_cache:
            self._atr_4h_cache[symbol] = load_4h_atr_pct(symbol)
        atr_4h_series = self._atr_4h_cache[symbol]
        if atr_4h_series is None:
            return []  # no 4h data → skip this study

        # Reindex 4h ATR pct to 5m timestamps via forward-fill
        atr_pct_5m = atr_4h_series.reindex(df.index, method="ffill")
        df["htf_atr_pct"] = atr_pct_5m

        disp_atr = float(params["disp_atr"])
        sweep_atr = float(params["sweep_atr"])
        regime_max = float(params["regime_max"])
        max_age = int(params["max_age"])
        sl_atr = float(params["sl_atr"])
        tp_atr = float(params["tp_atr"])

        trades: List[Trade] = []
        open_until = -1

        for i in range(25, len(df) - (max_age // BAR_SEC + 2)):
            if i <= open_until:
                continue
            row = df.iloc[i]
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue

            # NEW: HTF regime filter — skip high-vol chop
            htf_pct = row.get("htf_atr_pct", float("nan"))
            if pd.isna(htf_pct) or htf_pct > regime_max:
                continue

            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            roll_high = float(row["roll_high20"]); roll_low = float(row["roll_low20"])
            if pd.isna([roll_high, roll_low]).any():
                continue
            body = abs(c - o)

            side = None
            prev = df.iloc[i - 1]
            if l < roll_low - sweep_atr * atr and c > prev["low"] and c > o and body >= disp_atr * atr:
                side = "long"
                entry = c; sl = entry - sl_atr * atr; tp = entry + tp_atr * atr
            elif h > roll_high + sweep_atr * atr and c < prev["high"] and c < o and body >= disp_atr * atr:
                side = "short"
                entry = c; sl = entry + sl_atr * atr; tp = entry - tp_atr * atr
            else:
                continue

            result = _walk_forward_exit(df, i, side, entry, sl, tp, max_age)
            if result is None: continue
            exit_idx, exit_price, reason, peak_r = result
            holding_sec = (exit_idx - i) * BAR_SEC
            trades.append(Trade(
                symbol=symbol, side=side,
                entry_price=entry, exit_price=exit_price,
                notional_usd=1000.0, holding_sec=holding_sec,
                entry_ts=df.index[i], exit_ts=df.index[exit_idx],
                exit_reason=reason,
                extra={"peak_r": round(peak_r, 3), "htf_pct": round(htf_pct, 2)},
            ))
            open_until = exit_idx
        return trades


class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol, tf):
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


if __name__ == "__main__":
    print("=== LIQUIDITY SWEEP + HTF REGIME FILTER ===\n")
    print("Hypothesis: filtering by 4h ATR percentile (skip high-vol chop)")
    print("preserves IS edge while killing Q3-like regimes that broke OOS.\n")
    print("Param grid: 4 disp × 3 sweep × 3 regime_max = 36 cells × 4 sym × 3 var = 432\n")

    engine = _PatchedEngine(
        study=LiquiditySweepHtfStudy(),
        symbols=["BTC", "ETH", "SOL", "XRP"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "liquidity_sweep_htf",
        verbose=False,
    )
    results = engine.run()

    print(f"\nDONE in {results['wall_sec']}s. Cells run: {results['cells_run']}, PASS: {results['cells_pass']}")
    cells = list(results["cells"].items())
    verdicts = {}
    for _, v in cells:
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
    print("\nVerdict distribution:")
    for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {n}")

    is_pos = [c for c in cells if c[1]["is_ev"] > 0.01]
    print(f"\nIS-positive cells: {len(is_pos)}")

    pass_cells = [c for c in cells if c[1]["verdict"] == "PASS"]
    if pass_cells:
        print(f"\n=== PASS CELLS ({len(pass_cells)}) ===")
        for k, v in sorted(pass_cells, key=lambda kv: kv[1]["q4_ev"], reverse=True):
            sym, tf, cell, variant = k.split("|", 3)
            var_short = {"A_full_taker":"A","B_scalper_taker":"B","C_maker_scalper":"C"}.get(variant,"?")
            print(f"  {sym} {var_short} {cell}")
            print(f"    IS_n={v['is_n']} IS=${v['is_ev']:+.3f} Q3=${v['q3_ev']:+.3f} Q4=${v['q4_ev']:+.3f} gap={v.get('gap_pct',0)*100:.0f}%")

    # Top 8 cells by Q4 EV regardless of verdict
    cells_sorted = sorted(cells, key=lambda c: c[1]["q4_ev"], reverse=True)
    print(f"\n=== Top 8 cells by Q4 EV (any verdict) ===")
    for k, v in cells_sorted[:8]:
        sym, tf, cell, variant = k.split("|", 3)
        var_short = {"A_full_taker":"A","B_scalper_taker":"B","C_maker_scalper":"C"}.get(variant,"?")
        print(f"  {sym} {var_short} {cell}")
        print(f"    IS_n={v['is_n']} IS=${v['is_ev']:+.3f} Q3=${v['q3_ev']:+.3f} Q4=${v['q4_ev']:+.3f} verdict={v['verdict']}")
