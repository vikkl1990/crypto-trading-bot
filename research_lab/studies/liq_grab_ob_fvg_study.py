"""W/F STUDY: Liquidity Grab + OB Reclaim + FVG Fill (5-step SMC confluence).

Tests the user's proposed enhancement to the basic liq_sweep_htf paper engine
that W/F-passed today (7 PASS cells).

Setup (LONG; mirror for SHORT):
  Step 1 — LIQUIDITY GRAB: bar i sweeps below 20-bar low by >= sweep_atr × ATR
           AND bar i is a bearish displacement (close < open, body >= disp_atr × ATR)
  Step 2 — OB RECLAIM: bar i+1..i+lookback contains a bar that closes back ABOVE
           the LAST BULLISH bar's high before the sweep (the "proximal Order Block")
  Step 3 — FVG CONFIRMATION: a Fair Value Gap formed in the reclaim sequence,
           AND the FVG is filled with a bullish displacement candle
  Step 4 — ENTRY: at the FVG-fill candle close (maker limit at OB-high level)
  Step 5 — STOP: below swept low; TARGET: 1:2 RR

Compares against baseline (just liq_sweep_htf) by measuring whether adding the
OB+FVG gates produces higher OOS Q4 EV (acceptable to fewer trades if EV/trade
is significantly better).

If passes W/F, ship as Phase-2 paper engine `liq_grab_smc_paper_engine.py`.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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


def load_4h_atr_pct(symbol: str) -> Optional[pd.Series]:
    path = ROOT / "storage" / "candle_cache" / f"{symbol}_USDT_4h.parquet"
    if not path.exists(): return None
    df_4h = pd.read_parquet(path)
    if "datetime" in df_4h.columns: df_4h = df_4h.set_index("datetime")
    if df_4h.index.tz is None: df_4h.index = df_4h.index.tz_localize("UTC")
    df_4h = df_4h.sort_index()
    for c in ("high", "low", "close"):
        df_4h[c] = df_4h[c].astype(float)
    atr_4h = add_atr(df_4h, 14)
    return atr_4h.rolling(100, min_periods=50).rank(pct=True)


def find_proximal_ob_long(df: pd.DataFrame, sweep_idx: int, lookback: int = 5) -> Optional[Tuple[float, float]]:
    """Find last BULLISH bar (close > open) in lookback before sweep.
    Returns (ob_high, ob_low) — the OB candle's range. Long version."""
    for j in range(sweep_idx - 1, max(0, sweep_idx - lookback - 1), -1):
        bar = df.iloc[j]
        if float(bar["close"]) > float(bar["open"]):
            return (float(bar["high"]), float(bar["low"]))
    return None


def find_proximal_ob_short(df: pd.DataFrame, sweep_idx: int, lookback: int = 5) -> Optional[Tuple[float, float]]:
    """Find last BEARISH bar (close < open) in lookback before sweep — short version."""
    for j in range(sweep_idx - 1, max(0, sweep_idx - lookback - 1), -1):
        bar = df.iloc[j]
        if float(bar["close"]) < float(bar["open"]):
            return (float(bar["high"]), float(bar["low"]))
    return None


def detect_fvg_long(df: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[Tuple[int, float, float]]:
    """Find a bullish FVG (gap up) between start_idx and end_idx.

    A bullish FVG is: bar j+1 has low > bar j-1 high (3-candle pattern).
    The "gap" is from bar j-1.high to bar j+1.low. Returns (fvg_idx, gap_low, gap_high).
    """
    for j in range(start_idx + 1, min(end_idx, len(df) - 1)):
        prev_high = float(df.iloc[j - 1]["high"])
        next_low = float(df.iloc[j + 1]["low"])
        if next_low > prev_high:
            return (j, prev_high, next_low)
    return None


def detect_fvg_short(df: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[Tuple[int, float, float]]:
    """Find a bearish FVG (gap down). bar j+1 high < bar j-1 low."""
    for j in range(start_idx + 1, min(end_idx, len(df) - 1)):
        prev_low = float(df.iloc[j - 1]["low"])
        next_high = float(df.iloc[j + 1]["high"])
        if next_high < prev_low:
            return (j, next_high, prev_low)
    return None


def fvg_filled_long(df: pd.DataFrame, fvg_idx: int, gap_low: float, gap_high: float,
                     end_idx: int, atr: float, fill_disp: float) -> Optional[int]:
    """Bar after fvg_idx that fills (low <= gap_high) AND is bullish displacement
    (close > open, body >= fill_disp × atr). Returns the fill bar's index."""
    for j in range(fvg_idx + 1, min(end_idx, len(df))):
        bar = df.iloc[j]
        if float(bar["low"]) <= gap_high:
            body = abs(float(bar["close"]) - float(bar["open"]))
            if float(bar["close"]) > float(bar["open"]) and body >= fill_disp * atr:
                return j
    return None


def fvg_filled_short(df: pd.DataFrame, fvg_idx: int, gap_low: float, gap_high: float,
                      end_idx: int, atr: float, fill_disp: float) -> Optional[int]:
    for j in range(fvg_idx + 1, min(end_idx, len(df))):
        bar = df.iloc[j]
        if float(bar["high"]) >= gap_low:
            body = abs(float(bar["close"]) - float(bar["open"]))
            if float(bar["close"]) < float(bar["open"]) and body >= fill_disp * atr:
                return j
    return None


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


class LiquidityGrabOBFVGStudy(Strategy):
    """5-step SMC confluence: liquidity grab + OB reclaim + FVG fill.

    Param sweep:
      disp_atr: minimum body size for sweep candle (× ATR)
      sweep_atr: minimum sweep magnitude beyond 20-bar extreme (× ATR)
      reclaim_lookback: bars after sweep to look for OB reclaim
      fill_disp: minimum body for FVG-fill candle (× ATR)
      regime_max: 4h ATR pct max (HTF gate)
    """
    name = "liq_grab_ob_fvg"

    _atr_4h_cache: Dict[str, Optional[pd.Series]] = {}

    def param_grid(self):
        for disp_atr in [0.5, 0.7]:
            for sweep_atr in [0.1, 0.2]:
                for reclaim_lb in [3, 5]:
                    for fill_disp in [0.4, 0.6]:
                        for regime_max in [0.6, 0.8]:
                            yield {
                                "disp_atr": disp_atr,
                                "sweep_atr": sweep_atr,
                                "reclaim_lookback": reclaim_lb,
                                "fill_disp": fill_disp,
                                "regime_max": regime_max,
                                "max_age": 600,
                                "sl_atr": 1.0,
                                "tp_atr": 2.0,
                            }

    def simulate(self, df, params):
        if len(df) < 50: return []
        df = df.copy()
        df["atr"] = add_atr(df, 14)
        df["roll_high20"] = df["high"].rolling(20).max().shift(1)
        df["roll_low20"]  = df["low"].rolling(20).min().shift(1)
        symbol = df.attrs.get("symbol", "BTC")

        # HTF regime
        if symbol not in self._atr_4h_cache:
            self._atr_4h_cache[symbol] = load_4h_atr_pct(symbol)
        atr_4h = self._atr_4h_cache[symbol]
        if atr_4h is None: return []
        df["htf_atr_pct"] = atr_4h.reindex(df.index, method="ffill")

        disp_atr = float(params["disp_atr"])
        sweep_atr = float(params["sweep_atr"])
        reclaim_lb = int(params["reclaim_lookback"])
        fill_disp = float(params["fill_disp"])
        regime_max = float(params["regime_max"])
        max_age = int(params["max_age"])
        sl_atr = float(params["sl_atr"])
        tp_atr = float(params["tp_atr"])

        trades: List[Trade] = []
        open_until = -1

        for i in range(25, len(df) - (max_age // BAR_SEC + 10)):
            if i <= open_until:
                continue
            row = df.iloc[i]
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue

            htf_pct = row.get("htf_atr_pct", float("nan"))
            if pd.isna(htf_pct) or htf_pct > regime_max:
                continue

            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            roll_high = float(row["roll_high20"]); roll_low = float(row["roll_low20"])
            if pd.isna([roll_high, roll_low]).any():
                continue
            body = abs(c - o)

            # ─── LONG SETUP (bearish sweep below low + bullish reclaim) ───
            sweep_long = (l < roll_low - sweep_atr * atr) and (c < o) and (body >= disp_atr * atr)
            if sweep_long:
                ob = find_proximal_ob_long(df, i, lookback=5)
                if ob is None: continue
                ob_high, ob_low = ob

                # OB reclaim: any bar in next reclaim_lb closes >= ob_high
                reclaim_idx = None
                for j in range(i + 1, min(i + 1 + reclaim_lb, len(df))):
                    if float(df.iloc[j]["close"]) >= ob_high:
                        reclaim_idx = j; break
                if reclaim_idx is None: continue

                # FVG forms in [i, reclaim_idx], filled with bullish displacement
                fvg = detect_fvg_long(df, i, reclaim_idx + 2)
                if fvg is None: continue
                fvg_idx, gap_low, gap_high = fvg

                # FVG fill confirmation
                fill_idx = fvg_filled_long(df, fvg_idx, gap_low, gap_high,
                                            min(reclaim_idx + 5, len(df)), atr, fill_disp)
                if fill_idx is None: continue

                # Entry at fill candle close
                entry_idx = fill_idx
                entry = float(df.iloc[entry_idx]["close"])
                sl = entry - sl_atr * atr
                tp = entry + tp_atr * atr

                result = _walk_forward_exit(df, entry_idx, "long", entry, sl, tp, max_age)
                if result is None: continue
                exit_idx, exit_price, reason, peak_r = result
                trades.append(Trade(
                    symbol=symbol, side="long", entry_price=entry, exit_price=exit_price,
                    notional_usd=1000.0, holding_sec=(exit_idx - entry_idx) * BAR_SEC,
                    entry_ts=df.index[entry_idx], exit_ts=df.index[exit_idx],
                    exit_reason=reason,
                    extra={"peak_r": round(peak_r, 3), "reclaim_lag": fill_idx - i,
                           "ob_high": ob_high, "fvg_size": gap_high - gap_low},
                ))
                open_until = exit_idx
                continue

            # ─── SHORT SETUP (bullish sweep above high + bearish reclaim) ───
            sweep_short = (h > roll_high + sweep_atr * atr) and (c > o) and (body >= disp_atr * atr)
            if sweep_short:
                ob = find_proximal_ob_short(df, i, lookback=5)
                if ob is None: continue
                ob_high, ob_low = ob

                reclaim_idx = None
                for j in range(i + 1, min(i + 1 + reclaim_lb, len(df))):
                    if float(df.iloc[j]["close"]) <= ob_low:
                        reclaim_idx = j; break
                if reclaim_idx is None: continue

                fvg = detect_fvg_short(df, i, reclaim_idx + 2)
                if fvg is None: continue
                fvg_idx, gap_low, gap_high = fvg

                fill_idx = fvg_filled_short(df, fvg_idx, gap_low, gap_high,
                                             min(reclaim_idx + 5, len(df)), atr, fill_disp)
                if fill_idx is None: continue

                entry_idx = fill_idx
                entry = float(df.iloc[entry_idx]["close"])
                sl = entry + sl_atr * atr
                tp = entry - tp_atr * atr

                result = _walk_forward_exit(df, entry_idx, "short", entry, sl, tp, max_age)
                if result is None: continue
                exit_idx, exit_price, reason, peak_r = result
                trades.append(Trade(
                    symbol=symbol, side="short", entry_price=entry, exit_price=exit_price,
                    notional_usd=1000.0, holding_sec=(exit_idx - entry_idx) * BAR_SEC,
                    entry_ts=df.index[entry_idx], exit_ts=df.index[exit_idx],
                    exit_reason=reason,
                    extra={"peak_r": round(peak_r, 3), "reclaim_lag": fill_idx - i,
                           "ob_low": ob_low, "fvg_size": gap_high - gap_low},
                ))
                open_until = exit_idx

        return trades


class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol, tf):
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


if __name__ == "__main__":
    print("=== LIQUIDITY GRAB + OB RECLAIM + FVG FILL — W/F STUDY ===\n")
    print("Cell grid: 2 disp × 2 sweep × 2 reclaim × 2 fill × 2 regime = 32")
    print("Symbols: BTC, ETH, SOL, XRP × 5m × 3 fee variants = 384 cell-variants\n")

    engine = _PatchedEngine(
        study=LiquidityGrabOBFVGStudy(),
        symbols=["BTC", "ETH", "SOL", "XRP"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "liq_grab_ob_fvg",
        verbose=False,
    )
    results = engine.run()

    cells = list(results["cells"].items())
    verdicts = {}
    for _, v in cells:
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
    print(f"\n=== DONE ({results['wall_sec']}s) ===")
    print(f"Cells run: {results['cells_run']}  PASS: {results['cells_pass']}")
    print("\nVerdict distribution:")
    for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {n}")

    # Top 10 PASS cells
    pass_cells = [c for c in cells if c[1]["verdict"] == "PASS"]
    if pass_cells:
        print(f"\n=== TOP 10 PASS CELLS by Q4 EV ===")
        pass_cells.sort(key=lambda kv: kv[1]["q4_ev"], reverse=True)
        for k, v in pass_cells[:10]:
            sym, tf, cell, variant = k.split("|", 3)
            short = {"A_full_taker":"A","B_scalper_taker":"B","C_maker_scalper":"C"}.get(variant,"?")
            print(f"  {sym} {short}  {cell}")
            print(f"    IS_n={v['is_n']} IS=${v['is_ev']:+.3f}  Q3=${v['q3_ev']:+.3f}  Q4=${v['q4_ev']:+.3f}  WR_oos={v['win_rate_oos']*100:.0f}%")
    else:
        # Top 8 by IS_EV (any verdict)
        cells_sorted = sorted(cells, key=lambda c: c[1]["is_ev"], reverse=True)
        print(f"\n=== Top 8 by IS_EV (no PASS cells) ===")
        for k, v in cells_sorted[:8]:
            sym, tf, cell, variant = k.split("|", 3)
            short = {"A_full_taker":"A","B_scalper_taker":"B","C_maker_scalper":"C"}.get(variant,"?")
            print(f"  {sym} {short}  IS_n={v['is_n']} IS=${v['is_ev']:+.3f}  Q3=${v['q3_ev']:+.3f}  Q4=${v['q4_ev']:+.3f}  verdict={v['verdict']}")

    # Compare to baseline liq_sweep_htf
    print(f"\n=== BASELINE COMPARISON (liq_sweep_htf, 7 PASS cells) ===")
    print("  SOL liq_sweep_htf disp=0.6 sweep=0.2 regime=0.6: IS=$+0.292 Q4=$+0.238")
    print("  This study targets: equal-or-better Q4 EV with similar or fewer trades")
