"""W/F STUDY: Time-decay tightening — does max_age in [180,240,...,1200]s
improve net EV under proper Q1/Q2 IS + Q3/Q4 OOS validation?

Hypothesis (from 24h shadow data — INFORMS but does not VALIDATE):
  - 67% of losses come from time_decay_10m exits (avg peak 0.13R)
  - Tightening max_age 600 → 300s should euthanize the dud bucket while
    preserving winners (which avg peak-at-time = 269-298s)

This study tests the hypothesis on cached candles using a SIMPLIFIED
structure_bounce-like entry (wick rejection at recent extremes) so the
exit-rule sensitivity is isolated. The entry logic isn't a perfect replica
of production but is realistic enough that the exit-rule grid response
should be directionally accurate.

Pass criteria (per the harness):
  - IS EV per trade > $0.01
  - Q4 EV per trade > $0.10
  - OOS gap ≤ 50%
  - Q3 and Q4 both positive

If a max_age cell PASSes across BTC AND ETH on Variant C: legitimate W/F
evidence to ship. Otherwise: live observation was anecdotal, hold.
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


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_rel_vol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    return df["volume"] / df["volume"].rolling(lookback).median()


class TimeDecayStudy(Strategy):
    """Simplified structure_bounce + variable-exit grid.

    Entry rule (constant across all cells):
      - Bar i (last closed): rejection wick at 20-bar extreme
        - LONG:  low touches 20-bar min AND lower_wick > 0.30 × bar_range
                 AND close > open (bullish reversal candle)
        - SHORT: mirror at 20-bar max
      - Volume confirm: rel_vol >= 1.0 (the Wyckoff gate)

    Exit rule (param sweep):
      - Hard SL at 1.0 × ATR
      - TP at 1.5 × ATR (i.e. 1.5R)
      - Time decay: max_age sec (THE PARAM)
      - Peak-floor stall: at peak_floor_age sec, if peak_mfe_r < 0.15 → kill

    The study isolates exit-rule sensitivity because entry is fixed.
    """
    name = "time_decay_sweep"

    # 5-min bar count helpers
    BAR_SEC = 300

    def param_grid(self):
        for max_age in [180, 240, 300, 360, 420, 480, 600, 900, 1200]:
            yield {
                "max_age_sec": max_age,
                "peak_floor_age_sec": min(max_age, 300),  # peak floor at 300s or earlier
                "peak_floor_threshold_r": 0.15,
                "sl_atr_mult": 1.0,
                "tp_atr_mult": 1.5,
                "vol_threshold": 1.0,
            }

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        if len(df) < 50:
            return []
        df = df.copy()
        df["atr"] = add_atr(df, 14)
        df["rel_vol"] = add_rel_vol(df, 20)
        df["roll_high20"] = df["high"].rolling(20).max()
        df["roll_low20"] = df["low"].rolling(20).min()

        max_age = int(params["max_age_sec"])
        floor_age = int(params["peak_floor_age_sec"])
        floor_thr = float(params["peak_floor_threshold_r"])
        sl_mult = float(params["sl_atr_mult"])
        tp_mult = float(params["tp_atr_mult"])
        vol_thr = float(params["vol_threshold"])

        bar_sec = self.BAR_SEC
        max_bars_held = max(1, max_age // bar_sec) + 2  # extra bar for safety

        # Determine symbol from df attrs (set by harness or inferred from path)
        symbol = df.attrs.get("symbol")
        if not symbol:
            # Fall back: try to infer from index of harness call. Default BTC.
            symbol = "BTC"

        trades: List[Trade] = []
        # Don't open a new trade while one is open (single-position simulator)
        open_until_idx = -1

        # Iterate from index 25 to len-max_bars_held-1
        for i in range(25, len(df) - max_bars_held - 1):
            if i <= open_until_idx:
                continue
            row = df.iloc[i]
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue
            rel_vol = row.get("rel_vol", 0)
            if pd.isna(rel_vol) or rel_vol < vol_thr:
                continue

            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            bar_range = h - l
            if bar_range <= 0:
                continue
            lower_wick = min(o, c) - l
            upper_wick = h - max(o, c)

            roll_low = df.iloc[i]["roll_low20"]
            roll_high = df.iloc[i]["roll_high20"]

            side = None
            if l <= roll_low * 1.0005 and lower_wick > bar_range * 0.30 and c > o:
                side = "long"
                entry = float(c)
                sl = entry - sl_mult * atr
                tp = entry + tp_mult * atr
            elif h >= roll_high * 0.9995 and upper_wick > bar_range * 0.30 and c < o:
                side = "short"
                entry = float(c)
                sl = entry + sl_mult * atr
                tp = entry - tp_mult * atr
            else:
                continue

            entry_ts = df.index[i]
            initial_risk = abs(entry - sl)
            peak_mfe_r = 0.0

            # Walk forward bar-by-bar
            exit_idx = None
            exit_price = None
            exit_reason = None

            for j in range(i + 1, min(i + 1 + max_bars_held, len(df))):
                bar = df.iloc[j]
                age_sec = (j - i) * bar_sec

                # Track peak MFE
                if side == "long":
                    cur_high = float(bar["high"])
                    cur_r = (cur_high - entry) / initial_risk if initial_risk > 0 else 0
                    cur_low = float(bar["low"])
                    if cur_low <= sl:
                        exit_idx = j
                        exit_price = sl
                        exit_reason = "sl_hit"
                        break
                    if cur_high >= tp:
                        exit_idx = j
                        exit_price = tp
                        exit_reason = "tp_hit"
                        break
                else:
                    cur_low = float(bar["low"])
                    cur_r = (entry - cur_low) / initial_risk if initial_risk > 0 else 0
                    cur_high = float(bar["high"])
                    if cur_high >= sl:
                        exit_idx = j
                        exit_price = sl
                        exit_reason = "sl_hit"
                        break
                    if cur_low <= tp:
                        exit_idx = j
                        exit_price = tp
                        exit_reason = "tp_hit"
                        break

                peak_mfe_r = max(peak_mfe_r, cur_r)

                # Peak-floor stall
                if age_sec >= floor_age and peak_mfe_r < floor_thr:
                    exit_idx = j
                    exit_price = float(bar["close"])
                    exit_reason = "peak_floor_stall"
                    break

                # Time decay
                if age_sec >= max_age:
                    exit_idx = j
                    exit_price = float(bar["close"])
                    exit_reason = "time_decay"
                    break

            if exit_idx is None:
                # Forced exit at last available bar
                exit_idx = i + max_bars_held
                if exit_idx >= len(df):
                    exit_idx = len(df) - 1
                exit_price = float(df.iloc[exit_idx]["close"])
                exit_reason = "forced_end"

            holding_sec = (exit_idx - i) * bar_sec
            trades.append(Trade(
                symbol=symbol,
                side=side,
                entry_price=float(entry),
                exit_price=float(exit_price),
                notional_usd=1000.0,
                holding_sec=holding_sec,
                entry_ts=entry_ts,
                exit_ts=df.index[exit_idx],
                exit_reason=exit_reason,
                extra={"peak_mfe_r": round(peak_mfe_r, 3)},
            ))
            open_until_idx = exit_idx

        return trades


# Patch: pass symbol via df.attrs for the harness
class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol: str, tf: str) -> pd.DataFrame:
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


if __name__ == "__main__":
    print("=== TIME DECAY W/F STUDY ===\n")
    print("Hypothesis: max_age 600→300s tightening improves net EV.")
    print("Cell grid: 9 max_age values × 4 symbols × 1 TF (5m) × 3 fee variants = 108\n")

    engine = _PatchedEngine(
        study=TimeDecayStudy(),
        symbols=["BTC", "ETH", "SOL", "XRP"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "time_decay_sweep",
    )
    engine.run()
