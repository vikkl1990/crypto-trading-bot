"""W/F STUDIES: 3 scanner relaxations — find an entry rule with IS edge.

Each scanner is implemented as a Strategy with parameter relaxations sweeping
the gate that's most likely too tight. Common exit rule: 1× ATR SL / 1.5×
ATR TP / 600s time_decay (fixed — we already W/F-killed the exit-tweak hypothesis).

Studies:
  1. EMAMomentumStudy — fast/slow EMA cross + volume; sweeps EMA pairs
  2. LiquiditySweepStudy — sweep through 20-bar extreme + reversal; sweeps displacement
  3. BosChochStudy — break of N-bar high/low + displacement; sweeps lookback × displacement

If ANY cell passes IS EV > $0.01 AND OOS gap ≤50% AND Q4 EV > $0.10,
we have a non-broken entry to base scanner-relaxation patches on.
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


def add_rel_vol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    return df["volume"] / df["volume"].rolling(lookback).median()


def _walk_forward_exit(df, i, side, entry, sl, tp, max_age_sec):
    """Common exit walker: SL hit / TP hit / time_decay. Returns (exit_idx, exit_price, reason, peak_r)."""
    initial_risk = abs(entry - sl)
    if initial_risk <= 0:
        return None
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


# ──────────────────────────────────────────────────────────────────────
# Study 1 — EMA Momentum (your 5/8/13, 9/21, 20/50 EMA concepts)
# ──────────────────────────────────────────────────────────────────────
class EMAMomentumStudy(Strategy):
    """EMA cross + above-median volume. Skips structure_bounce's pullback wait.

    LONG  on bar i where: fast_ema[i] crosses above slow_ema[i] AND vol >= vol_thr
    SHORT on bar i where: fast_ema[i] crosses below slow_ema[i] AND vol >= vol_thr
    """
    name = "ema_momentum_relax"

    def param_grid(self):
        # User's reference strategies: 5/8/13, 9/21, 20/50
        # Also include 8/21 (current production)
        for (fast, slow) in [(5, 13), (8, 21), (9, 21), (13, 50), (20, 50)]:
            for vol_thr in [0.8, 1.0]:
                yield {"fast": fast, "slow": slow, "vol_thr": vol_thr,
                       "max_age": 600, "sl_atr": 1.0, "tp_atr": 1.5}

    def simulate(self, df, params):
        if len(df) < 100:
            return []
        df = df.copy()
        fast, slow = int(params["fast"]), int(params["slow"])
        df["ema_f"] = df["close"].ewm(span=fast, adjust=False).mean()
        df["ema_s"] = df["close"].ewm(span=slow, adjust=False).mean()
        df["atr"] = add_atr(df, 14)
        df["rel_vol"] = add_rel_vol(df, 20)

        symbol = df.attrs.get("symbol", "BTC")
        vol_thr = float(params["vol_thr"])
        max_age = int(params["max_age"])
        sl_atr = float(params["sl_atr"])
        tp_atr = float(params["tp_atr"])

        trades: List[Trade] = []
        open_until = -1

        for i in range(slow + 5, len(df) - (max_age // BAR_SEC + 2)):
            if i <= open_until:
                continue
            row = df.iloc[i]; prev = df.iloc[i - 1]
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue
            rel_vol = row.get("rel_vol", 0)
            if pd.isna(rel_vol) or rel_vol < vol_thr:
                continue

            ef, es, efp, esp = row["ema_f"], row["ema_s"], prev["ema_f"], prev["ema_s"]
            if pd.isna([ef, es, efp, esp]).any():
                continue

            side = None
            if efp <= esp and ef > es:
                side = "long"
                entry = float(row["close"]); sl = entry - sl_atr * atr; tp = entry + tp_atr * atr
            elif efp >= esp and ef < es:
                side = "short"
                entry = float(row["close"]); sl = entry + sl_atr * atr; tp = entry - tp_atr * atr
            else:
                continue

            result = _walk_forward_exit(df, i, side, entry, sl, tp, max_age)
            if result is None:
                continue
            exit_idx, exit_price, reason, peak_r = result
            holding_sec = (exit_idx - i) * BAR_SEC
            trades.append(Trade(
                symbol=symbol, side=side,
                entry_price=entry, exit_price=exit_price,
                notional_usd=1000.0, holding_sec=holding_sec,
                entry_ts=df.index[i], exit_ts=df.index[exit_idx],
                exit_reason=reason, extra={"peak_r": round(peak_r, 3)},
            ))
            open_until = exit_idx
        return trades


# ──────────────────────────────────────────────────────────────────────
# Study 2 — Liquidity Sweep (your "sweep + reclaim" concept)
# ──────────────────────────────────────────────────────────────────────
class LiquiditySweepStudy(Strategy):
    """Sweep through 20-bar extreme then snap-back with displacement.

    LONG: bar i swept below 20-bar low (low < 20-bar-low) AND closed above prev low
          AND body ≥ displacement_atr × ATR (real reversal, not just wick)
    SHORT: mirror at 20-bar high
    """
    name = "liquidity_sweep_relax"

    def param_grid(self):
        for disp_atr in [0.4, 0.6, 0.8, 1.0]:
            for sweep_atr in [0.1, 0.2, 0.3]:  # how far below extreme to count as a sweep
                yield {"disp_atr": disp_atr, "sweep_atr": sweep_atr,
                       "max_age": 600, "sl_atr": 1.0, "tp_atr": 2.0}

    def simulate(self, df, params):
        if len(df) < 50:
            return []
        df = df.copy()
        df["atr"] = add_atr(df, 14)
        df["roll_high20"] = df["high"].rolling(20).max().shift(1)  # exclude current bar
        df["roll_low20"]  = df["low"].rolling(20).min().shift(1)
        symbol = df.attrs.get("symbol", "BTC")

        disp_atr = float(params["disp_atr"])
        sweep_atr = float(params["sweep_atr"])
        max_age = int(params["max_age"])
        sl_atr = float(params["sl_atr"])
        tp_atr = float(params["tp_atr"])

        trades: List[Trade] = []
        open_until = -1

        for i in range(25, len(df) - (max_age // BAR_SEC + 2)):
            if i <= open_until:
                continue
            row = df.iloc[i]; prev = df.iloc[i - 1]
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue

            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            roll_high = float(row["roll_high20"]); roll_low = float(row["roll_low20"])
            if pd.isna([roll_high, roll_low]).any():
                continue
            body = abs(c - o)

            side = None
            # LONG: swept below 20-bar low + closed back inside, with body >= disp_atr × ATR
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
                exit_reason=reason, extra={"peak_r": round(peak_r, 3)},
            ))
            open_until = exit_idx
        return trades


# ──────────────────────────────────────────────────────────────────────
# Study 3 — BOS / CHoCH (your "trendline break + displacement" concept)
# ──────────────────────────────────────────────────────────────────────
class BosChochStudy(Strategy):
    """Break of N-bar structure with displacement bar + confirmation.

    LONG: bar i closes above N-bar rolling high AND bar i body ≥ disp_atr × ATR
          (CHoCH-flavor: the prior N bars were below high; this is the break)
    SHORT: mirror.

    The displacement gate replicates the "displacement bar" concept from your
    pin-bar / SMC framework: the break must be DECISIVE, not a weak push.
    """
    name = "bos_choch_relax"

    def param_grid(self):
        for lookback in [10, 20, 30]:
            for disp_atr in [0.5, 0.7, 1.0, 1.3]:
                yield {"lookback": lookback, "disp_atr": disp_atr,
                       "max_age": 600, "sl_atr": 1.0, "tp_atr": 2.0}

    def simulate(self, df, params):
        if len(df) < 50:
            return []
        df = df.copy()
        df["atr"] = add_atr(df, 14)
        n = int(params["lookback"])
        df["roll_high"] = df["high"].rolling(n).max().shift(1)
        df["roll_low"]  = df["low"].rolling(n).min().shift(1)
        symbol = df.attrs.get("symbol", "BTC")

        disp_atr = float(params["disp_atr"])
        max_age = int(params["max_age"])
        sl_atr = float(params["sl_atr"])
        tp_atr = float(params["tp_atr"])

        trades: List[Trade] = []
        open_until = -1

        for i in range(n + 5, len(df) - (max_age // BAR_SEC + 2)):
            if i <= open_until:
                continue
            row = df.iloc[i]
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            roll_high = float(row["roll_high"]); roll_low = float(row["roll_low"])
            if pd.isna([roll_high, roll_low]).any():
                continue
            body = abs(c - o)
            if body < disp_atr * atr:
                continue

            side = None
            if c > roll_high and c > o:  # bullish break
                side = "long"
                entry = c; sl = entry - sl_atr * atr; tp = entry + tp_atr * atr
            elif c < roll_low and c < o:  # bearish break
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
                exit_reason=reason, extra={"peak_r": round(peak_r, 3)},
            ))
            open_until = exit_idx
        return trades


class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol, tf):
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


if __name__ == "__main__":
    studies = [
        ("ema_momentum", EMAMomentumStudy()),
        ("liquidity_sweep", LiquiditySweepStudy()),
        ("bos_choch", BosChochStudy()),
    ]
    for label, study in studies:
        print(f"\n{'='*70}")
        print(f"  STUDY: {label}")
        print(f"{'='*70}\n")
        engine = _PatchedEngine(
            study=study,
            symbols=["BTC", "ETH", "SOL", "XRP"],
            timeframes=["5m"],
            out_dir=ROOT / "storage" / "wf_studies" / study.name,
            verbose=False,  # suppress per-cell prints; show summary only
        )
        results = engine.run()
        # Quick summary
        cells = list(results["cells"].items())
        passed = [c for c in cells if c[1]["verdict"] == "PASS"]
        is_pos = [c for c in cells if c[1]["is_ev"] > 0.01]
        print(f"Cells run: {len(cells)}  IS EV > $0.01: {len(is_pos)}  PASS: {len(passed)}")
        # Verdict distribution
        verdicts = {}
        for _, v in cells:
            verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
        for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
            print(f"  {v}: {n}")
        # Top 5 cells by IS EV (any verdict)
        cells_sorted = sorted(cells, key=lambda c: c[1]["is_ev"], reverse=True)
        print(f"\n  Top 5 cells by IS EV:")
        for k, v in cells_sorted[:5]:
            sym, tf, cell, variant = k.split("|", 3)
            short_var = variant.replace("_full_taker", "A").replace("_scalper_taker", "B").replace("_maker_scalper", "C")
            print(f"    {sym} {variant[:18]}  cell={cell[:40]}  IS_n={v['is_n']} IS=${v['is_ev']:+.3f} Q4=${v['q4_ev']:+.3f} verdict={v['verdict']}")

    print(f"\n{'='*70}")
    print("  ALL STUDIES COMPLETE")
    print(f"{'='*70}")
    print("Reports written to storage/wf_studies/<study_name>/")
