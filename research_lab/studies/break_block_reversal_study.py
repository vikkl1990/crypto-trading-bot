"""W/F STUDY: Break Block Reversal (multi-step SMC false-breakout pattern).

Idea: When price breaks a structural block then immediately reverses and
retests the broken level, the original break was likely a liquidity grab.
Entry is in the REVERSE direction of the initial break.

Setup (LONG-after-bearish-break example; mirror for SHORT):
  Step 1 — BEARISH BREAK: bar i closes BELOW rolling_low(window) with
           displacement >= break_atr_min × ATR.
  Step 2 — REVERSAL: within reversal_lookback bars, price re-enters the
           broken structure (close back ABOVE rolling_low) with body
           >= reversal_body_atr_min × ATR.
  Step 3 — RETEST: within retest_window bars after reversal, price retests
           the broken level (low touches rolling_low ± 0.3 × ATR).
  Step 4 — CONFIRMATION: at the retest, a bullish candle closes above
           rolling_low with body >= confirm_body_atr_min × ATR.
  Step 5 — ENTRY: at confirmation candle close.
  Step 6 — STOP: below retest low − 0.3 × ATR.
  Step 7 — TARGET: 1:tp_rr RR (TP at tp_rr × SL distance).

Hard time stop: 30 minutes (6 bars). Notional $400 fixed.

The pattern is a CONTRARIAN setup (anti-break) — captures false breakouts
that reverse. Should be naturally rare (target ≥5 trades/day at best params,
but ≤20/day or detection is too loose).

Compares cleanly against trend-following structure_bounce — would diversify
the bot's signal mix and likely improve maker fill rate (entries at known
structure levels).

If passes W/F, ship as Phase B paper engine `break_block_reversal_paper_engine.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.wf_harness import Strategy, Trade, WalkForwardEngine

BAR_SEC = 300
TIME_STOP_SEC = 1800  # 30 min
TIME_STOP_BARS = TIME_STOP_SEC // BAR_SEC  # 6 bars
NOTIONAL_USD = 400.0
RETEST_TOL_ATR = 0.3
SL_PAD_ATR = 0.3


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def _walk_forward_exit(
    df: pd.DataFrame,
    entry_idx: int,
    side: str,
    entry: float,
    sl: float,
    tp: float,
    max_bars: int,
):
    """Walk forward from entry_idx, return (exit_idx, exit_price, reason).

    Bar-by-bar TP/SL hit detection. If neither hit by max_bars, exit at
    that bar's close as 'time_stop'.
    """
    last = min(entry_idx + max_bars, len(df) - 1)
    for j in range(entry_idx + 1, last + 1):
        bar = df.iloc[j]
        if side == "long":
            if float(bar["low"]) <= sl:
                return (j, sl, "sl")
            if float(bar["high"]) >= tp:
                return (j, tp, "tp")
        else:
            if float(bar["high"]) >= sl:
                return (j, sl, "sl")
            if float(bar["low"]) <= tp:
                return (j, tp, "tp")
    return (last, float(df.iloc[last]["close"]), "time_stop")


class BreakBlockReversalStrategy(Strategy):
    """Break Block Reversal — false-breakout reversal pattern.

    Param sweep (216 cells per pair if rolling_window/confirm_body fixed,
    972 cells if full grid).
    """

    name = "break_block_reversal"

    # Reduced grid: fix rolling_window=20 and confirm_body_atr_min=0.65
    # Total cells = 3 × 3 × 3 × 2 × 2 × 1 × 1 = 108 (well under 216 budget)
    # Wait — task says 3×3×3×2×2×2 = 216 (drop one axis from 3→2).
    # Use full sweep on the 6 named axes; fix rolling_window=20 and
    # confirm_body=0.65 to keep grid lean per task instructions.
    USE_FULL_GRID = False  # if True: 972 cells; if False: 216 cells

    def param_grid(self) -> Iterator[Dict[str, Any]]:
        if self.USE_FULL_GRID:
            rolling_windows = [15, 20, 30]
            confirm_bodies = [0.5, 0.65]
        else:
            rolling_windows = [20]
            confirm_bodies = [0.65]
        for rolling_window in rolling_windows:
            for break_atr_min in [0.4, 0.6, 0.8]:
                for reversal_lookback in [3, 5, 8]:
                    for reversal_body in [0.4, 0.55, 0.65]:
                        for retest_window in [5, 8, 12]:
                            for confirm_body in confirm_bodies:
                                for tp_rr in [1.5, 2.0]:
                                    yield {
                                        "rolling_window": rolling_window,
                                        "break_atr_min": break_atr_min,
                                        "reversal_lookback": reversal_lookback,
                                        "reversal_body_atr_min": reversal_body,
                                        "retest_window": retest_window,
                                        "confirm_body_atr_min": confirm_body,
                                        "tp_rr": tp_rr,
                                    }

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        if len(df) < 60:
            return []
        df = df.copy()
        for c in ("open", "high", "low", "close"):
            df[c] = df[c].astype(float)

        rolling_window = int(params["rolling_window"])
        break_atr_min = float(params["break_atr_min"])
        reversal_lb = int(params["reversal_lookback"])
        reversal_body_min = float(params["reversal_body_atr_min"])
        retest_window = int(params["retest_window"])
        confirm_body_min = float(params["confirm_body_atr_min"])
        tp_rr = float(params["tp_rr"])

        df["atr"] = add_atr(df, 14)
        # Use shifted rolling so the level is "as of bar i-1" (no look-ahead)
        df["roll_high"] = df["high"].rolling(rolling_window).max().shift(1)
        df["roll_low"]  = df["low"].rolling(rolling_window).min().shift(1)

        symbol = df.attrs.get("symbol", "BTC")

        trades: List[Trade] = []
        # Cooldown: don't open another trade until the prior one exited.
        open_until = -1
        n = len(df)
        # Need room for: reversal window + retest window + max time-stop bars
        max_age_bars = TIME_STOP_BARS + 2
        max_lookahead = reversal_lb + retest_window + max_age_bars + 5
        start = max(rolling_window + 14, 30)
        end = n - max_lookahead

        for i in range(start, end):
            if i <= open_until:
                continue
            row = df.iloc[i]
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue
            roll_high = row["roll_high"]
            roll_low = row["roll_low"]
            if pd.isna(roll_high) or pd.isna(roll_low):
                continue
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            body = abs(c - o)

            # ─── LONG SETUP — bearish break, then reversal back UP ───
            # Step 1: bearish break — close below roll_low with displacement
            displacement = roll_low - c  # positive if close below the low
            bearish_break = (
                c < roll_low
                and displacement >= break_atr_min * atr
                and c < o  # bearish bar
            )
            if bearish_break:
                broken_level = roll_low
                # Step 2: scan reversal_lookback bars for re-entry
                reversal_idx: Optional[int] = None
                for j in range(i + 1, min(i + 1 + reversal_lb, n)):
                    rb = df.iloc[j]
                    rb_body = abs(float(rb["close"]) - float(rb["open"]))
                    if (
                        float(rb["close"]) > broken_level
                        and float(rb["close"]) > float(rb["open"])  # bullish bar
                        and rb_body >= reversal_body_min * atr
                    ):
                        reversal_idx = j
                        break
                if reversal_idx is None:
                    continue

                # Step 3: scan retest_window bars after reversal for retest
                # Retest = bar's low touches broken_level ± RETEST_TOL_ATR × atr
                retest_idx: Optional[int] = None
                retest_low: Optional[float] = None
                tol = RETEST_TOL_ATR * atr
                for j in range(reversal_idx + 1, min(reversal_idx + 1 + retest_window, n)):
                    bar = df.iloc[j]
                    bar_low = float(bar["low"])
                    # Touch the broken level: low <= level + tol AND low >= level - tol
                    # (i.e., bar tagged the level zone)
                    if bar_low <= broken_level + tol and bar_low >= broken_level - tol:
                        # Step 4: confirmation candle — same bar must close ABOVE
                        # broken_level (bullish reclaim) with body
                        bar_c = float(bar["close"])
                        bar_o = float(bar["open"])
                        bar_body = abs(bar_c - bar_o)
                        if (
                            bar_c > broken_level
                            and bar_c > bar_o
                            and bar_body >= confirm_body_min * atr
                        ):
                            retest_idx = j
                            retest_low = bar_low
                            break
                        # Else keep scanning — the retest must be confirmed.
                if retest_idx is None or retest_low is None:
                    continue

                # Entry at confirmation candle close
                entry_idx = retest_idx
                entry = float(df.iloc[entry_idx]["close"])
                sl = retest_low - SL_PAD_ATR * atr
                risk = entry - sl
                if risk <= 0:
                    continue
                tp = entry + tp_rr * risk

                exit_idx, exit_price, reason = _walk_forward_exit(
                    df, entry_idx, "long", entry, sl, tp, TIME_STOP_BARS
                )
                hold_sec = (exit_idx - entry_idx) * BAR_SEC
                trades.append(Trade(
                    symbol=symbol,
                    side="long",
                    entry_price=entry,
                    exit_price=exit_price,
                    notional_usd=NOTIONAL_USD,
                    holding_sec=hold_sec,
                    entry_ts=df.index[entry_idx],
                    exit_ts=df.index[exit_idx],
                    exit_reason=reason,
                    extra={
                        "break_idx": i,
                        "reversal_lag": reversal_idx - i,
                        "retest_lag": retest_idx - reversal_idx,
                        "broken_level": broken_level,
                        "atr": float(atr),
                    },
                ))
                open_until = exit_idx
                continue

            # ─── SHORT SETUP — bullish break, then reversal back DOWN ───
            # Step 1: bullish break — close above roll_high with displacement
            displacement_up = c - roll_high
            bullish_break = (
                c > roll_high
                and displacement_up >= break_atr_min * atr
                and c > o  # bullish bar
            )
            if bullish_break:
                broken_level = roll_high
                # Step 2: scan reversal_lookback bars for re-entry below
                reversal_idx = None
                for j in range(i + 1, min(i + 1 + reversal_lb, n)):
                    rb = df.iloc[j]
                    rb_body = abs(float(rb["close"]) - float(rb["open"]))
                    if (
                        float(rb["close"]) < broken_level
                        and float(rb["close"]) < float(rb["open"])  # bearish bar
                        and rb_body >= reversal_body_min * atr
                    ):
                        reversal_idx = j
                        break
                if reversal_idx is None:
                    continue

                # Step 3: scan retest_window bars after reversal for retest from below
                retest_idx = None
                retest_high: Optional[float] = None
                tol = RETEST_TOL_ATR * atr
                for j in range(reversal_idx + 1, min(reversal_idx + 1 + retest_window, n)):
                    bar = df.iloc[j]
                    bar_high = float(bar["high"])
                    if bar_high >= broken_level - tol and bar_high <= broken_level + tol:
                        bar_c = float(bar["close"])
                        bar_o = float(bar["open"])
                        bar_body = abs(bar_c - bar_o)
                        if (
                            bar_c < broken_level
                            and bar_c < bar_o
                            and bar_body >= confirm_body_min * atr
                        ):
                            retest_idx = j
                            retest_high = bar_high
                            break
                if retest_idx is None or retest_high is None:
                    continue

                entry_idx = retest_idx
                entry = float(df.iloc[entry_idx]["close"])
                sl = retest_high + SL_PAD_ATR * atr
                risk = sl - entry
                if risk <= 0:
                    continue
                tp = entry - tp_rr * risk

                exit_idx, exit_price, reason = _walk_forward_exit(
                    df, entry_idx, "short", entry, sl, tp, TIME_STOP_BARS
                )
                hold_sec = (exit_idx - entry_idx) * BAR_SEC
                trades.append(Trade(
                    symbol=symbol,
                    side="short",
                    entry_price=entry,
                    exit_price=exit_price,
                    notional_usd=NOTIONAL_USD,
                    holding_sec=hold_sec,
                    entry_ts=df.index[entry_idx],
                    exit_ts=df.index[exit_idx],
                    exit_reason=reason,
                    extra={
                        "break_idx": i,
                        "reversal_lag": reversal_idx - i,
                        "retest_lag": retest_idx - reversal_idx,
                        "broken_level": broken_level,
                        "atr": float(atr),
                    },
                ))
                open_until = exit_idx

        return trades


class _PatchedEngine(WalkForwardEngine):
    """Wrap base engine to attach symbol attribute on the loaded df."""

    def load_candles(self, symbol: str, tf: str) -> pd.DataFrame:
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


if __name__ == "__main__":
    print("=== BREAK BLOCK REVERSAL — W/F STUDY ===\n")
    grid_size = sum(1 for _ in BreakBlockReversalStrategy().param_grid())
    print(f"Grid size: {grid_size} cells (per symbol)")
    print(f"Symbols: BTC, ETH × 5m × 3 fee variants = {grid_size * 2 * 3} cell-variants\n")

    engine = _PatchedEngine(
        study=BreakBlockReversalStrategy(),
        symbols=["BTC", "ETH"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "break_block_reversal",
        verbose=False,
    )
    results = engine.run()

    cells = list(results["cells"].items())
    verdicts: Dict[str, int] = {}
    for _, v in cells:
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
    print(f"\n=== DONE ({results['wall_sec']}s) ===")
    print(f"Cells run: {results['cells_run']}  PASS: {results['cells_pass']}")
    print("\nVerdict distribution:")
    for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {n}")

    pass_cells = [c for c in cells if c[1]["verdict"] == "PASS"]
    if pass_cells:
        print("\n=== TOP 10 PASS CELLS by Q4 EV ===")
        pass_cells.sort(key=lambda kv: kv[1]["q4_ev"], reverse=True)
        for k, v in pass_cells[:10]:
            sym, tf, cell, variant = k.split("|", 3)
            short = {
                "A_full_taker": "A",
                "B_scalper_taker": "B",
                "C_maker_scalper": "C",
            }.get(variant, "?")
            print(f"  {sym} {short}  {cell}")
            print(
                f"    IS_n={v['is_n']} IS=${v['is_ev']:+.3f}  "
                f"Q3=${v['q3_ev']:+.3f}  Q4=${v['q4_ev']:+.3f}  "
                f"WR_oos={v['win_rate_oos']*100:.0f}%"
            )
    else:
        cells_sorted = sorted(cells, key=lambda c: c[1]["is_ev"], reverse=True)
        print("\n=== Top 8 by IS_EV (no PASS cells) ===")
        for k, v in cells_sorted[:8]:
            sym, tf, cell, variant = k.split("|", 3)
            short = {
                "A_full_taker": "A",
                "B_scalper_taker": "B",
                "C_maker_scalper": "C",
            }.get(variant, "?")
            print(
                f"  {sym} {short}  IS_n={v['is_n']} IS=${v['is_ev']:+.3f}  "
                f"Q3=${v['q3_ev']:+.3f}  Q4=${v['q4_ev']:+.3f}  "
                f"verdict={v['verdict']}"
            )
