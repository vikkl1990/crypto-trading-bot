"""W/F STUDY: SOL ensemble of liq_grab_ob_fvg + liquidity_sweep_htf.

Hypothesis: When BOTH liq_grab_ob_fvg (5-step SMC) AND liquidity_sweep_htf
(sweep + HTF gate) fire same-side on SOL within ~window_bars, the ensemble
agreement = higher-conviction trend continuation.

Both scanners are SOL-only W/F-validated continuation patterns with different
filters. Both belong to the same family (continuation) — ensemble_AND may
have very few trades if they overlap heavily; that itself is a finding.

Modes:
  - ob_fvg_only:    emit when liq_grab_ob_fvg fires
  - sweep_htf_only: emit when liquidity_sweep_htf fires
  - ensemble_AND:   emit when BOTH fired same-side within window_bars
  - ensemble_OR:    emit when either fires (sanity)

Locked passing params (from prior W/F results):
  liq_grab_ob_fvg: disp_atr=0.7, fill_disp=0.4, reclaim_lookback=5,
                    regime_max=0.6, sweep_atr=0.1
  liquidity_sweep_htf: disp_atr=0.6, sweep_atr=0.2, regime_max=0.6
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
NOTIONAL_USD = 1000.0
SL_ATR = 1.0  # per locked params

# Locked liq_grab_ob_fvg params
LG_DISP_ATR = 0.7
LG_FILL_DISP = 0.4
LG_RECLAIM_LB = 5
LG_REGIME_MAX = 0.6
LG_SWEEP_ATR = 0.1

# Locked liquidity_sweep_htf params
LS_DISP_ATR = 0.6
LS_SWEEP_ATR = 0.2
LS_REGIME_MAX = 0.6

MAX_AGE_SEC = 600  # 10 min hard time stop


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


def _walk_forward_exit_np(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                          i: int, side: str, entry: float, sl: float, tp: float,
                          max_age_sec: int = MAX_AGE_SEC) -> Tuple[int, float, str]:
    """Numpy bar-by-bar walk to SL/TP/time-stop. Mirrors original _walk_forward_exit
    (research_lab/studies/liq_grab_ob_fvg_study.py) exactly:
      max_bars = max_age_sec // BAR_SEC + 2
      time_decay triggers when (j-i)*BAR_SEC >= max_age_sec
    """
    n = len(close)
    max_bars = max_age_sec // BAR_SEC + 2
    end = min(i + 1 + max_bars, n)
    for j in range(i + 1, end):
        age_sec = (j - i) * BAR_SEC
        if side == "long":
            if low[j] <= sl: return (j, sl, "sl_hit")
            if high[j] >= tp: return (j, tp, "tp_hit")
        else:
            if high[j] >= sl: return (j, sl, "sl_hit")
            if low[j] <= tp: return (j, tp, "tp_hit")
        if age_sec >= max_age_sec:
            return (j, close[j], "time_decay")
    last = min(i + max_bars, n - 1)
    return (last, close[last], "forced_end")


def detect_liq_sweep_htf_signals(open_a, high_a, low_a, close_a,
                                  atr_a, htf_pct_a, roll_high20, roll_low20):
    """Vectorized detection of liquidity_sweep_htf signal candidates.
    Returns: side_arr (str array, 'long'/'short'/''), atr_arr at signal bar.
    Signal bar = bar i. Entry = close[i] (same as study).
    """
    n = len(close_a)
    side = np.full(n, "", dtype=object)

    # Skip if HTF gate fails or atr invalid
    valid = (~np.isnan(atr_a)) & (atr_a > 0) & (~np.isnan(htf_pct_a)) & (htf_pct_a <= LS_REGIME_MAX)
    valid &= (~np.isnan(roll_high20)) & (~np.isnan(roll_low20))

    body = np.abs(close_a - open_a)
    prev_low = np.roll(low_a, 1); prev_low[0] = np.nan
    prev_high = np.roll(high_a, 1); prev_high[0] = np.nan

    long_cond = valid & (low_a < roll_low20 - LS_SWEEP_ATR * atr_a) & \
                (close_a > prev_low) & (close_a > open_a) & (body >= LS_DISP_ATR * atr_a)
    short_cond = valid & (high_a > roll_high20 + LS_SWEEP_ATR * atr_a) & \
                 (close_a < prev_high) & (close_a < open_a) & (body >= LS_DISP_ATR * atr_a)

    side[long_cond] = "long"
    # short overrides shouldn't happen with mutually-exclusive conds, but safe:
    side[short_cond & (~long_cond)] = "short"
    return side


def detect_liq_grab_ob_fvg_signals(open_a, high_a, low_a, close_a,
                                    atr_a, htf_pct_a, roll_high20, roll_low20):
    """Detect liq_grab_ob_fvg signals.
    Returns: side_arr, entry_idx_arr (int, -1 = no signal at bar i),
             where entry_idx is the FVG-fill bar (entry happens later than detection bar).

    The 5-step pattern needs lookback through reclaim+fill — we partially vectorize
    sweep detection then use a tight loop for OB/FVG/fill confirmation.
    """
    n = len(close_a)
    side = np.full(n, "", dtype=object)
    entry_idx_arr = np.full(n, -1, dtype=np.int64)

    valid = (~np.isnan(atr_a)) & (atr_a > 0) & (~np.isnan(htf_pct_a)) & (htf_pct_a <= LG_REGIME_MAX)
    valid &= (~np.isnan(roll_high20)) & (~np.isnan(roll_low20))

    body = np.abs(close_a - open_a)
    sweep_long = valid & (low_a < roll_low20 - LG_SWEEP_ATR * atr_a) & \
                 (close_a < open_a) & (body >= LG_DISP_ATR * atr_a)
    sweep_short = valid & (high_a > roll_high20 + LG_SWEEP_ATR * atr_a) & \
                  (close_a > open_a) & (body >= LG_DISP_ATR * atr_a)

    sweep_long_idx = np.flatnonzero(sweep_long)
    sweep_short_idx = np.flatnonzero(sweep_short)

    OB_LOOKBACK = 5
    RECLAIM_WINDOW = LG_RECLAIM_LB
    FVG_WINDOW = RECLAIM_WINDOW + 2
    FILL_WINDOW = 5  # search up to reclaim_idx+5 for fill

    for i in sweep_long_idx:
        if i < 25 or i + 12 >= n: continue
        # find proximal bullish OB in lookback before i
        ob_high = None
        for j in range(i - 1, max(-1, i - OB_LOOKBACK - 1), -1):
            if close_a[j] > open_a[j]:
                ob_high = high_a[j]
                break
        if ob_high is None: continue

        # OB reclaim within next reclaim_lb bars
        reclaim_idx = -1
        for j in range(i + 1, min(i + 1 + RECLAIM_WINDOW, n)):
            if close_a[j] >= ob_high:
                reclaim_idx = j; break
        if reclaim_idx < 0: continue

        # Bullish FVG in [i, reclaim_idx + 2]: bar j+1.low > bar j-1.high
        fvg_idx = -1; gap_high = 0.0
        for j in range(i + 1, min(reclaim_idx + 2, n - 1)):
            ph = high_a[j - 1]; nl = low_a[j + 1]
            if nl > ph:
                fvg_idx = j; gap_high = nl  # j+1.low; gap_high = nl
                # Note: study uses gap_high = next_low, gap_low = prev_high
                break
        if fvg_idx < 0: continue

        # FVG fill: low <= gap_high AND bullish displacement candle
        fill_idx = -1
        atr_at_i = atr_a[i]
        end_search = min(reclaim_idx + 5, n)
        for j in range(fvg_idx + 1, end_search):
            if low_a[j] <= gap_high:
                bj = abs(close_a[j] - open_a[j])
                if close_a[j] > open_a[j] and bj >= LG_FILL_DISP * atr_at_i:
                    fill_idx = j; break
        if fill_idx < 0: continue

        # Signal recorded at sweep bar i; entry will be at fill_idx
        side[i] = "long"
        entry_idx_arr[i] = fill_idx

    for i in sweep_short_idx:
        if i < 25 or i + 12 >= n: continue
        ob_low = None
        for j in range(i - 1, max(-1, i - OB_LOOKBACK - 1), -1):
            if close_a[j] < open_a[j]:
                ob_low = low_a[j]
                break
        if ob_low is None: continue

        reclaim_idx = -1
        for j in range(i + 1, min(i + 1 + RECLAIM_WINDOW, n)):
            if close_a[j] <= ob_low:
                reclaim_idx = j; break
        if reclaim_idx < 0: continue

        # Bearish FVG: bar j+1.high < bar j-1.low; gap_low = next_high
        fvg_idx = -1; gap_low = 0.0
        for j in range(i + 1, min(reclaim_idx + 2, n - 1)):
            pl = low_a[j - 1]; nh = high_a[j + 1]
            if nh < pl:
                fvg_idx = j; gap_low = nh  # next_high
                break
        if fvg_idx < 0: continue

        fill_idx = -1
        atr_at_i = atr_a[i]
        end_search = min(reclaim_idx + 5, n)
        for j in range(fvg_idx + 1, end_search):
            if high_a[j] >= gap_low:
                bj = abs(close_a[j] - open_a[j])
                if close_a[j] < open_a[j] and bj >= LG_FILL_DISP * atr_at_i:
                    fill_idx = j; break
        if fill_idx < 0: continue

        side[i] = "short"
        entry_idx_arr[i] = fill_idx

    return side, entry_idx_arr


class ComboSOLSMCConfluenceStrategy(Strategy):
    """Ensemble of liq_grab_ob_fvg + liquidity_sweep_htf for SOL.

    The strategy precomputes both scanner signal arrays once per (symbol, df),
    then param_grid sweeps over (mode, window_bars, tp_rr).
    """
    name = "combo_sol_smc_confluence"

    _atr_4h_cache: Dict[str, Optional[pd.Series]] = {}
    _signal_cache: Dict[str, Dict[str, Any]] = {}

    def param_grid(self):
        for mode in ["ob_fvg_only", "sweep_htf_only", "ensemble_AND", "ensemble_OR"]:
            for window_bars in [1, 3, 6]:
                for tp_rr in [1.5, 2.0]:
                    yield {
                        "mode": mode,
                        "window_bars": window_bars,
                        "tp_rr": tp_rr,
                    }

    def _ensure_signals(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        symbol = df.attrs.get("symbol", "SOL")
        cache_key = f"{symbol}|{len(df)}|{df.index[0]}|{df.index[-1]}"
        if cache_key in self._signal_cache:
            return self._signal_cache[cache_key]

        if symbol not in self._atr_4h_cache:
            self._atr_4h_cache[symbol] = load_4h_atr_pct(symbol)
        atr_4h = self._atr_4h_cache[symbol]
        if atr_4h is None:
            self._signal_cache[cache_key] = None
            return None

        df_local = df.copy()
        df_local["atr"] = add_atr(df_local, 14)
        df_local["roll_high20"] = df_local["high"].rolling(20).max().shift(1)
        df_local["roll_low20"] = df_local["low"].rolling(20).min().shift(1)
        df_local["htf_atr_pct"] = atr_4h.reindex(df_local.index, method="ffill")

        open_a = df_local["open"].astype(float).to_numpy()
        high_a = df_local["high"].astype(float).to_numpy()
        low_a = df_local["low"].astype(float).to_numpy()
        close_a = df_local["close"].astype(float).to_numpy()
        atr_a = df_local["atr"].astype(float).to_numpy()
        htf_pct_a = df_local["htf_atr_pct"].astype(float).to_numpy()
        rh20 = df_local["roll_high20"].astype(float).to_numpy()
        rl20 = df_local["roll_low20"].astype(float).to_numpy()

        ls_side = detect_liq_sweep_htf_signals(open_a, high_a, low_a, close_a,
                                                atr_a, htf_pct_a, rh20, rl20)
        lg_side, lg_entry_idx = detect_liq_grab_ob_fvg_signals(open_a, high_a, low_a, close_a,
                                                                atr_a, htf_pct_a, rh20, rl20)

        cached = {
            "open": open_a, "high": high_a, "low": low_a, "close": close_a,
            "atr": atr_a, "ls_side": ls_side, "lg_side": lg_side,
            "lg_entry_idx": lg_entry_idx, "index": df_local.index,
            "symbol": symbol,
        }
        self._signal_cache[cache_key] = cached
        return cached

    def simulate(self, df, params):
        if len(df) < 50: return []
        cache = self._ensure_signals(df)
        if cache is None: return []

        mode = params["mode"]
        window_bars = int(params["window_bars"])
        tp_rr = float(params["tp_rr"])

        open_a = cache["open"]; high_a = cache["high"]; low_a = cache["low"]; close_a = cache["close"]
        atr_a = cache["atr"]; ls_side = cache["ls_side"]; lg_side = cache["lg_side"]
        lg_entry_idx = cache["lg_entry_idx"]; index = cache["index"]; symbol = cache["symbol"]

        n = len(close_a)
        max_bars = MAX_AGE_SEC // BAR_SEC + 2  # buffer for end-of-data check only

        trades: List[Trade] = []
        open_until = -1

        # Pre-compute trade entries by scanning all bars
        entries: List[Tuple[int, str, int]] = []  # (signal_bar_i, side, entry_idx)

        for i in range(25, n - max_bars - 5):
            ls = ls_side[i]
            lg = lg_side[i]

            # Determine if this bar fires per mode
            signal_side = ""
            entry_idx = -1

            if mode == "sweep_htf_only":
                if ls != "":
                    signal_side = ls
                    entry_idx = i  # sweep_htf entry is at signal bar close
            elif mode == "ob_fvg_only":
                if lg != "":
                    signal_side = lg
                    entry_idx = int(lg_entry_idx[i])
            elif mode == "ensemble_AND":
                # Both must fire same-side within window_bars on or before i
                # Check if at bar i we have a confirmed pair: one fires at i, the
                # other fires within [i-window_bars, i] same-side.
                if ls != "" and lg != "" and ls == lg:
                    # Both at same bar — definitely paired
                    signal_side = ls
                    # Entry: take later of the two entries (lg entry is fill candle, ls is i)
                    lg_e = int(lg_entry_idx[i])
                    entry_idx = max(lg_e, i)
                elif ls != "":
                    # ls fires now, look back for lg same-side within window
                    for k in range(max(0, i - window_bars), i):
                        if lg_side[k] == ls:
                            signal_side = ls
                            lg_e = int(lg_entry_idx[k])
                            entry_idx = max(lg_e, i)
                            break
                elif lg != "":
                    # lg fires now (entry at fill), look back for ls same-side
                    for k in range(max(0, i - window_bars), i):
                        if ls_side[k] == lg:
                            signal_side = lg
                            lg_e = int(lg_entry_idx[i])
                            entry_idx = max(lg_e, k)  # later of fill, or ls bar (already past)
                            entry_idx = max(entry_idx, lg_e)
                            break
            elif mode == "ensemble_OR":
                # Either fires; if both fire, prefer the later entry
                if ls != "" and lg != "":
                    if ls == lg:
                        signal_side = ls
                        lg_e = int(lg_entry_idx[i])
                        entry_idx = max(lg_e, i)
                    else:
                        # Conflict — skip
                        continue
                elif ls != "":
                    signal_side = ls
                    entry_idx = i
                elif lg != "":
                    signal_side = lg
                    entry_idx = int(lg_entry_idx[i])

            if signal_side == "" or entry_idx < 0: continue
            entries.append((i, signal_side, entry_idx))

        # Now process entries chronologically with open_until guard
        for sig_i, side, ent_i in entries:
            if ent_i <= open_until: continue
            if ent_i + max_bars >= n: continue
            atr_at = atr_a[sig_i]
            if np.isnan(atr_at) or atr_at <= 0: continue

            entry = float(close_a[ent_i])
            if side == "long":
                sl = entry - SL_ATR * atr_at
                tp = entry + tp_rr * SL_ATR * atr_at  # tp_rr × risk
            else:
                sl = entry + SL_ATR * atr_at
                tp = entry - tp_rr * SL_ATR * atr_at

            exit_idx, exit_price, reason = _walk_forward_exit_np(
                high_a, low_a, close_a, ent_i, side, entry, sl, tp, MAX_AGE_SEC)

            trades.append(Trade(
                symbol=symbol, side=side, entry_price=entry, exit_price=float(exit_price),
                notional_usd=NOTIONAL_USD, holding_sec=(exit_idx - ent_i) * BAR_SEC,
                entry_ts=index[ent_i], exit_ts=index[exit_idx],
                exit_reason=reason,
                extra={"signal_lag": ent_i - sig_i, "tp_rr": tp_rr, "mode": mode},
            ))
            open_until = exit_idx

        return trades


class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol, tf):
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


if __name__ == "__main__":
    print("=== COMBO SOL SMC CONFLUENCE — W/F STUDY ===\n")
    print("Cell grid: 4 mode × 3 window × 2 tp = 24 cells")
    print("Symbol: SOL only × 5m × 3 fee variants = 72 cell-variants\n")

    engine = _PatchedEngine(
        study=ComboSOLSMCConfluenceStrategy(),
        symbols=["SOL"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "combo_sol_smc_confluence",
        verbose=True,
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
        print(f"\n=== TOP 10 PASS CELLS by Q4 EV ===")
        pass_cells.sort(key=lambda kv: kv[1]["q4_ev"], reverse=True)
        for k, v in pass_cells[:10]:
            sym, tf, cell, variant = k.split("|", 3)
            short = {"A_full_taker": "A", "B_scalper_taker": "B", "C_maker_scalper": "C"}.get(variant, "?")
            print(f"  {sym} {short}  {cell}")
            print(f"    IS_n={v['is_n']} IS=${v['is_ev']:+.3f}  Q3=${v['q3_ev']:+.3f}  Q4=${v['q4_ev']:+.3f}  WR_oos={v['win_rate_oos']*100:.0f}%")
    else:
        cells_sorted = sorted(cells, key=lambda c: c[1]["is_ev"], reverse=True)
        print(f"\n=== Top 8 by IS_EV (no PASS cells) ===")
        for k, v in cells_sorted[:8]:
            sym, tf, cell, variant = k.split("|", 3)
            short = {"A_full_taker": "A", "B_scalper_taker": "B", "C_maker_scalper": "C"}.get(variant, "?")
            print(f"  {sym} {short}  IS_n={v['is_n']} IS=${v['is_ev']:+.3f}  Q3=${v['q3_ev']:+.3f}  Q4=${v['q4_ev']:+.3f}  verdict={v['verdict']}")
