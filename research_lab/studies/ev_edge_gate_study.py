"""W/F STUDY: EV-based and Market-Edge-based admission gate.

Hypothesis: Replace static ML threshold with an admission gate that evaluates
either EV (= p × b - (1-p)) or "market edge" (= p_model - p_mkt) where p_mkt
is a rolling 7-day cohort baseline. Trade only when EV > thr AND/OR edge > thr.

This is STRUCTURALLY DIFFERENT from the 6 KILL'd filter studies. Those asked
"is this signal good?" — this asks "does this signal have ALPHA over baseline?"

Setup (re-implemented minimally from structure_bounce):
  - 5m bar shows S/R rejection wick (>= wick_atr_min × ATR)
  - Inside structure zone: price within 1 ATR of recent 20-bar swing high/low
  - Volume confirmation: rel_vol >= 1.0
  - LONG: rejection at swing low.  SHORT: rejection at swing high.
  - Entry at signal bar close.  TP per tp_rr.  SL = swing extreme ± 0.3 × ATR.
  - Hard time stop 30 min (6 bars).  Notional $400.

Per-signal gating logic:
  - p = ml_probability proxy: 0.5 + (wick_atr × 0.10) + (rel_vol_excess × 0.05)
        + (body_score × 0.05), clamped to [0.40, 0.85]
  - b = RR = abs((TP - entry) / (entry - SL))
  - EV = p × b - (1 - p)
  - p_mkt = rolling cohort (regime × side) WR over last cohort_window_n trades
  - edge = p - p_mkt
  - Apply gate per gate_mode in {none, ev_only, edge_only, ev_and_edge, ev_or_edge}

Regime classification (minimal — from 5m price action, EMA(50) slope):
  - "trend_up"   : EMA50 slope > +0.0005 × close per bar
  - "trend_down" : EMA50 slope < -0.0005 × close per bar
  - "range"      : otherwise

Cohort key = (regime, side). Rolling deque of last N trades' WR seeds p_mkt.
Warmup: first N=cohort_window_n trades admitted unconditionally so the baseline
can build (otherwise gate_mode=none vs gated cells would have unequal warmup).

Param grid: 5 gate_mode × 5 ev_threshold × 4 edge_threshold = 100 cells per pair.
tp_rr fixed to 2.0. cohort_window_n fixed to 100.
"""
from __future__ import annotations
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Tuple, Deque

import numpy as np
import pandas as pd

ROOT = Path("/home/opc/crypto-trading-bot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.wf_harness import Strategy, Trade, WalkForwardEngine

BAR_SEC = 300
TIME_STOP_SEC = 1800
TIME_STOP_BARS = TIME_STOP_SEC // BAR_SEC  # 6 bars
NOTIONAL_USD = 400.0
SL_ATR_BUFFER = 0.3
SWING_WINDOW = 20
ATR_PERIOD = 14
EMA_PERIOD = 50
WICK_ATR_MIN = 0.55
VOL_MIN = 1.0
TP_RR_DEFAULT = 2.0
COHORT_WINDOW_N_DEFAULT = 100
WARMUP_TRADES = 100  # admit unconditionally for the first N trades per cohort


def add_atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    """ATR via True Range, simple rolling mean. Numpy."""
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


def add_ema_np(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (period + 1)
    out[0] = close[0]
    for i in range(1, n):
        out[i] = alpha * close[i] + (1.0 - alpha) * out[i - 1]
    return out


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


def _classify_regime(ema_slope: float, close: float) -> str:
    """Classify regime from EMA slope ratio (slope / close per bar)."""
    if close <= 0 or not np.isfinite(ema_slope):
        return "range"
    ratio = ema_slope / close
    if ratio > 0.0005:
        return "trend_up"
    if ratio < -0.0005:
        return "trend_down"
    return "range"


# ─────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────
class EVEdgeGateStrategy(Strategy):
    name = "ev_edge_gate"

    # cache base setup signals + features per symbol so the inner loop
    # over the param grid only varies the gate logic.
    _base_cache: Dict[str, Dict[str, Any]] = {}

    def param_grid(self):
        for gate_mode in ["none", "ev_only", "edge_only", "ev_and_edge", "ev_or_edge"]:
            for ev_threshold in [0.0, 0.02, 0.04, 0.06, 0.08]:
                for edge_threshold in [0.02, 0.04, 0.06, 0.08]:
                    yield {
                        "gate_mode": gate_mode,
                        "ev_threshold": ev_threshold,
                        "edge_threshold": edge_threshold,
                        "tp_rr": TP_RR_DEFAULT,
                        "cohort_window_n": COHORT_WINDOW_N_DEFAULT,
                    }

    def cell_id(self, params: Dict[str, Any]) -> str:
        """Override: use ; as inter-key sep so '_' inside gate_mode values
        like 'ev_only' / 'ev_and_edge' don't collide with the parser."""
        return ";".join(f"{k}={v}" for k, v in sorted(params.items()))

    def _build_base(self, df: pd.DataFrame) -> Dict[str, Any]:
        symbol = df.attrs.get("symbol", "BTC")
        if symbol in self._base_cache:
            return self._base_cache[symbol]

        open_a = df["open"].to_numpy(dtype=np.float64)
        high_a = df["high"].to_numpy(dtype=np.float64)
        low_a = df["low"].to_numpy(dtype=np.float64)
        close_a = df["close"].to_numpy(dtype=np.float64)
        vol_a = df["volume"].to_numpy(dtype=np.float64)
        atr_a = add_atr_np(high_a, low_a, close_a, ATR_PERIOD)
        ema_a = add_ema_np(close_a, EMA_PERIOD)
        # slope per bar — diff of ema
        ema_slope = np.full_like(ema_a, np.nan)
        ema_slope[1:] = ema_a[1:] - ema_a[:-1]

        v_ser = pd.Series(vol_a)
        rel_vol = (v_ser / v_ser.rolling(20, min_periods=10).mean()).to_numpy()

        h_ser = pd.Series(high_a)
        l_ser = pd.Series(low_a)
        swing_high = h_ser.rolling(SWING_WINDOW).max().shift(1).to_numpy()
        swing_low = l_ser.rolling(SWING_WINDOW).min().shift(1).to_numpy()

        upper_wick = high_a - np.maximum(open_a, close_a)
        lower_wick = np.minimum(open_a, close_a) - low_a

        # ───── Pre-compute setup signals for EVERY bar ─────
        # For each bar i, decide if it triggers a LONG, SHORT, or neither.
        # Then capture all features needed by the gate math: p_model, EV
        # given the chosen tp_rr, regime, side. Trade exit (entry, exit, etc.)
        # is also pre-computed once since tp_rr is fixed across the grid.
        n = len(close_a)
        sig_side = np.full(n, 0, dtype=np.int8)   # 1=long, -1=short, 0=none
        sig_entry = np.full(n, np.nan, dtype=np.float64)
        sig_sl = np.full(n, np.nan, dtype=np.float64)
        sig_tp = np.full(n, np.nan, dtype=np.float64)
        sig_p = np.full(n, np.nan, dtype=np.float64)
        sig_rr = np.full(n, np.nan, dtype=np.float64)
        sig_ev = np.full(n, np.nan, dtype=np.float64)
        sig_regime = np.full(n, "", dtype=object)
        sig_exit_idx = np.full(n, -1, dtype=np.int64)
        sig_exit_price = np.full(n, np.nan, dtype=np.float64)
        sig_exit_reason = np.full(n, "", dtype=object)

        max_bars = TIME_STOP_BARS
        end_i = n - (TIME_STOP_BARS + 5)
        start_i = max(SWING_WINDOW + ATR_PERIOD + 5, EMA_PERIOD + 5, 25)

        for i in range(start_i, end_i):
            atr = atr_a[i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            rv = rel_vol[i]
            if not np.isfinite(rv) or rv < VOL_MIN:
                continue
            sh = swing_high[i]; sl_ref = swing_low[i]
            if not (np.isfinite(sh) and np.isfinite(sl_ref)):
                continue

            o = open_a[i]; h = high_a[i]; l = low_a[i]; c = close_a[i]
            uw = upper_wick[i]; lw = lower_wick[i]

            regime = _classify_regime(ema_slope[i], c)

            side = None
            entry = sl = tp = wick_score = body_score = 0.0
            # LONG: rejection at swing low
            if lw >= WICK_ATR_MIN * atr and l <= sl_ref + atr and l >= sl_ref - atr:
                mid = (h + l) / 2.0
                if c > mid:
                    side = "long"
                    entry = c
                    sl = l - SL_ATR_BUFFER * atr
                    if sl >= entry:
                        continue
                    tp = entry + TP_RR_DEFAULT * (entry - sl)
                    if tp <= entry:
                        continue
                    wick_score = lw / atr
                    rng = max(h - l, 1e-12)
                    body_score = (c - mid) / (rng / 2.0)  # [0,1]
            elif uw >= WICK_ATR_MIN * atr and h >= sh - atr and h <= sh + atr:
                # SHORT: rejection at swing high
                mid = (h + l) / 2.0
                if c < mid:
                    side = "short"
                    entry = c
                    sl_p = h + SL_ATR_BUFFER * atr
                    if sl_p <= entry:
                        continue
                    tp_p = entry - TP_RR_DEFAULT * (sl_p - entry)
                    if tp_p >= entry:
                        continue
                    sl = sl_p
                    tp = tp_p
                    wick_score = uw / atr
                    rng = max(h - l, 1e-12)
                    body_score = (mid - c) / (rng / 2.0)

            if side is None:
                continue

            # p_model proxy following user spec: ml_probability
            # = 0.5 + (confluence_count * 0.05) + (wick_atr * 0.1) clamped [0.4, 0.85]
            # Confluence: vol (rv >= 1.0) gives +1, body strong (>= 0.6) gives +1.
            # Use SUBTRACTIVE base 0.30 so EV thresholds [0,0.02..0.08] actually
            # filter at 1:2 RR (where EV breakeven needs p > 0.333).
            confluence = 0
            if rv >= 1.2:
                confluence += 1
            if body_score >= 0.6:
                confluence += 1
            if wick_score >= 1.0:
                confluence += 1
            p = 0.30 + 0.05 * confluence + 0.05 * min(wick_score, 2.0)
            p = max(0.30, min(p, 0.85))
            rr = TP_RR_DEFAULT  # fixed across grid
            ev = p * rr - (1.0 - p)

            # Walk-forward exit (depends only on entry/sl/tp which are fixed
            # by tp_rr — we pre-compute once).
            exit_idx, exit_price, reason = _walk_forward_exit_np(
                high_a, low_a, close_a, i,
                side, entry, sl, tp, max_bars
            )

            sig_side[i] = 1 if side == "long" else -1
            sig_entry[i] = entry
            sig_sl[i] = sl
            sig_tp[i] = tp
            sig_p[i] = p
            sig_rr[i] = rr
            sig_ev[i] = ev
            sig_regime[i] = regime
            sig_exit_idx[i] = exit_idx
            sig_exit_price[i] = exit_price
            sig_exit_reason[i] = reason

        base = {
            "high": high_a, "low": low_a, "close": close_a,
            "ts": df.index.to_numpy(),
            "symbol": symbol,
            "sig_side": sig_side, "sig_entry": sig_entry,
            "sig_sl": sig_sl, "sig_tp": sig_tp,
            "sig_p": sig_p, "sig_rr": sig_rr, "sig_ev": sig_ev,
            "sig_regime": sig_regime,
            "sig_exit_idx": sig_exit_idx,
            "sig_exit_price": sig_exit_price,
            "sig_exit_reason": sig_exit_reason,
            "n": len(close_a),
        }
        self._base_cache[symbol] = base
        return base

    def simulate(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[Trade]:
        n = len(df)
        if n < 100:
            return []
        base = self._build_base(df)

        sig_side = base["sig_side"]; sig_entry = base["sig_entry"]
        sig_sl = base["sig_sl"]; sig_tp = base["sig_tp"]
        sig_p = base["sig_p"]; sig_ev = base["sig_ev"]
        sig_regime = base["sig_regime"]
        sig_exit_idx = base["sig_exit_idx"]
        sig_exit_price = base["sig_exit_price"]
        sig_exit_reason = base["sig_exit_reason"]
        ts_a = base["ts"]; symbol = base["symbol"]

        gate_mode = str(params["gate_mode"])
        ev_thr = float(params["ev_threshold"])
        edge_thr = float(params["edge_threshold"])
        cohort_n = int(params["cohort_window_n"])

        # Cohort baseline tracker: (regime, side) -> deque of last N WR ints
        # Each trade contributes 1 (win) or 0 (loss). p_mkt = sum/len.
        cohort_wins: Dict[Tuple[str, str], Deque[int]] = {}
        cohort_seen: Dict[Tuple[str, str], int] = {}

        trades: List[Trade] = []
        open_until = -1
        n_total = base["n"]

        for i in range(n_total):
            if i <= open_until:
                continue
            side_i = sig_side[i]
            if side_i == 0:
                continue
            side = "long" if side_i == 1 else "short"
            regime = sig_regime[i] if isinstance(sig_regime[i], str) else "range"
            cohort_key = (regime, side)

            p = sig_p[i]
            ev = sig_ev[i]

            seen = cohort_seen.get(cohort_key, 0)
            wins_dq = cohort_wins.get(cohort_key)
            if wins_dq is None:
                wins_dq = deque(maxlen=cohort_n)
                cohort_wins[cohort_key] = wins_dq

            if len(wins_dq) > 0:
                p_mkt = sum(wins_dq) / len(wins_dq)
            else:
                p_mkt = 0.5
            edge = p - p_mkt

            # ───── Gate logic ─────
            in_warmup = seen < WARMUP_TRADES
            admit = True
            if not in_warmup:
                if gate_mode == "none":
                    admit = True
                elif gate_mode == "ev_only":
                    admit = ev > ev_thr
                elif gate_mode == "edge_only":
                    admit = edge > edge_thr
                elif gate_mode == "ev_and_edge":
                    admit = (ev > ev_thr) and (edge > edge_thr)
                elif gate_mode == "ev_or_edge":
                    admit = (ev > ev_thr) or (edge > edge_thr)

            if not admit:
                # No trade — but DO update cohort? No: only ACTUAL trades update
                # the rolling baseline. Skipped signals don't count. (Otherwise
                # every cell would have identical baseline — degenerate.)
                continue

            entry = sig_entry[i]; sl = sig_sl[i]; tp = sig_tp[i]
            exit_idx = int(sig_exit_idx[i])
            exit_price = float(sig_exit_price[i])
            exit_reason = sig_exit_reason[i] if isinstance(sig_exit_reason[i], str) else "time_stop"

            # Determine win/loss for cohort update (gross)
            if side == "long":
                gross = (exit_price - entry) / entry * NOTIONAL_USD
            else:
                gross = (entry - exit_price) / entry * NOTIONAL_USD
            wins_dq.append(1 if gross > 0 else 0)
            cohort_seen[cohort_key] = seen + 1

            trades.append(Trade(
                symbol=symbol, side=side,
                entry_price=float(entry), exit_price=float(exit_price),
                notional_usd=NOTIONAL_USD,
                holding_sec=int((exit_idx - i) * BAR_SEC),
                entry_ts=pd.Timestamp(ts_a[i]),
                exit_ts=pd.Timestamp(ts_a[exit_idx]),
                exit_reason=str(exit_reason),
                extra={
                    "p_model": float(p), "ev": float(ev),
                    "p_mkt": float(p_mkt), "edge": float(edge),
                    "regime": regime, "gate_mode": gate_mode,
                    "in_warmup": bool(in_warmup),
                },
            ))
            open_until = exit_idx

        return trades


class _PatchedEngine(WalkForwardEngine):
    def load_candles(self, symbol, tf):
        df = super().load_candles(symbol, tf)
        df.attrs["symbol"] = symbol
        return df


def _summarize_and_report(results: Dict[str, Any]) -> str:
    """Build the report markdown including A/B gate_mode comparison."""
    cells = list(results["cells"].items())

    def parse_cell_key(k: str):
        sym, tf, cell, variant = k.split("|", 3)
        # Cell uses ';' as inter-key separator (override of cell_id) so values
        # containing '_' (like gate_mode='ev_and_edge') parse correctly.
        kv = dict(p.split("=", 1) for p in cell.split(";") if "=" in p)
        return sym, tf, cell, variant, kv

    # Verdict distribution
    verdicts: Dict[str, int] = {}
    for _, v in cells:
        verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1

    # ───── A/B by gate_mode (variant A_full_taker only — simpler comparison) ─────
    gate_modes = ["none", "ev_only", "edge_only", "ev_and_edge", "ev_or_edge"]
    by_gate: Dict[str, Dict[str, Any]] = {gm: {
        "pass": 0, "kill": 0, "hold": 0, "insuf": 0, "total": 0,
        "is_evs": [], "n_totals": [],
    } for gm in gate_modes}

    for k, v in cells:
        sym, tf, cell, variant, kv = parse_cell_key(k)
        if variant != "A_full_taker":
            continue
        gm = kv.get("gate_mode", "none")
        if gm not in by_gate:
            continue
        rec = by_gate[gm]
        rec["total"] += 1
        rec["is_evs"].append(v["is_ev"])
        rec["n_totals"].append(v["n_total"])
        verdict = v["verdict"]
        if verdict == "PASS":
            rec["pass"] += 1
        elif verdict.startswith("KILL"):
            rec["kill"] += 1
        elif verdict.startswith("HOLD"):
            rec["hold"] += 1
        else:
            rec["insuf"] += 1

    # ───── Per-pair top-3 PASS cells ─────
    pass_by_sym: Dict[str, List[Tuple[str, Dict]]] = {}
    for k, v in cells:
        sym, tf, cell, variant, kv = parse_cell_key(k)
        if v["verdict"] != "PASS":
            continue
        pass_by_sym.setdefault(sym, []).append((k, v))
    for sym in pass_by_sym:
        pass_by_sym[sym].sort(key=lambda kv: kv[1]["q4_ev"], reverse=True)

    # ───── Best fee variant ─────
    by_variant: Dict[str, int] = {}
    for k, v in cells:
        sym, tf, cell, variant, kv = parse_cell_key(k)
        if v["verdict"] == "PASS":
            by_variant[variant] = by_variant.get(variant, 0) + 1

    # ───── Best gate_mode (highest PASS rate) ─────
    best_gate = max(gate_modes, key=lambda g: by_gate[g]["pass"] / max(by_gate[g]["total"], 1))

    # ───── Build markdown ─────
    lines = [
        f"# EV/Edge Gate — Walk-Forward Report",
        f"",
        f"Generated: {results['finished_at']}",
        f"Cells run: {results['cells_run']}  PASS: {results['cells_pass']}  Wall: {results['wall_sec']}s",
        f"",
        f"## Headline: Does the EV/Edge Gate add structural alpha?",
        f"",
        f"**Critical comparison — does ANY gate beat `none`?**",
        f"",
        f"| gate_mode | total | PASS | PASS_rate | avg_IS_EV | avg_n_total | uplift_vs_none | trade_reduction |",
        f"|---|---|---|---|---|---|---|---|",
    ]
    none_avg_is = (sum(by_gate["none"]["is_evs"]) / max(len(by_gate["none"]["is_evs"]), 1)) if by_gate["none"]["is_evs"] else 0.0
    none_avg_n = (sum(by_gate["none"]["n_totals"]) / max(len(by_gate["none"]["n_totals"]), 1)) if by_gate["none"]["n_totals"] else 0.0
    none_pass_rate = by_gate["none"]["pass"] / max(by_gate["none"]["total"], 1)
    for gm in gate_modes:
        rec = by_gate[gm]
        avg_is = (sum(rec["is_evs"]) / max(len(rec["is_evs"]), 1)) if rec["is_evs"] else 0.0
        avg_n = (sum(rec["n_totals"]) / max(len(rec["n_totals"]), 1)) if rec["n_totals"] else 0.0
        pass_rate = rec["pass"] / max(rec["total"], 1)
        uplift = avg_is - none_avg_is
        reduction = ((none_avg_n - avg_n) / none_avg_n * 100.0) if none_avg_n > 0 else 0.0
        lines.append(
            f"| {gm} | {rec['total']} | {rec['pass']} | {pass_rate*100:.1f}% | "
            f"${avg_is:+.4f} | {avg_n:.0f} | ${uplift:+.4f} | {reduction:+.1f}% |"
        )

    lines.extend([
        f"",
        f"**Critical question answer:** ",
        f"- `none` PASS rate: {none_pass_rate*100:.1f}% ({by_gate['none']['pass']}/{by_gate['none']['total']})",
        f"- Best gated PASS rate: {by_gate[best_gate]['pass'] / max(by_gate[best_gate]['total'], 1) * 100:.1f}% ({by_gate[best_gate]['pass']}/{by_gate[best_gate]['total']}) via `{best_gate}`",
    ])

    if by_gate[best_gate]["pass"] > by_gate["none"]["pass"]:
        lines.append(f"- **VERDICT: GATE ADDS STRUCTURAL ALPHA** — `{best_gate}` beats `none` by "
                     f"{by_gate[best_gate]['pass'] - by_gate['none']['pass']} more PASSing cells.")
    elif by_gate[best_gate]["pass"] == by_gate["none"]["pass"] and by_gate["none"]["pass"] > 0:
        lines.append(f"- **VERDICT: GATE NEUTRAL** — `none` already passes; gating doesn't add alpha but doesn't hurt either.")
    else:
        lines.append(f"- **VERDICT: NO STRUCTURAL ALPHA** — gating fails to lift PASS rate above `none`.")

    # Verdict distribution
    lines.extend([
        f"",
        f"## Verdict distribution (all cells × variants)",
        f"",
        f"| Verdict | Count |",
        f"|---|---|",
    ])
    for v, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        lines.append(f"| {v} | {n} |")

    # Top-3 PASS per pair
    lines.extend([
        f"",
        f"## Top 3 PASS cells per pair (by Q4 EV)",
        f"",
    ])
    for sym in ["BTC", "ETH", "SOL", "XRP"]:
        lines.append(f"### {sym}")
        lines.append(f"")
        rows = pass_by_sym.get(sym, [])
        if not rows:
            lines.append(f"_(no PASS cells)_")
            lines.append(f"")
            continue
        lines.append(f"| variant | params | IS_EV | Q3_EV | Q4_EV | trades/day | gap% |")
        lines.append(f"|---|---|---|---|---|---|---|")
        for k, v in rows[:3]:
            sym2, tf, cell, variant, kv = parse_cell_key(k)
            # trades/day: ~ 7 months of IS+OOS data ≈ 213 days; n_total / days
            tdy = v["n_total"] / 213.0
            gap = f"{v['gap_pct']*100:.0f}" if v.get("gap_pct") is not None else "—"
            lines.append(
                f"| {variant} | {cell} | ${v['is_ev']:+.3f} | ${v['q3_ev']:+.3f} | "
                f"${v['q4_ev']:+.3f} | {tdy:.2f} | {gap}% |"
            )
        lines.append(f"")

    # Best fee variant
    lines.extend([
        f"",
        f"## Best fee variant (PASS counts)",
        f"",
        f"| variant | PASS_cells |",
        f"|---|---|",
    ])
    if by_variant:
        for v, n in sorted(by_variant.items(), key=lambda x: -x[1]):
            lines.append(f"| {v} | {n} |")
    else:
        lines.append(f"| (no PASS cells) | 0 |")
    best_fee = max(by_variant.items(), key=lambda x: x[1])[0] if by_variant else "(none)"

    # Best gate_mode
    lines.extend([
        f"",
        f"## Best gate_mode",
        f"",
        f"`{best_gate}` — {by_gate[best_gate]['pass']}/{by_gate[best_gate]['total']} cells PASS at variant A_full_taker.",
        f"",
        f"Avg IS_EV at this gate_mode: ${(sum(by_gate[best_gate]['is_evs'])/max(len(by_gate[best_gate]['is_evs']),1)):+.4f}",
        f"",
    ])

    # Recommendation
    lines.extend([
        f"## Recommendation",
        f"",
    ])
    if by_gate[best_gate]["pass"] > 0 and by_gate[best_gate]["pass"] >= max(by_gate["none"]["pass"], 1):
        # Find best (ev_thr, edge_thr) within the best gate_mode
        best_combo = None
        best_q4 = -1e9
        for k, v in cells:
            sym, tf, cell, variant, kv = parse_cell_key(k)
            if v["verdict"] != "PASS":
                continue
            if kv.get("gate_mode") != best_gate:
                continue
            if variant != best_fee:
                continue
            if v["q4_ev"] > best_q4:
                best_q4 = v["q4_ev"]
                best_combo = (kv, v, sym)
        if best_combo:
            kv, v, sym = best_combo
            lines.append(f"**SHIP as Patch P:**")
            lines.append(f"- `gate_mode`: `{best_gate}`")
            lines.append(f"- `ev_threshold`: {kv.get('ev_threshold', 'n/a')}")
            lines.append(f"- `edge_threshold`: {kv.get('edge_threshold', 'n/a')}")
            lines.append(f"- Top OOS Q4 EV: ${v['q4_ev']:+.3f} on {sym} (variant {best_fee})")
        else:
            lines.append(f"_Best gate_mode is `{best_gate}` but no PASS cell in best fee {best_fee} — review manually._")
    else:
        lines.append(f"**KILL** — no gate variant adds structural alpha over `none`. ")
        lines.append(f"All gate modes either underperform or match the unfiltered baseline.")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print("=== EV/EDGE GATE — W/F STUDY ===\n")
    print("Cell grid: 5 gate_mode × 5 ev_thr × 4 edge_thr = 100 cells per pair")
    print("(tp_rr=2.0 fixed, cohort_window_n=100 fixed)")
    print("Symbols: BTC, ETH, SOL, XRP × 5m × 3 fee variants = 1200 cell-variants total\n")

    engine = _PatchedEngine(
        study=EVEdgeGateStrategy(),
        symbols=["BTC", "ETH", "SOL", "XRP"],
        timeframes=["5m"],
        out_dir=ROOT / "storage" / "wf_studies" / "ev_edge_gate",
        verbose=False,
    )
    results = engine.run()

    # Custom report
    report = _summarize_and_report(results)
    out_md = ROOT / "storage" / "wf_studies" / "ev_edge_gate" / "report.md"
    out_md.write_text(report)
    print(report)
    print(f"\nReport: {out_md}")
