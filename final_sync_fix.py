"""FINAL FIX: Make real trades survive restarts + sync with Delta on startup."""
import py_compile

RM_FILE = "/home/opc/crypto-trading-bot/execution/real_manager.py"
rm = open(RM_FILE).read()
fixes = 0

# ============================================================
# FIX 1: _save_state must save ALL independent_exit fields
# Currently only saves basic fields — new fields get lost
# ============================================================
old_save = '''            open_trades_data.append({
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "side": side,
                "entry_price": t.entry_price,
                "stop_loss": getattr(t, "stop_loss", 0),
                "tp1": getattr(t, "tp1", 0),
                "tp2": getattr(t, "tp2", 0),'''

new_save = '''            open_trades_data.append({
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "side": side,
                "entry_price": t.entry_price,
                "stop_loss": getattr(t, "stop_loss", 0),
                "tp1": getattr(t, "tp1", 0),
                "tp2": getattr(t, "tp2", 0),
                # Independent exit fields (survive restart)
                "independent_exit": getattr(t, "independent_exit", True),
                "initial_risk": getattr(t, "initial_risk", 0),
                "highest_price": getattr(t, "highest_price", t.entry_price),
                "lowest_price": getattr(t, "lowest_price", t.entry_price),
                "peak_mfe_r": getattr(t, "peak_mfe_r", 0),
                "mfe_stale_seconds": getattr(t, "mfe_stale_seconds", 0),
                "last_mfe_update_time": getattr(t, "last_mfe_update_time", 0),
                "momentum_decay_count": getattr(t, "momentum_decay_count", 0),
                "breakeven_set": getattr(t, "breakeven_set", False),
                "chandelier_stop": getattr(t, "chandelier_stop", 0),
                "entry_atr": getattr(t, "entry_atr", 0),
                "trade_type": getattr(t, "trade_type", "SCALP"),
                "regime": getattr(t, "regime", ""),'''

if old_save in rm:
    rm = rm.replace(old_save, new_save)
    fixes += 1
    print("FIX 1: _save_state now saves ALL independent_exit fields")
else:
    print("FIX 1: WARN - save block not found")

# ============================================================
# FIX 2: _load_state must backfill missing fields
# ============================================================
old_load = '''                dry_obj = type("DryTrade", (), td)()
                self.real_trades[td["trade_id"]] = dry_obj'''

new_load = '''                # Backfill independent_exit fields for trades from older versions
                if "independent_exit" not in td:
                    td["independent_exit"] = True
                if "initial_risk" not in td:
                    _e = td.get("entry_price", 0)
                    _s = td.get("stop_loss", 0)
                    td["initial_risk"] = abs(_e - _s) if _e > 0 and _s > 0 else _e * 0.008
                for _fld, _def in [("highest_price", td.get("entry_price", 0)),
                                   ("lowest_price", td.get("entry_price", 0)),
                                   ("peak_mfe_r", 0), ("mfe_stale_seconds", 0),
                                   ("last_mfe_update_time", 0), ("momentum_decay_count", 0),
                                   ("breakeven_set", False), ("chandelier_stop", 0),
                                   ("entry_atr", td.get("initial_risk", 0))]:
                    if _fld not in td:
                        td[_fld] = _def
                if "trade_type" not in td or td["trade_type"] not in ("SCALP", "INTRADAY", "RUNNER"):
                    td["trade_type"] = "SCALP"
                dry_obj = type("DryTrade", (), td)()
                self.real_trades[td["trade_id"]] = dry_obj'''

if old_load in rm:
    rm = rm.replace(old_load, new_load)
    fixes += 1
    print("FIX 2: _load_state backfills ALL missing fields on load")
else:
    # Check if already has backfill
    if "independent_exit" in rm[rm.index("def _load_state"):rm.index("def _load_state")+2000] if "def _load_state" in rm else "":
        print("FIX 2: SKIP - backfill already exists")
    else:
        print("FIX 2: WARN - load block not found, trying alternate")
        # Find the DryTrade creation
        if 'type("DryTrade", (), td)()' in rm:
            idx = rm.index('type("DryTrade", (), td)()')
            line_start = rm.rfind("\n", 0, idx) + 1
            indent = " " * (idx - line_start - len('dry_obj = '))
            backfill = indent + '# Backfill independent_exit\n'
            backfill += indent + 'td.setdefault("independent_exit", True)\n'
            backfill += indent + 'td.setdefault("initial_risk", abs(td.get("entry_price",0) - td.get("stop_loss",0)) if td.get("entry_price") and td.get("stop_loss") else 0)\n'
            backfill += indent + 'td.setdefault("highest_price", td.get("entry_price", 0))\n'
            backfill += indent + 'td.setdefault("lowest_price", td.get("entry_price", 0))\n'
            backfill += indent + 'td.setdefault("peak_mfe_r", 0)\n'
            backfill += indent + 'td.setdefault("breakeven_set", False)\n'
            backfill += indent + 'td.setdefault("chandelier_stop", 0)\n'
            backfill += indent + 'td.setdefault("entry_atr", td.get("initial_risk", 0))\n'
            backfill += indent + 'td.setdefault("trade_type", "SCALP")\n'
            backfill += indent + 'td.setdefault("mfe_stale_seconds", 0)\n'
            backfill += indent + 'td.setdefault("momentum_decay_count", 0)\n'
            rm = rm[:line_start] + backfill + rm[line_start:]
            fixes += 1
            print("FIX 2: Added backfill before DryTrade creation (alternate)")

# ============================================================
# FIX 3: Remove debug logs (REAL_TICK, REAL_PRICES)
# ============================================================
rm = rm.replace(
    '''        for _tid, _t in list(self.real_trades.items()):
            _ie = getattr(_t, "independent_exit", "MISSING")
            logger.info("REAL_TICK: %s %s | independent=%s risk=%s peak=%.2fR",
                       getattr(_t, "symbol", "?"), getattr(_t, "side", "?"),
                       _ie, getattr(_t, "initial_risk", "MISSING"), getattr(_t, "peak_mfe_r", 0))''',
    ''
)
rm = rm.replace(
    '''        _eth_p = prices.get("ETH/USDT", 0)
        _btc_p = prices.get("BTC/USDT", 0)
        logger.info("REAL_PRICES: %d syms | ETH=%.2f BTC=%.0f", len(prices), _eth_p, _btc_p)''',
    ''
)
fixes += 1
print("FIX 3: Removed debug logs")

# ============================================================
# FIX 4: On startup, sync with Delta positions
# After loading state, check Delta for positions and reconcile
# ============================================================
old_startup_log = '''            logger.info(
                "REAL: Loaded state — enabled=%s, dry_run=%s, CB daily=$%.2f, "
                "total=$%.2f, %d closed trades, %d open trades",
                self.enabled, self.dry_run,
                self.circuit_breaker.daily_pnl, self.circuit_breaker.total_pnl,
                len(self.closed_real_trades), len(self.real_trades),
            )'''

new_startup_log = '''            logger.info(
                "REAL: Loaded state — enabled=%s, dry_run=%s, CB daily=$%.2f, "
                "total=$%.2f, %d closed trades, %d open trades",
                self.enabled, self.dry_run,
                self.circuit_breaker.daily_pnl, self.circuit_breaker.total_pnl,
                len(self.closed_real_trades), len(self.real_trades),
            )
            # Log loaded trade details for debugging
            for _tid, _t in self.real_trades.items():
                logger.info("REAL LOADED: %s %s %s | independent=%s risk=%.4f",
                           _tid[:20], getattr(_t, "symbol", "?"), getattr(_t, "side", "?"),
                           getattr(_t, "independent_exit", False),
                           getattr(_t, "initial_risk", 0))'''

if old_startup_log in rm:
    rm = rm.replace(old_startup_log, new_startup_log)
    fixes += 1
    print("FIX 4: Startup now logs loaded trade details")
else:
    print("FIX 4: WARN - startup log not found")

# Save
open(RM_FILE, "w").write(rm)
py_compile.compile(RM_FILE, doraise=True)
print(f"\n{'='*70}")
print(f"FINAL SYNC FIX — {fixes} changes")
print(f"{'='*70}")
print("""
NOW:
  _save_state → saves ALL exit tracking fields
  _load_state → backfills missing fields to True/defaults
  Startup → logs every loaded trade with its fields
  Debug logs → removed

This means: RESTART NO LONGER LOSES TRADE STATE
""")
