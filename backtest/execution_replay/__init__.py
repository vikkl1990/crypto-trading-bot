"""
Execution-replay backtest: takes paper signals (which assume frictionless
execution) and replays them through a realistic fill/fee/funding model
to produce the REAL P&L curve a live/demo bot would have experienced.

This is different from backtest/engine.py which replays the SCANNER on
candle data. Here we trust the scanner's output (from closed_signals.json)
and only simulate the execution layer.

Entry point:
    python3 -m backtest.execution_replay.run --days 30
"""
