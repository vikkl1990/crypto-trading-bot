"""Pre-train RL Shadow Agent on historical paper trades."""
import json
import sys
sys.path.insert(0, "/home/opc/crypto-trading-bot")

from ml_training.rl_shadow_agent import RLShadowAgent

def pretrain():
    with open("/home/opc/crypto-trading-bot/storage/closed_signals.json") as f:
        trades = json.load(f)
    
    agent = RLShadowAgent()
    print("Pre-training on %d trades..." % len(trades))
    
    for i, trade in enumerate(trades):
        pred = agent.predict(trade)
        realized_r = trade.get("mfe_r", 0)
        agent.record_outcome(trade, realized_r)
        
        if (i+1) % 200 == 0:
            print("  %d/%d | sizing=%.2f trail=%.2f" % (i+1, len(trades), pred["sizing_mult"], pred["trail_aggression"]))
    
    agent.save()
    print("Done! Saved to storage/ml_models/rl_shadow_weights.npz")
    
    # Test final predictions
    test = trades[-5:]
    for t in test:
        p = agent.predict(t)
        print("  %s %s | sizing=%.2fx trail=%.2f | grade=%s" % (
            t.get("symbol"), t.get("side"), p["sizing_mult"], p["trail_aggression"], t.get("grade")))

if __name__ == "__main__":
    pretrain()
