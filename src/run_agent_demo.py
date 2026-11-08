import json, shutil
import pandas as pd
from agent import run_agent, resume_agent, SimulatedCrash, base_dir, STATE_DIR, LEDGER_PATH

RUNS_PATH = base_dir / "outputs/agent_runs.jsonl"
for p in (RUNS_PATH, LEDGER_PATH):
    p.write_text("")
shutil.rmtree(STATE_DIR, ignore_errors=True); STATE_DIR.mkdir()

def refund_of(result):
    for step in reversed(result["steps"]):
        if "refund_inr" in step["observation"]:
            return step["observation"]["refund_inr"]
    return 0.0

scenarios = pd.read_csv(base_dir / "data/raw/agent_scenarios.csv")
table = []
for _, s in scenarios.iterrows():
    result = run_agent(s["message"], s["scenario_id"])
    with open(RUNS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"scenario_id": result["scenario_id"], "status": result["status"],
                            "final_answer": result["final_answer"], "llm_calls": result["llm_calls"],
                            "tokens_used": result["tokens_used"], "steps": result["steps"]}) + "\n")
    refund = refund_of(result)
    ok = result["status"].startswith(s["expected_outcome"]) and abs(refund - s["expected_refund_inr"]) <= 0.01
    table.append({"scenario": s["scenario_id"], "status": result["status"], "refund_inr": refund,
                  "model_calls": result["llm_calls"], "tokens": result["tokens_used"],
                  "expected": s["expected_outcome"], "matches": ok})
print(pd.DataFrame(table).to_string(index=False))

s1 = scenarios.iloc[0]
print("\nDEMO token_limit=200:", run_agent(s1["message"], "S1", token_limit=200)["status"])
print("DEMO time_limit_seconds=0:", run_agent(s1["message"], "S1", time_limit_seconds=0)["status"])

try:
    run_agent(s1["message"], "S1", crash_after_tool=2)
except SimulatedCrash as crash:
    print("\nCRASHED after tool 2 in run", crash.run_id)
    with open(STATE_DIR / f"{crash.run_id}.json") as f:
        before = json.load(f)["tools_executed"]
    result = resume_agent(crash.run_id)
    with open(STATE_DIR / f"{crash.run_id}.json") as f:
        after = json.load(f)["tools_executed"]
    print("tools before crash:", before)
    print("tools across crash + resume:", after)
    print("final:", result["status"], refund_of(result), "| model calls:", result["llm_calls"])