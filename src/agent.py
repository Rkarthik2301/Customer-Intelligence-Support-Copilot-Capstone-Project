import json, sqlite3, sys, time, uuid
from datetime import datetime
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))
import mock_llm
from rag import retrieve

DB_PATH = base_dir / "data/instamart.db"
LEDGER_PATH = base_dir / "outputs/refund_ledger.jsonl"
STATE_DIR = base_dir / "state"
(base_dir / "outputs").mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

def get_order(order_id):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                      (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}

def search_policy(query):
    return {"results": retrieve(query)}

def calculate_refund(order_amount, delivery_minutes):
    # late (> 45 minutes) -> 100% of the amount; otherwise 0. Escalation above INR 1000 is decided by the agent.
    if delivery_minutes is None or delivery_minutes <= 45:
        return {"refund_inr": 0.0}
    return {"refund_inr": round(float(order_amount), 2)}

def _write_ledger(tool_name, order_id, refund_inr):
    entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), "tool": tool_name,
             "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    return _write_ledger("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _write_ledger("escalate_to_human", order_id, refund_inr)

TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order's status, amount and delivery time by order id.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the best matching chunks.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund in INR from the order amount and delivery minutes.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                      "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an automatic refund of at most INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human lead and record it.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}}]

TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

class SimulatedCrash(Exception):
    def __init__(self, run_id):
        super().__init__(f"simulated crash in run {run_id}")
        self.run_id = run_id
        
#Checkpoints
def save_checkpoint(state):
    with open(STATE_DIR / f"{state['run_id']}.json", "w", encoding="utf-8") as f:
        json.dump(state, f)

def _fallback(state, status, reason):
    summary = (f"Handing over to a human agent: {reason}. Completed {len(state['steps'])} step(s): "
               f"{[s['tool'] for s in state['steps']]}. Scenario {state['scenario_id']}.")
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status,
            "final_answer": summary, "steps": state["steps"],
            "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"]}
    
#ReAct loop
def _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tool=None):
    start = time.monotonic()
    while True:
        if time.monotonic() - start >= time_limit_seconds:
            return _fallback(state, "FALLBACK_TIMEOUT", "time limit reached")
        if state["llm_calls"] >= max_iterations:
            return _fallback(state, "FALLBACK_MAX_ITERATIONS", f"no final answer after {max_iterations} model calls")
        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens
        if state["tokens_used"] > token_limit:
            return _fallback(state, "FALLBACK_TOKEN_LIMIT", f"token limit {token_limit} exceeded")

        message = response.choices[0].message
        if not message.tool_calls:   
            final_answer = message.content
            return {"run_id": state["run_id"], "scenario_id": state["scenario_id"],
                    "status": final_answer.split(":")[0], "final_answer": final_answer,
                    "steps": state["steps"], "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"]}

        call = message.tool_calls[0]
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": call.id, "name": call.name, "arguments": call.arguments}]})
        observation = TOOL_MAP[call.name](**call.arguments)  
        state["messages"].append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                  "content": json.dumps(observation)})
        state["steps"].append({"tool": call.name, "arguments": call.arguments, "observation": observation})
        state["tools_executed"].append(call.name)
        state["step"] += 1
        save_checkpoint(state) 
        if crash_after_tool is not None and state["step"] == crash_after_tool:
            raise SimulatedCrash(state["run_id"])

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              crash_after_tool=None):
    state = {"run_id": f"{scenario_id}-{uuid.uuid4().hex[:8]}", "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": "You are a refund AGENT. Use the tools to decide the request."},
                          {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "step": 0, "tools_executed": [], "steps": []}
    return _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tool)

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(STATE_DIR / f"{run_id}.json", encoding="utf-8") as f:
        state = json.load(f)
    return _loop(state, max_iterations, token_limit, time_limit_seconds)
