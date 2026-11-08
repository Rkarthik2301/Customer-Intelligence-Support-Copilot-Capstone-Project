import json, random, re, sys, time
from pathlib import Path
import pandas as pd
import os

base_dir = Path(__file__).resolve().parent.parent          
sys.path.insert(0, str(base_dir))                          # so "import mock_llm" finds the root file
import mock_llm
from mock_llm import TransientLLMError, est_tokens

os.makedirs(f"{base_dir}/outputs", exist_ok=True)

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
CATEGORIES = ["billing", "delivery", "technical", "fraud"]

#Redaction
PATTERNS = [("email", r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
            ("phone", r"\b[6-9]\d{9}\b", "[PHONE]"),
            ("order_id", r"ORD-\d{6}", "[ORDER_ID]")]

def redact(text):
    counts = {}
    for name, pattern, placeholder in PATTERNS:
        text, n = re.subn(pattern, placeholder, text)
        counts[name] = n
    return text, counts

def contains_pii(text):
    return any(re.search(pattern, text) for _, pattern, _ in PATTERNS)

# zero shot vs few - shot prompting
ZERO_SHOT = ("You CLASSIFY support tickets. Allowed categories: billing, delivery, technical, fraud. "
             'Reply with JSON only: {"category": "...", "reason": "..."}')

FEW_SHOT = ("You CLASSIFY support tickets. Allowed categories: billing, delivery, technical, fraud. "
            'Reply with JSON only: {"category": "...", "reason": "..."}\n'
            "EXAMPLES:\n"
            'Ticket: I was charged twice for [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}\n'
            'Ticket: The rider never delivered [ORDER_ID]. -> {"category": "delivery", "reason": "not received"}\n'
            'Ticket: The app freezes when I open the cart. -> {"category": "technical", "reason": "app error"}\n'
            'Ticket: Someone hacked my account and used my wallet. -> {"category": "fraud", "reason": "unauthorized access"}')

assert "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT

#resilient call
retry_log = []

def call_with_retry(messages, ticket_id):
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            if retry_number == MAX_RETRIES:
                raise                                           # out of retries: give up
            wait = BASE_DELAY * (2 ** retry_number) * random.uniform(0.9, 1.15)
            retry_log.append((ticket_id, retry_number, wait))
            print(f"  retry {ticket_id}: retry number {retry_number}, waiting {wait:.4f}s")
            time.sleep(wait)
            
#Answer validation
def validate_reply(text):
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (text or "").lower()))
        found = [c for c in CATEGORIES if c in words]
        if len(found) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", ""
    if isinstance(data, dict) and data.get("category") in CATEGORIES:
        return "classified_json", data["category"]
    return "flagged_invalid_output", ""         

#Pipeline
sent_prompts = []

def run_pipeline(tickets, system_prompt, token_budget=None):
    mock_llm.reset()                                            # immediately before each full run
    retry_log.clear(); sent_prompts.clear()
    running_tokens = 0
    rows = []
    for _, t in tickets.iterrows():
        row = {"ticket_id": t["ticket_id"], "customer_id": t["customer_id"], "status": "",
               "category": "", "tokens_used": 0, "true_category": t["true_category"]}
        text = t["text"]
        if text.strip() == "":                                  # guardrail (a)
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:               # guardrail (b)
            row["status"] = "flagged_long"
        elif token_budget is not None and running_tokens >= token_budget:
            row["status"] = "budget_exhausted"                  # no call is made
        else:
            clean_text, _ = redact(text)
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": clean_text}]   # EXACTLY the redacted text
            sent_prompts.append(messages)
            response = call_with_retry(messages, t["ticket_id"])
            used = response.usage.total_tokens
            running_tokens += used
            row["tokens_used"] = used
            row["status"], row["category"] = validate_reply(response.choices[0].message.content)
        rows.append(row)
    return pd.DataFrame(rows)

#routing with customer segments from part b
def add_routing(results):
    seg = pd.read_csv(base_dir / "outputs/customer_segments.csv")[["customer_id", "segment_name", "is_premium"]]
    out = results.merge(seg, on="customer_id", how="left")       # no row in the file -> NaN
    out["is_premium"] = out["is_premium"].fillna(False).astype(bool)   # treated as not premium
    routes, priorities = [], []
    for _, r in out.iterrows():
        if r["status"] in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
            routes.append("human_review"); priorities.append("NORMAL")
        elif r["category"] == "fraud":
            routes.append("security_team"); priorities.append("HIGH")
        elif r["is_premium"]:
            routes.append("premium_desk"); priorities.append("HIGH")
        else:
            routes.append("support_queue"); priorities.append("NORMAL")
    out["route"], out["priority"] = routes, priorities
    out["segment_name"] = out["segment_name"].fillna("")
    return out[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]

def report(results, label):
    ok = results[results["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((ok["category"] == ok["true_category"]).sum())
    total_tokens = int(results["tokens_used"].sum())
    print(f"\n[{label}] status counts: {results['status'].value_counts().to_dict()}")
    print(f"[{label}] classified: {len(ok)} | accuracy: {correct}/{len(ok)} = {correct/len(ok)*100:.1f}%")
    print(f"[{label}] total tokens: {total_tokens} | per classified ticket: {total_tokens/len(ok):.1f}")
    print(f"[{label}] retries logged: {len(retry_log)}")
    return ok

if __name__ == "__main__":
    tickets = pd.read_csv(f"{base_dir}/data/raw/tickets.csv", keep_default_na=False)   # keep "" as ""
    print("tickets:", len(tickets))
    
    empties = int((tickets["text"].str.strip() == "").sum())
    longs = int(tickets["text"].apply(lambda s: est_tokens(s) > MAX_INPUT_TOKENS).sum())
    print("empty:", empties, "| long:", longs, "| long estimate:", est_tokens(tickets[tickets['text'].apply(lambda s: est_tokens(s) > MAX_INPUT_TOKENS)].iloc[0]["text"]))
    totals = {"email": 0, "phone": 0, "order_id": 0}
    redacted_texts = []
    for text in tickets["text"]:
        clean, counts = redact(text)
        redacted_texts.append(clean)
        for k in totals: totals[k] += counts[k]
    print("substitutions:", totals, "| total:", sum(totals.values()))
    print("PII left after redaction:", sum(contains_pii(t) for t in redacted_texts))
    
    #zero-shot vs few-shot (no budget)
    zero = run_pipeline(tickets, ZERO_SHOT)
    for messages in sent_prompts:                               # assertion: no PII in any prompt
        assert not any(contains_pii(m["content"]) for m in messages)
    ok_zero = report(zero, "zero-shot")
    print("retry waits within range:", all(0.045 <= w <= 0.0575 for _, n, w in retry_log if n == 0))
    print("\nzero-shot confusion table (rows = true, columns = predicted):")
    print(pd.crosstab(ok_zero["true_category"], ok_zero["category"]))

    few = run_pipeline(tickets, FEW_SHOT)
    for messages in sent_prompts:
        assert not any(contains_pii(m["content"]) for m in messages)
    report(few, "few-shot")

    #budget demo (zero-shot) 
    budget = 6000
    demo = run_pipeline(tickets, ZERO_SHOT, token_budget=budget)
    print(f"\nbudget demo (TOKEN_BUDGET={budget}): budget_exhausted =", int((demo["status"] == "budget_exhausted").sum()))
    add_routing(demo).to_csv(f"{base_dir}/outputs/triage_budget_demo.csv", index=False)

    #Few shot (big budget)
    final = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
    final_out = add_routing(final)
    final_out.to_csv(f"{base_dir}/outputs/triage_results.csv", index=False)
    print("\ntriage_results.csv rows:", len(final_out))
    print(final_out["route"].value_counts().to_dict())
    print(final_out.groupby(["route", "priority"]).size())