import sys
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))
import mock_llm

POLICY_DIR = Path(base_dir) / "data" / "policies"

#chunking
def chunk_words(text, size=40, overlap=10):
    words = text.split()
    step = size - overlap                       # each chunk starts 30 words after the last
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):          # this chunk reaches the last word: stop
            break
        start += step
    return chunks

def load_chunks(size=40, overlap=10):
    records = []
    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for chunk in chunk_words(text, size, overlap):
            records.append({"source": path.name, "text": chunk})     # keep the file name with every chunk
    return records

#Vectorisation ansd retrieval
class PolicyIndex:
    def __init__(self, size=40, overlap=10):
        self.chunks = load_chunks(size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])  

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])            
        scores = cosine_similarity(query_vector, self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(scores[i]), 4),
                 "text": self.chunks[i]["text"]} for i in best]

default_index = PolicyIndex()                                       

def retrieve(query, k=3):
    return default_index.retrieve(query, k)

#Augument and answer
def answer(question, index=None):
    index = index or default_index
    hits = index.retrieve(question, k=3)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    messages = [
        {"role": "system", "content": "You are a POLICY ASSISTANT. Answer only from the supplied context. "
                                      "If the answer is not in the context, say you could not find the information."},
        {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
    response = mock_llm.chat(messages)
    return response.choices[0].message.content, [h["source"] for h in hits]

#evaluate retrieval and grounding 
def evaluate(index, label):
    eval_set = pd.read_csv(base_dir / "data/raw/policy_eval.csv", keep_default_na=False)
    retrieved_ok = fact_ok = fallback_ok = 0
    answerable = unanswerable = 0
    failures = []
    for _, row in eval_set.iterrows():
        text, sources = answer(row["question"], index)
        if row["expected_doc"] == "NONE":
            unanswerable += 1
            fallback_ok += text.startswith("I could not find")
        else:
            answerable += 1
            hit_doc = row["expected_doc"] in sources
            hit_fact = row["expected_fact"] in text                
            fact_ok += hit_fact
            if not (hit_doc and hit_fact):
                failures.append((row["question"], hit_doc, hit_fact))
    print(f"[{label}] chunks: {len(index.chunks)} | expected_doc in top 3: {retrieved_ok}/{answerable}"
          f" | answer has expected_fact: {fact_ok}/{answerable} | fallback: {fallback_ok}/{unanswerable}")
    for q, d, f in failures:
        print("   FAILS:", q, "| doc retrieved:", d, "| fact in answer:", f)
        
if __name__ == "__main__":
    counts = pd.Series([c["source"] for c in default_index.chunks]).value_counts().sort_index()
    print("total chunks:", len(default_index.chunks), counts.to_dict())
    print("sources for 'refund policy late delivery':", [h["source"] for h in retrieve("refund policy late delivery")])
    evaluate(default_index, "overlap 10")
    evaluate(PolicyIndex(size=40, overlap=0), "overlap 0")