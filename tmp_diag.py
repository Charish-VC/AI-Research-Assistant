import httpx, json

r = httpx.get("http://localhost:8000/health", timeout=10)
print("Health:", r.json())

r2 = httpx.post("http://localhost:8000/query",
    json={"query": "attention mechanism neural network", "top_k": 20, "threshold": 0.0},
    timeout=60)
data = r2.json()
print("Query total:", data["total"])
if data["results"]:
    for res in data["results"][:5]:
        print(f"  score={res['score']}, doc={res['doc_id'][:8]}, src={res['source']}")
else:
    print("No results — checking FAISS internals...")
