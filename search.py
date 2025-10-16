
"""search.py
Loads a FAISS index and metadata, generates an embedding for a user query, and performs a semantic search.
Usage:
  export OPENAI_API_KEY="sk-..."
  python search.py "How do computers learn patterns?"
"""

import os
import sys
import json
import numpy as np
from openai import OpenAI
import faiss

INDEX_FILE = "index.faiss"
VECTORS_FILE = "vectors.npy"
META_FILE = "docs_metadata.json"
MODEL = "text-embedding-3-large"

def main():
    if len(sys.argv) < 2:
        print("Usage: python search.py "your query here"")
        sys.exit(1)
    query = sys.argv[1]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable before running this script.")
    client = OpenAI(api_key=api_key)

    # Load index and metadata
    if not os.path.exists(INDEX_FILE):
        raise RuntimeError(f"Index file {INDEX_FILE} not found. Run embed_and_index.py first.")
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r") as f:
        metadata = json.load(f)

    # Embed the query
    resp = client.embeddings.create(input=query, model=MODEL)
    q_emb = np.array([resp.data[0].embedding], dtype="float32")

    k = 3
    distances, indices = index.search(q_emb, k)
    print(f"\nQuery: {query}\n\nTop {k} results:")
    for rank, idx in enumerate(indices[0]):
        doc = metadata.get(str(int(idx)), {}).get("text", "<unknown>")
        print(f"{rank+1}. (id={idx}) {doc} — distance: {distances[0][rank]:.6f}")

if __name__ == "__main__":
    main()
