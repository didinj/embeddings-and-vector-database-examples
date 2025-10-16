
"""embed_and_index.py
Generates embeddings for documents using OpenAI Embeddings API and stores them in a FAISS index.
Outputs:
  - index.faiss         : saved FAISS index
  - vectors.npy         : saved numpy array of vectors (float32)
  - docs_metadata.json  : mapping of doc_id -> text
Usage:
  export OPENAI_API_KEY="sk-..."
  python embed_and_index.py
"""

import os
import json
import numpy as np
from openai import OpenAI
import faiss

DATA_FILE = "simple_documents.json"
INDEX_FILE = "index.faiss"
VECTORS_FILE = "vectors.npy"
META_FILE = "docs_metadata.json"
MODEL = "text-embedding-3-large"

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable before running this script.")
    client = OpenAI(api_key=api_key)

    # Load documents
    with open(DATA_FILE, "r") as f:
        documents = json.load(f)

    # Generate embeddings
    embeddings = []
    for doc in documents:
        resp = client.embeddings.create(input=doc, model=MODEL)
        emb = resp.data[0].embedding
        embeddings.append(emb)

    vectors = np.array(embeddings, dtype="float32")
    dim = vectors.shape[1]
    print(f"Generated {vectors.shape[0]} vectors with dimension {dim}")

    # Create FAISS index (Flat L2 index) and save
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, INDEX_FILE)
    np.save(VECTORS_FILE, vectors)

    # Save metadata
    metadata = {str(i): {"text": documents[i]} for i in range(len(documents))}
    with open(META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved index -> {INDEX_FILE}, vectors -> {VECTORS_FILE}, metadata -> {META_FILE}")

if __name__ == "__main__":
    main()
