# Embeddings & Vector DB Tutorial - Example Code

This folder contains example code used in the Djamware tutorial [Everything You Need to Know About Embeddings and Vector Databases](https://www.djamware.com/post/68f0526b82144640bd8d2b45/everything-you-need-to-know-about-embeddings-and-vector-databases).

## Files

- `simple_documents.json` : Small example dataset (5 sample sentences).
- `embed_and_index.py` : Generate embeddings using OpenAI and store them in a FAISS index.
- `search.py` : Query the FAISS index with a natural language query.
- `requirements.txt` : Python dependencies.

## Quickstart

1. Create a Python virtual environment and install dependencies:

   ```
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key:

   ```
   export OPENAI_API_KEY="sk-..."   # macOS / Linux
   setx OPENAI_API_KEY "sk-..."     # Windows (or use set in a shell)
   ```

3. Generate embeddings and build the index:

   ```
   python embed_and_index.py
   ```

4. Run a semantic search:
   ```
   python search.py "How do computers learn patterns?"
   ```

## Notes

- The scripts use `text-embedding-3-large` as the embedding model. This is the same model used in the tutorial examples.
- The code intentionally reads the OpenAI API key from the `OPENAI_API_KEY` environment variable. Do **not** commit your secret key to version control.
- For larger datasets and production use, consider using a managed vector DB (Pinecone, Qdrant Cloud, Weaviate Cloud) or advanced FAISS indexes (IVF, HNSW) and add batching, caching, and persistence strategies.

## License

MIT
