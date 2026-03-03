# ai-postgres: AI-Powered PostgreSQL 16

`reinikp2/ai-postgres` is a specialized PostgreSQL 16 image designed for **AI, RAG (Retrieval-Augmented Generation), and Graph-based knowledge extraction**. It extends the standard PostgreSQL image with a pre-configured suite of powerful extensions.

## Included Extensions
- **[pgai](https://github.com/timescale/pgai)**: LLM integration directly in SQL (OpenAI, Anthropic, Ollama, etc.).
- **[pgvector](https://github.com/pgvector/pgvector)**: Vector similarity search (HNSW/IVFFlat indexing).
- **[Apache AGE](https://age.apache.org/)**: Graph database capabilities using Cypher queries.
- **[pg_search](https://github.com/paradedb/paradedb)**: Full-text search (BM25) powered by ParadeDB.
- **plpython3u**: Required for pgai and advanced Python-based procedures.

## Quick Start
To run a basic instance of the AI-powered database:

```bash
docker run -d 
  --name ai-postgres 
  -e POSTGRES_PASSWORD=mysecretpassword 
  -p 5432:5432 
  reinikp2/ai-postgres:latest
```

## How to use pgai Vectorizers
This image is designed to work seamlessly with the `pgai-vectorizer-worker`. Once your database is running, you can register a table for automatic embedding:

```sql
SELECT ai.create_vectorizer(
    'my_table'::regclass,
    loading => ai.loading_column('content_column'),
    embedding => ai.embedding_openai('text-embedding-3-small', 1536)
);
```

## Features
- **Semantic Search**: Use `pgvector` for high-performance vector similarity.
- **Hybrid RAG**: Combine semantic search with BM25 lexical search via `pg_search`.
- **Knowledge Graphs**: Store and query complex relationships using Apache AGE.
- **In-DB Summarization**: Use `ai.openai_chat_complete` to summarize or rerank documents directly in your SQL results.

## Environment Variables
- `POSTGRES_PASSWORD`: (Required) Sets the password for the postgres user.
- `POSTGRES_USER`: (Optional) Sets a custom superuser name.
- `POSTGRES_DB`: (Optional) Sets a custom default database name.
