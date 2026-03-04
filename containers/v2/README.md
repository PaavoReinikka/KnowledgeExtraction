# Knowledge Extraction containersV2

A PostgreSQL-based RAG environment featuring:
- **pgai**: LLM integration (OpenAI, Ollama, etc.) directly in SQL.
- **pgvector**: Vector storage and HNSW indexing.
- **Apache AGE**: Graph database capabilities for Knowledge Graphs.
- **pg_search (ParadeDB)**: BM25-based lexical search.

## Setup
1. Ensure your `.env` file (renamed from `project.env`) contains your API keys.
2. Start the stack:
   ```bash
   docker compose up --build -d
   ```

## Architecture
- **db** (`reinikp2/ai-postgres`): Custom PostgreSQL 16 image with `pgai`, `age`, and `pg_search`.
- **ai-worker**: Background worker that automatically manages embeddings for tables registered with `ai.create_vectorizer`.
- **flyway**: Manages database migrations in `database-migrations/`.
- **ollama**: Local LLM provider (optional, via `local-ai` profile).

## Features
- **Hybrid RAG**: Combine Vector search (Semantic) with BM25 (Lexical). See `sql/02_hybrid_rag_queries.sql`.
- **Auto-Embeddings**: Insert text into a table, and the worker automatically handles chunking and embedding. See `sql/01_setup_vectorizer.sql`.
- **In-DB LLM**: Summarize, translate, and rerank results using standard SQL. See `sql/03_llm_postprocessing.sql`.
