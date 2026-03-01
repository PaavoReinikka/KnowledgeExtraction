# Hybrid-Postgres (reinikp2/hybrid-postgres)

A custom PostgreSQL image tailored for modern AI applications and Hybrid RAG. It combines vector search, keyword search (BM25), and scheduled tasks.

## Included Extensions

- **[ParadeDB (pg_search)](https://github.com/paradedb/paradedb)**: High-performance BM25 and full-text search.
- **[pgvector](https://github.com/pgvector/pgvector)**: Vector similarity search (for LLM embeddings).
- **pg_cron**: For running periodic maintenance or extraction tasks within the database.

## Quick Start

```bash
docker run -d \
  --name hybrid-db \
  -e POSTGRES_PASSWORD=mypassword \
  -p 5432:5432 \
  reinikp2/hybrid-postgres:latest
```

## Usage in Docker Compose

```yaml
services:
  db:
    image: reinikp2/hybrid-postgres:latest
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: mypassword
      POSTGRES_DB: hybrid-db
    ports:
      - "5433:5432"
```
