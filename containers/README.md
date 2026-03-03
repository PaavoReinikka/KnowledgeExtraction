# Hybrid Postgres Container

This directory contains the custom Docker configuration for a "batteries-included" PostgreSQL image designed for knowledge extraction, vector search, and graph analysis.

## Features

- **Base Image**: PostgreSQL 16 (Bookworm)
- **Extensions included**:
    - `pg_vector` (v0.8.0): Vector similarity search.
    - `pg_search` (v0.21.10): Full-text search using BM25 (built on Tantivy).
    - `Apache AGE` (v1.6.0-rc0): Graph database functionality for PostgreSQL.
- **Auto-Initialization**: Extensions are automatically enabled in the default database on first start.
- **Preconfigured**: `shared_preload_libraries` is set to include `age` and `pg_search`.

## Getting Started

### Prerequisites

- Docker and Docker Compose
- `uv` (optional, for running test data scripts)

### Run with Docker Compose

```bash
docker compose up --build
```

### Accessing the Database

The container is configured with shortcuts for easier access:

```bash
docker exec -it knowledge_db psql
```

## Management & Development

### Database Migrations

Migrations are handled by Flyway. The migration files are located in `./database-migrations`.

- `V1__Initial_Schema.sql`: Creates core tables (e.g., `documents`).
- `V2__Enable_Extensions.sql`: Redundant but safe check to enable extensions.

### Test Data Seeding

To seed the database with synthetic documents (vectors) and a test graph (AGE):

```bash
uv run ../test_data/seed_data.py
```

## Publishing to Docker Hub

To build and publish a new version of the image:

1. **Login**: `docker login`
2. **Build and Tag**:
   ```bash
   docker build -t reinikp2/hybrid-postgres:latest -f Dockerfile.custom .
   ```
3. **Push**:
   ```bash
   docker push reinikp2/hybrid-postgres:latest
   ```

## Usage Notes

### Apache AGE

To use graph features, you must set the `search_path` and use the `cypher` function:

```sql
SET search_path = public, ag_catalog;
SELECT * FROM cypher('your_graph_name', $$ MATCH (n) RETURN n $$) as (v agtype);
```

### pg_search (ParadeDB)

Create indexes using:

```sql
CALL pg_search.create_bm25(index_name => 'idx_name', table_name => 'table_name', columns => '{col1}');
```

If you create a BM25 index with `USING bm25 ... WITH (key_field='id')`, include that key field in the indexed column list. Example:

```sql
CREATE INDEX idx_knowledge_bm25
ON knowledge_items USING bm25 (id, content)
WITH (key_field = 'id');
```

If the `key_field` is not present in the BM25 column list, inserts can fail with `ERROR: No key field defined.`

### Extension Management (`vector`, `age`, `pg_search`)

Extensions are enabled both in container init scripts and in Flyway migration `V1__Initial_Schema.sql` using `CREATE EXTENSION IF NOT EXISTS ...`.

- Keeping them in Flyway is recommended for portability and explicit schema setup.
- Keeping `IF NOT EXISTS` makes repeated runs safe.
- The startup warnings (`already exists, skipping`) are expected and harmless.

### Hybrid RAG Query Examples

Assume this table from `V3__Setup_Hybrid_Search.sql`:

- `knowledge_items(id uuid, content text, embedding vector(384), metadata jsonb, created_at timestamptz)`

#### 1) Simple keyword search (baseline)

```sql
SELECT id, content
FROM knowledge_items
WHERE content ILIKE '%hybrid%'
ORDER BY created_at DESC
LIMIT 10;
```

#### 2) Vector similarity search (`pgvector`)

Use your query embedding from the application (example below uses a dummy vector):

```sql
SELECT
   id,
   content,
   (embedding <=> array_fill(0.1, ARRAY[384])::vector) AS distance
FROM knowledge_items
ORDER BY embedding <=> array_fill(0.1, ARRAY[384])::vector
LIMIT 10;
```

- Lower `distance` means a better semantic match.

#### 3) BM25 full-text retrieval (`pg_search` / ParadeDB)

```sql
SELECT
   id,
   content,
   pdb.score(id) AS bm25_score
FROM knowledge_items
WHERE content ||| 'hybrid search bm25 vector'
ORDER BY bm25_score DESC
LIMIT 10;
```

- Use `|||` for OR-style match; use `&&&` for stricter AND-style matching.

#### 4) Hybrid retrieval with Reciprocal Rank Fusion (RRF)

This is a practical SQL-only pattern: generate two top-k lists (BM25 + vector), then fuse by rank.

```sql
DROP TABLE IF EXISTS tmp_bm25_hits;
DROP TABLE IF EXISTS tmp_vector_hits;

CREATE TEMP TABLE tmp_bm25_hits AS
SELECT
   id,
   ROW_NUMBER() OVER (ORDER BY bm25_score DESC) AS bm25_rank
FROM (
   SELECT id, pdb.score(id) AS bm25_score
   FROM knowledge_items
   WHERE content ||| 'hybrid search bm25 vector'
   ORDER BY bm25_score DESC
   LIMIT 50
) s;

CREATE TEMP TABLE tmp_vector_hits AS
SELECT
   id,
   ROW_NUMBER() OVER (ORDER BY vec_distance ASC) AS vec_rank
FROM (
   SELECT id, embedding <=> array_fill(0.1, ARRAY[384])::vector AS vec_distance
   FROM knowledge_items
   ORDER BY vec_distance ASC
   LIMIT 50
) s;

SELECT
   k.id,
   k.content,
   (
      COALESCE(1.0 / (60 + b.bm25_rank), 0.0) +
      COALESCE(1.0 / (60 + v.vec_rank), 0.0)
   ) AS rrf_score
FROM knowledge_items k
LEFT JOIN tmp_bm25_hits b ON k.id = b.id
LEFT JOIN tmp_vector_hits v ON k.id = v.id
WHERE b.id IS NOT NULL OR v.id IS NOT NULL
ORDER BY rrf_score DESC
LIMIT 10;
```

RRF tips:

- `LIMIT 50` controls candidate depth from each retriever.
- `60` is the RRF smoothing constant (`k`) and a good default.
- In production, use your real query embedding instead of `array_fill(...)`.


## Services

- **PostgreSQL (`knowledge_db`)**: Custom image (`reinikp2/hybrid-postgres`) based on ParadeDB (BM25) with `pgvector` installed.
- **Flyway**: Automated schema migration tool that applies scripts from `./database-migrations`.

## Quick Start (Docker Compose)

1. **Start services**:
   ```bash
   docker compose up -d
   ```

2. **Stop services**:
   ```bash
   docker compose down
   ```

3. **Reset everything** (Deletes all data!):
   ```bash
   docker compose down -v
   ```

## Debugging & Manual Access

To enter the database manually via `psql`:
```bash
docker exec -it knowledge_db psql -U postgres -d knowledge_base
```

To view migration logs:
```bash
docker compose logs flyway
```

## Connecting from Python

Use the following environment variables or connection string:

- **Host**: `localhost`
- **Port**: `5433` (Mapped from container 5432 to avoid local conflicts)
- **User**: `postgres`
- **Password**: `mypassword`
- **Database**: `knowledge_base`

### Connection String (SQLAlchemy/psycopg2)
`postgresql://postgres:mypassword@localhost:5433/knowledge_base`

## Adding Migrations
To evolve the schema, add new `.sql` files to `./database-migrations/` following the naming convention: `V2__description.sql`, `V3__...`. Flyway will apply them automatically on next start.

---

## Docker Hub Reference
This image is published as `reinikp2/hybrid-postgres:latest`. 