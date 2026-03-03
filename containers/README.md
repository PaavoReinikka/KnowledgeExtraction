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