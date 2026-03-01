# Database Infrastructure

This directory contains the Docker configuration for the hybrid search backend (Postgres + pgvector + pg_search).

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