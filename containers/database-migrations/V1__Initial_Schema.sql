-- V1__Initial_Schema.sql
-- create all extensions if not already created (for safety, since we also have them in Dockerfile)
CREATE EXTENSION IF NOT EXISTS vector CASCADE;
CREATE EXTENSION IF NOT EXISTS age CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_search CASCADE;

-- Create the initial documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(384), -- Example for MiniLM
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Note: In ParadeDB, indexes are often created via SQL commands like:
-- CALL pg_search.create_bm25(index_name => 'idx_docs_bm25', table_name => 'documents', columns => '{content}');
