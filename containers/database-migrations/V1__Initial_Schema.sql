-- V1__Initial_Schema.sql
-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS pg_search;
CREATE EXTENSION IF NOT EXISTS vector;

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
