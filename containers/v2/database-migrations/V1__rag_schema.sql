-- 1. Enable pgai (Vector and Search are already enabled by your base image)
CREATE EXTENSION IF NOT EXISTS plpython3u ;
CREATE EXTENSION IF NOT EXISTS ai CASCADE ;

-- 2. Create Document Chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
id UUID PRIMARY KEY DEFAULT gen_random_uuid (),
content TEXT NOT NULL,
metadata JSONB,
embedding vector (1536), -- Default size for OpenAI text-embedding-3-small
-- Generated column for BM25 (pg_search/ParadeDB)
search_vector tsvector GENERATED ALWAYS AS (to_tsvector ('english',
content)) STORED
) ;

-- 3. Create GIN index for Lexical Search
CREATE INDEX IF NOT EXISTS idx_chunks_search ON document_chunks USING GIN (search_vector) ;

-- 4. Create HNSW index for Semantic Search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops) ;

-- 5. Set up Apache AGE Graph for Knowledge Graph relationships
-- Entities and their relationships will be stored here
SET search_path = public, ag_catalog;
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'knowledge_graph') THEN
        PERFORM create_graph('knowledge_graph');
    END IF;
END $$ ;
