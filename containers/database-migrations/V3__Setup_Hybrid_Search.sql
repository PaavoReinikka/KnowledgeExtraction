-- V3__Setup_Hybrid_Search.sql

-- The pg_search extension is pre-installed in this image via Dockerfile.custom.
-- It maps to the 'paradedb' schema by default in recent versions.

-- Create a table specifically designed for hybrid search (BM25 + Vector)
CREATE TABLE IF NOT EXISTS knowledge_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding vector(384), -- Dimension for BGE-small or MiniLM
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- CREATE THE BM25 INDEX using ParadeDB's bm25 access method.
-- Important: key_field must also be included in the indexed column list.
CREATE INDEX idx_knowledge_bm25 ON knowledge_items USING bm25 (id, content) WITH (key_field = 'id');

-- Insert sample data for testing
INSERT INTO knowledge_items (content, embedding) VALUES 
('Hybrid search combines BM25 keyword matching with vector similarity for better retrieval.', array_fill(0.1, ARRAY[384])::vector),
('BM25 is a robust ranking function used by search engines to estimate the relevance of documents.', array_fill(0.2, ARRAY[384])::vector),
('Vector embeddings represent text as dense numerical vectors in a high-dimensional space.', array_fill(0.3, ARRAY[384])::vector),
('ParadeDB integrates tantivy for high-performance BM25 search directly inside Postgres.', array_fill(0.4, ARRAY[384])::vector);
