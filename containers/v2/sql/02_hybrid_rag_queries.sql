-- 02_hybrid_rag_queries.sql
-- Examples of how to perform Semantic, Lexical, and Hybrid search.

-- 1. Semantic Search (Vector only)
-- Search for a concept like "AI databases"
WITH query_embedding AS (
    SELECT ai.openai_embed('text-embedding-3-small', 'AI and vector databases') as emb
)
SELECT 
    v.chunk, 
    1 - (v.embedding <=> (SELECT emb FROM query_embedding)) as similarity
FROM demo_content_embedding v
ORDER BY similarity DESC
LIMIT 5;

-- 2. Lexical Search (Keyword only)
-- Note: 'pg_search' should be used for ParadeDB functionality, 
-- or standard tsvector/GIN index for native Postgres.
SELECT title, body
FROM demo_content
WHERE to_tsvector('english', body) @@ to_tsquery('vector & search')
LIMIT 5;

-- 3. Hybrid Search (Simplified RRF concept)
-- We combine scores from both vector and keyword searches
WITH semantic_search AS (
    SELECT 
        v.id, 
        v.chunk, 
        RANK() OVER (ORDER BY v.embedding <=> ai.openai_embed('text-embedding-3-small', 'Postgres AI tools')) as rank
    FROM demo_content_embedding v
    LIMIT 20
),
lexical_search AS (
    SELECT 
        id, 
        body, 
        RANK() OVER (ORDER BY ts_rank_cd(to_tsvector('english', body), to_tsquery('Postgres | AI')) DESC) as rank
    FROM demo_content
    WHERE to_tsvector('english', body) @@ to_tsquery('Postgres | AI')
    LIMIT 20
)
-- Reciprocal Rank Fusion (RRF) combines them: 1 / (60 + rank)
SELECT 
    COALESCE(s.chunk, l.body) as document,
    (1.0 / (60 + COALESCE(s.rank, 100))) + (1.0 / (60 + COALESCE(l.rank, 100))) as rrf_score
FROM semantic_search s
FULL OUTER JOIN lexical_search l ON s.id = l.id
ORDER BY rrf_score DESC
LIMIT 5;
