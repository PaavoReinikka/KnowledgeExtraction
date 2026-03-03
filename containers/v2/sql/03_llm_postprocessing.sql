-- 03_llm_postprocessing.sql
-- Use LLMs for summarization and reranking within the database.

-- 1. Summarization
-- Take a document and generate a 1-sentence summary
SELECT 
    title, 
    ai.openai_chat_complete_simple(
        'Summarize this into one short sentence: ' || body, 
        model => 'gpt-4o-mini'
    ) as summary
FROM demo_content
LIMIT 3;

-- 2. Reranking (LLM as a Judge)
-- Get the top 10 results from a fast search, 
-- then ask an LLM to pick the most relevant one.
WITH initial_search AS (
    SELECT v.id, v.chunk
    FROM demo_content_embedding v
    ORDER BY v.embedding <=> ai.openai_embed('text-embedding-3-small', 'PostgreSQL history')
    LIMIT 5
)
SELECT 
    id, 
    chunk,
    ai.openai_chat_complete_simple(
        'On a scale of 0 to 100, how relevant is this text to the query "PostgreSQL history"? 
        Respond ONLY with a number: ' || chunk
    )::numeric as relevance_score
FROM initial_search
ORDER BY relevance_score DESC;

-- 3. Complete RAG Chain (Prompt + Context)
-- Retrieve relevant chunks and generate an answer
WITH context AS (
    SELECT string_agg(chunk, '
---
') as doc_bundle
    FROM (
        SELECT v.chunk
        FROM demo_content_embedding v
        ORDER BY v.embedding <=> ai.openai_embed('text-embedding-3-small', 'What is Postgres?')
        LIMIT 3
    ) as subs
)
SELECT ai.openai_chat_complete_simple(
    'Answer the question: "What is Postgres?". Use only the provided context.
    Context: ' || doc_bundle,
    model => 'gpt-4o'
) as final_answer
FROM context;
