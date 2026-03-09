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

-- 4. Azure OpenAI / Azure Foundry (OpenAI-compatible) direct SQL embedding example
-- Use LiteLLM-based embedding directly in SQL for retrieval queries.
-- Replace <embedding-deployment-name> and <resource-name> before running.
WITH azure_search AS (
    SELECT v.id, v.chunk
    FROM demo_content_embedding v
    ORDER BY v.embedding <=> ai.litellm_embed(
            'azure/text-embedding-3-large',
            'PostgreSQL history',
            api_key_name => 'AZURE_API_KEY',
            extra_options => '{"api_base":"https://introduction-to-ai-spring26.cognitiveservices.azure.com","api_version":"2023-05-15"}'::jsonb
        )
    LIMIT 5
)
SELECT * FROM azure_search;

-- 5. Optional: Azure endpoint with OpenAI client-compatible call path
-- If your pgai extension version supports `client_config` on openai functions:
--
-- SELECT ai.openai_chat_complete(
--     'gpt-4.1-mini',
--     jsonb_build_array(
--         jsonb_build_object('role', 'system', 'content', 'you are a helpful assistant'),
--         jsonb_build_object('role', 'user', 'content', 'Give a one-line summary of PostgreSQL')
--     ),
--     api_key_name => 'AZURE_API_KEY',
--     client_config => jsonb_build_object(
--         'base_url', 'https://<resource-name>.openai.azure.com/openai/v1'
--     )
-- );
