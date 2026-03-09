-- 01_setup_vectorizer.sql
-- Run this to set up a table and its automatic vectorization.

-- 1. Create a content table (No embedding column needed here)
CREATE TABLE IF NOT EXISTS demo_content (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    category TEXT
);

-- 2. Register the table with a Vectorizer. 
-- The ai-worker will automatically create a secondary table: demo_content_embedding_store
SELECT ai.create_vectorizer(
    'demo_content'::regclass,
    loading => ai.loading_column('body'),
    embedding => ai.embedding_openai('text-embedding-3-small', 1536),
    chunking => ai.chunking_recursive_character_text_splitter('body', chunk_size => 500),
    formatting => ai.formatting_python_template('Title: $title
Category: $category

$body')
);

-- 2b. Azure OpenAI / Azure Foundry (OpenAI-compatible) example
-- Use this instead of the block above when your model is deployed in Azure.
-- Notes:
--  - model must be: 'azure/<deployment-name>'
--  - api_key_name should match your secret/env key name in pgai
--  - api_base is the resource endpoint root (no /openai/deployments/... suffix)
--  - api_version should match your Azure deployment API version
--
-- SELECT ai.create_vectorizer(
--     'demo_content'::regclass,
--     loading => ai.loading_column('body'),
--     embedding => ai.embedding_litellm(
--         'azure/<embedding-deployment-name>',
--         1536,
--         api_key_name => 'AZURE_API_KEY',
--         extra_options => '{"api_base":"https://<resource-name>.openai.azure.com","api_version":"2025-03-01-preview"}'::jsonb
--     ),
--     chunking => ai.chunking_recursive_character_text_splitter('body', chunk_size => 500),
--     formatting => ai.formatting_python_template('Title: $title
-- Category: $category
--
-- $body')
-- );

-- 3. Insert some data
INSERT INTO demo_content (title, body, category) VALUES 
('The History of PostgreSQL', 'PostgreSQL, also known as Postgres, is a free and open-source relational database management system emphasizing extensibility and SQL compliance.', 'Database'),
('Vector Search Explained', 'Vector search allows for searching based on semantic meaning rather than just keywords, which is core to RAG systems.', 'AI'),
('Hybrid Search Benefits', 'Hybrid search combines the best of both worlds: keyword-based BM25 and vector-based semantic search for higher accuracy.', 'AI');

-- 4. Check status (Wait a few seconds for ai-worker to process)
-- SELECT * FROM ai.vectorizer_status;
-- SELECT * FROM demo_content_embedding;
