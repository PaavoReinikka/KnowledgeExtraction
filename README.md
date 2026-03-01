# KnowledgeExtraction

A collection of utilities for knowledge extraction, focusing on hybrid RAG (Retrieval-Augmented Generation) systems combining semantic, lexical, and relational/graph retrieval.

This is a work in progress, and more tools and utilities will be added in stages; when I build new tools and llm applications, I will collect all the retrieval related utilities in here.

## Features

- **Chunking Utilities**: Tools for processing and segmenting text for indexing.
- **Extraction Utilities**: Utils for extracting entities, relations, and semantic information.
- **Embedding Wrappers**: Custom wrappers to make various embedding models (Sentence-Transformers, FastEmbed) compatible with LangChain's `Embeddings` interface.

## Project Structure

- `utils/`: Core utility functions.
  - `chunking_utils.py`: Text chunking and preprocessing.
  - `KG_extraction_utils.py`: Knowledge extraction logic.
  - `embeddings_wrappers.py`: Compatibility wrappers for embedding models.
- `test_data/`: Sample data for testing and development.

## License

This project is licensed under the MIT License.
