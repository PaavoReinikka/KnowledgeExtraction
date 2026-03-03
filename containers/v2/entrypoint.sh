#!/bin/bash

# Start Ollama server in the background
ollama serve &

# Wait for Ollama server to be ready
echo "Waiting for Ollama server..."
while ! curl -s http://localhost:11435/api/tags > /dev/null; do
    sleep 1
done

# Pre-fetch models
#models=("llama3.2" "mxbai-embed-large")
# Read models from the environment variable, defaulting to llama3.2
IFS=' ' read -r -a models <<< "${OLLAMA_MODELS:-llama3.2}"

for model in "${models[@]}"; do
    if ! ollama list | grep -q "$model"; then
        echo "Pulling model: $model"
        ollama pull "$model"
    else
        echo "Model $model already exists."
    fi
done

# Keep the script running so the container stays alive
wait
