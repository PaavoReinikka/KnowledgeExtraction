FROM reinikp2/hybrid-postgres:next

USER root

# 1. Install necessary system dependencies and PL/Python extension
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-plpython3-16 \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies for pgai
# --break-system-packages is required for Debian Bookworm when not using a venv
RUN pip3 install --no-cache-dir --break-system-packages \
    requests openai anthropic cohere backoff tiktoken voyageai datasets litellm jinja2 boto3 langchain-text-splitters

# 3. Manually install pgai extension files from source
# We copy the extension files, the PL/Python 'ai' package, and the vectorizer SQL
RUN git clone --depth 1 https://github.com/timescale/pgai.git /tmp/pgai \
    && cp /tmp/pgai/projects/extension/sql/output/*.sql /usr/share/postgresql/16/extension/ \
    && cp /tmp/pgai/projects/extension/sql/output/ai.control /usr/share/postgresql/16/extension/ \
    && sed -i "s/default_version='0.11.3-dev'/default_version='0.11.2'/g" /usr/share/postgresql/16/extension/ai.control \
    && cp -r /tmp/pgai/projects/extension/ai /usr/local/lib/python3.11/dist-packages/ \
    && mkdir -p /docker-entrypoint-initdb.d \
    && echo "CREATE EXTENSION IF NOT EXISTS ai CASCADE;" > /docker-entrypoint-initdb.d/01-init-ai.sql \
    && echo "BEGIN;" > /docker-entrypoint-initdb.d/02-vectorizer.sql \
    && sed "s/__version__/0.12.1/g" /tmp/pgai/projects/pgai/pgai/data/ai.sql | sed "s/@extschema:vector@/public/g" >> /docker-entrypoint-initdb.d/02-vectorizer.sql \
    && echo "COMMIT;" >> /docker-entrypoint-initdb.d/02-vectorizer.sql \
    && rm -rf /tmp/pgai

USER postgres
