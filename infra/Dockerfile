FROM python:3.11-slim

ARG PHOENIX_MCP_NPM_VERSION=4.0.13

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PHOENIX_MCP_COMMAND=phoenix-mcp \
    PHOENIX_MCP_ARGS=--apiKey,{PHOENIX_API_KEY}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && npm install -g @arizeai/phoenix-mcp@${PHOENIX_MCP_NPM_VERSION} \
    && phoenix-mcp --help >/dev/null \
    && npm cache clean --force \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
