# Pluginfer Devserver image.
#
# Build:
#   docker build -t pluginfer/devserver:latest -f v2/api/Dockerfile v2
# Run:
#   docker run --rm -p 11434:11434 \
#     -e PLUGINFER_DEV_PORT=11434 \
#     -e OPENAI_API_KEY=$OPENAI_API_KEY \
#     -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
#     pluginfer/devserver:latest
#
# Then any existing OpenAI/Anthropic-SDK app talks to the mesh by:
#   OPENAI_BASE_URL=http://localhost:11434/v1
#   ANTHROPIC_BASE_URL=http://localhost:11434

FROM python:3.12-slim AS base

# Drop privileges as soon as deps are installed. The shim only needs
# outbound HTTPS (to upstream LLM APIs) and the listener port — there
# is no reason it ever runs as root.
RUN groupadd --system pluginfer && useradd --system --gid pluginfer --create-home pluginfer

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PLUGINFER_DEV_PORT=11434

WORKDIR /app

# Two-step copy so dependency installs cache cleanly across rebuilds:
# only the requirements file invalidates the layer below.
COPY requirements-devserver.txt /tmp/req.txt
RUN pip install -r /tmp/req.txt

COPY api/ /app/api/
COPY core/ /app/core/
COPY ai/ /app/ai/

# Generate an empty Filum-Lite registry so the devserver doesn't
# panic on first launch when no providers are configured — it bids
# against zero providers and returns a structured 502, which is the
# correct behaviour for the operator to see.
RUN install -d -o pluginfer -g pluginfer /app/state \
 && chown -R pluginfer:pluginfer /app

USER pluginfer

EXPOSE 11434

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
  CMD python -c "import os,sys,urllib.request as u; p=os.environ.get('PLUGINFER_DEV_PORT','11434'); sys.exit(0 if u.urlopen(f'http://127.0.0.1:{p}/healthz', timeout=2).status==200 else 1)" \
  || exit 1

CMD ["python", "-m", "api.devserver", "--host", "0.0.0.0", "--port", "11434"]
