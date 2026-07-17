# Pluginfer Bootstrap Seed - slim Alpine image, < 50MB target.
#
# Build:   docker build -t pluginfer-seed -f Dockerfile ../..
# Run:     docker run -d --restart=always -p 9000:9000 \
#                    --name pluginfer-seed pluginfer-seed
# Health:  curl localhost:9000/  (raw socket; use the seed_client PING)

FROM python:3.12-alpine AS base

# cryptography needs libffi-dev + openssl-dev to build SECP256K1 wheels;
# we use prebuilt wheels for alpine via piwheels-style index when possible.
RUN apk add --no-cache --virtual .build-deps gcc musl-dev libffi-dev \
        openssl-dev rust cargo \
    && pip install --no-cache-dir cryptography==42.0.* \
    && apk del .build-deps

WORKDIR /app

# Only the files the seed actually needs. The seed_server is intentionally
# self-contained so we don't drag the entire core/ tree into the image.
COPY v2/infrastructure/seed_node/ /app/seed_node/

# A non-root user is good hygiene even on a single-purpose container.
RUN adduser -D -u 1000 pluginfer
USER pluginfer

EXPOSE 9000/tcp 9000/udp

# Healthcheck: a successful TCP connect is enough; the protocol is
# json-line-over-tcp so we can also send a PING to verify behaviour.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); \
        s.settimeout(2); s.connect(('127.0.0.1', 9000)); s.close()" \
        || exit 1

# seed_main runs both TCP (REGISTER / PEERS / PING) and UDP
# (REGISTER_UDP / INTRODUCE / RELAY) on the same port. UDP closes
# the symmetric-NAT survival path (~80% of consumer routers).
CMD ["python", "-m", "seed_node.seed_main", "--host", "0.0.0.0", "--port", "9000"]
