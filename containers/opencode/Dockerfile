# syntax=docker/dockerfile:1
# The base is immutable; Docker selects linux/amd64 from this pinned manifest.
FROM node:22.17.0-bookworm-slim@sha256:b04ce4ae4e95b522112c2e5c52f781471a5cbc3b594527bcddedee9bc48c03a0

RUN npm install --global --no-audit --no-fund opencode-ai@1.17.7 \
    && npm cache clean --force \
    && test "$(opencode --version)" = "1.17.7"

# The pinned Node base image already provides node as UID/GID 1000.
# Fail closed if a future base image changes that invariant.
RUN test "$(id -u node)" = "1000" \
    && test "$(id -g node)" = "1000"

USER 1000:1000
WORKDIR /home/opencode
ENTRYPOINT ["opencode"]