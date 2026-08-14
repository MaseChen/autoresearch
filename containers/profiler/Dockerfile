FROM --platform=linux/amd64 registry.cn-shanghai.aliyuncs.com/kcr-3rd/kesci_kernel_lab@sha256:5f1da890360acc5a81438d0e35a80079b34f30d1fa64fc954fef2e9a1ae45b64

ARG SOURCE_REVISION

LABEL org.opencontainers.image.source="https://github.com/masechen/autoresearch" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.title="autoresearch-metax-profiler" \
      org.opencontainers.image.description="Bounded advisory-only MetaX profiler"

COPY --chmod=0555 containers/profiler/bounded-profiler /opt/kernel-research/bin/bounded-profiler
COPY kernel_research /opt/kernel-research/lib/kernel_research

RUN test -n "${SOURCE_REVISION}" \
    && printf '%s' "${SOURCE_REVISION}" | grep -Eq '^[0-9a-f]{40}([0-9a-f]{24})?$' \
    && HOME=/tmp/profile-home TMPDIR=/tmp TRITON_CACHE_DIR=/tmp/triton-cache \
       /opt/kernel-research/bin/bounded-profiler verify-toolchain >/tmp/toolchain.json \
    && test -s /tmp/toolchain.json \
    && rm -f /tmp/toolchain.json

ENV HOME=/tmp/profile-home \
    TMPDIR=/tmp \
    TRITON_CACHE_DIR=/tmp/triton-cache \
    PYTHONPATH=/opt/kernel-research/lib

USER 1000:1000
WORKDIR /output
ENTRYPOINT ["/opt/kernel-research/bin/bounded-profiler"]
