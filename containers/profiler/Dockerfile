FROM --platform=linux/amd64 registry.cn-shanghai.aliyuncs.com/kcr-3rd/kesci_kernel_lab@sha256:5f1da890360acc5a81438d0e35a80079b34f30d1fa64fc954fef2e9a1ae45b64

ARG SOURCE_REVISION

LABEL org.opencontainers.image.source="https://github.com/masechen/autoresearch" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.title="autoresearch-metax-profiler" \
      org.opencontainers.image.description="Bounded advisory-only MetaX profiler"

COPY containers/profiler/bounded-profiler /opt/kernel-research/bin/bounded-profiler
COPY containers/profiler/normalize-library-tree /opt/kernel-research/bin/normalize-library-tree
COPY kernel_research /opt/kernel-research/lib/kernel_research

RUN chmod 0555 /opt/kernel-research/bin/bounded-profiler \
    && chmod 0555 /opt/kernel-research/bin/normalize-library-tree \
    && test "$(stat -c '%a %u:%g' /opt/kernel-research/bin/bounded-profiler)" = "555 0:0" \
    && test "$(stat -c '%a %u:%g' /opt/kernel-research/bin/normalize-library-tree)" = "555 0:0" \
    && normalizer_sha="$(sha256sum \
        /opt/kernel-research/bin/normalize-library-tree | cut -d ' ' -f 1)" \
    && test "${normalizer_sha}" = \
        "a64c60dc263489144af2a288b14c03d393b2eb5eb082e73bf54deb82de3e366f" \
    && python /opt/kernel-research/bin/normalize-library-tree \
    && rm -f /opt/kernel-research/bin/normalize-library-tree \
    && test -n "${SOURCE_REVISION}" \
    && printf '%s' "${SOURCE_REVISION}" | grep -Eq '^[0-9a-f]{40}([0-9a-f]{24})?$'

ENV HOME=/tmp/profile-home \
    TMPDIR=/tmp \
    TRITON_CACHE_DIR=/tmp/triton-cache \
    PYTHONPATH=/opt/kernel-research/lib

USER 1000:1000

RUN test "$(id -u):$(id -g)" = "1000:1000" \
    && mkdir -p "${HOME}" "${TRITON_CACHE_DIR}" \
    && chmod 0700 "${HOME}" "${TRITON_CACHE_DIR}" \
    && test "$(stat -c '%a %u:%g' "${HOME}")" = "700 1000:1000" \
    && cd "${HOME}" \
    && python -c 'import kernel_research; import kernel_research.profiler_contract' \
    && /opt/kernel-research/bin/bounded-profiler verify-toolchain >/tmp/toolchain.json \
    && test -s /tmp/toolchain.json \
    && rm -f /tmp/toolchain.json

WORKDIR /output
ENTRYPOINT ["/opt/kernel-research/bin/bounded-profiler"]
