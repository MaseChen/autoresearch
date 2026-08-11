# ADR-003: Candidate bundles and content addressing

## Status

Accepted

## Context

The V1 candidate is one `kernel.py`.  TileLang and native MACA CUDA require a
stable multi-file representation, while existing source hashes must remain
verifiable.

## Decision

Represent every new candidate as a validated `CandidateBundle`: a normalized
POSIX entrypoint and a bounded list of regular text files.  The host computes a
canonical manifest from raw file bytes and stores blobs in a content-addressed
object store.  Paths may not be absolute, contain `..`, repeat or imply links.

Use tagged artifact IDs:

- `source-sha256-v1:<digest>` for legacy single-source artifacts;
- `bundle-sha256-v1:<digest>` for canonical bundles.

ProposalV1 remains accepted only for the legacy Triton/Fused-MoE namespace and
is normalized immediately to a one-file bundle.

## Trade-offs

- Candidate build flags and capabilities are trusted profile data, never model
  supplied bundle metadata.
- Raw source bytes are not newline- or Unicode-normalized.
