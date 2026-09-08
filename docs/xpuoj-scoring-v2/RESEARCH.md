# XPU-OJ-Aligned Scoring V2 Research

## Overview

The current C500 protocol ranks candidates by the equal-weight geometric mean
of incumbent-relative case speedups.  That is a useful deployment comparison,
but it is not the anchored arithmetic-mean score used by SOL-ExecBench and
described by XPU-OJ.

## Recommended approach

Keep correctness, absolute scoring baseline, and deployment incumbent as three
separate identities.  Add a continuous XPU-OJ-aligned proxy score for search,
while retaining paired incumbent measurements, per-case regression limits,
independent confirmation, and fencing as promotion safety evidence.

The first version uses a qualified `torch.compile` baseline and `T_h=0`.  It is
therefore named `XPUOJ_TH0_PROXY_V1`; it is not represented as an official OJ
score or a measured hardware-efficiency score.

## Evidence and risks

- Existing winners improved the internal aggregate by roughly 1.7%, but the
  available XPU-OJ observations moved by only 0 to 0.25 displayed points.
- Host wall-clock timing around individual synchronized launches can dominate
  short decode cases; device-event, batched, balanced measurements are needed.
- A compiled-versus-eager pair has a useful qualification role but is a poor
  source for an absolute anchor when eager is substantially slower.  The
  absolute channel therefore self-pairs the compiled reference, completes all
  case anchors first, and keeps the eager comparison in a later proof-only
  phase.
- Keeping all full-case host datasets, device tensors, snapshots, and outputs
  alive across those two phases exceeded the evaluator's 24 GiB memory cgroup
  in the first real run.  Phase ordering is retained, but each deterministic
  case is now regenerated for the proof phase and released before the next
  case.  This is a resource-correctness requirement, not a scoring change.
- The current semantic reference may not compile as one full graph.  Any graph
  break, fallback, incorrect output, or unsupported W8A8 path must leave the
  scoring baseline unqualified rather than silently selecting another anchor.
- The current MACA stack differs from the current public OJ stack.  Results are
  explicitly a versioned current-environment proxy until external calibration.

## References

- NVIDIA SOL-ExecBench scorer and benchmark methodology
- KernelBench `fast_p` and iterative execution feedback
- PyTorch benchmark and `torch.compile` guidance
- Triton `do_bench` warmup/replicate contract
- MetaX XPU-OJ scoring and timing documentation
