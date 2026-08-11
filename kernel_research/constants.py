"""Dependency-light scientific and controller safety constants."""

from __future__ import annotations


EXPERT_TILE_ROWS = 128
REQUIRED_MATCH_RATIO = 0.99
C500_ALLOWED_NUM_WARPS = frozenset({1, 2, 4, 8, 16})

# Protocol identity and ordered case IDs intentionally live in this
# dependency-light module.  The trusted controller imports evaluation helpers
# but must never import NumPy merely to validate an evaluator envelope.
LEGACY_C500_EVALUATION_PROTOCOL_ID = "fused-moe-c500-v1"
CURRENT_C500_EVALUATION_PROTOCOL_ID = (
    "fused-moe-c500-v2-shadow-holdout"
)
LEGACY_C500_CASE_IDS = {
    "smoke": ("smoke_gate_up", "smoke_down"),
    "quick": (
        "quick_decode_gate_up",
        "quick_prefill_gate_up",
        "quick_decode_down",
        "quick_prefill_down",
    ),
    "full": (
        "full_decode_gate_up",
        "full_prefill_gate_up",
        "full_decode_down",
        "full_prefill_down",
    ),
}
CURRENT_C500_CASE_IDS = {
    **LEGACY_C500_CASE_IDS,
    "quick": (
        *LEGACY_C500_CASE_IDS["quick"],
        "quick_shadow_tiles_127_n2",
        "quick_shadow_tiles_128_n1",
        "quick_shadow_tiles_128_n2",
        "quick_shadow_tiles_129_n2",
    ),
}
CURRENT_C500_HOLDOUT_CASE_IDS = (
    "quick_shadow_tiles_128_n1",
    "quick_shadow_tiles_129_n2",
)

# These are trusted-controller ceilings. User configuration may lower them but
# may never raise them.
MAX_AUTORESEARCH_CANDIDATES = 5
MAX_AUTORESEARCH_HOURS = 6.0
MAX_CONSECUTIVE_FAILURES = 3
OPENCODE_PROPOSER_STEPS = 3
MAX_PROPOSER_ATTEMPTS = 2
MAX_FEEDBACK_CANDIDATES = 8
MAX_FEEDBACK_ERROR_CHARS = 2000
MAX_PROPOSER_TIMEOUT_SEC = 1200.0
MAX_PROPOSER_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_EVALUATOR_TIMEOUT_SEC = 2700.0
MAX_PROPOSER_CPUS = 2.0
MAX_EVALUATOR_CPUS = 8.0
PROPOSER_MEMORY_LIMIT = "2g"
EVALUATOR_MEMORY_LIMIT = "24g"
PROPOSER_PID_LIMIT = 256
EVALUATOR_PID_LIMIT = 768
DOCTOR_OUTPUT_LIMIT_BYTES = 2 * 1024 * 1024
EVALUATOR_OUTPUT_LIMIT_BYTES = 4 * 1024 * 1024
DOCTOR_TIMEOUT_SEC = 240.0

MAX_CANDIDATE_SOURCE_BYTES = 256 * 1024
POLICY_CPU_LIMIT_SEC = 2
POLICY_WALL_TIMEOUT_SEC = 5.0
POLICY_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
POLICY_OUTPUT_LIMIT_BYTES = 64 * 1024
POLICY_MAX_ERRORS = 100

PROPOSAL_HYPOTHESIS_HARD_LIMIT = 1000
PROPOSAL_HYPOTHESIS_RETRY_TARGET = 600
PROPOSAL_RATIONALE_HARD_LIMIT = 8000
PROPOSAL_RATIONALE_RETRY_TARGET = 6000

COMMAND_READ_CHUNK_BYTES = 64 * 1024
