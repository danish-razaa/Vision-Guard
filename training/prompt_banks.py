"""Disjoint neutral prompt banks for controlled perturbation research."""

TRAIN_PROMPTS = (
    "controlled spectral research pattern alpha",
    "neutral image integrity simulation bravo",
    "bounded frequency analysis sample charlie",
    "defensive visual perturbation study delta",
    "laboratory anomaly encoding echo",
    "synthetic residual measurement foxtrot",
    "controlled texture evaluation golf",
    "image security benchmark hotel",
)

VALIDATION_PROMPTS = (
    "validation-only spectral pattern india",
    "held-out integrity simulation juliet",
    "validation anomaly encoding kilo",
)

TEST_PROMPTS = (
    "unseen test frequency pattern lima",
    "final holdout visual simulation mike",
    "unseen prompt anomaly november",
)


def validate_prompt_banks() -> None:
    """Fail if a prompt appears in more than one experimental partition."""
    banks = [set(TRAIN_PROMPTS), set(VALIDATION_PROMPTS), set(TEST_PROMPTS)]
    if banks[0] & banks[1] or banks[0] & banks[2] or banks[1] & banks[2]:
        raise ValueError("prompt banks must be mutually disjoint")


validate_prompt_banks()
