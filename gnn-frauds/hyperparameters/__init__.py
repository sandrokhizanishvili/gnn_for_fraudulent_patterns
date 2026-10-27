"""Hyperparameter values. Each model folder matches conf/hyperparameters/."""


def check_fixed(fixed, known):
    """Reject a key the model does not have."""
    extra = set(fixed) - set(known)
    assert not extra, f'unknown hyperparameter keys {sorted(extra)}; known keys are {sorted(known)}'
