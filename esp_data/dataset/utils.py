import numpy as np


def generate_random_indices(
    n: int,
    L: int,
    probs: np.ndarray = None,
    normalize_probs: bool = False,
    with_replacement: bool = False,
    seed: int = None,
) -> np.ndarray:
    """
    Generate random indices from a dataset.

    Args:
        n: Number of indices to generate.
        L: Length of the dataset.
        probs: Probability vector to sample from.
        normalize_probs: Whether to normalize the probability vector.
        with_replacement: Whether to sample with replacement.
        seed: Random seed.

    Returns:
        Randomly sampled indices.
    """
    if seed and isinstance(seed, int) and seed >= 0:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if n > L:
        raise ValueError("n must be less than the length of the dataset.")

    if not probs:
        ids = rng.choice(L, size=n, replace=with_replacement)

    if probs and len(probs) != L:
        raise ValueError("Probability vector length should be the same as dataset length.")

    if probs and probs.sum() < 1.0:
        if normalize_probs:
            probs /= probs.sum()
        else:
            raise ValueError("Probability vector not normalized, set normalize_probs=True to normalize.")
        ids = rng.choice(L, size=n, replace=with_replacement, p=probs)

    return ids
