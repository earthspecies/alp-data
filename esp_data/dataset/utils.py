from functools import partial
from typing import Callable

import numpy as np

import esp_data.file_io.functional as F
from esp_data.config import DataSample
from esp_data.paths import AnyPath, is_cloud_path, is_local_path


def _make_file_opener(file_path: str | AnyPath, mode: str = "wb") -> Callable:
    """Make a file opener function for WebDataset"""
    file_path = AnyPath(file_path)

    if is_local_path(file_path):
        # Create parent directories if they don't exist
        parent_dir = file_path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        # Return a callable function that opens the file
        return partial(open, mode=mode)

    if is_cloud_path(file_path):
        return partial(F.open_file, mode=mode, use_fs=True)


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


def wrapped_apply(
    function: Callable,
    sample_config=DataSample,
    version_update_mode: str = "patch",
    **function_kwargs,
) -> Callable:
    """Creates a wrapper around a function that transforms each sample in the dataset
      to handle DataSample metadata updates.

      ASSUMPTION: This assumes that your function either implements:
        - A single sample transformation, where the function takes a sample as a dict and returns a transformed dict
        - A batch transformation, where the function takes a dict with keys = sample_config fields, and values = list of
            values for each field, length = batch_size, and returns a dict with the same structure.

    Args:
        function: The user's function to wrap. It should take a sample as a dict and return a transformed dict
        sample_config: The config class to use for samples
        version_update_mode: The mode to use for version updates
        **function_kwargs: Additional keyword arguments to pass to the function
    """

    def update_sample(transformed_sample: dict) -> dict:
        # remove the id and created_at fields, as they will be updated
        sample_id = transformed_sample.pop("id", None)
        transformed_sample.pop("created_at", None)

        transformed_sample = sample_config(
            **transformed_sample,  # generates id and created_at
        )

        transformed_sample.derived_from = sample_id
        if transformed_sample.version is not None:
            transformed_sample.increment_version(mode=version_update_mode)

        return transformed_sample.to_dict()

    def update_batch(transformed_batch: dict) -> dict:
        """When batched=True and batch_size > 1 is part of map_kwargs,
        the function will return a dict with keys = sample_config fields,
        and values = list of values for each field, length = batch_size.
        """
        some_key = list(transformed_batch.keys())[0]
        batch_size = len(transformed_batch[some_key])  # batch size
        new_dict = {k: [] for k in transformed_batch.keys()}

        for i in range(batch_size):
            sample = {k: transformed_batch[k][i] for k in transformed_batch.keys()}
            transformed_sample: dict = update_sample(sample)
            for k, v in transformed_sample.items():
                new_dict[k].append(v)

        return new_dict

    def wrapped_function(sample: dict) -> dict:
        # First apply the user's function to get transformed data
        transformed_sample: dict = function(sample, **function_kwargs)

        # Now update the sample metadata, namely id and created_at
        some_key = list(transformed_sample.keys())[0]
        if isinstance(transformed_sample[some_key], list):
            # If the function returns a batch of samples, apply the update_batch function
            transformed_sample = update_batch(transformed_sample)
        else:
            # If the function returns a single sample, apply the update function
            transformed_sample = update_sample(transformed_sample)

        return transformed_sample

    return wrapped_function
