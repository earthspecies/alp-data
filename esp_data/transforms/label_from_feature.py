import logging
from typing import Any, Literal

from pydantic import BaseModel

from esp_data.backends.protocol import DataBackend

from . import register_transform

logger = logging.Logger("esp_data")


class LabelFromFeatureConfig(BaseModel):
    type: Literal["label_from_feature"]
    feature: str
    label_map: dict[Any, int] | None = None
    output_feature: str = "label"
    override: bool = False


class LabelFromFeature:
    """Transform to create a label feature from an existing feature in a DataFrame.

    This transform maps the values of a specified feature to integer labels.

    Works with any backend (pandas, polars) through the DataBackend protocol.

    Parameters
    ----------
    feature: str
        The name of the feature in the DataFrame from which to create labels.
    label_map: dict[Any, int] | None
        A mapping of feature values to integer labels. If None, the labels will be
        created from the unique values in the feature.
    output_feature: str
        The name of the new feature to store the labels. Defaults to "label".
    override: bool
        If True, will override the output feature if it already exists in the DataFrame.
        If False, will raise an AssertionError if the output feature already exists.

    Examples
    -------
    >>> from esp_data.backends import PandasBackend
    >>> import pandas as pd
    >>> df = pd.DataFrame({"species": ["cat", "dog", "bird", "cat"]})
    >>> backend = PandasBackend(df)
    >>> transform = LabelFromFeature(feature="species", output_feature="label")
    >>> transformed_backend, metadata = transform(backend)
    """

    def __init__(
        self,
        *,
        feature: str,
        label_map: dict[Any, int] | None = None,
        output_feature: str = "label",
        override: bool = False,
    ) -> None:
        self.feature = feature
        self.label_map = label_map
        self.override = override
        self.output_feature = output_feature

    @classmethod
    def from_config(cls, cfg: LabelFromFeatureConfig) -> "LabelFromFeature":
        return cls(**cfg.model_dump(exclude=("type")))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        """Apply the transformation to the backend.

        Parameters
        ----------
        backend : DataBackend
            The backend wrapping the DataFrame to transform.

        Returns
        -------
        tuple[DataBackend, dict]
            A tuple containing the transformed backend and metadata about the labels.

        Raises
        -------
        AssertionError
            If the output feature already exists and override is False.
        """
        if self.output_feature in backend.columns and not self.override:
            raise AssertionError(
                "Feature already exists in DataFrame. Set `override=True` to replace it."
            )

        # Drop rows with null values in the feature column
        backend_clean = backend.dropna(subset=[self.feature])

        # Count dropped rows for logging
        # Note: In streaming mode (LazyFrame), this will trigger evaluation
        try:
            original_len = len(backend)
            clean_len = len(backend_clean)
            if clean_len != original_len:
                logger.warning(f"Dropped {original_len - clean_len} rows with {self.feature}=NaN")
        except Exception as e:
            logger.warning(f"Could not compute dropped rows: {e}")
            pass

        # Get unique values and create label map if not provided
        if self.label_map is None:
            uniques = backend_clean.get_unique(self.feature)
            label_map = {lbl: idx for idx, lbl in enumerate(uniques)}
        else:
            label_map = self.label_map

        # Map the feature to labels
        backend_with_labels = backend_clean.map_column(
            column=self.feature,
            mapping=label_map,
            output_column=self.output_feature,
        )

        metadata = {
            "label_feature": self.feature,
            "label_map": label_map,
            "num_classes": len(label_map),
        }

        return backend_with_labels, metadata


register_transform(LabelFromFeatureConfig, LabelFromFeature)
