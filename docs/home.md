# alp-data

### What is alp-data?

`alp-data` (<a href="https://earthspecies.org/2026/04/02/animal-language-processing-an-ai-convergence-in-animal-communication/" target="_blank" rel="noopener">Animal Language Processing</a> data) is a <a href="https://pypi.org/project/alp-data/" target="_blank" rel="noopener">Python package</a> that provides access to 35+ bioacoustic datasets behind one unified interface. Designed to support machine learning across species, the current datasets cover birds, marine mammals, primates, insects, anurans, and multi-taxon benchmarks.

### Why alp-data?

Today, bioacoustic datasets are distributed across many repositories — Zenodo, OSF, GBIF, and institutional archives — and each arrives with its own conventions for format, structure, sampling rate, and licensing. As a result, researchers working across multiple datasets must typically invest considerable effort in preprocessing: writing a custom loader for each dataset, then a separate step to combine them, before any of the data can be used together.

`alp-data` handles this overhead, letting you combine datasets and work through a single consistent interface from the start:

- **Unified dataset interface**: access datasets stored locally, on cloud storage, or in various formats (CSV, JSON, Parquet) through a consistent API, with iteration and random-access indexing.
- **Streaming support**: work with large datasets that don't fit into memory by streaming samples as you go.
- **Transformations**: apply transforms such as row/column filtering and label creation per sample.
- **Combinations**: concatenate or chain multiple datasets into a single unified dataset for training or evaluation.
