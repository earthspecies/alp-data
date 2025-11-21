# Benchmarks

This README documents how to use the benchmarking utilities in scripts/benchmarks/ and the SLURM job wrappers in jobs/.

Note: most benchmark default functionality is currently bucket‑oriented. The engineering team is migrating to a bucket‑first workflow (GCS), and several utilities, default paths, and upload/plot pipelines assume bucket/GCS semantics by default. However `yaml` config files can be used to work with different configurations and data location such as NFS (see **Configurations** section).

Contents
--------
- `benchmark_config.yaml` — config file listing the configuration of the experiment settings for `nfs` and `bucket` locations.
- `benchmark_dataset.py`, `benchmark_latency.py`, `loading_time.py` — main benchmark scripts to measure loading and latency characteristics.
- `benchmark_utils.py` — shared utilities used by the benchmarks.
- `dataset_list.txt`, `generate_list.py` — helpers for producing dataset lists used in jobs.
- `plot_loading_bench.py`, `plot_latency_bench.py` — plotting helpers that create figures from CSV results.
- `fig/` — default local directory where plots are written.

**Note: create the `fig/` folder to ensure plots are saved.**

Quick overview
--------------
- Benchmarks are run on a SLURM cluster.
- Results are saved as CSV files in a GCS bucket : <gs://esp-ci-cd-tests/esp-data-tests/benchmark_dataset>
    - benchmark_latency.csv
    - benchmark_loading_time.csv
- Plots are saved locally to `scripts/benchmarks/fig/`.

You can specify a dataset configuration via a YAML file or use the default configuration.

Cluster-first workflow (SLURM)
------------------------------
This repository’s benchmark tooling is intended to be run on a cluster using the SLURM-compatible job wrappers in the `esp-data/jobs/` folder. The instructions below focus on submitting and monitoring those job scripts.

Configurations
--------------
A YAML config file should follow the structure in `scripts/benchmarks/benchmark_config.yaml`. The file has two top-level sections: `nfs` and `bucket`. Each maps dataset names to their runtime configuration. Example:

```
nfs:
    dataset:
        dataset_name: beans
        split: validation
        sample_rate: 16000
        data_root: your/own/path

bucket:
    dataset:
        dataset_name: beans
        split: validation
        sample_rate: 16000
        data_root: gs://esp-ml-datasets/beans/v0.1.0/raw/
```

- `nfs` is used for local filesystems (NFS or mounted disks).
- `bucket` is used for GCS or other bucket-style paths.

Edit the file to point `data_root` to your dataset locations. The benchmark scripts choose `nfs` or `bucket` based on the `--data-location` option.

**Note: the `dataset` field under `nfs` or `bucket` is required.**

Running the job scripts with SLURM
---------------------------------
Job scripts are located in `jobs/` and are designed to be submitted with `sbatch` on a SLURM cluster.

Examples:

- With default configuration.
```
# dataloader latency over different parameters for a single dataset
sbatch jobs/benchmark_latency_default.sh --dataset beans

# dataset loading time and sample access for a single dataset
sbatch jobs/benchmark_loading_time_default.sh --dataset beans

# loading time over every available dataset in esp-data
sbatch jobs/multi_dataset_loading_time.sh
```

- With a provided config file.
```
# dataloader latency over different parameters for a single dataset
sbatch jobs/benchmark_latency_config.sh --config your/path/to/config --data-location nfs

# dataset loading time and sample access for a single dataset
sbatch jobs/benchmark_loading_time_config.sh --config your/path/to/config --data-location bucket
```


The value provided to `--dataset` must match the name registered in each DatasetInfo, for example:

```
info = DatasetInfo(
        name="beans",
        ...
)
```

Collecting results and plots
---------------------------
- Plots are written to `scripts/benchmarks/fig/` inside the job’s workspace; results are saved as CSV files to a GCS bucket.
- To copy plots to your local machine via SSH:
```bash
    scp -P 22 username@cluster:esp-data/scripts/benchmarks/fig/* .
```

What you can do
----------------
To adjust or expand experiments, edit the job script (for example `benchmark_latency.sh`) to change the tested parameter ranges. You can modify `--sleep` (simulated work per batch), `--batch-size`, and `--max-iterations` (total workload).
