"""Plot `benchmark_packed.py` results: packed against live, per dataset.

Reads one or more CSVs written by `scripts/benchmarks/benchmark_packed.py` and
writes three figures plus a markdown summary into `--out-dir`:

- `throughput.png`: samples per second against DataLoader workers, one panel per
  dataset, packed and live as separate lines, prefetch factor as line style.
- `speedup.png`: packed throughput divided by live throughput for the same
  configuration, so a value above one means the pack was faster.
- `startup_memory.png`: time to first batch and per-worker peak memory against
  workers, which is where the cost of spawning workers shows up.

Rows whose timed window never started (an epoch shorter than the warm-up) are
dropped with a note; sequential rows are reported in the summary only.

Examples
--------
uv run --group benchmark python scripts/benchmarks/plot_packed_bench.py \\
    --csv ~/outputs/benchmarks/benchmark_packed_92824.csv --out-dir docs/img/benchmark_packed
"""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SIDE_COLORS = {"packed": "#1b6ca8", "live": "#c2571a"}
PREFETCH_STYLES = {0: "-", 2: "-", 8: "--"}
PREFETCH_COLORS = {0: "#6d6d6d", 2: "#1b6ca8", 8: "#7fb3d5"}


def load(csvs: tuple[Path, ...], min_seconds: float) -> pd.DataFrame:
    """Concatenate benchmark CSVs and drop rows that were not really timed.

    Parameters
    ----------
    csvs : tuple[Path, ...]
        CSV files written by `benchmark_packed.py`.
    min_seconds : float
        Dataloader rows whose timed window is shorter than this are dropped:
        the epoch ended before the window had settled.

    Returns
    -------
    pd.DataFrame
        All rows that have a trustworthy throughput figure.
    """
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    short = (df["mode"] == "dataloader") & (df["measured_s"].fillna(0) < min_seconds)
    for (ds, w, pf), grp in df[short].groupby(["dataset", "workers", "prefetch"]):
        secs = grp["measured_s"].fillna(0).max()
        click.echo(
            f"note: dropping {ds} workers={w} prefetch={pf} ({', '.join(grp['side'])}): "
            f"timed window only {secs:.0f} s, the epoch ended during or just after the warm-up"
        )
    return df[~short & df["samples_per_s"].notna()].copy()


def _panels(datasets: list[str]) -> tuple[plt.Figure, list[plt.Axes]]:
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    return fig, list(axes[0])


def plot_throughput(df: pd.DataFrame, out: Path) -> None:
    """Samples per second against workers, per dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataloader rows.
    out : Path
        PNG to write.
    """
    datasets = list(dict.fromkeys(df["dataset"]))
    fig, axes = _panels(datasets)
    for ax, ds in zip(axes, datasets, strict=True):
        sub = df[df["dataset"] == ds]
        for (side, pf), grp in sub.groupby(["side", "prefetch"]):
            grp = grp.sort_values("workers")
            label = side if pf == 0 else f"{side}, prefetch {pf}"
            ax.plot(
                grp["workers"],
                grp["samples_per_s"],
                PREFETCH_STYLES.get(int(pf), ":"),
                marker="o",
                color=SIDE_COLORS[side],
                label=label,
            )
        ax.set_title(ds)
        ax.set_xlabel("DataLoader workers")
        ax.set_xscale("symlog", linthresh=4)
        ax.set_yscale("log")
        ax.set_xticks(sorted(sub["workers"].unique()))
        ax.set_xticklabels([str(w) for w in sorted(sub["workers"].unique())])
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("samples / s")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Throughput: packed vs live, batch of 32, 60 s timed window after warm-up")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def speedup_table(df: pd.DataFrame) -> pd.DataFrame:
    """Packed throughput divided by live throughput per configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Dataloader and sequential rows.

    Returns
    -------
    pd.DataFrame
        One row per (dataset, mode, workers, prefetch) with both sides and the ratio.
    """
    keys = ["dataset", "mode", "workers", "prefetch"]
    wide = df.pivot_table(index=keys, columns="side", values="samples_per_s").reset_index()
    wide = wide.dropna(subset=["packed", "live"])
    wide["speedup"] = wide["packed"] / wide["live"]
    return wide


def plot_speedup(wide: pd.DataFrame, out: Path, dataset_order: list[str]) -> None:
    """Bar chart of packed/live speed-up per dataset and worker count.

    Parameters
    ----------
    wide : pd.DataFrame
        Output of `speedup_table`, dataloader rows.
    out : Path
        PNG to write.
    dataset_order : list[str]
        Panel order, matching the throughput figure.
    """
    datasets = [d for d in dataset_order if d in set(wide["dataset"])]
    fig, axes = _panels(datasets)
    offsets = {0: 0.0, 2: -0.2, 8: 0.2}
    labels_seen: dict[str, object] = {}
    for ax, ds in zip(axes, datasets, strict=True):
        sub = wide[wide["dataset"] == ds]
        worker_levels = sorted(sub["workers"].unique())
        position = {w: i for i, w in enumerate(worker_levels)}
        for pf, grp in sub.groupby("prefetch"):
            label = f"prefetch {pf}" if pf else "in-process"
            x = [position[w] + offsets.get(int(pf), 0.0) for w in grp["workers"]]
            bars = ax.bar(x, grp["speedup"], width=0.4, label=label, color=PREFETCH_COLORS[int(pf)])
            labels_seen.setdefault(label, bars)
        ax.set_xticks(range(len(worker_levels)))
        ax.set_xticklabels([str(w) for w in worker_levels])
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_title(ds)
        ax.set_xlabel("DataLoader workers")
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both", alpha=0.3)
    axes[0].set_ylabel("packed / live throughput")
    fig.legend(labels_seen.values(), labels_seen.keys(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Speed-up of the pack over the live dataset (above 1 = pack faster)")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_startup_memory(df: pd.DataFrame, out: Path) -> None:
    """Time to first batch and per-worker peak memory against workers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataloader rows.
    out : Path
        PNG to write.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    for (ds, side), grp in df[df["prefetch"] != 8].groupby(["dataset", "side"]):
        grp = grp.sort_values("workers")
        ax1.plot(
            grp["workers"], grp["first_batch_s"], marker="o", color=SIDE_COLORS[side], alpha=0.7
        )
        ax2.plot(
            grp["workers"],
            grp["worker_max_peak_mb"],
            marker="o",
            color=SIDE_COLORS[side],
            alpha=0.7,
            label=f"{ds} {side}",
        )
    ax1.set_title("Time to first batch")
    ax1.set_ylabel("seconds")
    ax2.set_title("Peak RSS of the busiest worker")
    ax2.set_ylabel("MB")
    for ax in (ax1, ax2):
        ax.set_xlabel("DataLoader workers")
        ax.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, frameon=False)
    fig.suptitle("Start-up cost grows with workers on both sides (one line per dataset)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summary_markdown(df: pd.DataFrame, wide: pd.DataFrame) -> str:
    """Markdown tables: the headline per dataset and the full grid.

    Parameters
    ----------
    df : pd.DataFrame
        All rows with a throughput figure.
    wide : pd.DataFrame
        Output of `speedup_table`.

    Returns
    -------
    str
        Markdown text.
    """
    lines = [
        "| dataset | rows | sequential packed / live (ms) | best packed | best live "
        "| speed-up at best |",
        "|---|---|---|---|---|---|",
    ]
    for ds in dict.fromkeys(df["dataset"]):
        sub = df[df["dataset"] == ds]
        rows = int(sub["num_rows"].iloc[0])
        seq = sub[sub["mode"] == "sequential"].set_index("side")["batch_p50_s"] * 1000
        dl = wide[(wide["dataset"] == ds) & (wide["mode"] == "dataloader")]
        best = dl.sort_values("packed").iloc[-1] if len(dl) else None
        seq_txt = f"{seq.get('packed', float('nan')):.0f} / {seq.get('live', float('nan')):.0f}"
        if best is None:
            lines.append(f"| {ds} | {rows:,} | {seq_txt} | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {ds} | {rows:,} | {seq_txt} | {best['packed']:.0f}/s "
            f"({int(best['workers'])}w, pf {int(best['prefetch'])}) "
            f"| {best['live']:.0f}/s | {best['speedup']:.1f}x |"
        )
    lines += [
        "",
        "| dataset | workers | prefetch | packed /s | live /s | speed-up |",
        "|---|---|---|---|---|---|",
    ]
    grid = wide[wide["mode"] == "dataloader"].sort_values(["dataset", "workers", "prefetch"])
    for _, r in grid.iterrows():
        lines.append(
            f"| {r['dataset']} | {int(r['workers'])} | {int(r['prefetch'])} "
            f"| {r['packed']:.0f} | {r['live']:.0f} | {r['speedup']:.2f} |"
        )
    return "\n".join(lines) + "\n"


@click.command()
@click.option(
    "--csv", "csvs", type=click.Path(exists=True, path_type=Path), multiple=True, required=True
)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--min-seconds", default=30.0, show_default=True, help="Drop shorter timed windows")
def main(csvs: tuple[Path, ...], out_dir: Path, min_seconds: float) -> None:
    """Plot packed-vs-live benchmark CSVs into `--out-dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load(csvs, min_seconds)
    dl = df[df["mode"] == "dataloader"]
    wide = speedup_table(df)
    plot_throughput(dl, out_dir / "throughput.png")
    order = list(dict.fromkeys(dl["dataset"]))
    plot_speedup(wide[wide["mode"] == "dataloader"], out_dir / "speedup.png", order)
    plot_startup_memory(dl, out_dir / "startup_memory.png")
    summary = summary_markdown(df, wide)
    (out_dir / "summary.md").write_text(summary)
    click.echo(summary)
    click.echo(f"Wrote throughput.png, speedup.png, startup_memory.png, summary.md to {out_dir}")


if __name__ == "__main__":
    main()
