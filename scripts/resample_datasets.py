"""Resample audio files from esp_data datasets with configurable resampling backends.

This script provides multiple resampling methods:
1. librosa: High-quality resampling with various filter types (default: kaiser_best)
2. torchaudio: PyTorch-based resampling with efficient tensor operations
3. scipy: SciPy signal processing based resampling

All resampler arguments are exposed as CLI arguments for full control.
Resampled audio is exported to a GCS bucket with sample rate prefixes and
the new paths are added to the metadata.

Dependencies
------------
- librosa backend: Available by default (librosa is a core dependency)
- scipy backend: Available by default (scipy is a core dependency)
- torchaudio backend: Requires torchaudio, which is NOT included in the default
  dependencies. Use `uv run --with torchaudio` to temporarily add it.

Example Usage
-------------
Using librosa (default backend, no extra dependencies needed):

    uv run python scripts/resample_datasets.py --dataset birdset --split train \\
        --export-path gs://esp-ml-datasets/birdset/v0.1.0/audio \\
        --target-sr 16000

    # With custom librosa res_type
    uv run python scripts/resample_datasets.py --dataset beans --split train \\
        --export-path gs://esp-ml-datasets/beans/v0.1.0/audio \\
        --target-sr 16000 --librosa-res-type soxr_hq

Using scipy (no extra dependencies needed):

    uv run python scripts/resample_datasets.py --dataset beans --split train \\
        --export-path gs://esp-ml-datasets/beans/v0.1.0/audio \\
        --target-sr 16000 --backend scipy --scipy-domain poly --scipy-window hann

Using torchaudio (requires --with torchaudio flag):

    uv run --with torchaudio python scripts/resample_datasets.py \\
        --dataset inaturalist --split train \\
        --export-path gs://esp-ml-datasets/inaturalist/v0.1.0/audio \\
        --target-sr 32000 --backend torchaudio --torchaudio-rolloff 0.99

    # With custom torchaudio parameters
    uv run --with torchaudio python scripts/resample_datasets.py \\
        --dataset birdset --split train \\
        --export-path gs://esp-ml-datasets/birdset/v0.1.0/audio \\
        --target-sr 16000 --backend torchaudio \\
        --torchaudio-resampling-method sinc_interp_kaiser \\
        --torchaudio-lowpass-filter-width 6 --torchaudio-beta 14.77

List available datasets:

    uv run python scripts/resample_datasets.py --list-datasets

Output Structure
----------------
Given --export-path gs://esp-ml-datasets/dataset_name/v0.1.0/audio and --target-sr 16000,
the script will:
1. Create audio files at: gs://esp-ml-datasets/dataset_name/v0.1.0/audio/16KHz/<relative_path>.wav
2. Add a new column '16khz_path' to the metadata with the relative paths
3. Export updated metadata to: gs://esp-ml-datasets/dataset_name/v0.1.0/audio/dataset_name_split_metadata.csv
"""

import argparse
import threading
import time
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from tqdm import tqdm

from esp_data import dataset_class_from_name, list_registered_datasets
from esp_data.io import anypath, exists, filesystem_from_path


class ResampleBackend(str, Enum):
    """Available resampling backends."""

    LIBROSA = "librosa"
    TORCHAUDIO = "torchaudio"
    SCIPY = "scipy"


@dataclass
class LibrosaResamplerConfig:
    """Configuration for librosa resampler.

    Attributes
    ----------
    res_type : str
        Resampling filter type. Options include:
        - 'kaiser_best': High-quality Kaiser window (default)
        - 'kaiser_fast': Faster Kaiser window with lower quality
        - 'fft': FFT-based resampling
        - 'polyphase': Polyphase filtering
        - 'linear': Linear interpolation
        - 'zero_order_hold': Zero-order hold
        - 'sinc_best': High-quality sinc interpolation
        - 'sinc_medium': Medium-quality sinc interpolation
        - 'sinc_fastest': Fastest sinc interpolation
        - 'soxr_vhq': Very high quality (requires soxr)
        - 'soxr_hq': High quality (requires soxr)
        - 'soxr_mq': Medium quality (requires soxr)
        - 'soxr_lq': Low quality (requires soxr)
        - 'soxr_qq': Quick quality (requires soxr)
    scale : bool
        If True, scale the output to preserve RMS energy. Default True.
    fix : bool
        If True, fix the length of the output signal. Default True.
    """

    res_type: str = "kaiser_best"
    scale: bool = True
    fix: bool = True


@dataclass
class TorchaudioResamplerConfig:
    """Configuration for torchaudio resampler.

    Attributes
    ----------
    lowpass_filter_width : int
        Controls the width of the lowpass filter. A larger width gives a
        sharper transition band but is more computationally expensive.
        Default is 6.
    rolloff : float
        The roll-off frequency of the filter as a fraction of the Nyquist
        frequency. Lower values reduce aliasing but may attenuate high
        frequencies. Default is 0.99.
    resampling_method : str
        The resampling method to use. Options are:
        - 'sinc_interp_hann': Sinc interpolation with Hann window
        - 'sinc_interp_kaiser': Sinc interpolation with Kaiser window (default)
    beta : float | None
        The beta parameter for the Kaiser window. Only used when
        resampling_method is 'sinc_interp_kaiser'. Default is 14.769656459379492.
    dtype : str | None
        The dtype of the internal computation. If None, uses the input dtype.
        Options: 'float32', 'float64', None.
    """

    # Default values match librosa's kaiser_best settings
    # see https://docs.pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html#kaiser-best
    lowpass_filter_width: int = 64
    rolloff: float = 0.9475937167399596
    resampling_method: str = "sinc_interp_kaiser"
    beta: float | None = 14.769656459379492
    dtype: str | None = None


@dataclass
class ScipyResamplerConfig:
    """Configuration for scipy resampler.

    Attributes
    ----------
    window : str | tuple | array_like
        The window to use for the FIR filter. Default is 'hann'.
        Options include: 'hann', 'hamming', 'blackman', 'bartlett',
        'boxcar', 'triang', 'parzen', etc.
    domain : str
        The domain in which to resample. Options are:
        - 'time': Resample in time domain using scipy.signal.resample
        - 'poly': Resample using polyphase filtering (scipy.signal.resample_poly)
        Default is 'time'.
    padtype : str
        Only used when domain='poly'. The type of padding to use.
        Options: 'constant', 'line', 'mean', 'median', 'minimum',
        'maximum', 'reflect', 'symmetric', 'wrap', 'empty'.
        Default is 'constant'.
    """

    window: str = "hann"
    domain: str = "time"
    padtype: str = "constant"


def create_librosa_resampler(config: LibrosaResamplerConfig) -> Callable:
    """Create a librosa-based resampler function.

    Parameters
    ----------
    config : LibrosaResamplerConfig
        Configuration for the librosa resampler.

    Returns
    -------
    Callable
        A function that takes (audio, orig_sr, target_sr) and returns resampled audio.
    """
    import librosa

    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        return librosa.resample(
            y=audio,
            orig_sr=orig_sr,
            target_sr=target_sr,
            res_type=config.res_type,
            scale=config.scale,
            fix=config.fix,
        )

    return resample


def create_torchaudio_resampler(config: TorchaudioResamplerConfig) -> Callable:
    """Create a torchaudio-based resampler function.

    Parameters
    ----------
    config : TorchaudioResamplerConfig
        Configuration for the torchaudio resampler.

    Returns
    -------
    Callable
        A function that takes (audio, orig_sr, target_sr) and returns resampled audio.
    """
    import torch
    import torchaudio.transforms as T

    # Cache for resampler instances (keyed by (orig_sr, target_sr))
    resampler_cache: dict[tuple[int, int], T.Resample] = {}
    cache_lock = threading.Lock()

    def get_resampler(orig_sr: int, target_sr: int) -> T.Resample:
        key = (orig_sr, target_sr)
        with cache_lock:
            if key not in resampler_cache:
                kwargs = {
                    "orig_freq": orig_sr,
                    "new_freq": target_sr,
                    "lowpass_filter_width": config.lowpass_filter_width,
                    "rolloff": config.rolloff,
                    "resampling_method": config.resampling_method,
                }
                if config.beta is not None:
                    kwargs["beta"] = config.beta
                if config.dtype is not None:
                    dtype_map = {"float32": torch.float32, "float64": torch.float64}
                    kwargs["dtype"] = dtype_map.get(config.dtype)
                resampler_cache[key] = T.Resample(**kwargs)
        return resampler_cache[key]

    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        resampler = get_resampler(orig_sr, target_sr)
        # Convert to torch tensor, resample, convert back
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        resampled = resampler(audio_tensor)
        return resampled.squeeze(0).numpy()

    return resample


def create_scipy_resampler(config: ScipyResamplerConfig) -> Callable:
    """Create a scipy-based resampler function.

    Parameters
    ----------
    config : ScipyResamplerConfig
        Configuration for the scipy resampler.

    Returns
    -------
    Callable
        A function that takes (audio, orig_sr, target_sr) and returns resampled audio.
    """
    from math import gcd

    from scipy import signal

    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if config.domain == "poly":
            # Use polyphase filtering - more efficient for integer ratios
            g = gcd(orig_sr, target_sr)
            up = target_sr // g
            down = orig_sr // g
            return signal.resample_poly(
                audio, up, down, window=config.window, padtype=config.padtype
            )
        else:
            # Use FFT-based resampling
            num_samples = int(len(audio) * target_sr / orig_sr)
            return signal.resample(audio, num_samples, window=config.window)

    return resample


def get_resampler(
    backend: ResampleBackend,
    librosa_config: LibrosaResamplerConfig | None = None,
    torchaudio_config: TorchaudioResamplerConfig | None = None,
    scipy_config: ScipyResamplerConfig | None = None,
) -> Callable:
    """Get a resampler function based on the specified backend.

    Parameters
    ----------
    backend : ResampleBackend
        The resampling backend to use.
    librosa_config : LibrosaResamplerConfig | None
        Configuration for librosa backend. Uses defaults if None.
    torchaudio_config : TorchaudioResamplerConfig | None
        Configuration for torchaudio backend. Uses defaults if None.
    scipy_config : ScipyResamplerConfig | None
        Configuration for scipy backend. Uses defaults if None.

    Returns
    -------
    Callable
        A function that takes (audio, orig_sr, target_sr) and returns resampled audio.

    Raises
    ------
    ValueError
        If an unknown backend is specified.
    """
    if backend == ResampleBackend.LIBROSA:
        config = librosa_config or LibrosaResamplerConfig()
        return create_librosa_resampler(config)
    elif backend == ResampleBackend.TORCHAUDIO:
        config = torchaudio_config or TorchaudioResamplerConfig()
        return create_torchaudio_resampler(config)
    elif backend == ResampleBackend.SCIPY:
        config = scipy_config or ScipyResamplerConfig()
        return create_scipy_resampler(config)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def write_wav_to_gcs(
    audio: np.ndarray,
    sample_rate: int,
    path: str,
    fs: object,
) -> None:
    """Write audio data as uncompressed WAV to a path (local or GCS).

    Parameters
    ----------
    audio : np.ndarray
        The audio data to write.
    sample_rate : int
        The sample rate of the audio.
    path : str
        The full path to write to (including filename).
    fs : Any
        The filesystem object (from filesystem_from_path).
    """
    buffer = BytesIO()
    sf.write(buffer, audio, samplerate=sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    with fs.open(path, "wb") as f:
        f.write(buffer.getvalue())


def get_sample_rate_prefix(target_sr: int) -> str:
    """Get the folder prefix for a given sample rate.

    Parameters
    ----------
    target_sr : int
        Target sample rate in Hz.

    Returns
    -------
    str
        Folder prefix like '16KHz' or '32KHz'.
    """
    if target_sr % 1000 != 0:
        # Use Hz for non-standard rates
        return f"{target_sr}Hz"
    return f"{target_sr // 1000}KHz"


def process_dataset(
    dataset_name: str,
    split: str,
    export_path: str,
    target_sr: int,
    resample_fn: Callable,
    data_root: str | None = None,
    max_workers: int = 4,
    path_column: str | None = None,
    output_metadata_path: str | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Process a dataset split with the specified resampler.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (lowercase, as in DatasetInfo.name).
    split : str
        Dataset split to process.
    export_path : str
        GCS bucket path for exporting resampled audio.
        E.g., 'gs://esp-ml-datasets/birdset/v0.1.0/audio'
    target_sr : int
        Target sample rate.
    resample_fn : Callable
        The resampling function to use.
    data_root : str | None
        Optional data root override.
    max_workers : int
        Number of parallel workers.
    path_column : str | None
        Name of the column containing the audio path. If None, auto-detected.
    output_metadata_path : str | None
        Path to write the updated metadata CSV/JSONL. If None, derived from export_path.
    skip_existing : bool
        If True, skip files that already exist at the destination.

    Returns
    -------
    dict[str, Any]
        Processing statistics.

    Raises
    ------
    KeyError
        If the dataset is unknown or path column cannot be detected.
    ValueError
        If the path column is not found in the dataset.
    """
    from esp_data.io import audio_stereo_to_mono, read_audio

    total_start_time = time.time()

    # Get the dataset class using the registry
    try:
        dataset_cls = dataset_class_from_name(dataset_name)
    except KeyError as err:
        available = list_registered_datasets()
        raise KeyError(
            f"Unknown dataset: '{dataset_name}'. Available datasets: {available}"
        ) from err

    # Initialize dataset without sample_rate to get original audio
    print(f"Loading dataset '{dataset_name}', split='{split}'...")
    kwargs: dict[str, Any] = {"split": split}
    if data_root:
        kwargs["data_root"] = data_root

    dataset = dataset_cls(**kwargs)

    total_files = len(dataset)
    print(f"Processing {total_files} files from {dataset_name}/{split}")

    # Get filesystem for the export path
    fs = filesystem_from_path(export_path)

    # Determine the sample rate prefix (e.g., "16KHz")
    sr_prefix = get_sample_rate_prefix(target_sr)
    export_audio_dir = f"{export_path}/{sr_prefix}"

    # Auto-detect the path column if not provided
    if path_column is None:
        # Common path column names
        candidates = ["local_path", "path", "audio_path", "file_path", "originals_path"]
        for col in candidates:
            if col in dataset._data.columns:
                path_column = col
                break
        if path_column is None:
            raise ValueError(
                f"Could not auto-detect path column. Available columns: {dataset._data.columns}"
            )
    print(f"Using path column: '{path_column}'")

    # Prepare the new path column name
    new_path_column = f"{sr_prefix.lower()}_path"
    print(f"New paths will be stored in column: '{new_path_column}'")

    # Process files
    new_paths: list[str] = []
    successful = 0
    skipped = 0
    errors = 0
    error_details: list[dict[str, Any]] = []

    # Get the data root for reading audio
    read_data_root = anypath(data_root) if data_root else anypath(dataset.data_root)

    for idx in tqdm(range(total_files), desc=f"Resampling {dataset_name}/{split}"):
        try:
            # Get the row data directly from _data to avoid loading audio
            row = dataset._data[idx]

            # Get the original audio path
            orig_path = row.get(path_column)
            if orig_path is None:
                raise ValueError(f"No path found in column '{path_column}' for idx {idx}")

            # Construct the source audio path
            audio_source_path = read_data_root / orig_path

            # Construct the destination path
            # Keep the relative path structure but under the new sr folder
            relative_path = Path(orig_path)
            # Change extension to .wav
            dest_filename = relative_path.stem + ".wav"
            dest_relative = str(relative_path.parent / dest_filename)
            dest_full_path = f"{export_audio_dir}/{dest_relative}"

            # Store the relative path (without the base export path)
            # This is what gets stored in the metadata column
            new_relative_path = f"{sr_prefix}/{dest_relative}"

            # Check if file already exists
            if skip_existing and exists(anypath(dest_full_path)):
                new_paths.append(new_relative_path)
                skipped += 1
                continue

            # Read audio at original sample rate
            audio, orig_sr = read_audio(audio_source_path)
            audio = audio.astype(np.float32)
            audio = audio_stereo_to_mono(audio, mono_method="average")

            # Resample if needed
            if orig_sr != target_sr:
                audio = resample_fn(audio, orig_sr, target_sr)

            # Write the resampled audio
            write_wav_to_gcs(audio, target_sr, dest_full_path, fs)

            new_paths.append(new_relative_path)
            successful += 1

        except Exception as e:
            new_paths.append("")  # Empty path for failed files
            errors += 1
            error_details.append({"idx": idx, "error": str(e)})
            if errors <= 5:
                print(f"Error processing idx {idx}: {e}")

    # Add the new column to the dataset metadata
    print(f"\nAdding column '{new_path_column}' to metadata...")
    dataset._data = dataset._data.add_column(new_path_column, new_paths)

    # Export the updated metadata
    if output_metadata_path is None:
        # Default: export alongside the audio in the export path
        metadata_filename = f"{dataset_name}_{split}_metadata.csv"
        output_metadata_path = f"{export_path}/{metadata_filename}"

    print(f"Exporting metadata to: {output_metadata_path}")

    # Export as CSV using the backend's to_csv method
    dataset._data.to_csv(output_metadata_path)

    total_time = time.time() - total_start_time

    print(f"\nCompleted processing {dataset_name}/{split}:")
    print(f"  Successful: {successful}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Files per second: {total_files / max(total_time, 0.001):.2f}")

    return {
        "dataset": dataset_name,
        "split": split,
        "total_files": total_files,
        "successful": successful,
        "skipped": skipped,
        "errors": errors,
        "error_details": error_details,
        "total_time": total_time,
        "new_path_column": new_path_column,
        "metadata_path": output_metadata_path,
    }


def main() -> None:
    """Main entry point for the resampling script."""
    parser = argparse.ArgumentParser(
        description="Resample audio files from esp_data datasets with configurable backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using librosa (default) with kaiser_best filter
  python resample_datasets.py --dataset birdset --split train \\
      --export-path gs://esp-ml-datasets/birdset/v0.1.0/audio --target-sr 16000

  # Using torchaudio with custom parameters
  python resample_datasets.py --dataset inaturalist --split train \\
      --export-path gs://esp-ml-datasets/inaturalist/v0.1.0/audio --target-sr 32000 \\
      --backend torchaudio --torchaudio-rolloff 0.99

  # Using scipy with polyphase filtering
  python resample_datasets.py --dataset beans --split train \\
      --export-path gs://esp-ml-datasets/beans/v0.1.0/audio --target-sr 16000 \\
      --backend scipy --scipy-domain poly --scipy-window hann

  # List available datasets
  python resample_datasets.py --list-datasets
""",
    )

    # Utility options
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available registered datasets and exit",
    )

    # Required arguments
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset (lowercase, as in DatasetInfo.name)",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Dataset split to process (e.g., train, val, test)",
    )
    parser.add_argument(
        "--export-path",
        type=str,
        help="GCS bucket path for exporting resampled audio "
        "(e.g., gs://esp-ml-datasets/birdset/v0.1.0/audio)",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        help="Target sample rate in Hz",
    )

    # Backend selection
    parser.add_argument(
        "--backend",
        type=str,
        choices=["librosa", "torchaudio", "scipy"],
        default="librosa",
        help="Resampling backend to use (default: librosa)",
    )

    # General options
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override the dataset's default data root",
    )
    parser.add_argument(
        "--path-column",
        type=str,
        default=None,
        help="Name of the column containing audio paths (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output-metadata",
        type=str,
        default=None,
        help="Path to write the updated metadata file (default: derived from export-path)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-process files even if they already exist at the destination",
    )

    # Librosa-specific arguments
    librosa_group = parser.add_argument_group("librosa options")
    librosa_group.add_argument(
        "--librosa-res-type",
        type=str,
        default="kaiser_best",
        help="Librosa resampling filter type (default: kaiser_best). "
        "Options: kaiser_best, kaiser_fast, fft, polyphase, linear, "
        "zero_order_hold, sinc_best, sinc_medium, sinc_fastest, "
        "soxr_vhq, soxr_hq, soxr_mq, soxr_lq, soxr_qq",
    )
    librosa_group.add_argument(
        "--librosa-scale",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Scale output to preserve RMS energy (default: true)",
    )
    librosa_group.add_argument(
        "--librosa-fix",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Fix the length of the output signal (default: true)",
    )

    # Torchaudio-specific arguments
    torchaudio_group = parser.add_argument_group("torchaudio options")
    torchaudio_group.add_argument(
        "--torchaudio-lowpass-filter-width",
        type=int,
        default=6,
        help="Lowpass filter width (default: 6). Larger values give sharper "
        "transition bands but are more expensive.",
    )
    torchaudio_group.add_argument(
        "--torchaudio-rolloff",
        type=float,
        default=0.99,
        help="Roll-off frequency as fraction of Nyquist (default: 0.99). "
        "Lower values reduce aliasing but may attenuate high frequencies.",
    )
    torchaudio_group.add_argument(
        "--torchaudio-resampling-method",
        type=str,
        default="sinc_interp_kaiser",
        choices=["sinc_interp_hann", "sinc_interp_kaiser"],
        help="Torchaudio resampling method (default: sinc_interp_kaiser)",
    )
    torchaudio_group.add_argument(
        "--torchaudio-beta",
        type=float,
        default=14.769656459379492,
        help="Kaiser window beta parameter (default: 14.769656459379492). "
        "Only used with sinc_interp_kaiser.",
    )
    torchaudio_group.add_argument(
        "--torchaudio-dtype",
        type=str,
        default=None,
        choices=["float32", "float64"],
        help="Internal computation dtype (default: None, uses input dtype)",
    )

    # Scipy-specific arguments
    scipy_group = parser.add_argument_group("scipy options")
    scipy_group.add_argument(
        "--scipy-window",
        type=str,
        default="hann",
        help="Window function for FIR filter (default: hann). "
        "Options: hann, hamming, blackman, bartlett, boxcar, triang, parzen, etc.",
    )
    scipy_group.add_argument(
        "--scipy-domain",
        type=str,
        default="time",
        choices=["time", "poly"],
        help="Resampling domain (default: time). 'time' uses FFT-based resampling, "
        "'poly' uses polyphase filtering which is more efficient for integer ratios.",
    )
    scipy_group.add_argument(
        "--scipy-padtype",
        type=str,
        default="constant",
        help="Padding type for polyphase filtering (default: constant). "
        "Options: constant, line, mean, median, minimum, maximum, reflect, "
        "symmetric, wrap, empty.",
    )

    args = parser.parse_args()

    # Handle --list-datasets
    if args.list_datasets:
        print("Available datasets:")
        for name in sorted(list_registered_datasets()):
            print(f"  {name}")
        return

    # Validate required arguments
    if not args.dataset:
        parser.error("--dataset is required")
    if not args.split:
        parser.error("--split is required")
    if not args.export_path:
        parser.error("--export-path is required")
    if not args.target_sr:
        parser.error("--target-sr is required")

    # Build configuration based on backend
    backend = ResampleBackend(args.backend)

    librosa_config = None
    torchaudio_config = None
    scipy_config = None

    if backend == ResampleBackend.LIBROSA:
        librosa_config = LibrosaResamplerConfig(
            res_type=args.librosa_res_type,
            scale=args.librosa_scale,
            fix=args.librosa_fix,
        )
        print(f"Using librosa backend with res_type={librosa_config.res_type}")

    elif backend == ResampleBackend.TORCHAUDIO:
        torchaudio_config = TorchaudioResamplerConfig(
            lowpass_filter_width=args.torchaudio_lowpass_filter_width,
            rolloff=args.torchaudio_rolloff,
            resampling_method=args.torchaudio_resampling_method,
            beta=args.torchaudio_beta,
            dtype=args.torchaudio_dtype,
        )
        print(
            f"Using torchaudio backend with method={torchaudio_config.resampling_method}, "
            f"rolloff={torchaudio_config.rolloff}"
        )

    elif backend == ResampleBackend.SCIPY:
        scipy_config = ScipyResamplerConfig(
            window=args.scipy_window,
            domain=args.scipy_domain,
            padtype=args.scipy_padtype,
        )
        print(
            f"Using scipy backend with domain={scipy_config.domain}, window={scipy_config.window}"
        )

    # Create resampler
    resample_fn = get_resampler(
        backend=backend,
        librosa_config=librosa_config,
        torchaudio_config=torchaudio_config,
        scipy_config=scipy_config,
    )

    # Process dataset
    result = process_dataset(
        dataset_name=args.dataset,
        split=args.split,
        export_path=args.export_path,
        target_sr=args.target_sr,
        resample_fn=resample_fn,
        data_root=args.data_root,
        max_workers=args.workers,
        path_column=args.path_column,
        output_metadata_path=args.output_metadata,
        skip_existing=not args.no_skip_existing,
    )

    print(f"\nMetadata exported to: {result['metadata_path']}")
    print(f"New path column added: {result['new_path_column']}")


if __name__ == "__main__":
    main()
