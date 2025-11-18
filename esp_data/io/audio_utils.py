from functools import lru_cache

import librosa
import numpy as np
import torch
import torchaudio


@lru_cache(maxsize=8)
def _get_resampler(
    original_sample_rate: int,
    target_sample_rate: int,
    filter_type: str = "sinc_interp_kaiser",
    lowpass_filter_width: int = 16,
    beta: float | None = None,
    rolloff: float = 0.99,
) -> torchaudio.transforms.Resample:
    return torchaudio.transforms.Resample(
        orig_freq=original_sample_rate,
        new_freq=target_sample_rate,
        lowpass_filter_width=lowpass_filter_width,
        resampling_method=filter_type,
        beta=beta,
        rolloff=rolloff,
    )


def _resample_audio_with_torchaudio(
    waveform: np.ndarray | torch.Tensor,
    original_sample_rate: int,
    target_sample_rate: int,
    filter_type: str = "sinc_interp_kaiser",
    lowpass_filter_width: int = 16,
    beta: float | None = None,
    rolloff: float = 0.99,
    dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """Resample audio waveform to the target sample rate.
    Keeps the device of the input waveform.

    Parameters
    ----------
    waveform : torch.Tensor | np.ndarray
        Input audio waveform tensor of shape (channels, samples)
    original_sample_rate : int
        Original sample rate of the audio waveform
    target_sample_rate : int
        Target sample rate for resampling
    filter_type : str
        Resampling filter type. Options include 'sinc_interp_kaiser',
        'sinc_interp_hann'. 'sinc_interp_kaiser' provides higher quality but is slower.
    lowpass_filter_width : int
        Width of the lowpass filter. Larger values yield better quality but are slower.

    Returns
    -------
    np.ndarray
        Resampled audio waveform tensor
    """
    if original_sample_rate == target_sample_rate:
        return waveform

    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)

    waveform = waveform.to(dtype)

    resampler = _get_resampler(
        original_sample_rate=original_sample_rate,
        target_sample_rate=target_sample_rate,
        filter_type=filter_type,
        lowpass_filter_width=lowpass_filter_width,
        beta=beta,
        rolloff=rolloff,
    ).to(waveform.device)
    resampled_waveform = resampler(waveform).numpy()

    return resampled_waveform


def _resample_with_librosa(
    waveform: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int,
    scale: bool = True,
    filter_type: str = "kaiser_best",
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Resample audio waveform to the target sample rate using librosa.

    Parameters
    ----------
    waveform : np.ndarray
        Input audio waveform array of shape (samples,)
    original_sample_rate : int
        Original sample rate of the audio waveform
    target_sample_rate : int
        Target sample rate for resampling
    filter_type : str
        Resampling filter type. Options include 'kaiser_best', 'kaiser_fast', etc.
        "kaiser_best" provides higher quality but is slower. Try "soxr_hq"
        for a faster alternative with good quality.

    Returns
    -------
    np.ndarray
        Resampled audio waveform array
    """
    if original_sample_rate == target_sample_rate:
        return waveform

    resampled_waveform = librosa.resample(
        waveform.astype(dtype),
        orig_sr=original_sample_rate,
        target_sr=target_sample_rate,
        res_type=filter_type,
        scale=scale,
    )

    return resampled_waveform


def resample_audio(
    audio: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int,
    method: str = "torchaudio",
) -> np.ndarray:
    """Resample audio waveform to the target sample rate.
    Uses default parameters for the selected method.

    Parameters
    ----------
    audio : np.ndarray
        Input audio waveform array of shape (samples,)
    original_sample_rate : int
        Original sample rate of the audio waveform
    target_sample_rate : int
        Target sample rate for resampling
    method : str
        Resampling method to use. Options are 'torchaudio' or 'librosa'.

    Returns
    -------
    np.ndarray
        Resampled audio waveform array

    Raises
    ------
    ValueError
        If an unsupported resampling method is specified.

    Examples
    --------
    >>> import numpy as np
    >>> from esp_data.io import resample_audio
    >>> original_sr = 16000
    >>> t = np.linspace(0, 1, original_sr, endpoint=False)
    >>> audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
    >>> target_sr = 8000
    >>> resampled_data = resample_audio(
    ...     audio_data, original_sr, target_sr, method="torchaudio")
    >>> expected_length = int(len(audio_data) * target_sr / original_sr)
    >>> len(resampled_data) == expected_length
    True
    """
    if method == "torchaudio":
        return _resample_audio_with_torchaudio(
            waveform=audio,
            original_sample_rate=original_sample_rate,
            target_sample_rate=target_sample_rate,
        )
    elif method == "librosa":
        return _resample_with_librosa(
            waveform=audio,
            original_sample_rate=original_sample_rate,
            target_sample_rate=target_sample_rate,
        )
    else:
        raise ValueError(f"Unsupported resampling method: {method}")
