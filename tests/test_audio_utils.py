import numpy as np
from esp_data.io import resample_audio


def test_resample_audio_with_torchaudio() -> None:
    # Create a dummy audio signal (1 second of a 440 Hz sine wave at 16 kHz)
    original_sr = 16000
    t = np.linspace(0, 1, original_sr, endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Resample to 8 kHz
    target_sr = 8000
    resampled_data = resample_audio(
        audio_data, original_sr, target_sr, method="torchaudio"
    )

    # Check the length of the resampled data
    expected_length = int(len(audio_data) * target_sr / original_sr)
    assert len(resampled_data) == expected_length

    # Resample back to 16 kHz
    restored_data = resample_audio(resampled_data, target_sr, original_sr)

    # Check that the restored data is approximately equal to the original
    # WARNING! This is a lossy operation, so we use a higher tolerance
    np.testing.assert_allclose(restored_data, audio_data, atol=1e-1)


def test_resample_audio_with_librosa() -> None:
    # Create a dummy audio signal (1 second of a 440 Hz sine wave at 16 kHz)
    original_sr = 16000
    t = np.linspace(0, 1, original_sr, endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Resample to 8 kHz
    target_sr = 8000
    resampled_data = resample_audio(
        audio_data, original_sr, target_sr, method="librosa"
    )

    # Check the length of the resampled data
    expected_length = int(len(audio_data) * target_sr / original_sr)
    assert len(resampled_data) == expected_length

    # Resample back to 16 kHz
    restored_data = resample_audio(
        resampled_data, target_sr, original_sr, method="librosa"
    )

    # Check that the restored data is approximately equal to the original
    # WARNING! This is a lossy operation, so we use a higher tolerance
    np.testing.assert_allclose(restored_data, audio_data, atol=1e-1)
