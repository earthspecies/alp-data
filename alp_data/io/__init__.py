from alp_data.io.file_utils import exists, rm
from alp_data.io.filesystem import filesystem, filesystem_from_path
from alp_data.io.paths import (
    DATA_HOME,
    AnyPathT,
    PureGSPath,
    PureHTTPPath,
    PureHTTPSPath,
    PureR2Path,
    PureS3Path,
    anypath,
)
from alp_data.io.read_utils import (
    audio_stereo_to_mono,
    get_audio_info,
    read_audio,
    read_json,
    read_text,
    read_video,
    read_video_at_frames,
    read_yaml,
)

__all__ = [
    "anypath",
    "AnyPathT",
    "DATA_HOME",
    "PureGSPath",
    "PureHTTPPath",
    "PureHTTPSPath",
    "PureR2Path",
    "PureS3Path",
    "read_audio",
    "read_text",
    "audio_stereo_to_mono",
    "get_audio_info",
    "read_video",
    "read_video_at_frames",
    "get_video_info",  # noqa: F822 (lazy, see `__getattr__`)
    "filesystem",
    "filesystem_from_path",
    "exists",
    "rm",
    "read_json",
    "read_yaml",
]


def __getattr__(name: str) -> object:
    """Imports `get_video_info` on first access, so `torch` loads only for video.

    Parameters
    ----------
    name : str
        Attribute name looked up on `alp_data.io`.

    Returns
    -------
    object
        The requested attribute.

    Raises
    ------
    AttributeError
        If `name` is not a lazily exported attribute.
    """
    if name == "get_video_info":
        from alp_data.io.video_utils import get_video_info

        return get_video_info
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
