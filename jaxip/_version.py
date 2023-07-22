# This file includes the package version info
from typing import Tuple


def _version_as_tuple(version_str) -> Tuple[int, ...]:
    return tuple(int(i) for i in version_str.split(".") if i.isdigit())


__version__ = "0.7.3"

# __version_info__: tuple[int, ...] = _version_as_tuple(__version__)
