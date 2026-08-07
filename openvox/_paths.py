import os

from platformdirs import user_cache_dir

def cache_dir(sub: str = "") -> str:
    """Return a stable per-user cache directory under the OpenVox root.

    ``cache_dir("stt/models")`` -> ``<user cache>/openvox/stt/models``.
    The directory (including parents) is created if missing.
    """
    root = user_cache_dir("openvox")
    path = os.path.join(root, sub) if sub else root
    os.makedirs(path, exist_ok=True)
    return path
