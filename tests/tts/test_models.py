import os
import pytest
from openvox.tts import models

def test_voices_nonempty_and_known():
    vs = models.voices()
    assert isinstance(vs, list) and len(vs) >= 20
    assert "af_heart" in vs
    assert "am_michael" in vs
    assert vs == sorted(vs)

def test_validate_voice_ok():
    models.validate_voice("af_heart")   # no raise

def test_validate_voice_unknown_raises():
    with pytest.raises(ValueError):
        models.validate_voice("nonexistent_voice")

def test_ensure_assets_builds_paths_and_calls_download(tmp_path, monkeypatch):
    import openvox._paths as paths
    monkeypatch.setattr(paths, "user_cache_dir", lambda app: str(tmp_path))
    calls = []
    monkeypatch.setattr(models, "_download_if_missing",
                        lambda url, dest, **kwargs: calls.append((url, dest, kwargs)))
    model_path, voices_path = models.ensure_assets()
    assert model_path.endswith("kokoro-v1.0.onnx")
    assert voices_path.endswith("voices-v1.0.bin")
    assert os.path.join("tts", "models") in os.path.normpath(model_path)
    assert len(calls) == 2                          # both assets requested

def test_download_if_missing_skips_existing(tmp_path):
    dest = tmp_path / "asset.bin"
    dest.write_bytes(b"already here")
    models._download_if_missing("http://invalid.invalid/x", str(dest))  # no network
    assert dest.read_bytes() == b"already here"

def test_download_rejects_too_small(tmp_path, monkeypatch):
    import urllib.request
    from io import BytesIO

    dest = tmp_path / "asset.bin"

    # Monkeypatch urlopen to return a small response
    class FakeResponse:
        def __init__(self):
            self._read_count = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, size=-1):
            # Return content on first read, empty on subsequent reads (for copyfileobj)
            if self._read_count == 0:
                self._read_count += 1
                return b"tiny"
            return b""

    def fake_urlopen(url, timeout=None):
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    # Should raise RuntimeError because 4 bytes < 1000
    with pytest.raises(RuntimeError, match="too small"):
        models._download_if_missing("http://example.com/asset", str(dest), min_bytes=1000)

    # File should not exist (cleanup happened)
    assert not dest.exists()
