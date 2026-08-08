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
                        lambda url, dest: calls.append((url, dest)))
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
