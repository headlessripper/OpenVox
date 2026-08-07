import pytest
from nectarstt import models

def test_resolve_known_alias():
    assert models.resolve_model("distil-large-v3") == "distil-large-v3"
    assert models.resolve_model("large-v3") == "large-v3"

def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        models.resolve_model("gpt-9")

def test_download_root_is_stable_and_contains_app():
    root = models.download_root()
    assert isinstance(root, str)
    assert "nectarstt" in root.lower()
    assert models.download_root() == root  # deterministic
