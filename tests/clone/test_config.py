from openvox.clone.config import CloneConfig

def test_defaults():
    c = CloneConfig()
    assert c.device == "cuda"
    assert c.exaggeration == 0.5
    assert c.cfg == 0.5
