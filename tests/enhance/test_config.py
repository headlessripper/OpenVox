from openvox.enhance.config import EnhanceConfig

def test_defaults():
    c = EnhanceConfig()
    assert c.device == "cuda"
    assert c.nfe == 64
    assert c.solver == "midpoint"
    assert c.lambd == 0.9
    assert c.tau == 0.5
    assert c.denoise_only is False
