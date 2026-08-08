from openvox.tts.config import TTSConfig

def test_defaults():
    c = TTSConfig()
    assert c.device == "cuda"
    assert c.voice == "af_heart"
    assert c.speed == 1.0
    assert c.sample_rate == 24000
