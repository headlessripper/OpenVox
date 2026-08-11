from openvox.tts.config import TTSConfig

def test_defaults():
    c = TTSConfig()
    assert c.device == "cuda"
    assert c.voice == "af_heart"
    assert c.speed == 1.0
    assert c.sample_rate == 24000

def test_streaming_config_defaults():
    c = TTSConfig()
    assert c.segment_max_chars == 160
    assert c.stream_queue_size == 8
