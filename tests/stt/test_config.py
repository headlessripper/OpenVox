from openvox.stt.config import Config

def test_defaults():
    c = Config()
    assert c.device == "cuda"
    assert c.model == "distil-large-v3"
    assert c.sample_rate == 16000

def test_toml_overrides_defaults(tmp_path):
    f = tmp_path / "openvox.stt.toml"
    f.write_text('model = "small"\ndevice = "cpu"\n', encoding="utf-8")
    c = Config.load(path=str(f))
    assert c.model == "small"
    assert c.device == "cpu"
    assert c.language == "en"  # untouched default

def test_env_overrides_toml(tmp_path):
    f = tmp_path / "openvox.stt.toml"
    f.write_text('model = "small"\n', encoding="utf-8")
    c = Config.load(path=str(f), env={"OPENVOX_STT_MODEL": "tiny", "OPENVOX_STT_WINDOW_INTERVAL_MS": "250"})
    assert c.model == "tiny"
    assert c.window_interval_ms == 250  # coerced to int
