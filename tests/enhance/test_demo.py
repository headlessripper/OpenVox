from openvox.enhance import demo

def test_build_parser_defaults():
    a = demo.build_parser().parse_args(["--in", "a.wav", "--out", "b.wav"])
    assert a.input == "a.wav"
    assert a.output == "b.wav"
    assert a.device == "cuda"
    assert a.denoise_only is False
    assert a.nfe == 64

def test_parser_flags():
    a = demo.build_parser().parse_args(
        ["--in", "a.wav", "--out", "b.wav", "--device", "cpu", "--denoise-only", "--nfe", "32"])
    assert a.device == "cpu" and a.denoise_only is True and a.nfe == 32

def test_exports():
    from openvox.enhance import EnhanceEngine, TTSResult
    assert EnhanceEngine is not None and TTSResult is not None
