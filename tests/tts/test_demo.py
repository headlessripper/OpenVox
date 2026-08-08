from openvox.tts import demo

def test_build_parser_defaults():
    args = demo.build_parser().parse_args(["--text", "hi"])
    assert args.text == "hi"
    assert args.voice == "af_heart"
    assert args.device == "cuda"
    assert args.speed == 1.0
    assert args.out is None
    assert args.no_play is False

def test_parser_flags():
    args = demo.build_parser().parse_args(
        ["--text", "hi", "--voice", "am_michael", "--device", "cpu",
         "--speed", "1.2", "--out", "o.wav", "--no-play"])
    assert args.voice == "am_michael" and args.device == "cpu"
    assert args.speed == 1.2 and args.out == "o.wav" and args.no_play is True

def test_exports():
    from openvox.tts import TTSEngine, TTSResult
    assert TTSEngine is not None and TTSResult is not None
