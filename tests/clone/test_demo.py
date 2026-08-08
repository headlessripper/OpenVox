from openvox.clone import demo

def test_build_parser_defaults():
    args = demo.build_parser().parse_args(["--text", "hi", "--ref", "r.wav"])
    assert args.text == "hi"
    assert args.ref == "r.wav"
    assert args.exaggeration == 0.5
    assert args.cfg == 0.5
    assert args.device == "cuda"
    assert args.out is None
    assert args.no_play is False

def test_parser_flags():
    args = demo.build_parser().parse_args(
        ["--text", "hi", "--ref", "r.wav", "--exaggeration", "0.7",
         "--cfg", "0.3", "--device", "cpu", "--out", "o.wav", "--no-play"])
    assert args.exaggeration == 0.7 and args.cfg == 0.3
    assert args.device == "cpu" and args.out == "o.wav" and args.no_play is True

def test_exports():
    from openvox.clone import VoiceCloneEngine, TTSResult
    assert VoiceCloneEngine is not None and TTSResult is not None
