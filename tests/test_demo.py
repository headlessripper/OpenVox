from nectarstt import demo

def test_build_parser_defaults():
    args = demo.build_parser().parse_args([])
    assert args.model == "distil-large-v3"
    assert args.device == "cuda"
    assert args.language == "en"
    assert args.file is None

def test_parser_accepts_file():
    args = demo.build_parser().parse_args(["--file", "x.wav", "--device", "cpu"])
    assert args.file == "x.wav"
    assert args.device == "cpu"
