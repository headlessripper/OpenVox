from openvox.enroll.demo import build_parser


def test_enroll_parser_defaults():
    args = build_parser().parse_args(["--in", "a.wav", "b.wav", "--out", "v.ovx"])
    assert args.input == ["a.wav", "b.wav"]
    assert args.output == "v.ovx"
    assert args.quality == "balanced"
    assert args.device == "cuda"
    assert args.enhance is True


def test_enroll_parser_no_enhance_and_quality():
    args = build_parser().parse_args(
        ["--in", "a.wav", "--out", "v.ovx", "--quality", "thorough", "--no-enhance"])
    assert args.quality == "thorough" and args.enhance is False
