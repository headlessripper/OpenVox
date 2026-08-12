from openvox.agent.demo import build_parser, _echo_llm


def test_parser_defaults():
    a = build_parser().parse_args([])
    assert a.llm == "echo" and a.stt_model == "base" and a.barge_in is True


def test_parser_flags():
    a = build_parser().parse_args(["--llm", "ollama", "--model", "llama3.2", "--voice", "alice.ovx", "--no-barge-in"])
    assert a.llm == "ollama" and a.model == "llama3.2" and a.voice == "alice.ovx" and a.barge_in is False


def test_echo_llm_responds():
    assert isinstance(_echo_llm("hello", []), str)
    assert "hello" in _echo_llm("hello", [])
