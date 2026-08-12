"""OpenVox voice-assistant demo CLI with pluggable LLM support."""
import argparse
import sys

from openvox.agent import VoiceAgent


def _echo_llm(user_text, history):
    return f"You said: {user_text}. I am OpenVox, running entirely on this machine."


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="openvox-agent-demo",
        description="Offline voice assistant: mic -> STT -> your LLM -> streaming TTS."
    )
    p.add_argument("--llm", choices=["echo", "ollama", "openai"], default="echo")
    p.add_argument("--model", default="llama3.2")
    p.add_argument("--base-url", dest="base_url", default="http://localhost:8080/v1")
    p.add_argument("--voice", default="af_heart", help="A built-in voice name or an .ovx profile path.")
    p.add_argument("--stt-model", dest="stt_model", default="base")
    p.add_argument("--no-barge-in", dest="barge_in", action="store_false")
    p.set_defaults(barge_in=True)
    return p


def _make_llm(args):
    if args.llm == "ollama":
        from openvox.agent.llm import ollama
        return ollama(model=args.model)
    if args.llm == "openai":
        from openvox.agent.llm import openai_compatible
        return openai_compatible(base_url=args.base_url, model=args.model)
    return _echo_llm


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from openvox.stt import STTEngine
    from openvox.tts import TTSEngine

    stt = STTEngine(model=args.stt_model)
    tts = TTSEngine(voice=args.voice)
    agent = VoiceAgent(llm=_make_llm(args), stt=stt, tts=tts,
                       voice=args.voice, barge_in=args.barge_in)
    print("OpenVox assistant ready. Speak; Ctrl-C to quit.")
    agent.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
