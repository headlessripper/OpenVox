import argparse
import sys

from openvox.tts import TTSEngine

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openvox-tts-demo",
                                description="Offline OpenVox text-to-speech.")
    p.add_argument("--text", required=True, help="Text to speak.")
    p.add_argument("--voice", default="af_heart")
    p.add_argument("--device", default="cuda")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--out", default=None, help="Also save the audio to this WAV path.")
    p.add_argument("--no-play", action="store_true", help="Do not play the audio aloud.")
    return p

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = TTSEngine(voice=args.voice, device=args.device, speed=args.speed)
    result = engine.synthesize(args.text)
    if args.out:
        result.save_wav(args.out)
        print(f"Saved {args.out} ({result.duration:.1f}s)")
    if not args.no_play:
        engine.play(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())
