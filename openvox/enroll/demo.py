import argparse
import sys

from openvox.enroll import VoiceEnrollEngine


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="openvox-enroll-demo",
        description="Enroll a voice from clips into a reusable .ovx profile.")
    p.add_argument("--in", dest="input", nargs="+", required=True,
                   help="One or more clips (or a long recording) of the target voice.")
    p.add_argument("--out", dest="output", required=True, help="Output .ovx profile path.")
    p.add_argument("--quality", default="balanced",
                   choices=["fast", "balanced", "thorough"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-enhance", dest="enhance", action="store_false",
                   help="Skip reference-clip enhancement.")
    p.set_defaults(enhance=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = VoiceEnrollEngine(device=args.device, quality=args.quality)
    engine._config.enhance_clips = args.enhance
    profile = engine.enroll(args.input)
    profile.save(args.output)
    print(f"Saved {args.output} (stage {profile.metadata['stage']}, "
          f"speaker-similarity {profile.score:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
