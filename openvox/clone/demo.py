import argparse
import sys

from openvox.clone import VoiceCloneEngine

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openvox-clone-demo",
                                description="Offline zero-shot voice cloning.")
    p.add_argument("--text", required=True, help="Text to speak in the cloned voice.")
    p.add_argument("--ref", required=True, help="Reference audio clip of the voice to clone.")
    p.add_argument("--exaggeration", type=float, default=0.5)
    p.add_argument("--cfg", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="Also save the audio to this WAV path.")
    p.add_argument("--no-play", action="store_true", help="Do not play the audio aloud.")
    p.add_argument("--no-enhance", dest="enhance", action="store_false",
                   help="Do not auto-enhance the reference clip before cloning.")
    p.set_defaults(enhance=True)
    return p

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = VoiceCloneEngine(device=args.device, exaggeration=args.exaggeration, cfg=args.cfg)
    result = engine.clone(args.text, args.ref, enhance=args.enhance)
    if args.out:
        result.save_wav(args.out)
        print(f"Saved {args.out} ({result.duration:.1f}s)")
    if not args.no_play:
        engine.play(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())
