import argparse
import sys

from openvox.enhance import EnhanceEngine

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openvox-enhance-demo",
                                description="Offline speech denoise + restoration.")
    p.add_argument("--in", dest="input", required=True, help="Input audio file.")
    p.add_argument("--out", dest="output", required=True, help="Output WAV path.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoise-only", action="store_true",
                   help="Denoise without full restoration/bandwidth extension.")
    p.add_argument("--nfe", type=int, default=64)
    return p

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = EnhanceEngine(device=args.device, denoise_only=args.denoise_only)
    engine._config.nfe = args.nfe
    result = engine.enhance_file(args.input)
    result.save_wav(args.output)
    print(f"Saved {args.output} ({result.duration:.1f}s @ {result.sample_rate} Hz)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
