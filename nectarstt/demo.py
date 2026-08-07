import argparse
import sys

from nectarstt import STTEngine, FileSource

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nectarstt-demo",
                                description="Live NectarSTT transcription.")
    p.add_argument("--model", default="distil-large-v3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--language", default="en")
    p.add_argument("--file", default=None, help="Transcribe a WAV instead of the mic.")
    return p

def main(argv: list[str] | None = None) -> int:
    # Gracefully reconfigure stdout to UTF-8 if supported (for Unicode output).
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = build_parser().parse_args(argv)
    engine = STTEngine(model=args.model, device=args.device, language=args.language)
    source = FileSource(args.file) if args.file else None
    print("Listening… (Ctrl+C to stop)" if source is None else f"Transcribing {args.file}…")
    try:
        for event in engine.stream(source=source):
            if event.is_partial:
                print(f"~ {event.text}", end="\r", flush=True)
            else:
                print(f"\r✓ {event.text}")
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
