import argparse
import sys
import wave

import numpy as np

from nectarstt import STTEngine, ArraySource, MicSource


def load_audio_16k_mono(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load any audio file as a mono float32 array at ``sample_rate`` Hz.

    An already-conforming (mono, 16-bit, ``sample_rate`` Hz) WAV is read
    directly. Anything else — a different container (.ogg/.mp3/.m4a/.flac), a
    different sample rate, stereo, or another bit depth — is decoded and
    resampled on the fly via PyAV (which bundles ffmpeg). This convenience
    lives in the demo only; the engine's own sources stay strict 16k mono.
    """
    try:
        with wave.open(path, "rb") as w:
            if (w.getnchannels() == 1 and w.getsampwidth() == 2
                    and w.getframerate() == sample_rate):
                raw = w.readframes(w.getnframes())
                return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    except (wave.Error, EOFError):
        pass  # not a plain conforming WAV — fall through to PyAV

    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            f"'{path}' needs on-the-fly conversion to {sample_rate} Hz mono, "
            'which requires PyAV. Install it with: pip install "nectarstt[demo]" '
            "(or: pip install av)."
        ) from exc

    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    chunks: list[np.ndarray] = []
    with av.open(path) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for rframe in resampler.resample(frame):
                chunks.append(rframe.to_ndarray().reshape(-1))
        for rframe in resampler.resample(None):  # flush the resampler
            if rframe is not None:
                chunks.append(rframe.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def render_event(event, full: bool) -> str | None:
    """Return the line to print for an event, or None to suppress it.

    By default only finalized results are shown (clean transcript). Live
    partial hypotheses are shown only in full-transcript mode.
    """
    if event.is_partial:
        return f"~ {event.text}" if full else None
    return f"✓ {event.text}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nectarstt-demo",
                                description="Live NectarSTT transcription.")
    p.add_argument("--model", default="distil-large-v3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--language", default="en")
    p.add_argument("--file", default=None,
                   help="Transcribe an audio file (any format/rate) instead of the mic.")
    p.add_argument("--full", action="store_true",
                   help="Full transcript view: also show live partial hypotheses (~).")
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

    if args.file:
        audio = load_audio_16k_mono(args.file, sample_rate=engine._config.sample_rate)
        source = ArraySource(audio, sample_rate=engine._config.sample_rate)
        print(f"Transcribing {args.file}…")
    else:
        source = MicSource(sample_rate=engine._config.sample_rate)
        print("Listening… (Ctrl+C to stop)")

    try:
        for event in engine.stream(source=source):
            line = render_event(event, args.full)
            if line is None:
                continue
            if event.is_partial:
                print(line, end="\r", flush=True)
            else:
                # Overwrite any lingering partial line, then commit the final.
                prefix = "\r" if args.full else ""
                print(f"{prefix}{line}")
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
