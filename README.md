# 🍯 OpenVox

**The all-in-one, fully offline voice engine.**

OpenVox is a complete voice stack — speech-to-text *and* text-to-speech — that runs **entirely on your own hardware, with no cloud, no API keys, and no per-minute fees.** It's built for the places cloud voice services can't go: robots, embedded and edge devices, air-gapped systems, and any product where audio must never leave the machine.

The goal is simple and ambitious: **match the quality of cloud services like ElevenLabs, but 100% offline** — and then beat them on the things a cloud API structurally can't do (zero latency jitter, zero marginal cost, total privacy, and deep on-device integration).

Speech-to-text and text-to-speech are **available today**. Voice cloning is **available today**; ultra-low-latency streaming is **in active development** (see the roadmap below).

---

## 💡 Why OpenVox

| | Cloud voice APIs | **OpenVox** |
|---|---|---|
| **Connectivity** | Requires internet | **Fully offline / air-gapped** |
| **Cost** | Per-minute / per-character fees | **Zero marginal cost** — run it all day for free |
| **Privacy** | Audio leaves your device | **Audio never leaves the machine** |
| **Latency** | Network round-trip + jitter | **On-device, deterministic** |
| **Rate limits** | Throttled | **None** |
| **Deployment** | Someone else's servers | **Your robot, your edge box, your terms** |

This makes OpenVox a natural fit for **robotics, defense, medical, industrial, maritime, and privacy-sensitive** applications — anywhere a device needs to hear and speak without phoning home.

---

## 📦 Project status & roadmap

OpenVox is being built as a series of focused, independently-shipped engines under one package. Each is designed, planned, and reviewed before it lands (see [`docs/superpowers/`](docs/superpowers/)).

| Capability | Status |
|---|---|
| 🎙️ **Streaming speech-to-text** (live partials + finals, word timestamps) | ✅ **Available** |
| 🗣️ **Text-to-speech** — natural, human-sounding built-in voices | ✅ Available |
| 🎭 **Voice cloning** — a custom voice from a short sample, fully local | ✅ Available |
| 🧼 **Speech enhancement** — denoise + restore + bandwidth-extend a poor recording | ✅ Available |
| ⚡ **Ultra-low-latency streaming TTS** with barge-in (real-time robot speech) | 🔭 Planned |
| 🧩 **Headless daemon + ROS 2 node** for robot integration | 🔭 Planned |
| 🖥️ **Hardware backends** — CUDA today; Jetson / ARM / Raspberry Pi next | 🔭 Planned |

**The vision in four horizons:**

1. **Parity of plumbing** — an importable, offline library: streaming STT ✅, streaming TTS, a headless daemon, and a ROS 2 node.
2. **Parity of quality** — a full model ladder, GPU/Jetson/ARM backends, punctuation, diarization, wake-word, and command-grammar biasing.
3. **Surpass the cloud** — on-device voice cloning, a sub-100 ms full-duplex listen-and-speak loop, on-device adaptive fine-tuning (a robot that learns its operator's voice and its own noise profile), and mic-array direction-of-arrival.
4. **Platform** — a community voice-model hub, an eval harness proving OpenVox's accuracy/latency beats the cloud on real-world audio, and a hardened cross-platform SDK.

---

## 🧠 Architecture

OpenVox is one package, `openvox`, with each engine kept modular and independently installable so you only ship what a given device needs:

- **`openvox.stt`** — the speech-to-text engine (available today).
- **`openvox.tts`** — the text-to-speech engine (available today).
- **`openvox.clone`** — the zero-shot voice-cloning engine (available today).
- **`openvox.enhance`** — the offline speech-restoration engine (available today).

Each engine sits behind a **swappable backend interface**, so the underlying model can be upgraded — or replaced with a purpose-built one — without changing the code that uses it. `import openvox` stays lightweight; you pull in an engine (and only its dependencies) via `openvox.stt` / `openvox.tts`.

---

## 🎙️ Speech-to-Text (available today)

- Real-time **streaming** transcription with live partial results that firm up into finals, plus word-level timestamps.
- **Robust in noise** — neural voice-activity detection instead of a crude volume gate, so it holds up in real acoustic environments.
- **Bounded latency** — partial passes use a sliding window, so even long, continuous speech stays real-time.
- **Runs anywhere** — CUDA-accelerated with an automatic CPU fallback; the model auto-downloads once, then runs fully offline.
- **Multiple model sizes** (`tiny` → `large-v3`) to trade speed for accuracy.
- **Any input** — the demo accepts any audio format or sample rate (`.ogg`, `.mp3`, `.m4a`, 48 kHz, stereo, …), decoding and resampling on the fly.

### Installation

```bash
git clone https://github.com/headlessripper/OpenVox.git
cd OpenVox
pip install -e ".[stt,stt-demo]"
```

Add the `dev` extra to run the test suite:

```bash
pip install -e ".[stt,stt-demo,dev]"
```

The `stt-demo` extra adds on-the-fly audio decoding/resampling (via PyAV, which bundles ffmpeg). Without it, the demo still handles 16 kHz mono 16-bit WAV files, and the live microphone always works.

### Try it

```bash
# Live microphone — clean transcript (finalized lines only):
python -m openvox.stt.demo

# Transcribe any audio file:
python -m openvox.stt.demo --model base --device cpu --file recording.ogg

# Watch partial hypotheses update live as each sentence forms:
python -m openvox.stt.demo --model base --file clip.mp3 --full
```

By default the demo shows a **clean transcript** — only finalized lines (`✓`). Pass `--full` for the *Full Transcript View*, which also streams the live partials (`~`).

**Demo flags:** `--model` (`tiny`/`base`/`small`/`distil-large-v3`/`large-v3`, default `distil-large-v3`) · `--device` (`cuda` or `cpu`; auto-falls back to CPU) · `--language` (ISO 639-1, default `en`) · `--file` (any format/rate) · `--full` (show live partials).

### Use it in your own project

```python
from openvox.stt import STTEngine

engine = STTEngine(model="distil-large-v3", device="cuda", language="en")

# Live streaming from the microphone:
for event in engine.stream():
    if event.is_partial:
        print("~", event.text, end="\r")   # firms up as you speak
    else:
        print("✓", event.text)              # finalized line

# Or transcribe a whole file with word-level timestamps:
result = engine.transcribe_file("clip.wav")
print(result.text)
for word in result.words:
    print(f"  {word.word}: [{word.start:.2f}–{word.end:.2f}s, p={word.probability:.2f}]")
```

---

## 🗣️ Text-to-Speech (available today)

Genuinely human-sounding speech, fully offline, with built-in voices.

```bash
pip install -e ".[tts]"        # CPU
# or, for NVIDIA GPU acceleration (bundles the CUDA 12 / cuDNN 9 runtime, no system CUDA needed):
pip install -e ".[tts-gpu]"
```

The engine auto-selects the GPU when a working CUDA provider is available and falls back to CPU otherwise — `device="cuda"` is safe everywhere. (Install *either* `tts` or `tts-gpu`, not both.)

```bash
# Speak a line aloud (and optionally save it):
python -m openvox.tts.demo --text "Hello, I am OpenVox." --out hello.wav
```

```python
from openvox.tts import TTSEngine

engine = TTSEngine(voice="af_heart", device="cuda")   # falls back to CPU
engine.say("This runs entirely offline.")             # synthesize + speak

result = engine.synthesize("Save me to a file.")
result.save_wav("out.wav")

engine.voices()   # list the built-in voices
```

**Demo flags:** `--text` (required) · `--voice` (default `af_heart`) · `--device` (`cuda`/`cpu`) · `--speed` (default `1.0`) · `--out PATH` (save a WAV) · `--no-play` (skip playback).

---

## 🎭 Voice Cloning (available today)

Zero-shot voice cloning — give a short reference clip and speak any text in that voice, fully offline (via [Chatterbox](https://github.com/resemble-ai/chatterbox), MIT). Every generated clip carries an imperceptible neural watermark for traceability.

```bash
pip install -e ".[clone]"                 # CPU (pulls PyTorch)
# for NVIDIA GPU, also install a CUDA torch build:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

```bash
python -m openvox.clone.demo --text "This is my cloned voice." --ref myvoice.mp3 --out cloned.wav
```

```python
from openvox.clone import VoiceCloneEngine

engine = VoiceCloneEngine(device="cuda")            # falls back to CPU
result = engine.clone("Speak this in my voice.", reference_audio="myvoice.mp3")
result.save_wav("cloned.wav")
engine.say("Nice to meet you.", reference_audio="myvoice.mp3")   # clone + speak
```

**Demo flags:** `--text` (required) · `--ref PATH` (required, any audio format) · `--exaggeration` (default 0.5) · `--cfg` (default 0.5) · `--device` (`cuda`/`cpu`) · `--out PATH` · `--no-play`.

---

## 🧼 Speech Enhancement (available today)

Restore a poorly-recorded clip — denoise, enhance, and extend bandwidth (e.g. 16 kHz → 44.1 kHz) — fully offline (via [resemble-enhance](https://github.com/resemble-ai/resemble-enhance), MIT). The voice cloner uses it **automatically** to clean the reference clip before cloning (pass `--no-enhance` to skip).

Install is two steps (resemble-enhance ships incompatible pins, so it goes in `--no-deps`):

```bash
pip install -e ".[enhance]"
pip install resemble-enhance --no-deps
```

The `[enhance]` extra installs CPU torch; for NVIDIA GPU add a CUDA torch build: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`.

```bash
python -m openvox.enhance.demo --in poor.wav --out clean.wav
```

```python
from openvox.enhance import EnhanceEngine

engine = EnhanceEngine(device="cuda")            # falls back to CPU
result = engine.enhance_file("poor.wav")         # denoise + restore -> 44.1 kHz
result.save_wav("clean.wav")
```

**Demo flags:** `--in PATH` (required) · `--out PATH` (required) · `--device` (`cuda`/`cpu`) · `--denoise-only` · `--nfe` (default 64).

---

## 🗺️ Design docs

OpenVox is built spec-first. Full designs and step-by-step implementation plans for each engine:

- **Specifications:** [`docs/superpowers/specs/`](docs/superpowers/specs/)
- **Plans:** [`docs/superpowers/plans/`](docs/superpowers/plans/)

---

## 🤝 Contributing

Contributions are welcome — bug reports, feature ideas, and pull requests. Please open an issue to discuss any major change before starting work.

---

## 📜 License

OpenVox is released under the **OpenVox Proprietary License (Zashiron License v1.2)** — see [`LICENSE`](LICENSE). Commercial use requires written authorization; see the license for details.

---

## ⭐ Support

If OpenVox is useful to you:

- ⭐ Star the repository
- 🐞 Report issues
- 💬 Share feedback and ideas

---

**OpenVox — hear and speak, entirely offline.**
