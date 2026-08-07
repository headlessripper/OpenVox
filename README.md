# 🍯 NectarSTT  
**Nectar Speech-to-Text Engine**

NectarSTT (Nectar Speech To Text) is a Python-based speech recognition engine designed for real-time, offline-capable voice input. It is built to be modular, extensible, and suitable for AI assistants, automation systems, and accessibility tools.

This project focuses on **accurate speech recognition**, **low latency**, and **tight integration with AI pipelines**.

---

## ✨ Features

- 🎙️ Real-time streaming speech-to-text with live partial + final results
- 🧠 Modular engine design (swappable backend, easy to extend)
- ⚡ Low latency — partial passes are bounded to a sliding window so long speech stays real-time
- 🔌 Designed to integrate with AI / assistant systems
- 🖥️ Cross-platform (Windows, Linux)
- 🌍 Supports multiple Whisper model sizes (tiny through large-v3)
- 🎧 Demo accepts any audio format / sample rate (auto-decoded & resampled to 16 kHz)

---

## 🛠️ Installation

Clone the repository and install in editable mode with development dependencies:

```bash
git clone https://github.com/headlessripper/NectarSTT.git
cd NectarSTT
pip install -e ".[dev]"
```

The demo can transcribe **any** audio format or sample rate (`.ogg`, `.mp3`, `.m4a`, 48 kHz WAV, stereo, …) by decoding and resampling on the fly. That requires the optional `demo` extra (PyAV, which bundles ffmpeg):

```bash
pip install -e ".[dev,demo]"
```

Without it, the demo still works on files that are already 16 kHz mono 16-bit WAV, and the live mic always works.

---

## 📁 Usage

### Live mic transcription

```bash
python -m nectarstt.demo
```

By default the demo prints a **clean transcript** — only the finalized lines (`✓`). Pass `--full` for the *Full Transcript View*, which also streams the live partial hypotheses (`~`) as each sentence forms.

Available demo CLI flags:
- `--model`: model size (default: `distil-large-v3`; options: `tiny`, `base`, `small`, `distil-large-v3`, `large-v3`)
- `--device`: compute device (default: `cuda`; also: `cpu`). Falls back to CPU automatically if CUDA is unavailable.
- `--language`: ISO 639-1 language code (default: `en`)
- `--file`: transcribe an audio file instead of live mic input. **Any format / sample rate** is accepted — non-WAV or non-16 kHz files are decoded and resampled automatically (requires the `demo` extra; see Installation).
- `--full`: Full Transcript View — also show live partial hypotheses (`~`), not just finalized results.

Examples:
```bash
# Clean transcript (finals only) of any audio file:
python -m nectarstt.demo --model base --device cpu --file recording.ogg

# Watch partials update live as each sentence forms:
python -m nectarstt.demo --model base --file clip.mp3 --full
```

### In code

```python
from nectarstt import STTEngine

engine = STTEngine(model="distil-large-v3", device="cuda", language="en")

for event in engine.stream():
    if event.is_partial:
        print("~", event.text, end="\r")
    else:
        print("✓", event.text)
```

### Batch transcription

```python
from nectarstt import STTEngine

engine = STTEngine(model="distil-large-v3", device="cuda", language="en")

result = engine.transcribe_file("clip.wav")
print(result.text)
for word in result.words:
    print(f"  {word.word}: [{word.start:.2f}–{word.end:.2f}s, confidence={word.probability:.2f}]")
```

---

## 🚀 Roadmap

For detailed design specifications and implementation plans, see:
- **Specifications:** [`docs/superpowers/specs/`](docs/superpowers/specs/)
- **Plans:** [`docs/superpowers/plans/`](docs/superpowers/plans/)

---

## 🤝 Contributing

Contributions are welcome!

You can:

- 🐛 Report bugs
- 💡 Suggest new features
- 🔧 Submit pull requests

Please open an issue to discuss major changes before starting work.

---

## 📜 License
Use a Custom License
---

## ⭐ Support

If you find **NectarSTT** useful:

- ⭐ Star the repository
- 🐞 Report issues
- 💬 Share feedback and ideas

---

**Built with ❤️ in Python for high-quality, low-latency speech recognition**
