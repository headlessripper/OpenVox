# 🍯 NectarSTT  
**Nectar Speech-to-Text Engine**

NectarSTT (Nectar Speech To Text) is a Python-based speech recognition engine designed for real-time, offline-capable voice input. It is built to be modular, extensible, and suitable for AI assistants, automation systems, and accessibility tools.

This project focuses on **accurate speech recognition**, **low latency**, and **tight integration with AI pipelines**.

---

## ✨ Features

- 🎙️ Real-time streaming speech-to-text
- 🧠 Modular engine design (easy to extend)
- ⚡ Optimized for low latency
- 🔌 Designed to integrate with AI / assistant systems
- 🖥️ Cross-platform (Windows, Linux)
- 🌍 Supports multiple Whisper model sizes (tiny through large-v3)

---

## 🛠️ Installation

Clone the repository and install in editable mode with development dependencies:

```bash
git clone https://github.com/headlessripper/NectarSTT.git
cd NectarSTT
pip install -e ".[dev]"
```

---

## 📁 Usage

### Live mic transcription

```bash
python -m nectarstt.demo
```

Available demo CLI flags:
- `--model`: model size (default: `distil-large-v3`; options: `tiny`, `base`, `small`, `distil-large-v3`, `large-v3`)
- `--device`: compute device (default: `cuda`; also: `cpu`, `mps`)
- `--language`: ISO 639-1 language code (default: `en`)
- `--file`: transcribe a WAV file instead of live mic input

Example:
```bash
python -m nectarstt.demo --model base --device cpu --file clip.wav
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
