<div align="center">

<img src="docs/assets/openvox-icon.png" alt="OpenVox" width="120"/>

<h1>OpenVox</h1>

<h3>The all&#8209;in&#8209;one, fully offline voice engine</h3>

<p>Speech&#8209;to&#8209;text and text&#8209;to&#8209;speech that runs <b>entirely on your own hardware</b>.<br/>No cloud. No API keys. No per&#8209;minute fees.</p>

<p>
<img src="https://img.shields.io/badge/100%25-Offline-1E6FB8?style=for-the-badge" alt="Offline"/>
<img src="https://img.shields.io/badge/Python-3.11+-16264F?style=for-the-badge" alt="Python 3.11+"/>
<img src="https://img.shields.io/badge/Engines-6-2F86CF?style=for-the-badge" alt="6 engines"/>
<img src="https://img.shields.io/badge/License-Zashiron%20v1.2-6FC0F0?style=for-the-badge" alt="License"/>
</p>

<p>
<a href="#-engines"><b>Engines</b></a> &nbsp;·&nbsp;
<a href="#-quickstart"><b>Quickstart</b></a> &nbsp;·&nbsp;
<a href="#-streaming-tts-with-barge-in"><b>Barge&#8209;in</b></a> &nbsp;·&nbsp;
<a href="#-roadmap"><b>Roadmap</b></a> &nbsp;·&nbsp;
<a href="https://github.com/headlessripper/OpenVox"><b>GitHub&nbsp;↗</b></a>
</p>

</div>

<hr/>

OpenVox is a complete voice stack, speech&#8209;to&#8209;text *and* text&#8209;to&#8209;speech, built for the places cloud voice services can't go: robots, embedded and edge devices, air&#8209;gapped systems, and any product where audio must never leave the machine.

The goal is simple and ambitious: **match the quality of cloud services like ElevenLabs, but 100% offline**, then beat them on the things a cloud API structurally can't do, namely zero latency jitter, zero marginal cost, total privacy, and deep on&#8209;device integration.

<div align="center">

### Why OpenVox

</div>

| | Cloud voice APIs | **OpenVox** |
|---|---|---|
| **Connectivity** | Requires internet | **Fully offline / air-gapped** |
| **Cost** | Per-minute / per-character fees | **Zero marginal cost**, run it all day for free |
| **Privacy** | Audio leaves your device | **Audio never leaves the machine** |
| **Latency** | Network round-trip + jitter | **On-device, deterministic** |
| **Rate limits** | Throttled | **None** |
| **Deployment** | Someone else's servers | **Your robot, your edge box, your terms** |

A natural fit for **robotics, defense, medical, industrial, maritime, and privacy&#8209;sensitive** applications, anywhere a device needs to hear and speak without phoning home.

<hr/>

## 🧩 Engines

OpenVox is one package, `openvox`, with each engine kept modular and independently installable so you only ship what a given device needs. Every engine sits behind a **swappable backend interface**, so the model underneath can be upgraded or replaced without touching your code.

<div align="center">

| Engine | What it does | Status |
|:---|:---|:---:|
| 🎙️ **`openvox.stt`** | Streaming speech&#8209;to&#8209;text: live partials, finals, word timestamps | ✅ Available |
| 🗣️ **`openvox.tts`** | Natural, human&#8209;sounding text&#8209;to&#8209;speech | ✅ Available |
| ⚡ **`openvox.tts` · stream** | Real&#8209;time streaming with instant `stop()` barge&#8209;in | ✅ Available |
| 🎭 **`openvox.clone`** | Zero&#8209;shot voice cloning from a short sample | ✅ Available |
| 🧬 **`openvox.enroll`** | A reusable, higher&#8209;fidelity voice profile from several clips | ✅ Available |
| 🧼 **`openvox.enhance`** | Denoise, restore, and bandwidth&#8209;extend a poor recording | ✅ Available |

</div>

<hr/>

## 🚀 Quickstart

```bash
git clone https://github.com/headlessripper/OpenVox.git
cd OpenVox

# speech-to-text + text-to-speech (CPU)
pip install -e ".[stt,stt-demo,tts]"

# stream a line and interrupt it after 1.5s to see barge-in
python -m openvox.tts.demo --text "This gets cut off partway through." --stream --interrupt-after 1.5
```

Each capability is an optional extra, so you install only what you need: `stt`, `stt-demo`, `tts`, `tts-gpu`, `clone`, `enroll`, `enhance`, `dev`.

<hr/>

## 🎙️ Speech&#8209;to&#8209;Text

Real&#8209;time streaming transcription with live partials that firm up into finals, plus word&#8209;level timestamps. Neural voice&#8209;activity detection keeps it robust in noise, a sliding window bounds latency on long speech, and it runs CUDA&#8209;accelerated with an automatic CPU fallback.

```bash
pip install -e ".[stt,stt-demo]"
python -m openvox.stt.demo                                  # live mic, clean transcript
python -m openvox.stt.demo --model base --file clip.mp3 --full   # any file, show live partials
```

```python
from openvox.stt import STTEngine

engine = STTEngine(model="distil-large-v3", device="cuda", language="en")

for event in engine.stream():                 # live from the microphone
    if event.is_partial:
        print("~", event.text, end="\r")       # firms up as you speak
    else:
        print("OK", event.text)                 # finalized line

result = engine.transcribe_file("clip.wav")   # or a whole file with timestamps
for word in result.words:
    print(f"  {word.word}: [{word.start:.2f}s, p={word.probability:.2f}]")
```

<sub>**Flags:** `--model` (`tiny`/`base`/`small`/`distil-large-v3`/`large-v3`) · `--device` (`cuda`/`cpu`) · `--language` · `--file` (any format/rate) · `--full`.</sub>

<hr/>

## 🗣️ Text&#8209;to&#8209;Speech

Genuinely human&#8209;sounding speech, fully offline, with 28 built&#8209;in English voices at 24 kHz. The engine auto&#8209;selects the GPU when available and falls back to CPU.

```bash
pip install -e ".[tts]"       # CPU
pip install -e ".[tts-gpu]"   # NVIDIA GPU (bundles the CUDA 12 / cuDNN 9 runtime; no system CUDA needed)
```

```python
from openvox.tts import TTSEngine

engine = TTSEngine(voice="af_heart", device="cuda")   # falls back to CPU
engine.say("This runs entirely offline.")             # synthesize and speak
engine.synthesize("Save me to a file.").save_wav("out.wav")
engine.voices()                                       # list built-in voices
```

<sub>**Flags:** `--text` (required) · `--voice` · `--device` · `--speed` · `--out PATH` · `--no-play`.</sub>

<hr/>

## ⚡ Streaming TTS with Barge&#8209;in

Speech should start almost immediately and be **interruptible the instant the user speaks**, which is essential for robots and interactive agents. OpenVox streams synthesized audio segment by segment and exposes a `stop()` that cuts playback within a single audio block.

The same call works for a built&#8209;in voice **or a cloned voice profile**: pass `voice="af_heart"` or `voice="alice.ovx"`.

```python
from openvox.tts import TTSEngine

engine = TTSEngine(voice="af_heart", device="cuda")

# Stream chunks yourself (robot, socket, custom sink):
for chunk in engine.stream("Streamed as it is synthesized."):
    send_to_speaker(chunk.audio, chunk.sample_rate)

# Or play with barge-in support:
handle = engine.say_stream("I can be interrupted at any moment.")
handle.stop()     # cut audio within ~one audio block, safe to call from any thread
handle.wait()     # block until done (or already stopped)

# Stream in a cloned voice, same call:
engine.say_stream("Now in a cloned voice.", voice="alice.ovx")
```

`stop()` is signal&#8209;driven and thread&#8209;safe, so a future full&#8209;duplex loop (listening while OpenVox speaks) simply calls `handle.stop()` when it hears the user.

<sub>**New flags:** `--stream` · `--interrupt-after SECONDS` · `--voice` also accepts an `.ovx` profile path.</sub>

<hr/>

## 🎭 Voice Cloning

Zero&#8209;shot voice cloning: give a short reference clip and speak any text in that voice, fully offline (via [Chatterbox](https://github.com/resemble-ai/chatterbox), MIT). Every generated clip carries an imperceptible neural watermark for traceability.

```bash
pip install -e ".[clone]"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121   # for NVIDIA GPU
```

```python
from openvox.clone import VoiceCloneEngine

engine = VoiceCloneEngine(device="cuda")
engine.clone("Speak this in my voice.", reference_audio="myvoice.mp3").save_wav("cloned.wav")

# Or clone from a saved profile (see Voice Enrollment), no reference clip needed:
engine.clone("Speak this in the enrolled voice.", profile="alice.ovx").save_wav("out.wav")
```

<sub>**Flags:** `--text` · `--ref PATH` · `--profile PATH` · `--exaggeration` · `--cfg` · `--device` · `--out` · `--no-play`.</sub>

<hr/>

## 🧬 Voice Enrollment

Zero&#8209;shot cloning is only as good as one reference clip. **Enrollment** turns *several* clips of a voice into a saved, reusable **voice profile** (`.ovx`) that clones with materially higher, more consistent fidelity, and needs no reference clip at generation time.

Under the hood it builds a robust speaker representation from all the clips, then runs a speaker&#8209;similarity&#8209;guided search that optimizes the cloning conditioning to sound as close to the real voice as possible. No transcripts required.

```bash
pip install -e ".[enroll]"                # composes the clone + enhance engines
pip install resemble-enhance --no-deps
```

```python
from openvox.enroll import VoiceEnrollEngine

eng = VoiceEnrollEngine(device="cuda")
profile = eng.enroll(["clipA.wav", "clipB.wav", "long.m4a"])
print(profile.score)                      # achieved speaker-similarity
profile.save("alice.ovx")

# Use the profile anywhere a voice is accepted, cloning or streaming TTS.
```

<sub>**Flags:** `--in PATH [PATH ...]` · `--out PATH` · `--quality` (`fast`/`balanced`/`thorough`) · `--device` · `--no-enhance`. The optimization search runs on GPU; on a CPU&#8209;only machine enrollment uses the robust&#8209;baseline stage only.</sub>

<hr/>

## 🧼 Speech Enhancement

Restore a poorly&#8209;recorded clip, denoise, enhance, and extend bandwidth (16 kHz to 44.1 kHz), fully offline (via [resemble&#8209;enhance](https://github.com/resemble-ai/resemble-enhance), MIT). The cloner and enroller use it **automatically** to clean audio before use.

```bash
pip install -e ".[enhance]"
pip install resemble-enhance --no-deps
```

```python
from openvox.enhance import EnhanceEngine

engine = EnhanceEngine(device="cuda")
engine.enhance_file("poor.wav").save_wav("clean.wav")   # denoise + restore to 44.1 kHz
```

<sub>**Flags:** `--in PATH` · `--out PATH` · `--device` · `--denoise-only` · `--nfe`.</sub>

<hr/>

## 🗺️ Roadmap

The vision in four horizons:

1. **Parity of plumbing.** An importable, offline library: streaming STT ✅, streaming TTS ✅, a headless daemon, and a ROS 2 node.
2. **Parity of quality.** A full model ladder, GPU / Jetson / ARM backends, punctuation, diarization, wake&#8209;word, and command&#8209;grammar biasing.
3. **Surpass the cloud.** On&#8209;device voice cloning ✅, a sub&#8209;100 ms full&#8209;duplex listen&#8209;and&#8209;speak loop, on&#8209;device adaptive fine&#8209;tuning, and mic&#8209;array direction&#8209;of&#8209;arrival.
4. **Platform.** A community voice&#8209;model hub, an eval harness proving OpenVox beats the cloud on real&#8209;world audio, and a hardened cross&#8209;platform SDK.

<sub>Next up: an **LLM plug&#8209;and&#8209;play route**, a drop&#8209;in STT to your model to TTS loop with barge&#8209;in for building voice assistants.</sub>

<hr/>

## 🗂️ Design docs

OpenVox is built spec&#8209;first. Full designs and step&#8209;by&#8209;step implementation plans for each engine live under [`docs/superpowers/`](docs/superpowers/), split into [specs](docs/superpowers/specs/) and [plans](docs/superpowers/plans/).

<hr/>

## 🤝 Contributing

Contributions are welcome: bug reports, feature ideas, and pull requests. Please open an issue to discuss any major change before starting work.

## 📜 License

Released under the **OpenVox Proprietary License (Zashiron License v1.2)**, see [`LICENSE`](LICENSE). Commercial use requires written authorization.

<div align="center">

<br/>

<img src="docs/assets/openvox-logo.jpg" alt="OpenVox, Voice AI" width="280"/>

<b>OpenVox</b> &nbsp;·&nbsp; hear and speak, entirely offline.

</div>
