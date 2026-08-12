"""OpenVox TTS — offline text-to-speech."""
from openvox.tts.engine import TTSEngine
from openvox.tts.backend import TTSResult
from openvox.tts.stream import SpeechHandle

__all__ = ["TTSEngine", "TTSResult", "SpeechHandle"]
