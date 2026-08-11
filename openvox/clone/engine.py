import dataclasses
import hashlib
import logging
import os

from openvox.clone.config import CloneConfig
from openvox.clone.chatterbox_backend import ChatterboxBackend
from openvox.tts.backend import TTSResult, AudioDeviceError

log = logging.getLogger(__name__)

class VoiceCloneEngine:
    def __init__(self, device: str | None = None, exaggeration: float | None = None,
                 cfg: float | None = None, config: CloneConfig | None = None) -> None:
        c = dataclasses.replace(config) if config is not None else CloneConfig()
        if device is not None:
            c.device = device
        if exaggeration is not None:
            c.exaggeration = exaggeration
        if cfg is not None:
            c.cfg = cfg
        self._config = c
        self._backend = ChatterboxBackend(device=c.device)
        self._enhancer = None

    def clone(self, text: str, reference_audio: str | None = None,
              exaggeration: float | None = None, cfg: float | None = None,
              enhance: bool | None = None, profile=None) -> TTSResult:
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")
        if profile is not None and reference_audio is not None:
            raise ValueError("pass either reference_audio or profile, not both")
        e = exaggeration if exaggeration is not None else self._config.exaggeration
        g = cfg if cfg is not None else self._config.cfg

        if profile is not None:
            from openvox.enroll import VoiceProfile
            prof = profile if isinstance(profile, VoiceProfile) else VoiceProfile.load(os.fspath(profile))
            return self._backend.clone_from_profile(text, prof.conditionals, e, g)

        if reference_audio is None:
            raise ValueError("provide reference_audio or profile")
        if not os.path.isfile(reference_audio):
            raise FileNotFoundError(f"reference audio not found: {reference_audio}")
        if os.path.getsize(reference_audio) == 0:
            raise ValueError(f"reference audio is empty: {reference_audio}")
        do_enhance = enhance if enhance is not None else self._config.enhance
        ref_path = self._enhanced_reference(reference_audio) if do_enhance else reference_audio
        return self._backend.clone(text, ref_path, e, g)

    def _enhanced_reference(self, reference_audio: str) -> str:
        """Return a cached, cleaned copy of the reference; fall back to the raw
        path if enhancement is unavailable or fails (never breaks cloning)."""
        try:
            from openvox._paths import cache_dir
            st = os.stat(reference_audio)
            key = hashlib.sha1(
                f"{os.path.abspath(reference_audio)}|{st.st_mtime_ns}|{st.st_size}".encode()
            ).hexdigest()
            out = os.path.join(cache_dir("enhance/cache"), key + ".wav")
            if os.path.exists(out) and os.path.getsize(out) > 0:
                return out
            from openvox.enhance import EnhanceEngine
            if self._enhancer is None:
                self._enhancer = EnhanceEngine(device=self._config.device)
            tmp = out + ".tmp"
            self._enhancer.enhance_file(reference_audio).save_wav(tmp)
            os.replace(tmp, out)
            return out
        except Exception as exc:
            log.info("Reference enhancement unavailable (%s); cloning from the raw "
                     "reference. Install with: pip install -e \".[enhance]\" && "
                     "pip install resemble-enhance --no-deps", exc)
            return reference_audio

    def play(self, result: TTSResult) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioDeviceError(
                'Playback needs the audio extra: pip install "openvox[clone]".') from exc
        try:
            sd.play(result.audio, result.sample_rate)
            sd.wait()
        except sd.PortAudioError as exc:
            raise AudioDeviceError(str(exc)) from exc

    def say(self, text: str, reference_audio: str,
            exaggeration: float | None = None, cfg: float | None = None) -> None:
        self.play(self.clone(text, reference_audio, exaggeration, cfg))
