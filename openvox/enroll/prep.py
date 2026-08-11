import logging
import os

import numpy as np

log = logging.getLogger(__name__)

VE_SR = 16000
GEN_SR = 24000


def rms(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a ** 2)))


def passes_gate(a: np.ndarray, sr: int, min_rms: float, min_dur_s: float) -> bool:
    return len(a) >= min_dur_s * sr and rms(a) >= min_rms


def segment(a: np.ndarray, sr: int, max_clip_s: float, min_rms: float):
    a = np.asarray(a, dtype=np.float32)
    win = int(max_clip_s * sr)
    if len(a) <= win:
        return [a]
    out = []
    for start in range(0, len(a), win):
        chunk = a[start:start + win]
        if rms(chunk) >= min_rms:
            out.append(chunk)
    return out


def _resample(a: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return np.asarray(a, dtype=np.float32)
    import librosa
    return librosa.resample(np.asarray(a, dtype=np.float32),
                            orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def prepare(paths, *, enhance, device, min_rms, min_dur_s, max_clip_s):
    import soundfile as sf
    clips_16k, clips_24k = [], []
    enhancer = None
    for path in paths:
        audio, sr = sf.read(os.fspath(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if enhance:
            try:
                if enhancer is None:
                    from openvox.enhance import EnhanceEngine
                    enhancer = EnhanceEngine(device=device)
                res = enhancer.enhance(audio, sr)
                audio, sr = res.audio, res.sample_rate
            except Exception as exc:  # graceful: use the raw clip
                log.info("Clip enhancement unavailable (%s); using the raw clip.", exc)
        for seg in segment(audio, sr, max_clip_s, min_rms):
            if not passes_gate(seg, sr, min_rms, min_dur_s):
                continue
            clips_16k.append(_resample(seg, sr, VE_SR))
            clips_24k.append(_resample(seg, sr, GEN_SR))
    return clips_16k, clips_24k
