import contextlib
import io
import logging
import os

import numpy as np

from openvox.enroll.backend import EnrollBackend

log = logging.getLogger(__name__)

VE_SR = 16000


def _resolve_device(requested: str, cuda_available: bool) -> str:
    return "cuda" if requested == "cuda" and cuda_available else "cpu"


class ChatterboxEnrollBackend(EnrollBackend):
    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        import torch
        from chatterbox.tts import ChatterboxTTS
        dev = _resolve_device(self._device, torch.cuda.is_available())
        if self._device == "cuda" and dev != "cuda":
            log.info("CUDA not available to torch; loading Chatterbox on CPU (slow).")
        with contextlib.redirect_stdout(io.StringIO()):
            self._model = ChatterboxTTS.from_pretrained(dev)

    @property
    def cuda_available(self) -> bool:
        import torch
        return self._device == "cuda" and torch.cuda.is_available()

    @property
    def model_sr(self) -> int:
        self._ensure_loaded()
        return int(self._model.sr)

    def embed_clips(self, wavs_16k):
        self._ensure_loaded()
        wavs = [np.asarray(w, dtype=np.float32) for w in wavs_16k]
        embs = self._model.ve.embeds_from_wavs(wavs, sample_rate=VE_SR)
        return np.asarray(embs, dtype=np.float32)

    def reference_from_clip(self, clip_16k, clip_24k):
        self._ensure_loaded()
        import torch
        from chatterbox.tts import S3GEN_SR
        m = self._model
        s3gen_ref_wav = np.asarray(clip_24k, dtype=np.float32)[: m.DEC_COND_LEN]
        gen_ref = m.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=m.device)
        cond_prompt = None
        plen = m.t3.hp.speech_cond_prompt_len
        if plen:
            ref16 = np.asarray(clip_16k, dtype=np.float32)[: m.ENC_COND_LEN]
            tok, _ = m.s3gen.tokenizer.forward([ref16], max_len=plen)
            cond_prompt = torch.atleast_2d(tok).to(m.device)
        return {"gen": gen_ref, "cond_prompt": cond_prompt}

    def make_conditionals(self, speaker_emb, reference, exaggeration):
        self._ensure_loaded()
        import torch
        from chatterbox.tts import Conditionals
        from chatterbox.models.t3.modules.cond_enc import T3Cond
        m = self._model
        emb = torch.from_numpy(np.asarray(speaker_emb, dtype=np.float32)).reshape(1, -1).to(m.device)
        t3 = T3Cond(
            speaker_emb=emb,
            cond_prompt_speech_tokens=reference["cond_prompt"],
            emotion_adv=exaggeration * torch.ones(1, 1, 1, device=m.device),
        ).to(device=m.device)
        return Conditionals(t3, reference["gen"])

    def generate(self, conditionals, text, seed):
        self._ensure_loaded()
        import torch
        m = self._model
        torch.manual_seed(int(seed))
        m.conds = conditionals
        exaggeration = float(conditionals.t3.emotion_adv.reshape(-1)[0])
        wav = m.generate(text, exaggeration=exaggeration)
        return wav.squeeze(0).detach().cpu().numpy().astype(np.float32), int(m.sr)

    def ve_embed(self, wav, sr):
        self._ensure_loaded()
        w = np.asarray(wav, dtype=np.float32)
        if sr != VE_SR:
            import librosa
            w = librosa.resample(w, orig_sr=sr, target_sr=VE_SR).astype(np.float32)
        emb = self._model.ve.embeds_from_wavs([w], sample_rate=VE_SR)
        return np.asarray(emb, dtype=np.float32).reshape(-1)
