# OpenVox Voice Enrollment (2B.2b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `openvox.enroll` engine that turns several clips of a voice into a saved `.ovx` voice profile — a Chatterbox `Conditionals` blob optimized by a speaker-similarity objective — that `VoiceCloneEngine` clones from with no reference clip.

**Architecture:** A `VoiceEnrollEngine` façade orchestrates: prep+enhance clips → **Stage A** (robust multi-clip speaker-embedding centroid) → **Stage B** (gradient-free, similarity-guided search over a convex blend of the clip embeddings, scored by generating probe sentences and measuring voice-encoder cosine similarity to the target centroid). All algorithm modules are pure numpy behind an `EnrollBackend` ABC; only `chatterbox_backend.py` imports torch/chatterbox, lazily. The output `VoiceProfile` saves/loads an `.ovx` file; `VoiceCloneEngine.clone(profile=…)` injects it.

**Tech Stack:** Python ≥3.11, numpy, Chatterbox (torch, via the `[clone]` engine), resemble-enhance (via the `[enhance]` engine), soundfile/librosa (torch-free, lazy).

## Global Constraints

- Python ≥ 3.11. Copy verbatim from the spec.
- `import openvox` and `import openvox.enroll` MUST stay torch-free — heavy imports (`torch`, `chatterbox`, `librosa`, `soundfile`, `resemble_enhance`) are lazy, inside methods only. Enforced by a subprocess import-guard test (Task 8).
- Algorithm modules (`aggregate.py`, `scorer.py`, `search.py`, `prep.py`) are **pure numpy** — no torch/chatterbox imports anywhere in them.
- Only `openvox/enroll/chatterbox_backend.py` imports `torch`/`chatterbox`.
- Reuse `TTSResult` from `openvox.tts.backend` (do not redefine it).
- Reuse `cache_dir` from `openvox._paths` for any caching.
- Config objects are copied with `dataclasses.replace`, never mutated in place (match `EnhanceEngine`/`VoiceCloneEngine`).
- New `[enroll]` extra composes `openvox[clone,enhance]`; stays **out of `all`**. Entry point `openvox-enroll-demo`.
- `.ovx` files are written/read with `torch.save`/`torch.load(..., weights_only=False)`. `SCHEMA_VERSION = 1`.
- Coordinate ascent is the optimizer — **no new optimizer dependency**.
- Commit messages use the `feat(enroll):` / `test(enroll):` / `docs(enroll):` prefix.
- Stage B ships only if it beats Stage A by `accept_margin`; no GPU → Stage A only.

---

### Task 1: EnrollConfig + quality→evals mapping

**Files:**
- Create: `openvox/enroll/__init__.py`
- Create: `openvox/enroll/config.py`
- Test: `tests/enroll/__init__.py`, `tests/enroll/test_config.py`

**Interfaces:**
- Produces: `EnrollConfig` dataclass with fields `device: str = "cuda"`, `quality: str = "balanced"`, `max_evals: int | None = None`, `probes: list[str] | None = None`, `enhance_clips: bool = True`, `min_clips: int = 1`, `outlier_threshold: float = 0.6`, `realizability_lambda: float = 0.05`, `exaggeration: float = 0.5`, `seed: int = 0`, `accept_margin: float = 0.005`, `min_rms: float = 0.01`, `min_dur_s: float = 1.0`, `max_clip_s: float = 12.0`. Function `evals_for_quality(quality: str) -> int` → `fast=15`, `balanced=40`, `thorough=100` (raises `ValueError` on unknown).

- [ ] **Step 1: Write the failing test**

```python
# tests/enroll/test_config.py
import pytest
from openvox.enroll.config import EnrollConfig, evals_for_quality

def test_defaults():
    c = EnrollConfig()
    assert c.device == "cuda" and c.quality == "balanced"
    assert c.max_evals is None and c.enhance_clips is True
    assert c.min_clips == 1 and c.exaggeration == 0.5

def test_evals_for_quality():
    assert evals_for_quality("fast") == 15
    assert evals_for_quality("balanced") == 40
    assert evals_for_quality("thorough") == 100
    with pytest.raises(ValueError):
        evals_for_quality("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/enroll/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: openvox.enroll.config`

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/enroll/config.py
from dataclasses import dataclass

_QUALITY_EVALS = {"fast": 15, "balanced": 40, "thorough": 100}

def evals_for_quality(quality: str) -> int:
    try:
        return _QUALITY_EVALS[quality]
    except KeyError:
        raise ValueError(
            f"unknown quality {quality!r}; choose from {sorted(_QUALITY_EVALS)}"
        ) from None

@dataclass
class EnrollConfig:
    device: str = "cuda"
    quality: str = "balanced"
    max_evals: int | None = None
    probes: list[str] | None = None
    enhance_clips: bool = True
    min_clips: int = 1
    outlier_threshold: float = 0.6
    realizability_lambda: float = 0.05
    exaggeration: float = 0.5
    seed: int = 0
    accept_margin: float = 0.005
    min_rms: float = 0.01
    min_dur_s: float = 1.0
    max_clip_s: float = 12.0
```

```python
# openvox/enroll/__init__.py
from openvox.enroll.engine import VoiceEnrollEngine
from openvox.enroll.profile import VoiceProfile

__all__ = ["VoiceEnrollEngine", "VoiceProfile"]
```

Note: `__init__.py` imports `engine`/`profile` which don't exist yet. To keep this task green in isolation, create `__init__.py` now but leave the two imports commented with a `# added in Task 2 / Task 8` marker, OR create empty stub modules. **Do this:** create `openvox/enroll/__init__.py` containing only a module docstring for Task 1; the exports line is added in Task 8 once `engine`/`profile` exist. Also create empty `tests/enroll/__init__.py`.

```python
# openvox/enroll/__init__.py  (Task 1 version)
"""OpenVox per-voice enrollment engine (embedding optimization)."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/enroll/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/enroll/__init__.py openvox/enroll/config.py tests/enroll/__init__.py tests/enroll/test_config.py
git commit -m "feat(enroll): add EnrollConfig and quality->evals mapping"
```

---

### Task 2: VoiceProfile artifact (payload pack/unpack + save/load)

**Files:**
- Create: `openvox/enroll/profile.py`
- Test: `tests/enroll/test_profile.py`

**Interfaces:**
- Produces: `VoiceProfile` dataclass with attributes `conditionals` (opaque, the Chatterbox `Conditionals` object), `score: float`, `metadata: dict`. Class attr `SCHEMA_VERSION: int = 1`. Methods: `_to_payload() -> dict`, classmethod `_from_payload(payload: dict) -> VoiceProfile` (raises `ValueError` on schema mismatch), `save(path: str) -> None` (lazy `torch.save`), classmethod `load(path: str) -> VoiceProfile` (lazy `torch.load(..., weights_only=False)`).
- Consumes: nothing.

- [ ] **Step 1: Write the failing test**

```python
# tests/enroll/test_profile.py
import pytest
from openvox.enroll.profile import VoiceProfile

def test_payload_roundtrip_is_torch_free():
    sentinel = object()  # stands in for a Conditionals object
    p = VoiceProfile(conditionals=sentinel, score=0.87, metadata={"stage": "B"})
    restored = VoiceProfile._from_payload(p._to_payload())
    assert restored.conditionals is sentinel
    assert restored.score == 0.87
    assert restored.metadata == {"stage": "B"}

def test_from_payload_rejects_bad_schema():
    bad = {"schema": 999, "conditionals": object(), "score": 0.0, "metadata": {}}
    with pytest.raises(ValueError, match="schema"):
        VoiceProfile._from_payload(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/enroll/test_profile.py -v`
Expected: FAIL with `ModuleNotFoundError: openvox.enroll.profile`

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/enroll/profile.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

@dataclass
class VoiceProfile:
    SCHEMA_VERSION = 1

    conditionals: Any            # Chatterbox Conditionals (opaque here)
    score: float
    metadata: dict = field(default_factory=dict)

    def _to_payload(self) -> dict:
        return {
            "schema": self.SCHEMA_VERSION,
            "conditionals": self.conditionals,
            "score": self.score,
            "metadata": self.metadata,
        }

    @classmethod
    def _from_payload(cls, payload: dict) -> "VoiceProfile":
        schema = payload.get("schema")
        if schema != cls.SCHEMA_VERSION:
            raise ValueError(
                f"unsupported .ovx schema {schema!r} (expected {cls.SCHEMA_VERSION}); "
                "regenerate the profile with the current version"
            )
        return cls(
            conditionals=payload["conditionals"],
            score=payload["score"],
            metadata=payload.get("metadata", {}),
        )

    def save(self, path: str) -> None:
        import torch
        tmp = os.fspath(path) + ".tmp"
        torch.save(self._to_payload(), tmp)
        os.replace(tmp, os.fspath(path))

    @classmethod
    def load(cls, path: str) -> "VoiceProfile":
        import torch
        payload = torch.load(os.fspath(path), map_location="cpu", weights_only=False)
        return cls._from_payload(payload)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/enroll/test_profile.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/enroll/profile.py tests/enroll/test_profile.py
git commit -m "feat(enroll): add VoiceProfile artifact with .ovx save/load"
```

---

### Task 3: Stage A — robust centroid (aggregate.py)

**Files:**
- Create: `openvox/enroll/aggregate.py`
- Test: `tests/enroll/test_aggregate.py`

**Interfaces:**
- Produces:
  - `l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray` (shared vector helper).
  - `cosine(a: np.ndarray, b: np.ndarray) -> float` (shared vector helper).
  - `robust_centroid(embs: np.ndarray, outlier_threshold: float) -> tuple[np.ndarray, np.ndarray, int]` — takes `(N, D)` embeddings, returns `(centroid (D,), kept_mask (N,) bool, representative_idx int)`. Normalizes internally; drops clips whose cosine to the initial centroid is below `outlier_threshold` (but never drops all — if every clip would be dropped, keep them all); recomputes the centroid over survivors; `representative_idx` is the surviving clip closest to the final centroid.
- Consumes: nothing.

- [ ] **Step 1: Write the failing test**

```python
# tests/enroll/test_aggregate.py
import numpy as np
from openvox.enroll.aggregate import l2_normalize, cosine, robust_centroid

def test_l2_normalize_and_cosine():
    v = np.array([3.0, 4.0])
    n = l2_normalize(v)
    assert np.isclose(np.linalg.norm(n), 1.0)
    assert np.isclose(cosine(np.array([1.0, 0.0]), np.array([2.0, 0.0])), 1.0)

def test_robust_centroid_drops_outlier():
    # three tight vectors + one far outlier
    base = np.array([1.0, 0.0, 0.0])
    embs = np.stack([
        base + 0.01, base - 0.01, base,
        np.array([0.0, 1.0, 0.0]),   # outlier
    ])
    centroid, kept, rep = robust_centroid(embs, outlier_threshold=0.6)
    assert kept[3] == False            # outlier dropped
    assert kept[:3].all()
    assert np.isclose(np.linalg.norm(centroid), 1.0)
    assert rep in (0, 1, 2)

def test_robust_centroid_never_drops_all():
    embs = np.stack([np.array([1.0, 0.0]), np.array([-1.0, 0.0])])
    centroid, kept, rep = robust_centroid(embs, outlier_threshold=0.99)
    assert kept.all()                  # would drop everything -> keep all
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/enroll/test_aggregate.py -v`
Expected: FAIL with `ModuleNotFoundError: openvox.enroll.aggregate`

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/enroll/aggregate.py
import numpy as np

def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, 1e-9, None)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / denom)

def robust_centroid(embs: np.ndarray, outlier_threshold: float):
    unit = l2_normalize(np.asarray(embs, dtype=np.float32), axis=1)   # (N, D)
    initial = l2_normalize(unit.mean(axis=0))
    sims = unit @ initial
    kept = sims >= outlier_threshold
    if not kept.any():
        kept = np.ones(len(unit), dtype=bool)   # never drop everything
    centroid = l2_normalize(unit[kept].mean(axis=0))
    # representative = surviving clip closest to the final centroid
    surviving_idx = np.where(kept)[0]
    rep = int(surviving_idx[np.argmax(unit[kept] @ centroid)])
    return centroid.astype(np.float32), kept, rep
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/enroll/test_aggregate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/enroll/aggregate.py tests/enroll/test_aggregate.py
git commit -m "feat(enroll): add Stage A robust speaker-embedding centroid"
```

---

### Task 4: Scorer — generate probes → embed → similarity (scorer.py)

**Files:**
- Create: `openvox/enroll/scorer.py`
- Test: `tests/enroll/test_scorer.py`

**Interfaces:**
- Produces: `score_embedding(backend, emb, reference, target_centroid, probes, realizability_lambda, clip_embs, exaggeration, seed) -> float`. Builds `conds = backend.make_conditionals(emb, reference, exaggeration)`; for each probe text: `wav, sr = backend.generate(conds, probe, seed)`, `out = backend.ve_embed(wav, sr)`, collect `cosine(out, target_centroid)`; the score is `mean(cosines) - realizability_lambda * min_euclidean(l2_normalize(emb), l2_normalize(clip_embs rows))`.
- Consumes: `l2_normalize`, `cosine` from `openvox.enroll.aggregate`. A `backend` object exposing `make_conditionals(emb, reference, exaggeration)`, `generate(conds, text, seed) -> (wav, sr)`, `ve_embed(wav, sr) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

```python
# tests/enroll/test_scorer.py
import numpy as np
from openvox.enroll.scorer import score_embedding

class FakeBackend:
    """generate() echoes the conditioning; ve_embed() returns it -> score == cos(emb, target)."""
    def make_conditionals(self, emb, reference, exaggeration):
        return np.asarray(emb, dtype=np.float32)
    def generate(self, conds, text, seed):
        return conds, 16000
    def ve_embed(self, wav, sr):
        return np.asarray(wav, dtype=np.float32)

def test_closer_embedding_scores_higher():
    target = np.array([1.0, 0.0, 0.0])
    clips = np.stack([target, target])           # on-manifold
    be = FakeBackend()
    kw = dict(reference=None, target_centroid=target, probes=["hi"],
              realizability_lambda=0.0, clip_embs=clips, exaggeration=0.5, seed=0)
    near = score_embedding(be, np.array([0.9, 0.1, 0.0]), **kw)
    far = score_embedding(be, np.array([0.0, 1.0, 0.0]), **kw)
    assert near > far

def test_realizability_penalty_reduces_offmanifold_score():
    target = np.array([1.0, 0.0, 0.0])
    clips = np.stack([target])
    be = FakeBackend()
    emb = np.array([1.0, 0.0, 0.0])
    no_pen = score_embedding(be, emb, reference=None, target_centroid=target,
                             probes=["hi"], realizability_lambda=0.0,
                             clip_embs=np.stack([np.array([0.0, 1.0, 0.0])]),
                             exaggeration=0.5, seed=0)
    with_pen = score_embedding(be, emb, reference=None, target_centroid=target,
                               probes=["hi"], realizability_lambda=1.0,
                               clip_embs=np.stack([np.array([0.0, 1.0, 0.0])]),
                               exaggeration=0.5, seed=0)
    assert with_pen < no_pen
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/enroll/test_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: openvox.enroll.scorer`

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/enroll/scorer.py
import numpy as np

from openvox.enroll.aggregate import l2_normalize, cosine

def score_embedding(backend, emb, reference, target_centroid, probes,
                    realizability_lambda, clip_embs, exaggeration, seed) -> float:
    conds = backend.make_conditionals(emb, reference, exaggeration)
    sims = []
    for text in probes:
        wav, sr = backend.generate(conds, text, seed)
        out = backend.ve_embed(wav, sr)
        sims.append(cosine(out, target_centroid))
    similarity = float(np.mean(sims))
    unit_emb = l2_normalize(np.asarray(emb, dtype=np.float32))
    unit_clips = l2_normalize(np.asarray(clip_embs, dtype=np.float32), axis=1)
    penalty = float(np.min(np.linalg.norm(unit_clips - unit_emb, axis=1)))
    return similarity - realizability_lambda * penalty
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/enroll/test_scorer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/enroll/scorer.py tests/enroll/test_scorer.py
git commit -m "feat(enroll): add speaker-similarity scorer with realizability penalty"
```

---

### Task 5: Stage B optimizer — coordinate ascent (search.py)

**Files:**
- Create: `openvox/enroll/search.py`
- Test: `tests/enroll/test_search.py`

**Interfaces:**
- Produces: `maximize(score_fn, x0, max_evals, seed=0, init_step=0.5, min_step=0.03) -> tuple[np.ndarray, float]`. Gradient-free coordinate ascent: from `x0`, try `±step` on each coordinate (in a seeded random order), keep any improvement, halve the step when a full pass yields no improvement, stop at `max_evals` evaluations or when `step < min_step`. Returns the best `x` (a numpy array) and its score. `score_fn(x: np.ndarray) -> float`.
- Consumes: nothing (pure).

- [ ] **Step 1: Write the failing test**

```python
# tests/enroll/test_search.py
import numpy as np
from openvox.enroll.search import maximize

def test_maximize_improves_toward_optimum():
    target = np.array([0.3, -0.7, 0.5])
    def score_fn(x):
        return -float(np.sum((x - target) ** 2))   # optimum at x == target
    x0 = np.zeros(3)
    best_x, best_score = maximize(score_fn, x0, max_evals=200, seed=0)
    assert best_score > score_fn(x0)               # strictly better than start
    assert np.linalg.norm(best_x - target) < np.linalg.norm(x0 - target)

def test_maximize_respects_eval_budget():
    calls = {"n": 0}
    def score_fn(x):
        calls["n"] += 1
        return -float(np.sum(x ** 2))
    maximize(score_fn, np.zeros(4), max_evals=10, seed=0)
    assert calls["n"] <= 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/enroll/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: openvox.enroll.search`

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/enroll/search.py
import numpy as np

def maximize(score_fn, x0, max_evals, seed=0, init_step=0.5, min_step=0.03):
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=np.float64).copy()
    best = score_fn(x)
    evals = 1
    step = init_step
    dim = x.size
    while evals < max_evals and step >= min_step:
        improved = False
        for i in rng.permutation(dim):
            if evals >= max_evals:
                break
            for sign in (1.0, -1.0):
                if evals >= max_evals:
                    break
                cand = x.copy()
                cand[i] += sign * step
                s = score_fn(cand)
                evals += 1
                if s > best:
                    best, x, improved = s, cand, True
                    break
        if not improved:
            step *= 0.5
    return x, best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/enroll/test_search.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/enroll/search.py tests/enroll/test_search.py
git commit -m "feat(enroll): add gradient-free coordinate-ascent optimizer"
```

---

### Task 6: Clip prep — gate + segment (prep.py)

**Files:**
- Create: `openvox/enroll/prep.py`
- Test: `tests/enroll/test_prep.py`

**Interfaces:**
- Produces:
  - `rms(a: np.ndarray) -> float`.
  - `passes_gate(a: np.ndarray, sr: int, min_rms: float, min_dur_s: float) -> bool`.
  - `segment(a: np.ndarray, sr: int, max_clip_s: float, min_rms: float) -> list[np.ndarray]` — returns `[a]` if `a` is within `max_clip_s`; otherwise splits into consecutive windows of `max_clip_s` seconds and keeps only windows whose `rms >= min_rms`.
  - `prepare(paths, *, enhance, device, min_rms, min_dur_s, max_clip_s) -> tuple[list[np.ndarray], list[np.ndarray]]` — full loader (integration-exercised): decode each path to mono float32, optionally enhance (graceful), segment, gate, resample survivors to 16 kHz and 24 kHz. Returns parallel lists `(clips_16k, clips_24k)`.
- Consumes: nothing pure; lazy `soundfile`/`librosa`/`openvox.enhance` inside `prepare` only.

- [ ] **Step 1: Write the failing test**

```python
# tests/enroll/test_prep.py
import numpy as np
from openvox.enroll.prep import rms, passes_gate, segment

def test_rms_and_gate():
    loud = 0.5 * np.ones(16000, dtype=np.float32)
    quiet = 1e-4 * np.ones(16000, dtype=np.float32)
    assert rms(loud) > 0.4
    assert passes_gate(loud, 16000, min_rms=0.01, min_dur_s=0.5) is True
    assert passes_gate(quiet, 16000, min_rms=0.01, min_dur_s=0.5) is False   # too quiet
    assert passes_gate(loud[:4000], 16000, min_rms=0.01, min_dur_s=0.5) is False  # too short

def test_segment_splits_long_and_drops_silence():
    sr = 16000
    short = 0.5 * np.ones(sr * 3, dtype=np.float32)
    assert len(segment(short, sr, max_clip_s=12.0, min_rms=0.01)) == 1
    # 20s: 12s loud + 8s silence -> two windows, silent one dropped
    long = np.concatenate([0.5 * np.ones(sr * 12, dtype=np.float32),
                           np.zeros(sr * 8, dtype=np.float32)])
    segs = segment(long, sr, max_clip_s=12.0, min_rms=0.01)
    assert len(segs) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/enroll/test_prep.py -v`
Expected: FAIL with `ModuleNotFoundError: openvox.enroll.prep`

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/enroll/prep.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/enroll/test_prep.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/enroll/prep.py tests/enroll/test_prep.py
git commit -m "feat(enroll): add clip prep (gate + segment + resample)"
```

---

### Task 7: EnrollBackend ABC + ChatterboxEnrollBackend + import guard

**Files:**
- Create: `openvox/enroll/backend.py`
- Create: `openvox/enroll/chatterbox_backend.py`
- Test: `tests/enroll/test_backend.py`

**Interfaces:**
- Produces:
  - `EnrollBackend` ABC with abstract methods: `embed_clips(wavs_16k: list) -> np.ndarray` `(N,256)`; `reference_from_clip(clip_16k, clip_24k) -> object`; `make_conditionals(speaker_emb, reference, exaggeration) -> object`; `generate(conditionals, text, seed) -> tuple[np.ndarray, int]`; `ve_embed(wav, sr) -> np.ndarray`; and properties `cuda_available -> bool`, `model_sr -> int`.
  - `ChatterboxEnrollBackend(device="cuda")` implementing it, importing torch/chatterbox lazily.
- Consumes: nothing from earlier enroll tasks (the engine wires it to the algorithm modules).

- [ ] **Step 1: Write the failing test**

```python
# tests/enroll/test_backend.py
import subprocess
import sys

from openvox.enroll.backend import EnrollBackend

def test_abc_cannot_instantiate():
    import pytest
    with pytest.raises(TypeError):
        EnrollBackend()

def test_backend_module_is_torch_free_at_import():
    # importing the ABC + concrete backend module must not require torch
    code = (
        "import sys\n"
        "for m in ('torch', 'chatterbox'):\n"
        "    sys.modules[m] = None\n"
        "import openvox.enroll.backend\n"
        "import openvox.enroll.chatterbox_backend\n"
        "from openvox.enroll.chatterbox_backend import ChatterboxEnrollBackend\n"
        "ChatterboxEnrollBackend(device='cpu')\n"   # construct without loading
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/enroll/test_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: openvox.enroll.backend`

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/enroll/backend.py
from abc import ABC, abstractmethod

import numpy as np

class EnrollBackend(ABC):
    @abstractmethod
    def embed_clips(self, wavs_16k: list) -> np.ndarray:
        """Voice-encoder embeddings for a list of 16 kHz mono clips -> (N, 256)."""
        raise NotImplementedError

    @abstractmethod
    def reference_from_clip(self, clip_16k, clip_24k):
        """Build the s3gen reference + cond-prompt tokens for one clip."""
        raise NotImplementedError

    @abstractmethod
    def make_conditionals(self, speaker_emb, reference, exaggeration):
        """Assemble a Chatterbox Conditionals from a speaker embedding + reference."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, conditionals, text, seed):
        """Generate speech for text under conditionals -> (wav float32, sample_rate)."""
        raise NotImplementedError

    @abstractmethod
    def ve_embed(self, wav, sr) -> np.ndarray:
        """Voice-encoder embedding of a generated waveform -> (256,)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def cuda_available(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_sr(self) -> int:
        raise NotImplementedError
```

```python
# openvox/enroll/chatterbox_backend.py
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
        return torch.cuda.is_available()

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
        wav = m.generate(text)
        return wav.squeeze(0).detach().cpu().numpy().astype(np.float32), int(m.sr)

    def ve_embed(self, wav, sr):
        self._ensure_loaded()
        w = np.asarray(wav, dtype=np.float32)
        if sr != VE_SR:
            import librosa
            w = librosa.resample(w, orig_sr=sr, target_sr=VE_SR).astype(np.float32)
        emb = self._model.ve.embeds_from_wavs([w], sample_rate=VE_SR)
        return np.asarray(emb, dtype=np.float32).reshape(-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/enroll/test_backend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/enroll/backend.py openvox/enroll/chatterbox_backend.py tests/enroll/test_backend.py
git commit -m "feat(enroll): add EnrollBackend ABC and Chatterbox backend"
```

---

### Task 8: VoiceEnrollEngine orchestration + import-lean guard

**Files:**
- Create: `openvox/enroll/engine.py`
- Modify: `openvox/enroll/__init__.py` (add the exports deferred from Task 1)
- Test: `tests/enroll/test_engine.py`, `tests/enroll/test_import_lean.py`

**Interfaces:**
- Produces: `VoiceEnrollEngine(device=None, quality=None, config=None)` with `enroll(clips: list[str]) -> VoiceProfile`. Also `DEFAULT_PROBES: list[str]` (module-level, ~6 phonetically-varied sentences).
- Consumes: `EnrollConfig`, `evals_for_quality` (Task 1); `VoiceProfile` (Task 2); `robust_centroid`, `l2_normalize` (Task 3); `score_embedding` (Task 4); `maximize` (Task 5); `prepare` (Task 6); `ChatterboxEnrollBackend` (Task 7).

- [ ] **Step 1: Write the failing test**

The engine is unit-tested with a fake backend and monkeypatched `prepare`, so no torch is needed. It must run Stage A always, run Stage B only when `cuda_available`, and never ship a profile worse than Stage A.

```python
# tests/enroll/test_engine.py
import numpy as np
import pytest

from openvox.enroll import VoiceEnrollEngine, VoiceProfile
from openvox.enroll.config import EnrollConfig

class FakeBackend:
    def __init__(self, cuda):
        self._cuda = cuda
    @property
    def cuda_available(self):
        return self._cuda
    @property
    def model_sr(self):
        return 24000
    def embed_clips(self, wavs_16k):
        # two identical clips pointing at a fixed identity
        return np.stack([np.array([1.0, 0.0, 0.0], dtype=np.float32)] * len(wavs_16k))
    def reference_from_clip(self, clip_16k, clip_24k):
        return {"gen": "ref", "cond_prompt": None}
    def make_conditionals(self, speaker_emb, reference, exaggeration):
        return np.asarray(speaker_emb, dtype=np.float32)
    def generate(self, conds, text, seed):
        return conds, 16000
    def ve_embed(self, wav, sr):
        return np.asarray(wav, dtype=np.float32)

@pytest.fixture
def patched_prep(monkeypatch):
    def fake_prepare(paths, **kw):
        one = np.ones(16000, dtype=np.float32)
        return [one, one], [one, one]      # clips_16k, clips_24k
    monkeypatch.setattr("openvox.enroll.engine.prepare", fake_prepare)

def _engine(cuda, patched_prep):
    eng = VoiceEnrollEngine(device="cpu", config=EnrollConfig(enhance_clips=False, seed=0))
    eng._backend = FakeBackend(cuda=cuda)
    return eng

def test_enroll_stage_a_only_without_gpu(patched_prep):
    eng = _engine(cuda=False, patched_prep=patched_prep)
    prof = eng.enroll(["a.wav"])
    assert isinstance(prof, VoiceProfile)
    assert prof.metadata["stage"] == "A"
    assert prof.score > 0.9                    # matches the fixed identity

def test_enroll_runs_stage_b_with_gpu(patched_prep):
    eng = _engine(cuda=True, patched_prep=patched_prep)
    prof = eng.enroll(["a.wav"])
    assert prof.metadata["stage"] in ("A", "B")   # B only ships if it wins
    assert prof.score >= 0.9

def test_enroll_requires_survivors(patched_prep, monkeypatch):
    monkeypatch.setattr("openvox.enroll.engine.prepare", lambda paths, **kw: ([], []))
    eng = _engine(cuda=False, patched_prep=patched_prep)
    with pytest.raises(ValueError, match="clip"):
        eng.enroll(["a.wav"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/enroll/test_engine.py -v`
Expected: FAIL with `ImportError`/`ModuleNotFoundError` for `VoiceEnrollEngine`

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/enroll/engine.py
import dataclasses
import hashlib
import logging
import os

import numpy as np

from openvox.enroll.aggregate import l2_normalize, robust_centroid
from openvox.enroll.backend import EnrollBackend
from openvox.enroll.chatterbox_backend import ChatterboxEnrollBackend
from openvox.enroll.config import EnrollConfig, evals_for_quality
from openvox.enroll.prep import prepare
from openvox.enroll.profile import VoiceProfile
from openvox.enroll.scorer import score_embedding
from openvox.enroll.search import maximize

log = logging.getLogger(__name__)

DEFAULT_PROBES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells sea shells by the shore on a bright morning.",
    "How much wood would a woodchuck chuck if it could?",
    "We drove through six zigzagging mountain villages.",
    "Please call Stella and ask her to bring these things.",
    "Authentic voices carry warmth, weight, and quiet detail.",
]

def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)

class VoiceEnrollEngine:
    def __init__(self, device: str | None = None, quality: str | None = None,
                 config: EnrollConfig | None = None) -> None:
        c = dataclasses.replace(config) if config is not None else EnrollConfig()
        if device is not None:
            c.device = device
        if quality is not None:
            c.quality = quality
        self._config = c
        self._backend: EnrollBackend = ChatterboxEnrollBackend(device=c.device)

    def enroll(self, clips: list[str]) -> VoiceProfile:
        if not clips:
            raise ValueError("provide at least one audio clip to enroll")
        for p in clips:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"clip not found: {p}")
            if os.path.getsize(p) == 0:
                raise ValueError(f"clip is empty: {p}")

        c = self._config
        probes = c.probes if c.probes else DEFAULT_PROBES
        clips_16k, clips_24k = prepare(
            clips, enhance=c.enhance_clips, device=c.device,
            min_rms=c.min_rms, min_dur_s=c.min_dur_s, max_clip_s=c.max_clip_s,
        )
        if len(clips_16k) < c.min_clips:
            raise ValueError(
                f"only {len(clips_16k)} usable clip(s) after prep; need >= {c.min_clips}"
            )

        embs = self._backend.embed_clips(clips_16k)                 # (N, 256)
        centroid, kept, rep = robust_centroid(embs, c.outlier_threshold)
        kept_embs = l2_normalize(embs[kept], axis=1)
        reference = self._backend.reference_from_clip(clips_16k[rep], clips_24k[rep])

        def score_of(emb):
            return score_embedding(
                self._backend, emb, reference, centroid, probes,
                c.realizability_lambda, kept_embs, c.exaggeration, c.seed,
            )

        score_a = score_of(centroid)
        final_emb, final_score, stage = centroid, score_a, "A"

        if self._backend.cuda_available:
            max_evals = c.max_evals or evals_for_quality(c.quality)
            def weight_score(w):
                emb = l2_normalize(kept_embs.T @ _softmax(w))
                return score_of(emb)
            best_w, score_b = maximize(
                weight_score, np.zeros(len(kept_embs)), max_evals=max_evals, seed=c.seed,
            )
            if score_b >= score_a + c.accept_margin:
                final_emb = l2_normalize(kept_embs.T @ _softmax(best_w))
                final_score, stage = score_b, "B"
        else:
            log.info("No CUDA; enrolling with Stage A only (the search needs a GPU).")

        conds = self._backend.make_conditionals(final_emb, reference, c.exaggeration)
        metadata = {
            "stage": stage,
            "algorithm": "embedding-optimization",
            "schema_note": "openvox.enroll 2B.2b",
            "sample_rate": self._backend.model_sr,
            "n_clips_used": int(kept.sum()),
            "clip_sha1": [self._sha1(p) for p in clips],
        }
        return VoiceProfile(conditionals=conds, score=float(final_score), metadata=metadata)

    @staticmethod
    def _sha1(path: str) -> str:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
```

Then update `openvox/enroll/__init__.py` to the full export form:

```python
# openvox/enroll/__init__.py
"""OpenVox per-voice enrollment engine (embedding optimization)."""
from openvox.enroll.engine import VoiceEnrollEngine
from openvox.enroll.profile import VoiceProfile

__all__ = ["VoiceEnrollEngine", "VoiceProfile"]
```

Also add the import-lean guard test:

```python
# tests/enroll/test_import_lean.py
import subprocess
import sys

def test_import_openvox_enroll_without_heavy_deps():
    code = (
        "import sys\n"
        "for m in ('torch', 'chatterbox', 'librosa', 'soundfile', 'resemble_enhance'):\n"
        "    sys.modules[m] = None\n"
        "import openvox\n"
        "import openvox.enroll\n"
        "from openvox.enroll import VoiceEnrollEngine, VoiceProfile\n"
        "VoiceEnrollEngine(device='cpu')\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/enroll/test_engine.py tests/enroll/test_import_lean.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/enroll/engine.py openvox/enroll/__init__.py tests/enroll/test_engine.py tests/enroll/test_import_lean.py
git commit -m "feat(enroll): add VoiceEnrollEngine orchestration (Stage A + Stage B)"
```

---

### Task 9: Clone-from-profile integration

**Files:**
- Modify: `openvox/clone/chatterbox_backend.py` (add `clone_from_profile`)
- Modify: `openvox/clone/engine.py` (add `profile` param + mutual exclusion)
- Test: `tests/clone/test_profile_clone.py`

**Interfaces:**
- Consumes: `VoiceProfile` (Task 2), whose `.conditionals` is a Chatterbox `Conditionals` with a `.to(device)` method.
- Produces: `ChatterboxBackend.clone_from_profile(text: str, conditionals, exaggeration: float, cfg: float) -> TTSResult`; `VoiceCloneEngine.clone(text, reference_audio=None, exaggeration=None, cfg=None, enhance=None, profile=None) -> TTSResult`. `profile` accepts a `VoiceProfile`, a path string, or `os.PathLike`. `profile` and `reference_audio` are mutually exclusive.

- [ ] **Step 1: Write the failing test**

```python
# tests/clone/test_profile_clone.py
import numpy as np
import pytest

from openvox.clone import VoiceCloneEngine
from openvox.enroll import VoiceProfile

class _Conds:
    def to(self, device):
        return self

def test_profile_and_reference_are_mutually_exclusive():
    eng = VoiceCloneEngine(device="cpu")
    with pytest.raises(ValueError, match="either"):
        eng.clone("hi", reference_audio="ref.wav", profile=VoiceProfile(_Conds(), 0.9, {}))

def test_clone_from_profile_injects_conditionals(monkeypatch):
    eng = VoiceCloneEngine(device="cpu")
    captured = {}
    from openvox.tts.backend import TTSResult
    def fake_cfp(text, conditionals, exaggeration, cfg):
        captured["conds"] = conditionals
        captured["text"] = text
        return TTSResult(np.zeros(10, dtype=np.float32), 24000)
    monkeypatch.setattr(eng._backend, "clone_from_profile", fake_cfp)
    conds = _Conds()
    out = eng.clone("speak this", profile=VoiceProfile(conds, 0.9, {}))
    assert out.sample_rate == 24000
    assert captured["conds"] is conds and captured["text"] == "speak this"

def test_clone_requires_a_source():
    eng = VoiceCloneEngine(device="cpu")
    with pytest.raises(ValueError, match="reference_audio or profile"):
        eng.clone("hi")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/clone/test_profile_clone.py -v`
Expected: FAIL — `clone()` has no `profile` parameter / no `clone_from_profile`.

- [ ] **Step 3: Write minimal implementation**

Add to `openvox/clone/chatterbox_backend.py` (new method on `ChatterboxBackend`):

```python
    def clone_from_profile(self, text: str, conditionals, exaggeration: float,
                           cfg: float) -> TTSResult:
        self._ensure_loaded()
        self._model.conds = conditionals.to(self._model.device)
        wav = self._model.generate(text, exaggeration=exaggeration, cfg_weight=cfg)
        audio = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return TTSResult(audio=audio, sample_rate=int(self._model.sr))
```

Rewrite `VoiceCloneEngine.clone` in `openvox/clone/engine.py`:

```python
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
```

Note: `say()` still calls `clone(text, reference_audio, …)` positionally — unaffected because `reference_audio` remains the second positional parameter.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/clone/test_profile_clone.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/clone/chatterbox_backend.py openvox/clone/engine.py tests/clone/test_profile_clone.py
git commit -m "feat(clone): clone from a saved voice profile (profile= param)"
```

---

### Task 10: Enroll demo CLI + packaging + clone demo `--profile`

**Files:**
- Create: `openvox/enroll/demo.py`
- Modify: `openvox/clone/demo.py` (add `--profile`, make `--ref` optional)
- Modify: `pyproject.toml` (`[enroll]` extra + `openvox-enroll-demo` entry point)
- Test: `tests/enroll/test_demo.py`, `tests/clone/test_demo.py` (extend)

**Interfaces:**
- Produces: `openvox/enroll/demo.py` with `build_parser() -> argparse.ArgumentParser` and `main(argv=None) -> int`. Clone demo `build_parser` gains `--profile`.

- [ ] **Step 1: Write the failing test**

```python
# tests/enroll/test_demo.py
from openvox.enroll.demo import build_parser

def test_enroll_parser_defaults():
    args = build_parser().parse_args(["--in", "a.wav", "b.wav", "--out", "v.ovx"])
    assert args.input == ["a.wav", "b.wav"]
    assert args.output == "v.ovx"
    assert args.quality == "balanced"
    assert args.device == "cuda"
    assert args.enhance is True

def test_enroll_parser_no_enhance_and_quality():
    args = build_parser().parse_args(
        ["--in", "a.wav", "--out", "v.ovx", "--quality", "thorough", "--no-enhance"])
    assert args.quality == "thorough" and args.enhance is False
```

Extend the clone demo test:

```python
# tests/clone/test_demo.py  (add this test)
from openvox.clone.demo import build_parser as clone_parser

def test_clone_parser_accepts_profile():
    args = clone_parser().parse_args(["--text", "hi", "--profile", "alice.ovx"])
    assert args.profile == "alice.ovx"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/enroll/test_demo.py tests/clone/test_demo.py -v`
Expected: FAIL — `openvox.enroll.demo` missing; clone parser has no `--profile`.

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/enroll/demo.py
import argparse
import sys

from openvox.enroll import VoiceEnrollEngine

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="openvox-enroll-demo",
        description="Enroll a voice from clips into a reusable .ovx profile.")
    p.add_argument("--in", dest="input", nargs="+", required=True,
                   help="One or more clips (or a long recording) of the target voice.")
    p.add_argument("--out", dest="output", required=True, help="Output .ovx profile path.")
    p.add_argument("--quality", default="balanced",
                   choices=["fast", "balanced", "thorough"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-enhance", dest="enhance", action="store_false",
                   help="Skip reference-clip enhancement.")
    p.set_defaults(enhance=True)
    return p

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = VoiceEnrollEngine(device=args.device, quality=args.quality)
    engine._config.enhance_clips = args.enhance
    profile = engine.enroll(args.input)
    profile.save(args.output)
    print(f"Saved {args.output} (stage {profile.metadata['stage']}, "
          f"speaker-similarity {profile.score:.3f})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

In `openvox/clone/demo.py`: make `--ref` optional and add `--profile`, and pass `profile` through to `clone()`/`say()`. Find the `--ref` argument (currently `required=True`) and change it, then add:

```python
    p.add_argument("--ref", help="Reference audio (any format). Omit if using --profile.")
    p.add_argument("--profile", help="A saved .ovx voice profile (mutually exclusive with --ref).")
```

And in the clone demo's `main`, replace the clone call so it forwards `profile=args.profile` (and passes `reference_audio=args.ref`); when `--profile` is set, playback/`say` uses the profile path. Keep the existing `--out`/`--no-play` behavior.

Add the `[enroll]` extra and entry point to `pyproject.toml`:

```toml
clone    = ["chatterbox-tts>=0.1", "sounddevice>=0.4.6"]
enhance  = ["torch", "torchaudio", "matplotlib", "omegaconf", "pandas", "celluloid", "resampy", "soundfile", "rich", "tabulate"]
enroll   = ["openvox[clone,enhance]"]
dev      = ["pytest>=8.0"]
all      = ["openvox[stt,stt-demo,tts]"]
```

```toml
[project.scripts]
openvox-stt-demo = "openvox.stt.demo:main"
openvox-tts-demo = "openvox.tts.demo:main"
openvox-clone-demo = "openvox.clone.demo:main"
openvox-enhance-demo = "openvox.enhance.demo:main"
openvox-enroll-demo = "openvox.enroll.demo:main"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/enroll/test_demo.py tests/clone/test_demo.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/enroll/demo.py openvox/clone/demo.py pyproject.toml tests/enroll/test_demo.py tests/clone/test_demo.py
git commit -m "feat(enroll): add enroll demo CLI, [enroll] extra, clone --profile"
```

---

### Task 11: Real end-to-end integration test

**Files:**
- Test: `tests/enroll/test_integration.py`

**Interfaces:**
- Consumes: the whole stack (`VoiceEnrollEngine`, `VoiceProfile`, `VoiceCloneEngine`). Marked `@pytest.mark.integration` — downloads the Chatterbox model and really generates audio (GPU strongly preferred; skips if no reference clips are present).

- [ ] **Step 1: Write the test**

```python
# tests/enroll/test_integration.py
import glob
import os

import numpy as np
import pytest

pytestmark = pytest.mark.integration

REF_CANDIDATES = sorted(glob.glob("reference_hq.wav") + glob.glob("*.wav"))

@pytest.mark.skipif(not REF_CANDIDATES, reason="no local reference wav to enroll from")
def test_enroll_then_clone_from_profile(tmp_path):
    from openvox.enroll import VoiceEnrollEngine, VoiceProfile
    from openvox.clone import VoiceCloneEngine

    clips = REF_CANDIDATES[:2] or REF_CANDIDATES
    eng = VoiceEnrollEngine(device="cuda", quality="fast")
    profile = eng.enroll(clips)
    assert 0.0 < profile.score <= 1.0
    assert profile.metadata["stage"] in ("A", "B")

    out = tmp_path / "voice.ovx"
    profile.save(str(out))
    reloaded = VoiceProfile.load(str(out))
    assert reloaded.metadata["stage"] == profile.metadata["stage"]

    clone = VoiceCloneEngine(device="cuda")
    result = clone.clone("This is a profile-based clone.", profile=reloaded)
    assert result.sample_rate == 24000
    assert result.audio.size > 0
    assert float(np.sqrt(np.mean(result.audio ** 2))) > 1e-4   # non-silent
```

- [ ] **Step 2: Run it**

Run: `pytest tests/enroll/test_integration.py -v -m integration`
Expected: PASS on a machine with a CUDA torch build and a local `.wav`. (On CPU it still passes but is slow and runs Stage A only.)

- [ ] **Step 3: Commit**

```bash
git add tests/enroll/test_integration.py
git commit -m "test(enroll): end-to-end enroll -> save -> clone-from-profile integration"
```

---

## Self-Review

**Spec coverage:**
- §3 in-scope `enroll(clips) -> VoiceProfile` → Task 8. Prep/segment/enhance → Task 6. Stage A → Task 3. Stage B (search + scorer) → Tasks 4–5. `VoiceProfile` `.ovx` save/load → Task 2. Clone `profile=` integration → Task 9. Demo + `[enroll]` extra + clone `--profile` → Task 10. Tests (unit no-torch, import guards, heavy integration) → every task + Tasks 7/8/11. ✓
- §7 config fields → Task 1 (all present, incl. `outlier_threshold`, `realizability_lambda`, `accept_margin`, `seed`, gate params). ✓
- §8 error handling: no-GPU→Stage A (Task 8), Stage B beats A by margin (Task 8), too-few clips `ValueError` (Task 8), `profile`+`reference_audio` mutual exclusion (Task 9), `.ovx` schema mismatch (Task 2), enhance-missing graceful (Task 6), missing/empty file (Tasks 8/9). ✓
- §4 boundaries: only `chatterbox_backend.py` imports torch/chatterbox; algorithm modules pure numpy; import-lean enforced (Tasks 7/8). ✓
- §10 packaging: `[enroll] = openvox[clone,enhance]`, out of `all`, entry point (Task 10). ✓

**Placeholder scan:** No TBD/TODO; every code and test step is concrete. ✓

**Type consistency:** `robust_centroid` returns `(centroid, kept_mask, rep)` used exactly so in Task 8. `score_embedding(backend, emb, reference, target_centroid, probes, realizability_lambda, clip_embs, exaggeration, seed)` — same argument order in Tasks 4 and 8. `maximize(score_fn, x0, max_evals, seed=...)` — same in Tasks 5 and 8. `make_conditionals(speaker_emb, reference, exaggeration)` / `generate(conditionals, text, seed)` / `ve_embed(wav, sr)` — consistent across Tasks 4, 7, 8. `VoiceProfile(conditionals, score, metadata)` — consistent Tasks 2, 8, 9, 11. `clone_from_profile(text, conditionals, exaggeration, cfg)` — consistent Tasks 9 and its test. ✓

**Note on structure:** `prep.py` is added beyond the spec's §4 file list — a deliberate decomposition so `engine.py` stays focused (the spec's §6 folds prep into the engine; splitting it keeps each file single-purpose per the plan's file-structure guidance). All other files match §4.
