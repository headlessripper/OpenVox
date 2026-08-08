# OpenVox — Per-Voice Enrollment / Fine-Tuning Engine (Sub-project 2B.2b) — Design

**Date:** 2026-08-08
**Status:** Approved (design), proceeding to implementation plan
**Scope:** Turn a set of clips of a target voice into a saved, reusable **voice
profile** that clones that voice with materially higher fidelity than single-shot
zero-shot cloning — via **embedding optimization** (no model-weight training). The
profile is a Chatterbox `Conditionals` blob optimized by a speaker-similarity
objective. Delivered as a standalone `openvox.enroll` engine that the voice cloner
can clone *from*.

---

## 1. Goal & positioning

Zero-shot cloning (2B) is only as good as one reference clip. This engine pushes
toward the "indistinguishable from the original" ceiling by **enrolling** a voice:
it ingests several clips, builds a robust speaker representation, and then
**optimizes the cloning conditioning to maximize measured speaker similarity**
between what the model actually generates and the real voice. The result is a
per-voice profile (`.ovx`) that the cloner loads to speak any text in that voice —
no reference clip at generation time, higher and more consistent fidelity.

It reuses, rather than duplicates, the shipped engines: `openvox.enhance` cleans
the input clips; the Chatterbox model from `openvox.clone` is the generator and
its voice encoder is the similarity scorer.

## 2. Decisions locked during brainstorming

| Decision | Choice |
| --- | --- |
| Mechanism | **Embedding optimization** (per-voice conditioning), not model-weight fine-tuning |
| Algorithm | **B built on A**: robust multi-clip centroid (A) → gradient-free similarity-guided search (B) |
| Feasibility | **Proven by live spike** (see §3) — conditioning is settable, saveable/loadable, and the similarity objective is measurable |
| Artifact | A saved **`VoiceProfile` (`.ovx`)** = optimized `Conditionals` + metadata + achieved score |
| Structure | Standalone `openvox/enroll/` engine; `VoiceCloneEngine` learns to clone from a profile |
| Transcripts | **Not required** — the objective is text-independent (voice-encoder embeddings) |
| Probes | Baked-in default set (~6 phonetically-varied sentences); optional `probes=` override |
| Search budget | `quality` knob → `fast`/`balanced`/`thorough` ≈ 15/40/100 evals (default `balanced`) |
| Safety net | Stage B ships **only if it beats Stage A** by a margin; else fall back to A |
| Hardware | Windows x86 + CUDA (torch); CPU falls back to **Stage A only** (search too slow on CPU) |

**Feasibility spike result (live, on this machine).** Using Chatterbox directly:
`Conditionals.save()/.load()` roundtrips a `.pt` file (per-voice profiles are
persistable); `model.conds` is a settable attribute (conditioning is injectable);
`ve.embeds_from_wavs([clip1, clip2, …])` embeds a **list** of clips and averaging
them yields one 256-d speaker vector (multi-clip aggregation works); generating
from an injected/aggregated profile produced real audio; and the optimization
objective is real — cosine similarity between the voice-encoder embedding of the
*generated* speech and the target centroid measured **0.838** at baseline. Full
weight fine-tuning is intentionally rejected: Chatterbox's autoregressive, discrete
T3 token sampling is not end-to-end differentiable, so gradient descent on a
reconstruction loss is not tractable — a gradient-free search over the conditioning
is the correct tool.

## 3. Scope

### In scope

- `openvox/enroll/` with a `VoiceEnrollEngine` façade: `enroll(clips) -> VoiceProfile`.
- Input prep: accept any format/rate; auto-segment a long recording into clips;
  clean each clip via `openvox.enhance`; quality-gate and drop bad clips.
- **Stage A** — robust multi-clip speaker-embedding centroid with outlier
  rejection; pick the most-representative clip as the s3gen/prompt reference;
  assemble a baseline `Conditionals`.
- **Stage B** — gradient-free, similarity-guided optimization of the speaker
  conditioning (generate probes → embed → score vs. target centroid, with a
  realizability penalty), bounded by an eval budget; keep the running best.
- `VoiceProfile` artifact with `save()/load()` of an `.ovx` file (Conditionals +
  metadata + score).
- Clone integration: `VoiceCloneEngine.clone(text, profile=…)` loads a profile and
  clones from it (mutually exclusive with `reference_audio=`).
- A demo CLI (`python -m openvox.enroll.demo`) and a `--profile` flag on the clone
  demo.
- A `[enroll]` optional extra (composes `[clone,enhance]`), out of `all`.
- Tests: unit without torch/chatterbox; one heavy integration that really enrolls
  and proves a score gain.

### Out of scope (deferred)

- Model-weight fine-tuning / LoRA / training loops (rejected above).
- Optimizing the s3gen reference or cond-prompt tokens via search (Stage B tunes
  the **speaker embedding**; the s3gen/prompt reference is *selected* in Stage A,
  not searched — a possible later extension).
- Cross-model profiles (an `.ovx` is tied to the Chatterbox conditioning schema and
  versioned accordingly).
- A voice-profile hub / sharing UI (H4, later).
- Real-time / streaming enrollment.

## 4. Architecture

### Package layout

```
openvox/enroll/
  __init__.py            # exports VoiceEnrollEngine, VoiceProfile
  config.py              # EnrollConfig (device, quality, max_evals, probes, enhance_clips,
                         #   min_clips, outlier_threshold, realizability_lambda, seed, accept_margin)
  profile.py             # VoiceProfile: wraps Conditionals + metadata; save()/load() an .ovx file
  aggregate.py           # Stage A: robust speaker-embedding centroid + representative-clip pick
  scorer.py              # generate probes -> voice-encoder embed -> cosine sim to target centroid
  search.py              # Stage B: gradient-free optimizer (coordinate ascent default; CMA-ES optional)
  backend.py             # EnrollBackend ABC (model handle: embed, build_conditionals, generate, ve_embed)
  chatterbox_backend.py  # the ONLY file importing torch/chatterbox (lazy, after HF_HUB_DISABLE_SYMLINKS)
  engine.py              # VoiceEnrollEngine.enroll(clips) -> VoiceProfile
  demo.py                # python -m openvox.enroll.demo
tests/enroll/
```

### Components (each independently testable)

- **EnrollBackend (ABC)** + **ChatterboxEnrollBackend** — the only place torch /
  chatterbox load (lazily, inside the model-load method, after setting
  `HF_HUB_DISABLE_SYMLINKS`). Exposes the primitives the algorithm needs, keeping
  the algorithm files torch-free and unit-testable:
  - `embed_clips(wavs_16k) -> np.ndarray (N, 256)` — voice-encoder embeddings.
  - `build_reference(clip_24k) -> gen_ref_dict` — s3gen reference for a chosen clip.
  - `make_conditionals(speaker_emb, gen_ref, cond_prompt) -> Conditionals`.
  - `generate(conditionals, text) -> wav @ model.sr`.
  - `ve_embed(wav, sr) -> np.ndarray (256,)` — embed generated audio for scoring.
  - `resolve_device(requested, cuda_available)`.
- **aggregate.py (Stage A)** — pure numpy: L2-normalize embeddings, compute
  centroid, reject outliers (cosine-to-centroid below `outlier_threshold`),
  recompute; return the centroid and the index of the most-representative clip.
- **scorer.py** — given a candidate speaker embedding and the fixed probe set:
  build `Conditionals`, generate each probe (fixed seed), `ve_embed` the outputs,
  return the mean cosine to the target centroid **minus** the realizability penalty
  `realizability_lambda * dist(candidate, nearest real clip embedding)`.
- **search.py (Stage B)** — gradient-free optimizer over the speaker embedding,
  parameterized as `centroid + perturbation` (or convex blend-weights over the
  per-clip embeddings — lower-dimensional, inherently realizable; the default
  starting parameterization). Coordinate ascent is the zero-dependency default;
  CMA-ES is an optional path if a small library is warranted. Bounded by
  `max_evals`; returns the best-scoring embedding and its score.
- **VoiceProfile (profile.py)** — holds the final `Conditionals` + metadata (source
  clip hashes, achieved score, algorithm + schema version, sample rate, stage that
  won). `save(path)` writes an `.ovx` (torch-saved dict); `load(path)` restores it
  and validates the schema version.
- **VoiceEnrollEngine (engine.py)** — façade: validate inputs → prep+enhance clips →
  Stage A → (GPU? Stage B : skip) → accept best (A or B by `accept_margin`) →
  return `VoiceProfile`. Copies its config.
- **Clone integration** (`openvox/clone/engine.py`): `clone()` gains
  `profile: str | Path | VoiceProfile | None`. When set, it lazily imports
  `openvox.enroll`, loads the profile, injects its `Conditionals` as `model.conds`,
  and generates with no reference/enhancement. `profile` and `reference_audio` are
  mutually exclusive (explicit error if both). The `openvox.enroll` import stays
  lazy inside `clone()` so `import openvox.clone` remains torch-free.

`import openvox`, `import openvox.enroll` stay lean (no torch/chatterbox at import).

## 5. Public API

```python
from openvox.enroll import VoiceEnrollEngine

eng = VoiceEnrollEngine(device="cuda")                  # falls back to CPU (Stage A only if no GPU)
profile = eng.enroll(["a.wav", "b.mp3", "long.m4a"])    # prep -> Stage A -> Stage B
print(profile.score)                                    # achieved speaker-similarity
profile.save("alice.ovx")

from openvox.enroll import VoiceProfile
profile = VoiceProfile.load("alice.ovx")

from openvox.clone import VoiceCloneEngine
clone = VoiceCloneEngine(device="cuda")
clone.clone("Speak any text in this voice.", profile="alice.ovx").save_wav("out.wav")
clone.clone("Zero-shot still works.", reference_audio="ref.wav")   # unchanged path
```

## 6. Data flow

```
enroll(clips):
  1. resolve device; validate at least one input exists
  2. prep: decode any format; segment long files on silence; resample 16k (VE) + 24k (s3gen)
     clean each clip via openvox.enhance (enhance_clips, graceful if deps absent)
     quality-gate; drop bad/short/too-noisy clips; require >= min_clips survivors
  3. Stage A (aggregate.py): VE-embed survivors -> robust centroid (+outlier reject)
     pick most-representative clip -> build_reference; assemble baseline Conditionals
     score baseline -> score_A
  4. GPU present (or forced)?  Stage B (search.py + scorer.py):
       optimize speaker embedding to maximize scorer(); bounded by max_evals
       -> best_emb, score_B
     else: skip (Stage A only)
  5. accept: (score_B present and score_B >= score_A + accept_margin) ? B : A
  6. VoiceProfile(final Conditionals, metadata, winning score) -> return / save(.ovx)

clone(text, profile):
  load .ovx -> validate schema -> model.conds = profile.conditionals -> generate -> TTSResult(24000)
```

## 7. Configuration

`EnrollConfig`: `device="cuda"`, `quality="balanced"`, `max_evals=None` (derived
from `quality`: fast≈15 / balanced≈40 / thorough≈100 unless set explicitly),
`probes=None` (use the baked-in default set), `enhance_clips=True`, `min_clips=1`,
`outlier_threshold=0.6`, `realizability_lambda` (small, tuned so the penalty guards
the manifold without dominating), `seed=0` (fixed per-candidate RNG for comparable
scores), `accept_margin` (small positive; Stage B must clear Stage A by this to
win). `VoiceEnrollEngine(device=…, quality=…, config=…)` overrides fields when
provided; a passed config is copied, not mutated. `CloneConfig`/`clone()` gain the
`profile` parameter.

## 8. Error handling

- `device="cuda"` but no CUDA in torch → CPU with an info notice; Stage B is
  **skipped** (Stage A only) unless explicitly forced, because per-candidate
  generation is too slow on CPU.
- Stage B raises, or fails to beat Stage A by `accept_margin` → ship the Stage A
  profile (the output is **never worse** than the deterministic baseline).
- All clips fail the quality gate / fewer than `min_clips` survivors → clear
  `ValueError` before any model load.
- `clone(profile=…, reference_audio=…)` both provided → explicit `ValueError`.
- `.ovx` schema/version or tensor-shape mismatch on load → clear error naming the
  regenerate-the-profile fix.
- `openvox.enhance` deps missing → clips used raw with a one-line notice (same
  graceful degradation pattern as the cloner's auto-enhance).
- Missing/empty input file → `FileNotFoundError` before any model load.

## 9. Testing

- **Unit (no torch/chatterbox):**
  - `EnrollConfig` defaults and the `quality → max_evals` mapping.
  - `aggregate.py`: robust centroid + outlier rejection on synthetic vectors (a
    planted outlier is dropped; centroid is L2-normalized).
  - `scorer.py`: cosine + realizability-penalty math with a stubbed backend
    (injected fake embeddings) — higher similarity and lower off-manifold distance
    score higher.
  - `search.py`: on a synthetic scorer with a known optimum, the optimizer improves
    monotonically toward it within budget (deterministic with fixed seed).
  - `VoiceProfile.save/load` roundtrip with a mocked `Conditionals` (metadata +
    score preserved; schema-version mismatch raises).
  - Import-guard subprocess test: `import openvox.enroll` and construct
    `VoiceEnrollEngine` with `torch`/`chatterbox` import-blocked → succeeds.
  - Clone unit (patched, no torch): `clone(profile=…)` injects the profile's
    conditioning and generates; `clone(profile=…, reference_audio=…)` raises;
    `clone(reference_audio=…)` path unchanged.
- **Integration (heavy, real, GPU-preferred):** enroll from 2–3 real clips of one
  voice → an `.ovx` whose winning score ≥ the Stage-A baseline; then
  `clone(profile=…)` produces non-silent 24 kHz `TTSResult`. A CPU-only variant
  asserts Stage B is skipped and a valid Stage-A profile is still produced.
- Demo `build_parser` unit tests (enroll demo + the clone demo's `--profile`).

## 10. Dependencies & packaging

New `[enroll]` extra composes the engines it builds on:
`enroll = ["openvox[clone,enhance]"]` — the Chatterbox model (generator + voice
encoder) and the enhancer are the only heavy pieces; the algorithm itself is numpy.
Coordinate ascent is the **zero-extra-dependency** default optimizer; a CMA-ES
library is pulled in **only** if it clearly earns its place (kept out otherwise).
`[enroll]` stays **out of `all`** (heavy/torch). Install is documented alongside
clone/enhance (two-step, since enhance needs `resemble-enhance --no-deps`). New
demo entry point `openvox-enroll-demo`.

## 11. Demo CLI

`python -m openvox.enroll.demo` (entry point `openvox-enroll-demo`):
`--in PATH [PATH …]` (required, one or more clips / a long recording), `--out PATH`
(required, the `.ovx`), `--quality` (`fast`/`balanced`/`thorough`, default
`balanced`), `--device` (default cuda), `--no-enhance`. Enrolls and writes the
profile, printing the achieved speaker-similarity score. The clone demo gains
`--profile PATH` (mutually exclusive with `--ref`).

## 12. Definition of done

1. `from openvox.enroll import VoiceEnrollEngine; enroll(clips)` → a saved `.ovx`
   voice profile, fully offline (after one-time model downloads).
2. `python -m openvox.enroll.demo --in a.wav b.wav --out alice.ovx` enrolls and
   reports a score.
3. `VoiceCloneEngine.clone(text, profile="alice.ovx")` clones from the profile with
   no reference clip; `profile` + `reference_audio` together error clearly.
4. Stage B never ships a profile worse than Stage A; no-GPU degrades to Stage A.
5. Unit tests green with no torch/chatterbox; integration proves a real score gain
   and a working profile-clone; `import openvox` / `import openvox.enroll` stay lean.
