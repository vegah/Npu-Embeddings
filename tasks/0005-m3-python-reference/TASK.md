# 0005 — M3: Python reference encoder + golden vectors

- **Date** 2026-08-17
- **Milestone** M3
- **Status** done (**gate passed**)

## Goal

Build the oracle. A pure-numpy fp32 implementation of all-MiniLM-L6-v2 that
agrees with HuggingFace tensor-by-tensor, plus stored golden vectors, so that
every NPU kernel in M5 has something exact to be checked against.

**Gate:** a layer-by-layer oracle exists and provably matches HuggingFace.

## Context

M2 finished with the array **data-movement bound** — per-core compute healthy at
~54% of peak, but compute only ~7% of NPU time ([0004](../0004-m2-multicore-gemm/TASK.md)).
Nothing more can be decided about kernels until there is a correctness oracle,
because from M5 onward every optimisation must be provably lossless.

M2 also left an explicit open question, recorded in CLAUDE.md: the bfp16 error
of **1.040e-02** was measured on `iron.rand` uniform [0,1) inputs, which was
*assumed to be adversarial* for a block float format. M3 is the first point at
which real activations exist, so it is the first point that question can be
answered. It is answered below — and the assumption was **backwards**.

## What was done

### 1. The reference environment

`.venv-ref` already existed (created from the conda `iron` interpreter with
`--system-site-packages`, so torch and numpy are inherited rather than
duplicated) but contained only numpy/torch. Installed the reference stack into
it. **mteb was deliberately not installed** — it belongs to M8 and drags in a
large dependency tree we do not need to carry through M4–M7.

### 2. Fetch and *verify* the checkpoint

`reference/fetch_model.py` downloads only the 9 files we consume (the ONNX and
OpenVINO exports and the duplicate `pytorch_model.bin` are several hundred MB of
nothing) and then **asserts the checkpoint against `docs/04-model/README.md`**:
every config value, all 101 architecture tensors with exact shapes and F32
dtype, and no unexpected tensors beyond the three known-ignorable ones.

This is not ceremony. A silently different checkpoint would poison every golden
downstream, so the sha256 is pinned into `models/<name>/CHECKPOINT.json` and
carried in the golden metadata; `check_reference.py` refuses to run if the file
on disk no longer matches.

Result: **104 tensors, 6 layers, all shapes as documented.** The doc's claim of
104 is confirmed exactly — 101 architecture + `position_ids` buffer + 2 dead
pooler tensors.

### 3. A dependency-free safetensors reader/writer

`reference/safetensors_io.py`, ~80 lines of numpy. Written rather than
`pip install safetensors` because golden data has to cross an environment
boundary as *files*: produced in `.venv-ref` (torch, transformers), consumed in
the `iron` env, which must stay clean. The consuming side now needs nothing but
numpy. M4/M7 have to parse this format from C++ anyway, so writing it once in
Python is the cheap way to be sure we understand it.

### 4. The reference encoder

`reference/encoder.py`. Every numerical decision from `docs/04-model` is
implemented at the point it matters, with the reason in a comment:

- LayerNorm in **fp64**, `eps=1e-12` **inside** the sqrt, **biased** variance
  (÷N). The fp64 is not paranoia — post-LN BERT has hidden dims carrying ±50–100
  while the rest sit near ±1.
- **Exact erf** GELU, never the tanh approximation.
- Softmax in fp64 with max subtraction.
- Additive attention mask `(1-mask) * finfo(f32).min` — large *finite* negative,
  never `-inf`.
- Mean pool with the `clamp(min=1e-9)` denominator.
- **No pooler.** 147,840 dead params, correctly not implemented.

Two structural choices worth recording:

- **QKV is fused** into one `[384, 1152]` GEMM, because that is what M4 will bake
  and M5 will dispatch. An oracle with different seams than the implementation
  validates a different program.
- **`1/sqrt(head_dim)` is NOT folded into Q.** M4 folds it offline; keeping the
  reference canonical is exactly what will let us *prove* the fold is exact.

### 5. The GEMM hook

After the gate passed, every matmul that will run on the NPU — the four
projection/FFN GEMMs plus QKᵀ and A·V — was routed through a swappable
`self.gemm(a, b)` primitive taking `(M,K) x (K,N)`, the same 2D shape the kernel
receives. Bias add, LayerNorm, softmax, GELU and pooling deliberately stay fp32
outside it.

The gate was re-run after the refactor and produced **identical numbers**, which
is the only reason to believe the refactor was neutral.

### 6. Goldens

`reference/make_goldens.py` writes two files, split by size:

- `minilm_l6_s64_boundary.safetensors` — **3.2 MB, committed.** Tokenizer output,
  every layer boundary, `last_hidden_state`, pooled and normalized embedding,
  plus the real sentence-transformers embedding as an *independent second
  oracle*. This is the contract M5 is checked against.
- `minilm_l6_s64_taps.safetensors` — **54 MB, gitignored.** All 75 intermediates
  including attention scores and the FFN interior. A deterministic CPU-only
  derivative of a sha256-pinned input, so it is regenerated rather than stored,
  the same call we made for the parsed Perfetto traces in
  [0004](../0004-m2-multicore-gemm/TASK.md).

The corpus (`reference/corpus.py`) is four frozen sentences chosen to hit the
tokenizer landmines rather than four generic sentences: accented Latin, CJK, a
long `##`-decomposing word, and a short one so the batch is ragged and padding
is actually exercised. It confirmed all three documented traps in one run —
`café → cafe` (accents ARE stripped, so `strip_accents: null` does inherit from
`do_lower_case`), `北 京` split per codepoint, and
`anti ##dis ##est ##ab ##lish ##ment ##arian ##ism`.

### 7. The gate

`reference/check_reference.py` runs **in the iron env, numpy only** — proof the
env boundary holds.

### 8. Extra: what the NPU's number format costs the embedding

`reference/precision_study.py`, answering the M2 carry-forward. Structured so
the second half is only trustworthy because of the first:

1. **Calibrate** a bfp16 simulation against M2's *hardware* number under M2's own
   conditions. A simulation that cannot reproduce the measurement has no
   standing to predict anything.
2. **Apply** the calibrated format to the full encoder on real activations and
   report what actually matters — cosine, and distortion of the
   sentence-similarity matrix.

Labelled a **simulation** everywhere, per `docs/05-measurement`. It predicts;
M5 measures.

## Commands

```powershell
# --- env (from the repo root) ---
& ".\.venv-ref\Scripts\python.exe" -m pip install --disable-pip-version-check `
    transformers safetensors sentence-transformers huggingface_hub

# --- fetch + verify the checkpoint ---
& ".\.venv-ref\Scripts\python.exe" reference\fetch_model.py

# --- goldens (must run in .venv-ref: torch/transformers) ---
& ".\.venv-ref\Scripts\python.exe" reference\make_goldens.py
& ".\.venv-ref\Scripts\python.exe" reference\make_goldens.py --taps

# --- THE GATE (must run in the iron env: numpy only) ---
& "C:\Users\vegar\.conda\envs\iron\python.exe" reference\check_reference.py

# --- precision study (iron env, numpy only) ---
& "C:\Users\vegar\.conda\envs\iron\python.exe" reference\precision_study.py
```

## Result

### Checkpoint verification

```
  model.safetensors     : 90.9 MB
  sha256                : 53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db
  tensors in checkpoint : 104
    required by arch    : 101
    ignorable present   : ['embeddings.position_ids', 'pooler.dense.bias', 'pooler.dense.weight']
  layers                : 6

OK -- checkpoint matches docs/04-model/README.md
```

### Tokenizer behaviour (all three documented traps confirmed)

```
  [0]   8 tokens  ['[CLS]', 'a', 'man', 'is', 'eating', 'food', '.', '[SEP]']
  [1]  22 tokens  ['[CLS]', 'le', 'cafe', 'co', '##ute', '5', 'euros', '—', 'c', "'", 'est', 'bien',
                   'tr', '##op', 'cher', 'pour', 'un', 'es', '##press', '##o', '.', '[SEP]']
  [2]  17 tokens  ['[CLS]', 'the', '北', '京', 'office', 'opens', 'at', '9', '##am', ',',
                   '上', '海', 'at', '10', '##am', '.', '[SEP]']
  [3]  33 tokens  ['[CLS]', 'anti', '##dis', '##est', '##ab', '##lish', '##ment', '##arian', '##ism',
                   ',', 'trans', '##ub', '##stan', '##tia', '##tion', ',', 'and', 'other', 'un',
                   '##ne', '##ces', '##sari', '##ly', 'long', 'words', ';', 'do', 'they', 'token',
                   '##ize', 'correctly', '?', '[SEP]']

sentence-transformers vs manual mean-pool: max abs diff 9.383e-08
```

### **GATE — reference vs HuggingFace** (iron env, numpy only)

```
tensor                                                shape     max_abs     rel_fro
emb.ln                                         (4, 64, 384)   1.431e-06   8.494e-08   ok
L0.ln2                                         (4, 64, 384)   3.114e-06   4.698e-07   ok
L1.ln2                                         (4, 64, 384)   6.229e-06   6.405e-07   ok
L2.ln2                                         (4, 64, 384)   3.219e-06   7.260e-07   ok
L3.ln2                                         (4, 64, 384)   4.113e-06   7.999e-07   ok
L4.ln2                                         (4, 64, 384)   5.245e-06   8.396e-07   ok
L5.ln2                                         (4, 64, 384)   4.053e-06   9.928e-07   ok
last_hidden_state                              (4, 64, 384)   4.053e-06   9.928e-07   ok
pool.mean                                          (4, 384)   6.855e-07   7.291e-07   ok
out.embedding                                      (4, 384)   1.267e-07   7.153e-07   ok
out.embedding vs sentence-transformers             (4, 384)   1.304e-07   7.297e-07   ok

cosine vs HF                : min 0.999999977643  (1-cos max 2.236e-08)
cosine vs sentence-transf.  : min 0.999999995862  (1-cos max 4.138e-09)

PASS -- all 11 comparisons within rel_fro 2e-05, cosine within 1e-06
```

**The error profile is the point, not just the pass.** It grows monotonically
with depth — 4.7e-07 at layer 0 to 9.9e-07 at layer 5 — which is the signature
of fp32 accumulation order differing between numpy and torch. A formula error
(tanh GELU, unbiased variance, eps outside the sqrt) does not look like that; it
shows up as a **constant floor around 1e-3** at every layer. The tolerance of
2e-05 is set precisely in that gap: ~20× above the observed noise and ~50× below
the cheapest formula mistake.

Two independent oracles agree: the manual `BertModel` + mean-pool path and the
real sentence-transformers pipeline, the latter including its own tokenization.

### Precision study — **calibration**

```
  bf16 + fp32 accum      sim 1.209e-07   hw 1.213e-07   ratio  1.00x
```

The bf16 path reproduces M2's hardware number to **1.00×** on the first try.
That is the control: it says the simulation harness itself is sound.

bfp16 did not. Sweeping block size and mantissa width against the measured
1.040e-02 (block size shown at 8; 16/32/64 differ by <10% throughout):

| block | bits/element | sim rel_fro | ratio vs hw |
|---|---|---|---|
| 8 | 3 | 1.201e-01 | 11.54× |
| 8 | 4 | 3.140e-02 | 3.02× |
| 8 | **5** | **8.982e-03** | **0.86×** |
| 8 | 6 | 2.676e-03 | 0.26× |
| 8 | 7 | 1.103e-03 | 0.11× |
| 8 | 8 | 5.560e-04 | 0.05× |
| 8 | 9 | 1.878e-04 | 0.02× |

**Block size is not identifiable from this experiment** — 8 through 64 differ by
under 10% at every mantissa width, because uniform [0,1) has the same dynamic
range in a block of 8 as in a block of 64. So block was fixed at 8 (physically
plausible for an AIE bfp16 datapath) and only the mantissa width fitted, which
the data *does* resolve at roughly a factor 3 per bit.

**Finding: the hardware bfp16 path behaves as if it keeps ~5 bits per element**
(1 sign + 4 magnitude). This is a *phenomenological* fit, not a claim about the
datapath — something else could produce the same error magnitude. But it is
far coarser than the name "bfp16" suggests, and it is the number to plan against
until M5 measures the real thing on real activations.

### Precision study — **was uniform [0,1) the adversarial case?**

One GEMM, `256×384×1152` (layer 0 QKV), identical bfp16 model, only the data
differs:

| inputs | rel_fro | median within-block max/min |
|---|---|---|
| uniform [0,1) (M2's test data) | 8.416e-03 | 10.7 |
| real activations × real weights | **5.052e-02** | **18.2** |

**Real activations are 6.0× WORSE, not better.** The assumption carried forward
from M2 was backwards, and the mechanism is already documented one file over:
`docs/04-model` records that post-LN BERT has outlier dimensions carrying
±50–100 among values near ±1. Block floating point sets one shared exponent from
the largest magnitude in the block, so a single outlier crushes the other seven
elements. Uniform [0,1) is *benign* for block FP by comparison — it is bounded
and has no outliers. The measured within-block dynamic range, 18.2 vs 10.7,
is the mechanism showing up directly in the data.

### Precision study — **end-to-end, on real activations**

```
  tensor                 bf16 rel_fro   bfp16 rel_fro
  emb.ln                    0.000e+00       0.000e+00
  L0.ln2                    2.367e-03       8.410e-02
  L1.ln2                    3.072e-03       1.160e-01
  L2.ln2                    3.561e-03       1.403e-01
  L3.ln2                    3.994e-03       1.638e-01
  L4.ln2                    4.379e-03       1.779e-01
  L5.ln2                    5.719e-03       2.164e-01

  metric                                      bf16         bfp16
  1 - cos vs fp32 (worst)                1.271e-05     1.823e-02
  1 - cos vs HuggingFace (worst)         1.275e-05     1.823e-02
  max shift in sentence similarity       5.869e-04     1.409e-02
  embedding rel_fro                      4.695e-03     1.791e-01
```

- **bf16 + fp32 accumulate is effectively free.** Worst-case `1-cos` of
  **1.3e-05** against HuggingFace, and the sentence-similarity matrix moves by at
  most 5.9e-04. Nothing downstream can see that.
- **bfp16 is not free.** `1-cos` of **1.8e-02** and similarity shifts up to
  **1.4e-02**, with error compounding through depth (8.4e-02 → 2.2e-01 across six
  layers). MTEB scores are cosine-ranking-based; a 1.4e-02 perturbation of the
  similarity matrix is the same order as the gaps between neighbouring models on
  the leaderboard. The M8 budget is **0.3 MTEB points**.

Note both bfp16 figures are *conservative in the wrong direction* if anything:
they use the fitted 5-bit model, calibrated on the distribution that has now been
shown to be the benign one.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| bfp16 simulation gave 5.6e-04 against a measured 1.040e-02 — **0.05×**, a 19× miss | Assumed "bfp16" means 1 sign + 7 magnitude bits per element. It does not behave that way | Stopped guessing the mechanism and swept mantissa width instead, fitting the width the hardware *behaves as if* it keeps: ~5 bits. **The failed first attempt is the reason the second is believable** — had the first guess landed near 1.0× by luck, we would have shipped an unvalidated model |
| `models/all-MiniLM-L6-v2/CHECKPOINT.json` would not stage despite an explicit `!` negation | `.gitignore` excluded `models/`, and **git does not descend into an excluded directory**, so a negation for a file inside it can never match | Ignore the *contents* — `models/**` plus `!models/`, `!models/*/`, `!models/*/CHECKPOINT.json` |
| `scipy.special.erf` unavailable in the iron env | scipy is pulled in by sentence-transformers, so it exists in `.venv-ref` but not in `iron` — and the oracle must run in `iron` | Fall back to `np.frompyfunc(math.erf, ...)` in fp64. Slower, but it is the same C library erf, so it is exact — and this is an oracle, not a kernel |
| Not a problem, recorded because it was a real risk: routing all six GEMMs through a swappable hook could have silently changed the maths | — | Re-ran the gate after the refactor and confirmed **byte-identical** comparison output before trusting it |

## Artifacts

Committed:

- `reference/{safetensors_io,encoder,corpus,fetch_model,make_goldens,check_reference,precision_study}.py`
- `reference/goldens/minilm_l6_s64_boundary.safetensors` — 3.2 MB, 14 tensors, the contract
- `reference/goldens/precision_study.json` — full sweep and per-tensor results
- `models/all-MiniLM-L6-v2/CHECKPOINT.json` — the pinned sha256

Not committed, regenerable:

- `models/all-MiniLM-L6-v2/` — 90.9 MB checkpoint → `reference\fetch_model.py`
- `reference/goldens/minilm_l6_s64_taps.safetensors` — 54 MB, 75 tensors →
  `reference\make_goldens.py --taps`

## Next

M3 unblocks M4 (offline weight pre-tiling → `.npue`), which is doubly required:
it is the only way to express `ffn_down` (K=1536 > the 1023 DMA BD limit) and it
is the main performance lever now that M2 has shown we are data-movement bound.

Carried forward:

1. **bfp16 is now a serious accuracy risk, not merely a tradeoff.** M2 priced it
   at 5.5× throughput; M3 prices it at `1-cos ≈ 1.8e-02` on real data — and shows
   the M2 error figure was measured on the *benign* distribution. Both paths
   still stay selectable, but the M8 default should be assumed **bf16 + fp32
   accumulate** unless MTEB says otherwise. bf16 costs `1-cos ≈ 1.3e-05`, which
   is free.
2. **Confirm the 5-bit fit against hardware in M5**, on real activations rather
   than `iron.rand`. If it holds, block floating point is likely finished for
   this model; if the real datapath is finer than the fit suggests, the 5.5× is
   back on the table. This is the single highest-value measurement in M5.
3. **M4 should verify its offline fusions against these goldens**, not just
   round-trip them: the `1/sqrt(32)` fold into Q and the fused QKV must
   reproduce `L{i}.qkv` exactly.
4. The reference is deliberately slow (looped per-head GEMMs, fp64 LayerNorm,
   scalar erf). If M6 wants it as a working encoder rather than an oracle, that
   is a separate, and separately validated, fast path.

---

## Correction — the bfp16 findings above are superseded by [0008](../0008-m5-bfp16-real-data/TASK.md)

Added 2026-08-17, after the same question was measured on hardware. **The parts
of this task that concern bfp16 were wrong. The gate, the oracle, the goldens and
the bf16 results are unaffected.**

| this task claimed | hardware says |
|---|---|
| Real activations are **6.0× worse** than uniform for block FP | **0.85× — indistinguishable.** 9.02e-03 real vs 1.06e-02 uniform |
| The hardware behaves as if it keeps **~5 bits/element** | **~7 bits** when fitted against the real-data measurement |
| bfp16 costs `1-cos ≈ 1.8e-02` end to end | **≈1.8e-03**, ~10× less |
| bfp16 is "a serious accuracy risk, not merely a tradeoff" | A **genuine tradeoff**: 6.5e-03 similarity shift for 5.5× throughput. M8 decides |

**Why it was wrong, and what still stands.** The block-float model was fitted
against the only hardware number available at the time — M2's *uniform* input
measurement — and then applied to a different distribution. The model's
sensitivity to within-block dynamic range was its own property, not the
datapath's. The input statistics this task reported are correct and were
confirmed on hardware (median within-block range 18.2 real vs 10.8 uniform); the
inference from them to error was not.

The methodology that produced the error is worth keeping: the calibration step
is what made the claim checkable at all, and it is why the refutation was cheap.
What was missing was a second anchor on a different distribution.
