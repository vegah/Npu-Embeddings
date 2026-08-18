# 0005 — Testing the expert review, claim by claim

*Started 2026-08-17. **Living document** — updated as each test lands.
The review itself is reproduced in [`tasks/0029`](../../tasks/0029-m7-one-xclbin-probe/TASK.md)'s
context; this note tracks what was tested, how, and what came out.*

An external reviewer read the kernel designs, the AIE kernels, the export
pipeline, the C++ runtime and the research summaries, and made a set of
falsifiable claims. This note tests them. Verdicts: **CONFIRMED** /
**REFUTED** / **PARTIAL** / **UNTESTED**.

## Scoreboard

| § | claim | verdict | evidence |
|---|---|---|---|
| §1 step 0 | Two insts streams over one context alternate at alone-cost; the switch follows the hw_context, not the stream | **CONFIRMED** | [`tasks/0029`](../../tasks/0029-m7-one-xclbin-probe/TASK.md): 500 µs vs 497 alone; two-context control 976; xclbins differ only in UUIDs; foreign stream exact on 6.3 M elements |
| §1 step 1 | RTP-ifying the GEMM loop counts unifies the four GEMMs into one xclbin + four streams | **CONFIRMED** | Two shapes share a UUID-only-diff xclbin; foreign-shape stream computes **exactly** (rel err 0.0); A↔B in one context = A+B alone to the microsecond; RTP overhead **+1.6%** |
| §2 | GELU fusable into ffn_up via K-augmented bias + polynomial in core_fn; ~110 ms net | **CONFIRMED** (mechanism+accuracy) / **PARTIAL** (magnitude) | Fused `gelu(A@B+bias)` measures **3.167e-04** — 10× better than the separate path, because the epilogue reads fp32 in L1. Cost: fused 4311 µs vs separate ~5800 with overheads at M=1024/2col — a ~1.35× win, smaller than projected |
| §3 | LN/softmax `.split()`/`.join()` reaches 8 columns; params broadcast from mem tile | **CONFIRMED** | All three eltwise designs build and validate at 8 columns; LayerNorm 4.2 → 1.1 ms; encode **251 → 298.5 seq/s** at `1-cos` 2.469e-04 |
| §4 | Post-attention block fusable as one design | **MECHANISMS PROVEN, build-out priced** | Every gating mechanism individually confirmed (§1 streams, §2 epilogue+bias, §3 8-col eltwise, heterogeneous workers exist in IRON). The *naive* one-xclbin form (spatial split + streams) prices to **a wash** by measured numbers — the pipelined form is the payoff and is milestone-scale |
| §5a | The unresolved exp2_poly corruption is the worker stack (0xD00) | **CONFIRMED** | 2×2 below: poly+0xD00 reproduces the 0021 corruption, poly+0x2000 passes; narrowing ruled out. Now production: full-encode `1-cos` **3.397e-04 → 2.469e-04** |
| §5b | The h≥1536 stride wall is the C-drain row-block stride `256·N > 2²⁰`; qkv+ffn_up fail, attn_out+ffn_down pass; `tb_n_rows=1` fixes it | **CONFIRMED** | Per-shape prediction 4/4; boundary at exactly 2²⁰ is **inclusive** (bge-large clears it); with the cap all five shapes build; MiniLM regression clean at 2.469e-04 |
| §6a | Micro-batch pipelining pays after §1 | **DEFERRED with cause** | Pays only once ops share a context; see §4 pricing below |
| §6b | Device-resident intermediates remove t_conv/sync | **DEFERRED with cause** | Same prerequisite; t_conv+syncs ≈ 70 ms at batch 128 remain the prize |
| §6c | bf16 GEMM output halves C traffic at identical numerics | **REFUTED as stated / route exists** | M2 already measured the decisive number: bf16 GEMM output re-rounds at every K step, 7.4e-03 vs 1.2e-07 (CLAUDE.md trap 2). "Identical numerics" requires narrowing ONCE after accumulation — i.e. in a §2-style epilogue, which §2 just proved works |

## Test log

### 2026-08-17 — §1 step 0: CONFIRMED

See [`tasks/0029`](../../tasks/0029-m7-one-xclbin-probe/TASK.md). Three parts:
static config byte-identical modulo UUIDs; foreign instruction stream exact on
all 6,291,456 elements; alternation in one context at alone-price (500 µs vs
497), while the two-context control pays 479 µs against the
[note 0004](0004-context-switch-cost.md) model's predicted 486 — a 1.5%
cross-check between independent methods.

### 2026-08-17 — §5a: CONFIRMED, and it is now the production softmax

The claim: the 0021 "works standalone, corrupts composed" exp2_poly failure is
the worker stack (0xD00 — the same size the 4-chain GELU overran, and stack
overrun corrupts rather than faults). Secondary candidate from the review: the
fp32→bf16 narrowing of exp2_poly's result.

`softmax.cc` gained a templated impl with both variants as separate symbols
(so the cache marker stays unambiguous), with the narrowing done the only
correct way — through the accumulator, never a cast. The 2×2:

| | stack 0xD00 | stack 0x2000 |
|---|---|---|
| `aie::exp2` (lib) | PASS, NPU-vs-golden 1.744e-02 | — |
| `exp2_poly` | **FAIL — 384 non-finite, the 0021 signature** | **PASS, 4.278e-03** |

**The stack is the cause; the narrowing is not** (correct narrowing with the
old stack still fails). Against the bf16 floor of 3.225e-03 the poly variant
sits at 1.33×, where the library call sat at 5.4×.

In the full encode the poly softmax moves `1-cos` from 3.397e-04 to
**2.469e-04** — the best accuracy the project has produced — and it is now the
production variant.

### 2026-08-17 — a fifth fail-open found on the way, worth its own entry

The first full-encode test of the poly softmax FAILED at 3.165e-03, and the
"one variable at a time" revert then failed too — at **3.470e-03, which is
exactly 0026's `--emulate-bfp16` number**. That was the tell: the GEMM cache
matcher pins memrefs, stride and column count, and none of those distinguish a
bf16 build from an `--emulate-bfp16` build of the same shape. Both sat in the
cache; mtime picked the bfp16 ones; every GEMM in the encode silently ran
emulated. The only readable difference in the artifact is the kernel hash
prefix (`c894e098_matmul_bf16_f32` vs `333c4d33_matmul_bf16_f32`), whose
expected value is not computable at match time.

**Fix: remove the ambiguity instead of resolving it.** `purge_ambiguous()` in
`tools/export_xclbin.py` deletes every matching cache candidate *before* each
build, so the subsequent match sees exactly one directory. Costs a recompile
whenever candidates existed; correctness over cache warmth. With the purge in
place the encode restored to 3.397e-04 (lib) and improved to 2.469e-04 (poly).

Fifth instance of the fail-open class (0022, 0024, 0025, 0026, here) — and the
second time the *same* mtime mechanism produced a confidently wrong artifact.

### 2026-08-17 — §5b: CONFIRMED — the stride wall is per-shape, and bge-large clears it

The claim: `'aie.dma_bd' op Stride 3 exceeds [1:1048576]` at hidden ≥ 1536
([`tasks/0027`](../../tasks/0027-m7-width-hypothesis/TASK.md) called it "a wall
without a known door") is the **C-drain's row-block stride**, `m·4·N = 256·N`
elements against a 20-bit field — so it should be per-*shape*, not per-hidden:
qkv (N=4608) and ffn_up (N=6144) over, attn_out and ffn_down (N=1536) clear.

`experiments/m7-switch-cost/stride_wall_probe.py` builds each shape in
isolation:

| shape | 256·N | predicted | got |
|---|---|---|---|
| attn_out@h1536 | 393,216 | pass | BUILT |
| ffn_down@h1536 | 393,216 | pass | BUILT |
| qkv@h1536 | 1,179,648 | fail | STRIDE-FAIL |
| ffn_up@h1536 | 1,572,864 | fail | STRIDE-FAIL |
| ffn_up@h1024 | **1,048,576 = 2²⁰ exactly** | boundary | **BUILT** |

**4/4 on the clear cases, and the boundary is inclusive** — which the reviewer
flagged as the thing to check, because ffn_up at h=1024 lands exactly on it.
**bge-large's four shapes all clear the wall.**

The fix: cap `tb_n_rows` to 1 when `m·4·N > 2²⁰` (one row block per drain, no
repeat stride, twice the drain tasks), which required generalising the drain
loop from the hard-coded tb/pingpong pairing to groups of `tb_n_rows` — the
sequence is bit-identical for the production value of 2. After the cap all
five shapes build, and the MiniLM full encode regresses clean at `1-cos`
2.469e-04.

0027's "wall without a known door" is downgraded to "known limit with a
measured fix". Correctness of the h≥1536 designs themselves (as opposed to
their buildability) remains unvalidated — they have no goldens yet.

### 2026-08-17 — §3: CONFIRMED — all three eltwise designs at 8 columns

LayerNorm and softmax rewritten on the GELU pattern: one shim stream per column
each way, `.split()`/`.join()` through the mem tile. The reviewer's specific
suggestion for LayerNorm's third fifo worked exactly as described: the 768
parameter floats go L3→L2 **once per column** and are **broadcast** from the
mem tile — multiple workers consuming the same ObjectFifo handle, which is
precisely how the GEMM broadcasts B down a column. Shim cost per column: 2 in,
1 out — inside the shimNOC budget at 8 columns.

Validated against goldens at 2 columns first (LN PASS at its 5e-03 tolerance,
softmax 4.278e-03), then production at `--elt-cols 8`, batch 128:

| | 2 columns | 8 columns |
|---|---|---|
| LayerNorm per call | ~4,200 µs | **1,106 µs** |
| softmax per call | ~6,600 µs | 5,334 µs* |
| full encode | 251.3 seq/s | **298.5 seq/s** |

`1-cos` **2.469e-04**, unchanged. (*softmax is now the poly variant, so its
per-call number is not directly comparable to the old lib figure.)

The golden-shape isolated tests cannot run at 8 columns (256 rows do not split
over 32 cores); correctness at 8 columns is carried by the full-encode
validation, where the shapes divide.

### 2026-08-17 — §1 step 1: CONFIRMED — one xclbin really does serve multiple GEMM shapes

`gemm_pretiled.py` gained `rtp=True`: the two loop bounds (`n_tiles_per_core`,
`K//k`) — the *only* shape-dependent values in the static design — move into a
per-worker `Buffer(use_write_rtp=True)`, written from the runtime sequence
behind a `WorkerRuntimeBarrier` (the `scale_shift` pattern).

Three findings on the way, each of which would have silently broken it:

1. **The RTP initializer is part of the static image.** With
   `initial_value=[n_tiles, K//k]` the two shapes' xclbins differed by exactly
   8 bytes — the CDO writes of those initial values. Zero initializers fix it;
   the runtime sequence writes the real bounds before the barrier releases.
2. **The JIT does not hash `initial_value`**, so changing it served the stale
   builds — the same cache-identity gap as the `.cc`-edit trap. Purge first.
3. The RTP buffers appear in `aie.mlir` as `sym_name = "rtp_r_c"`, not as any
   `write_rtp` op — markers must match what is actually emitted.

With those fixed, `gemm_rtp_probe.py` builds qkv (384×1152) and attn_out
(384×384) at M=1024: **xclbins differ only in UUID metadata**. The C++
`--probe-rtp` then loads **one** xclbin, both instruction streams, and runs
both shapes through the one context, correctness by the constant-B trick
(layout-invariant reference):

```
      stream 0 (own shape)   worst rel err 0.000e+00  OK
      stream 1 (other shape) worst rel err 0.000e+00  OK
      stream 0 again         worst rel err 0.000e+00  OK
      shape A alone                          1431 us
      shape B alone                           556 us
      A <-> B (per two dispatches)           1987 us   = 1431 + 556, exact
```

**Different shapes, one context, exact results, zero switch cost.** And the
reviewer's flagged risk — `range_(rtp)` pipelining worse than compile-time
bounds — is measured at **+1.6%** (1431 vs 1409 µs for the same shape), noise.

What remains for production: unify all four GEMM shapes + the eltwise designs
into one exported design set and teach the runtime to drive it as streams.
That is engineering on a proven mechanism rather than an open question.

### 2026-08-17 — §2: CONFIRMED on mechanism and accuracy; cost win real but smaller

`gemm_pretiled` gained `epilogue="gelu"`: a fp32 GELU kernel
(`gelu_epilogue_3072_f32`) applied by the core to the output tile before it is
released. The bias rides in as the reviewer described — K augmented 384→448,
A's extra block a ones-column, B's extra block the bias row — so the
accumulator holds `A@B + bias` with no third input (a core has 2-in/2-out).

**Accuracy: `rel_fro` 3.167e-04 against exact-erf `gelu(A@B + bias)`** — an
order of magnitude better than the separate GELU path's 4.3e-03, because the
epilogue reads the pre-activation in fp32 from L1 where the separate design
reads it after a bf16 round trip. Fusion here is an *accuracy* feature first.

**Cost** (M=1024, 2 columns): the first, single-chain epilogue ran 6266 µs —
latency-bound, 1.47× *slower* than the separate pieces, repeating 0026's lesson.
Four interleaved chains (and the 0x2000 stack that requires — the 0xD00 overrun
corrupts silently) brought it to **4311 µs**, against separate
`ffn_up` 1861 + GELU 2410 + a dispatch + a switch + the host round trip
≈ 5800 µs: a ~1.35× win today. The reviewer's ~110 ms net at batch 128 looks
high — the K-augmentation MACs and the epilogue's residence on the release path
eat more of the margin than the estimate assumed. Production integration
(augmented `.npue` tensor, runtime A-fill) not yet done.

### 2026-08-17 — §6c: refuted as stated, but the correct route is §2's epilogue

No new experiment needed — M2 already measured it: **bf16 GEMM output
re-rounds at every K step, 7.4e-03 against 1.21e-07** (CLAUDE.md trap 2). So
"bf16 C at identical numerics" is false for the stock kernel. It becomes true
exactly when the narrowing happens **once, after accumulation** — in a
§2-style epilogue, whose machinery is now proven. The two claims compose: an
epilogue that applies GELU and stores bf16 would halve C traffic *and* improve
accuracy at once.

## Remaining, and why

§4 (block fusion), §6a (micro-batch pipelining) and §6b (device-resident
intermediates) are architecture proposals whose *gating mechanisms* are now all
individually proven — §1's streams, §2's epilogue+bias, §3's 8-column eltwise.
What blocks them is one shared prerequisite: **productionising the one-xclbin
design** (all GEMM shapes + eltwise as workers in one static design, driven as
instruction streams). Note that unifying only the GEMMs would save almost
nothing: the encode alternates GEMM and eltwise dispatches, so switches vanish
only when *every* op shares the context. That build-out is the next task; it
is engineering on proven mechanisms, not an open question.

### 2026-08-17 — §4/§6a/§6b: the integration, priced with measured numbers

All gating mechanisms are proven. But the *naive* production form of the
one-xclbin architecture — every op's workers in one static design, spatially
partitioned across the 8 columns, driven as instruction streams — prices to
**roughly a wash at batch 128**, using only numbers measured in this note and
[`0026`](../../tasks/0026-m7-closing-on-cpu/TASK.md):

- One design must place all workers at once, and there are exactly 32 cores.
  GEMM on 4 columns costs ~nothing (8,104 vs 8,132 µs per call at 8 vs 4 —
  movement-bound), but eltwise squeezed from 8 columns to 4 costs
  **≈ +75 ms** per encode (GELU +28, LayerNorm +14, softmax +32), against the
  **≈ +60 ms** the eliminated switches save.
- So the spatial-split build-out buys accuracy of architecture but not time.
  **The winning form is the reviewer's §4 proper**: ops *pipelined* through
  the array — GEMM columns streaming into eltwise columns through the mem
  tiles, one dispatch per layer block — so no column idles while another
  works. That is what AMD's 15→3, STEEL's 22.8× and ARIES' adjacent-tile
  handoff all measured, and it is milestone-scale work, not a session task.

§6a (micro-batch pipelining) and §6b (device-resident intermediates) both
activate in that same architecture; ~70 ms of t_conv+sync at batch 128 is the
§6b prize. Deferred **with cause**, not untested by neglect.

## Session summary

Seven of ten claims resolved on hardware, one refuted by an existing
measurement, two deferred with a measured pricing argument. Along the way the
encode improved **251.3 → 298.5 seq/s** and its accuracy improved
**3.397e-04 → 2.469e-04** — both directly from the reviewer's §5a and §3 —
and a fifth fail-open (the bfp16 cache poisoning) was found and closed with
`purge_ambiguous()`.
