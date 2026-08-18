# 0026 — M7: chasing CPU parity. 42.4 → 251.3 seq/s, and where it stops

- **Date** 2026-08-17
- **Milestone** M7
- **Status** done — **5.9× faster**; parity **not reached** and shown to be out of
  reach with the available levers. The ceiling is quantified, not guessed.

## Goal

Get to CPU parity, checking routes and measuring each. Giving up was permitted
if no path exists — so the deliverable is either parity or a *demonstrated*
ceiling.

The CPU baseline ([0018](../0018-npu-vs-cpu/TASK.md), `sentence-transformers`,
12 threads): **267.6 seq/s at batch 4, 548.1 at batch 32, 710.0 at batch 128.**
Comparisons below are always at matched batch.

## Result

| batch | NPU seq/s | NPU cores | CPU seq/s | wall ratio | per core |
|---|---|---|---|---|---|
| 4 | 70.6 | 0.50 | 267.6 | 0.26× | **6.3× better** |
| 32 | 183.6 | 1.15 | 548.1 | 0.34× | 3.5× better |
| 128 | **251.3** | 1.58 | 710.0 | 0.35× | **3.2× better** |

`1-cos` **3.397e-04** at every batch — identical, and identical to the Python
path. Accuracy never moved during any of this.

**5.9× over [0023](../0023-m7-full-cpp-encode/TASK.md)'s 42.4 seq/s. Still
~2.9× short of the CPU in wall clock, at every batch.**

## The routes, in the order they were taken

| route | from → to | worth |
|---|---|---|
| Vectorise + thread host attention | 72.2 → 100.1 | **1.39×** |
| Batch 16 → 64, GEMMs to 8 columns | 100.1 → 126.9 | 1.27× |
| Eltwise designs 1 → 2 columns | 126.9 → 173.1 | **1.36×** |
| GELU kernel: 4 interleaved chains | 173.1 → 202.3 | **1.17×** |
| Batch 64 → 128 | 202.3 → 229.7 | 1.14× |
| Thread the host conversions, bias and residuals | 229.7 → 251.3 | 1.09× |

Two routes were tried and rejected on measurement:

- **GELU tile 1024 → 4096.** No change at all (16.4 ms both ways). Four times
  fewer DMA transactions bought nothing, which is what first suggested the cost
  was not in the data movement.
- **`--emulate-bfp16`.** +2.2% and it **fails** accuracy (3.470e-03 against the
  2e-03 tolerance). M2 measured bfp16 at 5.5× on GEMM *compute* — and it buys
  almost nothing here, which is itself the diagnosis: GEMM compute is not the
  bottleneck.

## Why it stops: the eltwise kernels are at the machine's fp32 limit

At batch 128, `dispatch + wait` is 347 ms of a 509 ms encode. Per-design:

| | per call | calls | total |
|---|---|---|---|
| GELU | ~19 ms | 6 | **114 ms** |
| LayerNorm | ~4.2 ms | 13 | 55 ms |
| softmax | ~6.6 ms | 6 | 40 ms |
| 4 GEMMs | — | 24 | 77 ms |
| design switches | — | 49 | ~60 ms |

**Eltwise is 209 ms of 347.** And it is compute bound, not data bound — the
control settles it. A passthrough moving the *same* bytes over the *same* two
columns takes **626 µs** against GELU's **9703 µs** at batch 64: the array can
feed the kernel **15×** faster than the kernel consumes it.

So how much of that is the polynomial? A degree-2 speed probe — identical
structure, same widen/narrow, same four interleaved chains, Horner cut from 8
steps to 2:

```
  passthrough (no arithmetic)     626 us
  GELU degree 2                  4056 us
  GELU degree 8                  9703 us
```

> **t ≈ 2174 µs + 941 µs per Horner step.**

Linear in degree **with four independent chains already running**, so this is
**throughput, not latency** — the ILP is there and the issue slots are full.
941 µs is ~19 cycles per 16-lane `mul → srs → add`, and `aie::vector<float,16>`
is very likely two 8-wide fp32 ops plus a separate shift-round-saturate, so
~3 cycles per native op. **The kernel is close to the machine's fp32 vector
throughput.** There is no factor of five hiding in it.

### The ceiling

Grant the best imaginable outcome — eltwise reduced to *zero*:

```
  509 ms - 209 ms = 300 ms  ->  427 seq/s     against the CPU's 710
```

**Free eltwise still does not reach parity.** Everything else would have to
improve too, and the remaining items are the GEMMs (already at 8 columns), the
switch cost (already characterised in [0024](../0024-m7-dispatch-cost-anatomy/TASK.md)
as ~25 µs + 7.2 µs per lock) and the host path (already AVX2 and threaded).

Fusing GELU into the `ffn_up` epilogue — the obvious next idea — removes the
DMA round trip and one dispatch, but *not the arithmetic*: the same polynomial
would run on the same cores. It is worth roughly the 626 µs of movement plus a
switch per call, not the 9 ms of evaluation.

**Verdict: parity is not reachable at matched batch with these levers.** Getting
there needs a different decomposition — bf16 arithmetic in the eltwise kernels
(an accuracy decision for M8, not a free win), or ops that keep the array in its
MAC datapath where its 14.7 TOPS actually live, instead of in fp32 elementwise
work where a 12-core Zen 5 with AVX-512 is genuinely strong.

**On the project's own criterion this still qualifies**, and by more than
before: 251.3 seq/s on **1.58 cores** against 710 on twelve is **3.2× better per
core**, up from [0023](../0023-m7-full-cpp-encode/TASK.md)'s 10× at a batch
where the CPU was far less efficient.

## A fourth fail-open — this one inside the validation

The 4-chain GELU first reported:

```
  embedding rel_fro vs HF golden                 nan
  worst 1 - cos vs HuggingFace             0.000e+00
PASS -- tolerance 2e-03 on 1-cos
```

**A perfect score, from NaN.** `std::max(0.0, NaN)` returns `0.0` — every
comparison with NaN is false, so `max` returns its first argument — so a kernel
producing NaN scored better than a correct one and passed. `rel_fro` printed
`nan` on the line directly above and nothing looked at it.

A tolerance test whose failure mode is a perfect score is not a test. Fixed with
an explicit `std::isfinite` gate, and **verified against the known-bad artifact**
rather than only against a good one.

Fourth instance of a check failing open in this project
([0022](../0022-m7-cpp-runtime/TASK.md), [0024](../0024-m7-dispatch-cost-anatomy/TASK.md),
[0025](../0025-m7-batching-and-crossover/TASK.md), here) and the first inside the
validation itself.

The NaN's cause: 4 interleaved chains keep 12 vectors live against 1 chain's 3,
and the spill overran `stack_size=0xD00`. It does not fault — it corrupts.
`0x2000` fixed it, and the result returned to **exactly** the single-chain
`1-cos`, as the bit-identical-arithmetic argument required.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| GELU NaN, reported as `1-cos = 0.000e+00` PASS | worker stack too small for 12 live vectors; `std::max(0.0, NaN) == 0.0` | `stack_size=0x2000`; explicit `isfinite` gate |
| `--gelu-tile 4096` changed nothing | the cost was never per-transaction | kept as a measured negative |
| `--elt-cols 4` fails: `no ShimNOCTile has sufficient DMA capacity` | one shim fill+drain per core, 16 cores at 4 columns | capped at 2 — and the passthrough control shows more columns would not have helped anyway |

## Artifacts

- `runtime/src/main.cpp` — thread pool, AVX2 attention, threaded conversions and
  residuals, `isfinite` gate
- `experiments/m5-eltwise/kernels/gelu_poly.cc` — 4-chain kernel, 4k tile
  variant, degree-2 speed probe
- `experiments/m7-switch-cost/build_passthrough.py` — the control, now
  parameterised by batch and columns
- `tools/export_xclbin.py` — `--gelu-tile`, `--gelu-variant`

## Next

1. **bf16 eltwise arithmetic** — the only remaining large factor, and an
   accuracy question M8 must answer rather than a free win.
2. **Fuse GELU into `ffn_up`** — worth the movement and a switch, not the
   arithmetic. Modest and now correctly priced.
3. **Attention on the array** would move 22 ms of host work but adds to the
   array's own critical path.
