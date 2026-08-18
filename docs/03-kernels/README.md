# Kernels

Design notes and measured results, one page per kernel. Nothing here is written from
theory — a kernel gets a page once it has been built and traced.

**Status:** [saxpy](#saxpy--m1-toolchain-proof) (M1, toolchain proof) and
[bf16 GEMM](#bf16-gemm--single-core-m2) (M2, single core). Multi-core GEMM is blocked
on adding trace support to `whole_array`.

## bf16 GEMM — single core (M2)

`experiments/m2-bf16-gemm/` · full write-up in
[`tasks/0003`](../../tasks/0003-m2-bf16-gemm/TASK.md)

**Configuration that works:** bf16 in → **f32 accumulate**, tile **64×64×32**, one
compute tile. Measured from hardware traces.

| variant | avg cycles | MACs/cycle | % of 256 peak | rel-Frobenius |
|---|---|---|---|---|
| native `(4,8,8)` | 5250.9 | 25.0 | 9.8% | **1.21e-07** |
| **bfp16 emul `(8,8,8)`** | **954.7** | **137.3** | **53.6%** | 1.04e-02 |

### The four things that matter

**1. `--emulate-bf16-mmul-with-bfp16` is worth 5.5×** — not the 2× the geometry change
suggests. It takes single-core bf16 GEMM from 9.8% to **53.6% of peak**, which is at or
above what MLIR-AIR reports for well-compiled *multi-core* matmul. It is the single
highest-value knob found so far.

**2. It costs ~10⁵× accuracy** (1.21e-07 → 1.04e-02), because bfp16 shares one exponent
per 8-element block. **Not yet adopted by default** — at ~1% per GEMM over six layers
this may exceed our ≤5e-3 embedding budget. Keep both paths; decide with MTEB in M8.

**3. fp32 accumulation is mandatory.** With bf16 output the L1 accumulator re-rounds at
every K step: error 7.4e-3. With f32 output, same kernel: **1.21e-07**. But it doubles
the C tile, and `64×64×64` with f32 then needs exactly 64 KB of L1 and fails to
allocate — hence 64×64×32.

**4. Per-tile cycles are shape-independent** — 949.8 to 953.7 cycles (0.4% spread)
across N ∈ {384, 1152, 1536} and K ∈ {384, 960}. Only the invocation count changes.
Empirical support for **one tile size across every GEMM in the model**, varying only
shim DMAs and loop bounds (Rösti's minimal-reconfiguration strategy).

### Two hard limits discovered

**L1 budget** (ObjectFifos are double-buffered, 64 KB per tile):
```
2 * (m*k*in + k*n*in + m*n*out)  must be < 65536
64x64x64 bf16->f32 = 65536  ->  fails ('Basic sequential allocation also failed')
64x64x32 bf16->f32 = 40960  ->  fits
```

**DMA buffer-descriptor size field is 10 bits — max 1023.** Bisected: K=960 compiles,
**K=1024 fails**. So MiniLM's `ffn_down` (K=1536) is **unrepresentable** with a naive
access pattern:
```
error: 'aie.dma_bd' op Size 1 exceeds the [0:1023] range.
```
This makes offline weight pre-tiling (M4) a **requirement**, not an optimisation —
exactly TileFuse's design challenge #3.

### Multi-core (`gemm_whole_array.py`)

Our traced copy of `whole_array`. bf16→f32 + bfp16, 512³, tile 64×64×32:

| config | cores | per-core MACs/cyc | % peak | array MACs/cyc | scaling |
|---|---|---|---|---|---|
| 1 core | 1 | 137.3 | 53.6% | — | — |
| 2 cols | 8 | 142.0 | 55.5% | 1135.9 | — |
| 4 cols | 16 | 141.7 | 55.3% | 2266.7 | **99.8%** |

**The compute side of the array is healthy and scales.** The problem is elsewhere:

| cols | cores | npu time (wall clock) | TFLOP/s | scaling |
|---|---|---|---|---|
| 1 | 4 | 785.9 µs | 0.34 | 1.00× |
| 4 | 16 | 304.6 µs | 0.88 | 2.58× (64%) |
| 8 | 32 | 243.0 µs | 1.10 | 3.23× (**40%**) |

At 32 cores the traced compute time is **16.4 µs** against a measured **243 µs** —
**compute is ~7% of NPU-side time**. Adding cores shrinks the 7% and leaves the rest,
which is why core scaling reads 99.8% while wall-clock scaling reads 40%.

Raising arithmetic intensity helps but hits a wall — cols=8, K=N=512:

| M | TFLOP/s |
|---|---|
| 512 | 1.17 |
| 1024 | 1.74 |
| 2048 | 2.25 |
| 4096 | 2.61 |

Linear fit: **≈0.168 µs per M row + ≈144 µs fixed per dispatch**, asymptote **~3.1
TFLOP/s** versus a compute-bound ~16 TFLOP/s. At M=4096 the achieved bandwidth is only
~24 GB/s of the ~40–60 GB/s available — so there is real headroom, and the next wins
are in **data movement**: B-reuse across row blocks, larger L2 megatiles, offline
pre-tiling.

8 columns is worth **1.44×** over 4 at M=4096 (2.56 vs 1.78 TFLOP/s), so we cannot drop
to the traceable width for convenience. See
[`../05-measurement/`](../05-measurement/README.md) for the resulting two-track policy.

### Note on efficiency methodology

`aie::mmul` is **not** a single hardware instruction. The disassembly shows the inner
loop as ~36 bundles containing ~32 `vmac.f` plus `vextbcst`/`vshuffle` operand
preparation, so "intrinsics per cycle" is meaningless. Efficiency is reported as
**MACs/cycle against the datapath peak** (256 for bf16 on aie2p). The static estimate
(~1 `vmac.f`/cycle × ~32 MACs ≈ 28 MACs/cycle) agrees with the measured 25.0.

## saxpy — M1 toolchain proof

`experiments/m1-hello-npu/` · full write-up in
[`tasks/0002`](../../tasks/0002-m1-hello-npu/TASK.md)

**What it computes:** `z = 3x + y`, bf16 in/out, fp32 accumulate, N=4096, one compute
tile, 64 bf16 lanes per iteration.

**Why it exists:** not for performance — it is the smallest kernel that still exercises
the entire chain (IRON design, ObjectFifo DMA, Peano compile, xclbin, XRT dispatch,
hardware trace). It is the M1 gate.

**Measured** (bit-exact vs NumPy, `max_abs_err = 0`):

| variant | cycles | vector fraction |
|---|---|---|
| vector | **335** | 0.382 |
| scalar | **541,662** | 0.0076 |

**1,617×.** The cause is not vector width: the scalar variant calls **`__mulsf3`** —
a software float multiply — **once per element**. See
[research note 0001](../../research/notes/0001-aie-kernel-pitfalls.md).

**Both measurement signals agree:** the emitted loop is a 5-bundle zero-overhead loop
with trip count 61 (`add.nc lc, r2, #-0x3`, `r2=0x40` — three iterations peeled for
software pipelining), so `61 × 5 + prologue ≈ 335`, matching the trace exactly.

**What it tells us going into M2:** the loop body carries 8 real vector ops across ~25
VLIW slots, the rest `nop` — issue-limited, not compute-limited. Fine for a
memory-bound elementwise op, fatal for GEMM. This is precisely why Rösti's
four-independent-accumulator technique exists, and it is the first thing to check in
the M2 disassembly.

## What each kernel page must contain

1. **What it computes**, with exact shapes and dtypes.
2. **Why this design** — the tiling, the dataflow, the accumulator strategy, and what
   was rejected.
3. **The IRON design** — where the `.py` lives, the ObjectFifo topology, the
   `dims_to_stream` layouts.
4. **Measured results** — cycles (min/avg/max) from `get_trace_summary.py`,
   vector-unit utilisation from `get_vector_time()`, and % of the 14.7 TOPS bf16
   ceiling. Cross-checked against `llvm-objdump` instruction counts.
   **See [`../05-measurement/`](../05-measurement/README.md) — wall clock is not a
   valid claim here.**
5. **Correctness** — which M3 golden it was validated against and at what tolerance.
6. **What was tried and rejected**, with numbers. This is the part worth keeping.

## Building blocks we already have

`C:\dev\mlir-aie\aie_kernels\aie2p\` — all bf16, all already bracketed with
`event0()`/`event1()` for tracing:

| File | Symbol | Notes |
|---|---|---|
| `mm.cc` | `matmul_vectorized_2x2_mmul` | bf16→bf16/f32 at (4,8,8); i8→i32 at (8,8,8). Compile-time `-DDIM_M/K/N`, `-DB_COL_MAJ`, `-DC_COL_MAJ` |
| `zero.cc` | `zero_*` | clears C before each K reduction — `mm` accumulates in place |
| `softmax.cc` | `softmax_bf16` | 3-pass (max, exp2 after ×log2e, normalize), `SM_VEC_LEN=32` |
| `layer_norm.cc` | `layer_norm`, `layer_norm_welford` | vec width 16 |
| `gelu.cc` | `gelu_bf16` | **input_size hard-coded to 1024** |
| `rms_norm.cc`, `rope.cc`, `silu.cc`, `swiglu.cc`, `bf16_exp.cc` | | not needed for BERT |

Device math helpers: `aie_runtime_lib/AIE2P/vec_math.h` (`getRsqrtBf16`, `getSqrtBf16`,
`getErfBf16`, `getSigmoidBf16`, …) and `lut_based_ops.h` (`getExpBf16` via
`aie::lut` + `aie::parallel_lookup`).

**`getErfBf16` matters**: BERT uses *exact erf* GELU, not the tanh approximation.

Loop-pragma macros: `aie_kernels/aie_kernel_utils.h` — `AIE_PREPARE_FOR_PIPELINING`,
`AIE_LOOP_MIN_ITERATION_COUNT(n)`, `AIE_LOOP_FLATTEN`. These map to Peano
`#pragma clang loop` on our toolchain. Note Peano treats the pipelining hint as a
**trip-count hint only** and still emits a runtime loop.

Design templates worth copying:
- `programming_examples\basic\matrix_multiplication\whole_array\whole_array.py` (605
  lines) — 4 rows × N columns, three ObjectFifo layers, the `dims_to_stream` pre-packing.
- `programming_examples\ml\norm\` — RMS/LayerNorm over sequence × embedding_dim, 8 cores.
- `programming_examples\basic\transposes\` — four transpose strategies including the
  `shuffle` (16×16 VSHUFFLE) approach.

## What does not exist and we must write

| Kernel | Why it's missing | Milestone |
|---|---|---|
| **Fused GEMM + bias + activation** | mlir-aie has GEMM and GELU as separate dispatches | M5 |
| **Fused attention** | no attention example at all in `programming_examples` | M5 |
| **Row-wise softmax across > 1024 elements** | `softmax.cc` is per-1024-tile with no cross-tile reduction | M5 |
| **Mean/CLS pooling + L2 normalize** | no pooling head anywhere | M5 |

The fusion work is not optional polish — finding **F1** says per-dispatch overhead is
the dominant cost, and MLIR-AIR measured **2.24×** from fusing five kernels into one.
The target is **one dispatch per encoder layer**.

## Optimisation method

**Read `C:\dev\mlir-aie\skills\aie-kernel-opt\SKILL.md` before touching kernel
performance.** It is AMD's own measure-first playbook for Peano-compiled kernels:
establish a baseline over ≥20 iterations, gate on bit-exactness, **ablate to attribute**
each change, change one thing at a time, and verify in the emitted `.o`.

Its lever catalogue, in priority order: loop hints → compile-time constants → killing
`__divsi3` → branch-splitting → vectorised epilogue → operand-layout pre-pack →
explicit wide packing → wider `mmul` → DMA-layout offload.

Two project-specific levers on top:

- **`--emulate-bf16-mmul-with-bfp16`** upgrades bf16 mmul geometry from `4×8×8` to
  `8×8×8` — a 2× MAC improvement, Peano-compatible. First thing to A/B in M2.
- **Four independent accumulators** in the inner loop to avoid RAW hazards on the
  accumulator (Rösti's method; verify by the absence of no-ops in the disassembly).
