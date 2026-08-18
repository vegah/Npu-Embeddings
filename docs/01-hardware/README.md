# The hardware: AMD XDNA2 / AIE2P (Strix Point)

Everything here is **verified on this machine** unless marked otherwise. Sources are
either direct queries (`xrt-smi`, `get_target_model()`) or the mlir-aie source tree.

## This machine

| | |
|---|---|
| CPU | **AMD Ryzen AI 9 HX 370 w/ Radeon 890M** — Strix Point, 12C/24T |
| NPU | **XDNA2**, internal name **AIE2P**, a.k.a. **NPU2**. Marketed 50 TOPS INT8 |
| PnP device | `NPU Compute Accelerator Device`, `PCI\VEN_1022&DEV_17F0`, service `IpuMcdmDriver` |
| NPU driver | **32.0.20102.3930** (2026-05-07) |
| NPU firmware | **1.1.2.64** |
| XRT | **2.21.0** |
| `xrt-smi` reports | `[00c6:00:01.1] NPU Strix` |
| OS | Windows 11 Pro 10.0.26200 |
| RAM | 29.6 GiB |
| GPU | Radeon 890M iGPU only, no discrete |

```powershell
C:\Windows\System32\AMD\xrt-smi.exe examine     # not on PATH by default
```

## Naming, once and for all

The same silicon has four names depending on who is speaking. This table resolves
nearly every confusing document you will read:

| mlir-aie internal | Marketing | Generation | Chips |
|---|---|---|---|
| `aie` | AIE | — | Versal VCK5000 |
| `aie2` | AIE-ML / XDNA | **NPU1** | Phoenix, Hawk Point |
| **`aie2p`** | **XDNA2** | **NPU2** | **Strix Point**, Strix Halo, Krackan/Kraken |
| `aie2ps` | AIE-MLv2 | — | Telluride |

**We target `aie2p` / `npu2`.** `NPU2=1` is exported by `iron_env.ps1`.

Phoenix's "10 TOPS" and Hawk Point's "16 TOPS" are the *same array* at 1.0 vs 1.6 GHz.

## How to think about this machine

Worth reading before the tables, because almost every mistake we have made came from
carrying over a GPU intuition.

**It is a spatial dataflow array, not a throughput processor.** A GPU hides latency by
oversubscribing threads; you hand it a kernel and the hardware finds parallelism. Here
there is no scheduler, no cache, no branch predictor, and **cores do not stall**. You
place data at specific addresses and you schedule its movement explicitly. Nothing is
discovered at runtime.

Three consequences that shape everything in this repo:

1. **Execution time is countable.** Fixed instruction latencies mean the emitted loop
   body *is* the cost. That is why static instruction counts and hardware traces are
   both valid, and why they must agree — see
   [`../05-measurement/`](../05-measurement/README.md).
2. **The program is mostly data movement.** A "kernel" here is a small compute inner
   loop wrapped in a large amount of explicit DMA choreography: DDR → mem tile → core,
   and back. Most of our effort, and all of our failures so far, have been in that
   choreography rather than in the arithmetic.
3. **The compiler will not rescue a bad placement.** If a tile does not fit in L1, or a
   fifo needs more descriptors than a mem tile has, you get a resource-allocation error
   — not a slower program. The limits below are hard walls, not gradients.

**The mental model that has held up:** picture 32 small cores in 8 columns of 4. Each
column has one mem tile (L2) and one shim (to DDR). Data flows *up* a column — DDR →
L2 → L1 — and results flow back down. Cores in a column share their column's B operand
by broadcast; cores across columns work on different output tiles. Your job is to keep
every core's inner loop fed without exceeding, at any level, the bytes, the buffer
descriptors, or the DMA channels available there.

## Array structure

Verified live via `get_target_model()` and from
`C:\dev\mlir-aie\include\aie\Dialect\AIE\IR\AIETargetModel.h`:

| | **npu1** (Phoenix) | **npu2** (Strix — ours) |
|---|---|---|
| Columns | 4 addressable | **8** |
| Rows | 6 | 6 |
| Row layout | row 0 = shim DMA, row 1 = memtile, rows 2–5 = compute | same |
| **Compute cores** | 4 × 4 = 16 | **8 × 4 = 32** |
| L1 per compute tile | **64 KB** (`0x10000`) | **64 KB** |
| L2 per memtile | **512 KB** (`0x80000`) | **512 KB** |
| Memtile rows | 1 | 1 |
| Arch enum | `AIE2` (2) | `AIE2p` (3) |
| Neighbour mem base | south `0x40000`, west `0x50000`, north `0x60000`, east `0x70000` | same |

Also: **16 KB program memory** per compute tile; L1's 64 KB is split into **8 banks,
each independently DMA-accessible**. Clock measured at **1.808 GHz** on a Krackan SKU
(external source; not separately verified for HX 370).

`docs/Devices.md` in mlir-aie notes: *"The hidden zeroth-column of Phoenix NPUs is
irregular and no longer exposed through MLIR-AIE."*

## Compute: MACs per cycle per core

| dtype | XDNA1 (aie2) | XDNA2 (aie2p) |
|---|---|---|
| int8 × int8 | 256 | **512** |
| int16 × int8 | 128 | ≥128 |
| int16 × int16 | 64 | — |
| **bf16 × bf16** | 128 | **256** |
| **bfp16** (block FP) | emulated | **native 8×8×8 = 512 MACs** — the peak mode |
| fp8 | no | yes |
| MX9 / MX6 / MX4 | no | yes |
| fp32 | emulated | emulated |
| **int4** | **not a native MAC dtype** — storage only, dequantised on-tile | same |

Sanity check: 50 TOPS ≈ `2 × 512 MACs × 32 tiles × ~1.53 GHz`. The arithmetic works.

> ⚠️ **bfp16 is the peak mode but is unreachable for us.** Native bfp16 kernels
> (`mm_bfp.cc`) require the Chess compiler, which does not exist on native Windows.
> The Peano-compatible consolation prize is `--emulate-bf16-mmul-with-bfp16`, which
> upgrades bf16 matmul *geometry* to 8×8×8. See [`../02-toolchain/`](../02-toolchain/README.md).

## Matmul primitive geometry

From `aie_kernels/aie2p/mm.cc` and `python/iron/kernels/linalg.py`. `(r,s,t)` is the
`aie::mmul` shape.

| in → out | aie2 (NPU1) | **aie2p (NPU2 — ours)** |
|---|---|---|
| i8 → i8 / i16 / i32 | 4×8×8 | **8×8×8** |
| i16 → i16 / i32 | 4×4×4 | **4×4×8** |
| **bf16 → bf16 / f32** | 4×8×4 | **4×8×8** |
| bf16 → bf16 / f32, *bfp16-emulated* | — | **8×8×8** |

**There is no int4 matmul anywhere in `aie_kernels/`.** This is why int4 is deferred:
it would be a storage format requiring an in-core dequant to bf16 first.

**8×8 is the universal block granularity** — independently confirmed by TileFuse's L1
layout, Zen-Attention's shuffle transpose, and Rösti's 4×8/8×4 VMAC.

## Memory bandwidth — real, but not the binding constraint at our sizes

**The NPU gets only ~40–60 GB/s of DRAM read bandwidth.** The iGPU gets ~130 GB/s on
the same chip. Zen-Attention (AMD) quotes ~60 GB/s combined read+write; the Gemma3
team measured **below 40 GB/s** on Krackan and reported that *every* one of their
kernels was therefore memory-bound.

For MiniLM specifically: 21.3 MB of bf16 encoder weights streamed once per sequence
costs ~0.18 ms, which **equals the entire theoretical compute time for one sequence**.
At batch 1 we are purely memory-bound. This is why batching is an architectural
requirement, not an optimisation.

> ### What we measured ourselves, and why it changes the conclusion
>
> This section used to be titled *"the binding constraint"*. Our own numbers say
> that is wrong at MiniLM's shapes.
> ([`tasks/0010`](../../tasks/0010-m5-b-reuse-and-cost-model/TASK.md))
>
> Fitting runtime against DDR traffic across the four MiniLM GEMMs, then validating
> on a sweep of M it was **not** fitted to (±1.4% at the two largest sizes):
>
> ```
> t = 150.4 µs  +  traffic / 33.0 GB/s
> ```
>
> Two things follow, and both are easy to get backwards:
>
> - **Naively dividing traffic by runtime gives 8.6–19.0 GB/s**, which looks like
>   catastrophic bandwidth starvation. It is an artifact. That average folds the
>   fixed 150 µs into the rate. The **marginal** bandwidth is **33 GB/s**, inside
>   the expected band.
> - **The fixed cost is 40–73% of runtime at M=512.** We are **dispatch-bound**,
>   not bandwidth-bound, at the sizes a single MiniLM encode actually produces.
>
> This is why pre-tiled weights bought nothing
> ([`tasks/0007`](../../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md)): layout and
> bandwidth optimisations attack something that is not the limit here.
>
> **Practical rule.** Before optimising data movement, check whether the transfer
> is large enough for bandwidth to matter at all. Below roughly 5 MB per dispatch
> on this part, the 150 µs dominates and the only lever that helps is doing more
> work per dispatch.

## The three memory levels, and what actually runs out

Most surprises on this hardware are not about bytes. They are about *how many
descriptors and channels* you have to move the bytes. Worth internalising in this
order:

| level | where | size | what runs out first |
|---|---|---|---|
| **L1** | per core | 64 KB | **bytes.** Budget `2·(m·k·in + k·n·in + m·n·out) < 65536` |
| **L2** | per mem tile (one per column) | 512 KB | **buffer descriptors**, long before bytes |
| **L3** | DDR | — | bandwidth, and the 10-bit BD size field |

**L1 — bytes.** The factor 2 is double buffering. Exceeding it gives the opaque
`'aie.tile' op Basic sequential allocation also failed`. A 64×64×48 bf16→fp32 tile
costs 53,248 B and fits; there is not much headroom.

**L2 — descriptors, not bytes.** This is the counter-intuitive one and it cost us a
whole milestone attempt. An ObjectFifo's `depth` maps **1:1 to mem-tile buffer
descriptors**, and the pool is shared with every other fifo on that tile (A in, C out,
and their splits/joins). Measured ceiling for staging B tiles
([`tasks/0010`](../../tasks/0010-m5-b-reuse-and-cost-model/TASK.md)):

| columns | tiles we wanted to stage | max depth that compiles |
|---|---|---|
| 4 | 48 | **6** |
| 8 | 24 | **4** |

288 KB of a 512 KB mem tile would have fit comfortably. The descriptors did not.
Symptom: `no space for this BD`.

The obvious workaround — one large object instead of many small ones, so the same
bytes cost one descriptor — hits a *different* wall: `number of input DMA channel
exceeded`, on a **core** tile. Feeding tile-sized L1 consumers from one big L2 object
needs more input DMA channels than a core tile has.

**A core tile has 2 input and 2 output DMA channels.** That caps how many
distinct streams one kernel can take, independently of any size or descriptor
budget. A LayerNorm taking `input`, `gamma`, `beta` and `output` separately is
already too many:

```
error: tile (0, 3) requires 3 input/1 output DMA channels,
       but only 2 input/2 output available
```

The fix is to pack — gamma and beta became one buffer
([`tasks/0020`](../../tasks/0020-m5-layernorm-kernel/TASK.md)). **This is the
first constraint a fused layer will hit**, because fusion is precisely the act
of giving one kernel more inputs.

**L3 — the 10-bit BD size field (max 1023).** Any single dimension of a DMA access
pattern must stay under 1024. **This is a property of the access pattern, not of the
tensor.** The same `[1536, 384]` operand fails in a single-core design, where B is
walked as one column strip so the k-blocks collapse into `size=1536`, and compiles fine
in the whole-array design, where the tiler keeps k-blocks as their own dimension
(`<size=24, stride=24576>`). Getting this backwards cost us a milestone premise —
see [`tasks/0007`](../../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md).

## Tracing costs routing, and a full array cannot be traced

Adding a single trace flow to a fully-packed 8-column design exhausts circuit-switch
routing (`Unable to find a legal routing`, or `max number of packet IDs reached` with
`--packet-sw-objFifos`). Traceable widths are **2 columns** at
`(trace_col=1, egress_shim_col=1)` and **4 columns** at `(0,0)`, found by exhaustive
search.

Since 8 columns is worth 1.44× over 4 and is the production configuration, this is
permanent, and it forces the two-track measurement policy in
[`../05-measurement/`](../05-measurement/README.md): **per-core cycles at 4 columns
(traced), end-to-end throughput at 8 columns (wall clock)**. That is legitimate only
because per-core cost is shape- and width-independent — 137.3 → 142.0 → 141.7
MACs/cycle from 1 to 8 to 16 cores.

## DMA constraints

These bite in practice and are worth memorising:

- **Minimum stride / reorder granularity is 4 bytes.** DMA therefore *cannot*
  transpose bf16 (2 bytes) on its own. The workaround is 8×8 block-transpose in DMA
  plus an in-core `SHUFFLE` within each block.
- **Payloads want 128-byte divisibility.** TileFuse duplicates 64 zero-points per tile
  purely to satisfy this.
- **Buffer-descriptor stride registers are limited.** A naive row-major weight layout
  breaks past ~8K dims; interleaved column-major ordering (so one AIE column's tiles
  are contiguous) lifts it to 32K and improves burst efficiency.
- **XDNA2 L2 MM2S (read) channels support hardware padding across {D0,D1,D2}** — use
  it to pad odd shapes for free instead of emitting a `Pad` operator.
- **`VSHUFFLE` issues on a different VLIW slot than `VMAC`**, so in-core swizzle can be
  genuinely free. Rösti verified this by disabling it and measuring no change.

## Realistic performance expectations

Do **not** plan against 50 TOPS.

| Figure | Value | Source |
|---|---|---|
| Marketing peak int8 | 50 TOPS | AMD |
| Theoretical peak int8 | ~59 TOPS | `2 × 512 × 32 × 1.808 GHz` |
| **Attainable int8 GEMM** | **38.05 TOPS** | arXiv 2512.13282 |
| **Attainable bf16 GEMM** | **14.71 TOPS** | arXiv 2512.13282 |
| bf16 GEMM, 512³ megatile | 13.7 TOPS | [2602.06063](https://arxiv.org/abs/2602.06063) |
| bf16 GEMM, 128×512×512 megatile | 5.9 TOPS | same |
| Compiled matmul efficiency, bf16 | ~48–50% of peak | [2510.14871](https://arxiv.org/abs/2510.14871) |
| Compiled matmul efficiency, i8 / i16 | 59% / 78.7% | same |

**Plan against 14.7 TOPS bf16** and treat 25–40% end-to-end as an ambitious target for
a full encoder (which has LayerNorm, softmax, GELU and inter-GEMM DMA, not just matmul).

## Other hardware notes

- **9 × 512-bit accumulator registers** per core (mlir-aie programming guide,
  section-4c). For 16×16 you get 64 MACs for matmul but only **32 accumulator lanes
  for elementwise**, capping eltwise MAC utilisation at 50%. One external source
  claims XDNA2 has 5 × 2048-bit accumulators instead — **treat as uncertain**.
- Core-to-core **cascade streams** exist for spatial reduction (used by Zen-Attention
  for cross-core softmax).
- Two inter-core paths: direct neighbour load/store into adjacent L1, and AXI4 stream
  DMA to non-adjacent tiles (with broadcast).
- **AIE cores do not stall** — no caches, no out-of-order execution, no branch
  prediction, fixed instruction latencies. This is why counting instructions in the
  emitted loop body predicts execution time, and it is one half of our measurement
  doctrine.
