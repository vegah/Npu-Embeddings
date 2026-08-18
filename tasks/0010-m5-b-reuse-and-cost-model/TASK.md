# 0010 — M5: B reuse is blocked, and a cost model that says what it is worth

- **Date** 2026-08-17
- **Milestone** M5
- **Status** done — **B reuse not expressible in this design; a validated cost
  model quantifies it at 1.25–1.7× and reframes what "starved" means**

## Goal

Attack the one lever [0007](../0007-m5-pretiled-gemm-on-npu/TASK.md) identified
and nothing has touched: **B is re-streamed from DDR once per row block.** M2
estimated 16× re-streaming at M=4096.

## Context

Pre-tiling turned out to be a throughput wash
([0007](../0007-m5-pretiled-gemm-on-npu/TASK.md),
[0008](../0008-m5-bfp16-real-data/TASK.md)), which left B reuse as the last
standing explanation for M2's "the array is starved, not slow".

## What was done

### 1. The mechanism exists, and is documented for exactly this

`ObjectFifo(..., repeat_count=N)`:

> *"causes the MemTile DMA to replay the buffer descriptor this many times
> **without a new DMA transfer from L3**"*

So: stage a column's whole B slice in L2 once, let the mem tile replay it for
every row block, and fill B from DDR exactly once per column instead of once per
row block.

Implemented as `--b-reuse` in `gemm_pretiled.py`: the L3→L2 fifo depth becomes
the slice size, the L2→L1 forward gets `repeat_count = M // m // n_aie_rows`,
and the `rt.fill(B...)` moves out of the row-block loop.

### 2. It does not compile — limit 1, the mem-tile BD pool

```
error: 'aie.tile' op ... no space for this BD
note: aie.objectfifo @B_L3L2_1(..., 48 : i32) : !aie.objectfifo<memref<3072xbf16>>
Resource allocation pipeline failed
```

Not L2 *memory* — 288 KB of a 512 KB mem tile would have fit. **Buffer
descriptors.** Depth maps 1:1 to BDs, and the pool is shared with the A and C
fifos on the same tile.

Swept the ceiling directly:

| columns | full column B slice | max depth that compiles |
|---|---|---|
| 4 | **48 tiles** | **6** |
| 8 | **24 tiles** | **4** |

An order of magnitude short at both widths.

### 3. The obvious workaround — limit 2, core-tile DMA channels

Fewer, larger objects use fewer descriptors for the same bytes, so try one L2
object holding the entire slice (`b_reuse="mega"`, depth 2 → 2 BDs):

```
error: 'aie.tile' op number of input DMA channel exceeded!
note: see current operation: "aie.tile"() <{col = 3 : i32, row = 1 : i32}>
```

Row 1 is a **core** tile. Handing one big L2 object to tile-sized L1 consumers
makes the lowering instantiate more input DMA channels than a core tile has.

**Two different resource limits, at two different levels, both hard.** B reuse is
not reachable by adjusting fifo shapes in this design. It needs a dataflow
change — core-side blocking over row blocks, so a core computes several output
row-tiles from one resident B tile — which requires several C accumulators live
in L1 at once and is a redesign of `core_fn`, not of the fifos.

### 4. So: what would it have been worth?

Rather than guess, build a cost model from measurements already in hand — the
four MiniLM shapes at M=512, 8 columns, one process per measurement
([0008](../0008-m5-bfp16-real-data/TASK.md)) — and fit runtime against DDR
traffic:

```
traffic = M·K·2·(N/n/cols)      A, re-streamed per n-block group
        + K·N·2·(M/m/rows)      B, re-streamed per row block
        + M·N·4                 C
```

| fit | fixed | slope | R² |
|---|---|---|---|
| **vs DDR traffic (MB)** | **150.4 µs** | 30.3 µs/MB → **33.0 GB/s** | **0.902** |
| vs MACs | 154.9 µs | 0.611 µs/MMAC | 0.859 |

### 5. The model validates on data it was not fitted to

Held the shape fixed and swept M — a different axis entirely:

| M | row blocks | traffic | predicted | measured | error | TFLOP/s |
|---|---|---|---|---|---|---|
| 512 | 2 | 4.7 MB | 293.4 µs | 325.9 µs | +11.1% | 1.85 |
| 1024 | 4 | 9.4 MB | 436.4 µs | 412.5 µs | −5.5% | 2.93 |
| 2048 | 8 | 18.9 MB | 722.4 µs | 723.5 µs | **+0.2%** | 3.34 |
| 4096 | 16 | 37.7 MB | 1294.3 µs | 1275.8 µs | **−1.4%** | 3.79 |

Within 1.4% at the two largest sizes, on a model fitted to four different shapes
at a single M.

### 6. What B reuse would buy

Streaming B once instead of once per row block saves `K·N·2·(rowblocks−1)`:

| M | B traffic saved | measured | predicted with reuse | speedup |
|---|---|---|---|---|
| 512 | 1.2 MB | 325.9 µs | 257.6 µs | **1.26×** |
| 1024 | 3.5 MB | 412.5 µs | 329.1 µs | 1.25× |
| 2048 | 8.3 MB | 723.5 µs | 472.1 µs | 1.53× |
| 4096 | 17.7 MB | 1275.8 µs | 758.1 µs | **1.68×** |

**Worth a redesign, and worth more the more we batch** — which is the direction
F2 already pushes.

## The reframing, which matters more than the blocked lever

Naively dividing traffic by runtime gives an alarming number:

| shape | effective GB/s | B share of traffic |
|---|---|---|
| `qkv` | 19.0 | 33% |
| `proj` | **8.6** | 33% |
| `ffn_up` | 18.9 | 33% |
| `ffn_down` | 15.1 | 50% |

Against the ~40–60 GB/s the NPU is thought to get, that reads as badly
bandwidth-starved. **It is an artifact of the fixed cost.** The *marginal*
bandwidth is **33 GB/s**, comfortably inside the expected band. What the average
is measuring is 150 µs of overhead spread over a small transfer.

At M=512, that fixed cost is **40–73% of total runtime** (73% for `proj`). It
independently reproduces M2's ~144 µs from a completely different fit.

Three consequences:

1. **We are not bandwidth-starved at MiniLM's natural sizes. We are
   dispatch-bound.** F1 all over again, now with our own number: 150 µs.
2. **This is why pre-tiling was a wash.** It optimises layout and bandwidth —
   neither is the binding constraint here. [0007](../0007-m5-pretiled-gemm-on-npu/TASK.md)'s
   negative result stops being surprising.
3. **Batching beats every layout change tried so far.** M=512 → 4096 takes
   `ffn_down` from 1.85 to 3.79 TFLOP/s — 2.05×, purely from amortising the
   150 µs. No kernel change, no layout change.

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

# B reuse, full slice in L2 -- fails on the BD pool
python -c "
import sys; sys.path.insert(0,'experiments/m5-pretiled-gemm')
import aie.iron as iron
from aie.iron.device import from_name
iron.set_current_device(from_name('npu2', n_cols=None))
import gemm_pretiled as g
g.run_one(512,1536,384,64,64,48,8,True,262144,pretiled=False,b_reuse=True,trace=False)
"

# the BD ceiling sweep, and the one-big-object variant, use b_reuse=<int> and
# b_reuse='mega' in the same call.

# time vs M at 8 columns (one process per point)
#   -> artifacts/scaling_M_ffn_down_c8.json
```

## Result

| claim | verdict |
|---|---|
| B reuse is expressible with `repeat_count` | **No.** Blocked by the mem-tile BD pool, and by core-tile DMA channels when worked around |
| The array is bandwidth-starved | **No.** Marginal bandwidth is 33 GB/s, inside the expected band |
| Runtime is dominated by a fixed per-dispatch cost | **Yes — 150 µs**, 40–73% of runtime at M=512, reproducing M2's ~144 µs independently |
| B reuse would still be worth having | **Yes — 1.26× at M=512, 1.68× at M=4096**, from a model validated to ±1.4% |
| Batching is the cheapest available win | **Yes — 2.05×** from M=512 to M=4096 with no code change |

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| `no space for this BD` at depth 48, and still at 24 | ObjectFifo depth maps 1:1 to mem-tile buffer descriptors, and the pool is shared with A and C | None. Ceiling measured at 6 tiles (4 cols) / 4 tiles (8 cols) |
| `number of input DMA channel exceeded` on a core tile | One large L2 object feeding tile-sized L1 consumers needs more input DMA channels than a core tile has | None. Recorded as the second independent limit |
| Effective bandwidth looked catastrophic (8.6 GB/s) | Dividing by total runtime folds the 150 µs fixed cost into the rate | Fit fixed + marginal separately. The marginal rate is 33 GB/s |

## Artifacts

- `experiments/m5-pretiled-gemm/gemm_pretiled.py` — `b_reuse` flag (`False` |
  `int` | `"mega"`), kept because the failure modes are the finding
- `experiments/m5-pretiled-gemm/artifacts/scaling_M_ffn_down_c8.json`
- `experiments/m5-pretiled-gemm/artifacts/bench_all_shapes_c8_isolated.json`
  — the data the model is fitted to

## Next

1. **Attack the 150 µs, not the data movement.** It is 40–73% of runtime at
   MiniLM's sizes and the largest single lever. F1's prescription — one resident
   `.xclbin`, fused whole layers, one dispatch per encoder layer — is now backed
   by our own number.
2. **Batch.** 2.05× is available today with no code change, and it makes
   everything else worth more.
3. **B reuse stays on the list but needs a core-side redesign**, not a fifo
   change: block over row blocks inside `core_fn` with several C accumulators
   resident in L1. Worth 1.68× at M=4096, on top of batching.
4. The cost model should be re-fitted once any of the above lands — it is
   currently anchored to one design at 8 columns.
