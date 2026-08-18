# 0004 — Switching design costs more than computing, and it scales with descriptors

*Found during [M7](../../tasks/0024-m7-dispatch-cost-anatomy/TASK.md), 2026-08-17.*

**Severity: defines the performance model.** The most expensive single operation
anywhere in this stack is not a kernel and not a dispatch — it is changing which
design the array is configured for. It costs **89 µs to 2.4 ms**, against ~150 µs
for a whole dispatch and ~20 µs of actual compute for a MiniLM GEMM tile.

Nothing reports it. It appears as the kernel simply taking longer.

## The measurement

`runtime/src/main.cpp --probe*`. Every number is with **no host work in the
loop**: buffers are staged once, then `dispatch_only()` in a tight loop.

Repeating **one** design, times scale with work and match
[0010](../../tasks/0010-m5-b-reuse-and-cost-model/TASK.md)'s validated cost
model. Alternating **two** designs, they do not:

```
    same design repeatedly:            alternating:
      qkv (4 col)       280 us           qkv <-> ffn_up          1519 us
      attn_out          165 us           qkv <-> gelu            1980 us
      ffn_up            340 us           layernorm <-> softmax    721 us
      ffn_down          343 us
```

## What it is not

**Not reconfiguration in the sense of "different data must be loaded."** Load
the *same xclbin* into two contexts and alternate — identical configuration
bytes — and it costs the same as two different designs:

| | 2 cols | 4 cols | 8 cols |
|---|---|---|---|
| `qkv` alone, one context | 451 µs | 288 µs | 221 µs |
| **`qkv` ↔ `qkv′`, same xclbin** | **1061 µs** | **1478 µs** | **2569 µs** |
| `qkv` ↔ `ffn_up` | 1122 µs | 1527 µs | 2612 µs |

**Not resource pressure from too many resident contexts.** Two designs loaded
instead of seven: 1509 µs vs 1519 µs.

**Not eviction.** `xrt-smi examine --report aie-partitions` sampled twice during
an alternating run shows all seven `npuembed.exe` contexts simultaneously
`Active`, with **`Suspensions = 0` and `Migrations = 0`** throughout, while
submission counters climb (180 → 388 per context; 389 → 840 for LayerNorm,
which is dispatched 13 times per encode against 6 for the rest). The contexts
are never evicted. Switching between co-resident contexts still costs the full
amount.

## What it is

The partition report shows **one partition spanning all eight columns**:

```
  Partition Index   : 0
    Columns: [0, 1, 2, 3, 4, 5, 6, 7]
```

If that is the granularity, only one design's configuration can be live on the
array at a time no matter how narrow the designs are or how many contexts the
driver keeps resident. The context object stays; the array configuration is
rewritten.

And it is rewritten **unconditionally** — that is what A→A′ proves. The driver
does not take the shortcut that the incoming configuration is byte-identical to
the outgoing one. This is a missed fast path, not inevitable work.

### The serialisation was an inference. Now it is measured.

Two processes, each with its own context, both dispatching (cheaper than
instrumenting FastFlowLM, and cleaner than threads, which share a context unless
explicitly given two):

| designs | 1 process | 2 processes | aggregate |
|---|---|---|---|
| 1 column | 51.9 | 38.2 + 38.4 | **1.48×** |
| 2 columns | 52.8 | 39.0 + 39.2 | **1.48×** |
| 4 columns | 46.3 | 33.1 + 33.5 | **1.44×** |

Neither 1.0× nor 2.0× — and, decisively, **independent of design width**.
Spatial partitioning requires narrow designs to scale better; they do not. With
dispatch+wait at 72% of wall, full serialisation on the array predicts
`1/0.72 = 1.39×`, and the extra is the submit and sync that also overlap. So the
1.46× is host-side overlap, not array concurrency, and serialisation stands with
evidence rather than by assumption. By proxy it also answers the FastFlowLM
question: their six active contexts are almost certainly serialised too.
→ [`tasks/0025`](../../tasks/0025-m7-batching-and-crossover/TASK.md)

### What that does *not* claim

**Observed:** partition index 0 covers all eight columns, on this machine, with
this driver, for contexts created the way `npu_device.cpp` creates them
(`xrt::hw_context(device, uuid)`).

**Not shown:** that this is the only possible partitioning. Whether it is a
driver policy, a firmware property, or a consequence of how the context is
created is **untested**, and the difference decides whether this is a wall or a
knob. Three things would close it, in increasing cost:

1. **Is the partition width settable at context creation?** *Still open, but
   the search was in the wrong place.* `xrt::hw_context` takes
   `cfg_param_type = std::map<std::string, uint32_t>`, an untyped map with no
   relevant key in the public headers (only priority is evidenced, via
   `xrt-smi`) — but the column count is not decided at XRT level. It is decided
   in the driver's create-hwctx call. On Linux that is
   `amdxdna_drm_create_hwctx` in the `amd/xdna-driver` UAPI header, which
   carries a tile count the driver converts to columns. If so, partition width
   is a **parameter at creation, not a fixed policy** — a knob. No such header
   exists locally: the Windows driver is closed and its XRT shim takes another
   path. But that public header is the best available documentation of the same
   firmware interface, and reading it answers a sharper question than the C++
   header did, more cheaply than either alternative below.
2. ~~**Does FastFlowLM's six-active-context arrangement actually run
   concurrently?**~~ **Answered by proxy** (above): two of our own processes
   serialise on the array, width-independently, so six of theirs almost
   certainly do too. High submission counts never excluded serialisation.
3. **Does asynchronous dispatch from two threads behave differently from
   synchronous?** Every measurement here dispatches synchronously from one
   thread. If the partition really is the whole array, threading cannot help —
   which makes this a test of the driver's partitioning policy rather than of
   our threading model.

The diagnosis stands either way. Its *consequences* depend on which of these is
true.

## It scales with descriptor count, not with columns

The obvious model is per-column, and it fits three points well
(≈ 55 µs + 286 µs/column). It is wrong, and a control refutes it.

`experiments/m7-switch-cost/` builds a minimal 1-column design — one worker, two
ObjectFifos, a vector copy — to compare against a 1-column GEMM at **identical
width**. A per-column model requires them to cost the same:

| design | cols | `aie.dma_bd` | `aie.lock` | switch |
|---|---|---|---|---|
| passthrough | 1 | 6 | 8 | **89 µs** |
| gelu | 1 | 24 | 32 | **298 µs** |
| qkv | 1 | 63 | 48 | **365 µs** |
| qkv | 2 | 110 | 88 | **628 µs** |
| qkv | 4 | 204 | 168 | **1190 µs** |
| qkv | 8 | 388 | 320 | **2348 µs** |

**4.1× spread at one column.** The per-column model is refuted. What does hold,
across a 65× range of descriptor counts and an 8× range of widths:

> **switch ≈ 25 µs + 7.2 µs per lock**
> (equivalently ≈ 49 µs + 5.8 µs per DMA buffer descriptor) — within ~15%
> at every point.

Columns only ever correlated because more columns means more descriptors.

The magnitude corroborates the mechanism. 286 µs per column is far too much to
be bulk DMA of state — a column's ELFs are tens of KB — and entirely plausible
as a few thousand individual MMIO writes to BD registers, stream switches and
locks. Per-descriptor scaling is what that predicts.

## Consequences

**The one sentence that binds this to the rest of the project:** data-movement
optimisation and switch optimisation spend the *same budget* — every descriptor
added to feed the array better is paid for again at every switch.

That trade is invisible in the literature this project indexes, because those
papers measure kernels in isolation. This is an end-to-end pipeline with 49
switches per encode, and the arithmetic comes out differently. It also reframes
M4: pre-tiling was measured as a throughput wash
([0007](../../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md)), and the reason to
keep it is now partly that it keeps the runtime's descriptors simple.

1. **Simpler dataflow buys cheaper switches.** The cost of a design is not only
   what it computes but how many descriptors it programs.
2. **Narrower is faster while switches dominate** — but only down to a point,
   and the bottom of the curve is flat. Three repeats per width:

   | GEMM columns | 1 | 2 | 4 | 8 |
   |---|---|---|---|---|
   | seq/s (mean of 3) | 52.1 | 52.8 | 46.0 | 34.7 |
   | range | 0.7 | 1.3 | 0.7 | 0.8 |

   **1 and 2 columns are indistinguishable** — 0.7 apart with overlapping
   ranges. 4 and 8 are decisively worse. So the honest claim is "≤ 2 columns",
   not "2 beats 1"; a single run had suggested 53.0 vs 51.4 and that gap did not
   survive repetition. (The same lesson as
   [0008](../../tasks/0008-m5-bfp16-real-data/TASK.md): one isolated run is not
   a measurement.)

   All of this is a consequence of the switch:compute ratio, not a property of
   the hardware, and it must be re-measured after any change that lowers the
   switch count — batching moves the ratio monotonically, so the optimum will
   walk back toward wider designs.
3. **Operator-major reordering does not apply.** Layer *L*+1 depends on layer
   *L*, so an encoder's dispatches are a chain, not a set. Batching and fusion
   are the two levers that survive.
4. **The NPU is genuinely shared.** The same report shows
   `WorkloadsSessionHost.exe` holding three resident contexts with non-zero
   suspension counts (308, 3, 32) during our runs. CLAUDE.md rule 1 is not
   hypothetical.

## How to measure it yourself

```powershell
runtime\build\npuembed.exe .. --probe                      # repeat vs alternate
runtime\build\npuembed.exe .. --probe-pair                 # 2 resident vs 7
runtime\build\npuembed.exe .. --probe-ctx                  # same xclbin, 2 ctx
runtime\build\npuembed.exe .. --probe-design <dir>         # one design in isolation
C:\Windows\System32\AMD\xrt-smi.exe examine --report aie-partitions
```

Descriptor counts come from the **placed** MLIR, `input_with_addresses.mlir` —
never `aie.mlir`, which is pre-placement and contains no coordinates or
descriptors at all. Same file distinction the trace flow depends on.

Related: [[0003-two-designs-per-process]] — the *other* thing that looks like
"two designs interfere", which turned out to be our own missing sync.
