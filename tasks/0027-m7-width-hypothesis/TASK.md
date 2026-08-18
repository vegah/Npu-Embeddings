# 0027 — M7: is MiniLM simply the wrong shape for this machine?

- **Date** 2026-08-17
- **Milestone** M7
- **Status** done — hypothesis **confirmed** to within 7% at two widths;
  and [0026](../0026-m7-closing-on-cpu/TASK.md)'s ceiling claim **corrected**

## Two things prompted this

[0026](../0026-m7-closing-on-cpu/TASK.md) concluded that eltwise sits at the
machine's fp32 limit and that "parity needs work that stays in the MAC
datapath". Two objections to that, both fair:

1. **The ceiling argument overreached.** The degree probe shows the kernel is
   throughput bound *per core*. Compute-bound work **scales with cores** — so
   the fp32 limit bounds one core, not the design. That is a different claim
   with a different consequence, and 0026 conflated them.
2. **"Stays in the MAC datapath" was a post-hoc explanation.** It can be made
   quantitative, and then it can be tested.

## The correction: eltwise scales cleanly with columns

0026 reported the 1→2 column change as 1.36×, which is the *end-to-end* number
and therefore Amdahl-diluted. Per design, batch 64:

| | 1 column | 2 columns | |
|---|---|---|---|
| GELU | 32,762 µs | 16,527 µs | **1.98×** |
| LayerNorm | 4,219 µs | 2,122 µs | **1.99×** |
| softmax | 6,481 µs | 3,298 µs | **1.97×** |

Linear in cores, exactly as compute-bound work should be. **Nothing about this
stops at 2 columns** — so the ceiling in 0026 is not the arithmetic, it is that
the design would not compile wider.

### Why it would not compile wider — and I misread which design failed

The eltwise designs gave every core its own `rt.fill`/`rt.drain`: 4 shim streams
per column, 32 at eight columns, and `aiecc` refuses with *no ShimNOCTile has
sufficient DMA capacity*. That was recorded as a hardware limit. **It is not** —
the GEMM in this same repo runs at 8 columns by routing L3→L2→L1, and
`ObjectFifo` has the operators for it: `.split()` fans one mem-tile buffer to
four cores, `.join()` collects them.

**Correction to the first version of this note.** After the rewrite the build
still failed with an identical message, and I recorded that as "the placer will
not do 4 or 8 columns, unresolved". That attribution was wrong. The failing
frame is **`ln_array`, not `gelu_array`** — GELU builds fine at 8 columns;
LayerNorm was never rewritten and still opens **three** fifos per core (in,
params, out), i.e. 96 shim streams at 8 columns.

That the message did not change should have been the tell: a rewrite that
genuinely halved a stream count would move the numbers. It did not, because the
rewrite was not in the design that was failing.

**GELU at 8 columns, measured:**

| | 2 columns | 8 columns | |
|---|---|---|---|
| GELU alone | 9,280 µs | **2,398 µs** | **3.87×** |
| + its design switch | 9,802 µs | 4,621 µs | 2.12× |
| full encode (batch 64) | 214.8 seq/s | **242.9 seq/s** | **1.13×** |

`1-cos` **3.397e-04**, unchanged. The switch cost rises with width exactly as
[0024](../0024-m7-dispatch-cost-anatomy/TASK.md) says it must (522 → 2,223 µs),
which eats half the gain and is why the per-dispatch figure is 2.12× and not
3.87×.

**Still to do:** LayerNorm and softmax need the same `.split()`/`.join()`
rewrite. They are 95 ms of the 209 ms eltwise at batch 128.

## The width hypothesis, made quantitative

Per layer and token, a BERT-family encoder has

- **4·h** GELU elements (the FFN intermediate), and
- **12·h² MACs** — 4h² for QKVO, 8h² for the FFN.

So the elementwise share of the work is **1 : 3h**, *independent of
implementation*. At h = 384 that is 1 : 1152; at h = 4096 it is 1 : 12288.

MiniLM is therefore not merely small, it is **structurally elementwise-heavy**,
and elementwise is precisely the work that does not live in the MAC datapath
where the NPU's 14.7 TOPS are. The CPU has the same asymmetry, far more mildly.

### Tested

`tools/export_xclbin.py --hidden H` generates the whole shape set from one
parameter (QKV is 3h, FFN is 4h). Synthetic weights; only the shapes matter.
Per design, batch 16, 2 columns:

| | h = 384 | h = 768 | growth | structural prediction |
|---|---|---|---|---|
| qkv | 1,409 µs | 5,302 µs | 3.76× | 4× (MACs ∝ h²) |
| attn_out | 548 µs | 1,838 µs | 3.35× | 4× |
| ffn_up | 1,855 µs | 6,994 µs | 3.77× | 4× |
| ffn_down | 1,855 µs | 6,948 µs | 3.75× | 4× |
| **GEMM total** | **5,667 µs** | **21,082 µs** | **3.72×** | **4×** |
| **GELU** | **2,389 µs** | **4,674 µs** | **1.96×** | **2×** (elements ∝ h) |
| **GELU share** | **29.7%** | **18.1%** | | 17.4% |

**Both growth rates land within 7% of the structural prediction.** The
hypothesis is confirmed: the elementwise fraction falls as 1/h, and it does so
on this hardware at the rate the arithmetic says it must.

**The 18.1% share is NOT a third confirmation.** It is arithmetically determined
by the two growth rates — given 3.72× and 1.96×, 18.1% follows. There are two
measurements here, not three, and the row is kept only because it is the number
the encode actually cares about. Counting it as independent evidence would make
the model look better supported than it is.

### The GEMM rate improvement cannot continue

GEMM grew 3.72× for 4× the MACs, so the rate *improved* with width — which is
expected on a movement-bound datapath as arithmetic intensity rises. But that
improvement is bounded, and it is nearly spent:

| | h = 384 | h = 768 |
|---|---|---|
| MACs | 1,812 M | 7,248 M |
| GEMM rate | **0.639 TFLOP/s** | **0.688 TFLOP/s** |

**+7.7%.** [0010](../0010-m5-b-reuse-and-cost-model/TASK.md) measured 3.79
TFLOP/s at **8 columns**; this sweep runs at **2**, and per-core rate is flat in
core count (M2: 137–142 MACs/cycle from 1 to 16 cores), so the comparable
ceiling here is ~0.95 TFLOP/s. **We are already at 72% of it, rising to 88%.**

So future doublings of h give GEMM ×4, not ×3.72, while eltwise keeps growing a
clean ×2 with no corresponding ceiling. The gain per doubling therefore *falls*,
and any parity estimate built on the h=384→768 rates is optimistic.

### The parity-h projection is WITHDRAWN

The first version of this note projected parity at h ≈ 1300–2000 by scaling our
breakdown with the measured rates and assuming CPU time grows ~4× per doubling.

**That assumption is wrong, and it is the load-bearing one.** The CPU runs the
same encoder and therefore has the *same* 1 : 3h structure: its GEMM work also
grows 4× while its elementwise work grows 2×, so its total also grows
sub-quadratically. The gap closes only to the extent that we are *relatively*
worse at eltwise than the CPU is — which is true (eltwise is 209 ms against
77 ms of GEMM here, a ratio no tuned CPU encoder has) but is not what the
projection computed.

Without a measured CPU breakdown into GEMM and elementwise time there is no
basis for a parity-h number at all, so the number is withdrawn rather than moved.
Two further reasons it would have been optimistic even with that measurement:
the GEMM rate improvement is nearly exhausted (above), and the switch cost is
flat in h and so becomes *relatively* cheaper for us — the one term that does
favour us.

**h = 1536 did not build**: `'aie.dma_bd' op Stride 3 exceeds the [1:1048576]
range`. This is *not* the K=1536 buffer-descriptor **size** wall of
[0003](../0003-m2-bf16-gemm/TASK.md) / CLAUDE.md trap 4 that M4 pre-tiling was
scoped to fix — the exporter already builds `pretiled=True`, and it failed
anyway. It is the **stride** field, a different limit, and it needs its own
investigation. So nothing beyond h = 768 is verified, and this is a wall without
a known door rather than a known door not yet opened.

What survives is the measured part: **the 1 : 3h structure holds on this
hardware to within 7%**, and the elementwise share falls 29.7% → 18.1% from
h = 384 to h = 768.

## Conclusion

**MiniLM at h = 384 is close to the worst case for this machine**, and that is
now a measured property rather than an excuse. Roughly 30% of its per-layer
array work is elementwise, and that share falls as 1/h.

But **"bge-large wins" would be overclaiming**, and the first version of this
note came close to it. bge-large is h = 1024 — only 1.4 doublings from here,
and with the parity projection withdrawn there is no defensible number for what
it would achieve. The honest headline is narrower: **the crossover lies
somewhere inside the range of model widths people actually use, above
h = 384**, and where exactly is unmeasured.

The claim that does hold today is the per-core one, and it holds without any
projection: **3.2× better per core at h = 384**, and the 1 : 3h structure says
that argument only strengthens with width. Core efficiency and offload are the
case that stands now; throughput parity is the case that might stand at the
next model size.

The levers that remain, in order:

1. **`.split()`/`.join()` for LayerNorm and softmax.** GELU proved it: 3.87× at
   8 columns and 1.13× on the full encode. These two are 95 ms of the 209 ms of
   eltwise at batch 128.
2. **A wider model.** [`docs/04-model`](../../docs/04-model/README.md) already
   designed for bge-small as a drop-in; bge-large is the interesting one, with
   the caveat above.
3. **The stride wall at h ≥ 1536**, which currently blocks testing any of this
   at the widths that matter.

## Artifacts

- `tools/export_xclbin.py` — `--hidden`
- `experiments/m5-eltwise/gelu_kernel.py` — mem-tile `split()`/`join()` routing,
  pinned worker placement
- `runtime/artifacts_h{384,768}/` — the width sweep
- `runtime/artifacts_g8/` — 8-column GELU in the full encode
