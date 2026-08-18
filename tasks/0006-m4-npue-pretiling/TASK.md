# 0006 — M4: offline weight pre-tiling → `.npue`

- **Date** 2026-08-17
- **Milestone** M4
- **Status** done (**gate passed**)

## Goal

Move every weight transformation offline. Produce a pre-tiled, pre-fused,
`mmap`-able container so the runtime never transposes, converts, concatenates or
re-tiles — and so `ffn_down` becomes expressible at all.

**Gate:** format spec + round-trip verified.

## Context

Two independent forces made this the next milestone:

1. **[0004](../0004-m2-multicore-gemm/TASK.md) showed we are data-movement
   bound** — traced compute is ~7% of NPU time, core scaling 99.8% but
   wall-clock scaling 40%. Pre-tiling is the main lever.
2. **`ffn_down` cannot be expressed without it.** The DMA BD size field is 10
   bits (max 1023) and `ffn_down` is `[1536, 384]`
   ([0003](../0003-m2-bf16-gemm/TASK.md)).

And [0005](../0005-m3-python-reference/TASK.md) supplied the thing that makes
this checkable: an fp32 oracle with stored goldens. Its carry-forward item 3 was
explicit — *M4 must verify its offline fusions against these goldens, not just
round-trip them*.

## What was done

### 1. The format

`tools/npue.py` — header, JSON directory, tiling, reader, writer. numpy only, so
it runs in the iron env. It is also the reference for the C++ loader in M7.

Spec written up in [`docs/04-model/npue-format.md`](../../docs/04-model/npue-format.md).

### 2. What pre-tiling actually absorbs

Worth stating precisely, because "pre-tiling" undersells it. The M2 design did
**two** separate re-layouts on the B operand at runtime:

| stage | M2 | now |
|---|---|---|
| L3→L2 | `TensorTiler2D.step_tiler((K,N),(k,n),…)` gathering a tile out of row-major DDR | a contiguous read |
| L2→L1 | `dims_to_stream=[(k//s, s*n), (n//t, t), (s, n), (t, 1)]` — the intrinsic sub-tile order | baked into the file |

`tile_b()` does both. The result is that access-pattern dimensions become *tile
counts* rather than tensor extents, which is what unblocks `ffn_down`:

| operand | `[K, N]` | k-blocks | n-blocks | max BD dim |
|---|---|---|---|---|
| `qkv` | `[384, 1152]` | 6 | 24 | 24 |
| `attn_out` | `[384, 384]` | 6 | 8 | 8 |
| `ffn_up` | `[384, 1536]` | 6 | 32 | 32 |
| `ffn_down` | `[1536, 384]` | 24 | 8 | **24** |

`ffn_down` needed 1536 as a BD dimension. It needs 24.

### 3. `mac_s = mac_t = 8` for both bf16 paths — so the format is stable

The inner sub-tile order depends on `mac_dims`. From M2 those are `(4,8,8)` for
plain bf16 on npu2 and `(8,8,8)` under bfp16 emulation — and **only `s` and `t`
appear in the B layout**, both 8 either way. So the bf16-vs-bfp16 question, which
M3 reopened as an accuracy risk, **does not force a repack**. That was worth
checking before baking anything.

### 4. `tile_n = 48`, not M2's 32 — M2's tiling is illegal at 8 columns

The whole-array design asserts `N % (tile_n · n_cols) == 0`. MiniLM's N dims are
384 / 1152 / 1536:

```
cols 4  gcd(N/4) = 96 -> tile_n in {8, 16, 24, 32, 48, 96}
cols 8  gcd(N/8) = 48 -> tile_n in {8, 16, 24, 48}
```

**`tile_n = 32` fails at 8 columns**: `1152 / (32·8) = 4.5`. It was M2's best at
four columns, and it would have been the obvious thing to carry forward. 48 is
the largest legal choice, needs **zero padding** on all three shapes, and fits
L1 at `2·(64·64·2 + 64·48·2 + 64·48·4) = 53,248 < 65,536`.

This is arithmetic against documented constraints — **not a hardware result**.
M5 compiles and traces it.

### 5. The five fusions

All from `docs/04-model`, all pure weight rewrites: Q/K/V fused to `[384,1152]`;
everything transposed to `[K,N]`; GEMM operands to bf16 with **round-to-nearest-even**
(truncation would bias 10.6M weights toward zero systematically rather than
randomly); LayerNorm γ/β and every bias kept fp32; `1/√32` folded into the Q
block; position embeddings pre-sliced to 256, MiniLM's real max, not the 512 in
`config.json`.

The embedding table is stored **un-tiled and fp32** — gathered, never multiplied.
The pooler is not stored at all.

### 6. The gate

`tools/verify_npue.py`. Five checks, ordered by what they would catch. The point
is check D: A–C would all pass on a file whose fusions are mathematically wrong.

To make D possible, `reference/encoder.py` gained two small hooks — `qkv_w`/`qkv_b`
overrides and `qk_scale` — so the *identical* forward pass can run off packed
weights. Nothing in check D is re-derived from the original checkpoint.

## Commands

```powershell
# pack (iron env, numpy only)
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\pack_npue.py

# the gate
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\verify_npue.py

# retune the tiling and repack (the layout descriptor is data, not code)
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\pack_npue.py --tile-n 32 --out models\minilm_n32.npue
```

## Result

### Packing

```
  tile (64, 48), mac (s=8, t=8), 1/sqrt(32) = 0.17677669529663687 folded into Q

  operand               [K,N]  k-blocks  n-blocks   tiles  max BD dim
  qkv             [384, 1152]         6        24     144          24
  attn_out         [384, 384]         6         8      48           8
  ffn_up          [384, 1536]         6        32     192          32
  ffn_down        [1536, 384]        24         8     192          24
  (row-major DDR would need K=1536 as a BD dimension for ffn_down -- over 1023,
   which is why it could not be expressed before)

  tensors    : 77  (24 pre-tiled GEMM operands)
  json       : 16346 B at 64
  data       : 68.77 MB at 20480
  file       : 68.79 MB
```

68.79 MB = 21.3 MB encoder weights (bf16) + 46.9 MB embedding table (fp32),
which is exactly what `docs/04-model` predicted.

### **GATE**

```
A. spec conformance
   ok    header is exactly 64 bytes
   ok    magic == b'NPUE'
   ok    version == 1
   ok    flags has bit0 (pre-tiled)
   ok    reserved is zero
   ok    data_offset 20480 is 4096-aligned
   ok    json sits between header and data
   ok    file size == data_offset + data_length
   ok    all 77 tensors 4096-aligned
         alignment + json overhead: 150.0 KB on 68.8 MB (0.223%)

B. round-trip (de-tile == bf16 of the fused source, bit-exact)
   ok    24 operands, 10,616,832 bf16 elements, 0 differing

C. layout_hash guard
   ok    matching layout accepted
   ok    tile_n 48 -> 64 refused

D. the encoder, running off packed weights, vs the M3 goldens
   tensor                 bf16 weights only   + bf16 activations
   emb.ln                         8.494e-08            8.494e-08
   L0.ln2                         1.651e-03            2.367e-03
   L1.ln2                         2.023e-03            3.037e-03
   L2.ln2                         2.378e-03            3.575e-03
   L3.ln2                         2.710e-03            3.985e-03
   L4.ln2                         3.070e-03            4.372e-03
   L5.ln2                         3.965e-03            5.647e-03
   last_hidden_state              3.965e-03            5.647e-03

   fp32 activations   1-cos 7.698e-06   similarity shift 5.847e-04
   bf16 activations   1-cos 1.165e-05   similarity shift 4.058e-04
                      vs M3's end-to-end bf16 (1.27e-05): 0.92x

E. what the 1/sqrt(head_dim) fold costs, in isolation
   1 - cos, scale folded into Q       : 1.165e-05
   1 - cos, scale applied at runtime  : 1.275e-05
   folding costs 0.914x -- free

PASS -- spec conformant, round-trip bit-exact, layout guarded, goldens reproduced
```

**Readings:**

- **Round-trip is bit-exact**, not approximate: 0 of 10,616,832 bf16 elements
  differ. Tiling is a permutation, so anything else would have been a bug.
- **The fusions cost nothing beyond bf16.** Packed weights on the real datapath
  land at 0.92× M3's end-to-end bf16 figure. Since bf16 alone was already
  measured as free (`1-cos = 1.3e-05`), the whole offline pipeline is free.
- **Folding `1/√32` into Q is free**, which was not obvious: `1/√32` is not a
  power of two, so folding changes which bf16 grid point each Q weight rounds to.
  Measured 0.914×. That is *not* evidence folding is better — on a four-sentence
  corpus it is indistinguishable from 1.0. The claim is only that it is not
  worse, which is what the fusion needed.
- **Alignment costs 0.223%** — 150 KB on 68.8 MB, for page-aligned DMA.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| `struct.calcsize` of the header from `docs/04-model` gave **72**, not the 64 the same paragraph required | The draft spec declared `uint8_t reserved[24]`, but the preceding fields already sum to 48 | Reserved is **16**. Kept the 64-byte total. Found by *asserting* the size rather than trusting the draft — the assert was written before the first pack |
| Carrying M2's winning `tile_n = 32` forward would have produced a file unusable at 8 columns | `N % (tile_n · n_cols) == 0`, and `1152 / (32·8) = 4.5`. M2 measured at 4 columns, where 32 is legal | Computed the legal set per column count; `tile_n = 48`. **The trap is that it would have compiled and run fine at 4 columns**, i.e. exactly where M2 tested |
| `PermissionError: [WinError 32]` deleting the temporary unfolded `.npue` | `np.memmap` holds a Windows file lock, and `del reader` does not release it — arrays derived from the mapping keep it alive | `Reader.close()` + context manager, and `tensor()` now returns a real `.copy()` rather than a view into the mapping |
| First golden check reported **0.61×** M3's bf16 baseline — i.e. packing looked *better* than bf16 | Not comparable. That run had bf16 **weights** but fp32 **activations**; M3's figure is bf16 on both | Report both runs explicitly. The comparable number is **0.92×**. A flattering number that answers a different question is worse than no number |

## Artifacts

Committed:

- `tools/{npue,pack_npue,verify_npue}.py`
- `docs/04-model/npue-format.md` — the authoritative spec for the M7 C++ loader

Not committed, regenerable:

- `models/all-MiniLM-L6-v2.npue` — 68.79 MB → `tools\pack_npue.py`

## Next

**M5 — encoder ops onto the NPU, one at a time**, each validated against the M3
goldens and traced. Carried forward:

1. **`tile_n = 48` at 8 columns is arithmetic, not a measurement.** The first M5
   task is to compile and trace it. M2 measured per-core cost as essentially
   shape-independent (137.3 → 142.0 → 141.7 MACs/cycle), so 48 should behave
   like 32 — but that is a prediction.
2. **`ffn_down` is the first thing to try**, because it is the layer that could
   not previously be expressed at all. It is the direct proof that M4 did its
   job.
3. **Confirming the 5-bit bfp16 fit on hardware with real activations remains
   the highest-value single measurement in M5** ([0005](../0005-m3-python-reference/TASK.md)).
   The format is stable either way — `s = t = 8` for both paths — so this is a
   kernel-flag decision, not a repack.
4. **Feed pre-tiled operands to the M2 design and re-measure the starvation.**
   M2's 40% wall-clock scaling and ~144 µs fixed dispatch cost are the numbers
   pre-tiling is supposed to move. Nothing here has measured that yet.
