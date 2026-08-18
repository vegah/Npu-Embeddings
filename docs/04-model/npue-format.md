# The `.npue` runtime weight container — v1

**Implemented and verified in M4** ([`tasks/0006`](../../tasks/0006-m4-npue-pretiling/TASK.md)).
Reference implementation: [`tools/npue.py`](../../tools/npue.py). This document is
what the C++ loader in M7 must implement.

> ### Corrected by M5 — read this first
>
> [`tasks/0007`](../../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md) took this
> layout to hardware and **refuted two of the claims below**:
>
> 1. **"`ffn_down` cannot be expressed without pre-tiling" is false for the
>    whole-array design.** That BD failure belongs to the *single-core* design,
>    where B is walked as one full column strip so the k-blocks collapse into a
>    single `size=1536`. The whole-array design keeps k-blocks as their own
>    dimension (`<size=24, stride=24576>`) and K never appears as a size.
>    Verified by running `ffn_down` row-major on the unmodified M2 design.
> 2. **Pre-tiling is not a performance win here.** Per-core it is equal at best
>    and −11% on average with up to 22% run-to-run spread (row-major: 0.2%);
>    end-to-end at 8 columns it is a ±9% wash across all four MiniLM GEMMs.
>    Measured against the same design with the same BD dimension count and the
>    same sizes — only the strides differ.
>
> **What survives:** `tile_n = 48` is confirmed necessary and workable at 8
> columns (n=32 is impossible for three of the four shapes), the offline
> fusions are free, the round-trip is bit-exact, and baking the sub-tile order
> into the file costs nothing. The container is worth having for the fusions,
> bf16 conversion, fp32 biases, `mmap` and the sha256 pin — not for the DMA
> reasons originally given.

Input is HuggingFace **safetensors**. Output is this: a pre-tiled, pre-fused,
`mmap`-able container that the runtime hands to DMA descriptors without touching.

## Why a custom format

The runtime must not transpose, must not convert dtypes, must not concatenate,
and must not re-tile. Every one of those is a pure function of the weights, so
every one belongs offline. What is left at load time is `mmap` and pointer
arithmetic.

There was believed to be a second, harder reason: the DMA BD size field is 10
bits (max 1023), and a row-major `[1536, 384]` `ffn_down` walked as one column
strip needs `size = 1536`. **M5 showed that applies to the single-core design
only** — see the correction above. The remaining reasons are the ones at the top
of this section, and they are enough on their own.

**GGUF was considered and rejected**: we need neither llama.cpp's quant zoo nor
its metadata system. Two of its ideas are worth stealing and are here — explicit
alignment padding, and room to embed the tokenizer vocabulary in the same file.

## File layout

```
[0 : 64]                       FileHeader, exactly 64 bytes, little-endian
[json_offset : +json_length]   UTF-8 JSON: config + tensor directory
[data_offset : +data_length]   raw tensor data, each tensor 4096-byte aligned
```

```c
struct FileHeader {           // exactly 64 bytes at offset 0
  char     magic[4];          // "NPUE"
  uint32_t version;           // 1
  uint32_t arch;              // 0 = BERT_ABS_GELU_POSTLN
  uint32_t flags;             // bit0 = pre-tiled
  uint64_t json_offset;       // 64
  uint64_t json_length;
  uint64_t data_offset;       // 4096-aligned
  uint64_t data_length;
  uint8_t  reserved[16];      // zero
};
```

> **Correction to the M0 draft.** The planned spec in
> [`README.md`](README.md) declared `reserved[24]` while also requiring exactly
> 64 bytes. Those are inconsistent: `4 + 4 + 4 + 4 + 8·4 = 48`, so reserved must
> be **16**, not 24. The 64-byte total is what matters (it keeps the JSON start
> cache-line aligned), so reserved was shortened. Found by asserting
> `struct.calcsize(...) == 64` rather than trusting the draft.

## Directory entry

```json
{ "name": "layer.0.qkv", "role": "gemm_b", "dtype": "BF16",
  "logical_shape": [384, 1152], "padded_shape": [384, 1152],
  "layout": {"kind": "block_panel", "tile_k": 64, "tile_n": 48,
             "order": "k,n,kt,nt", "inner": "s,t",
             "mac_s": 8, "mac_t": 8, "dtype": "BF16"},
  "layout_hash": "94266693ea31aa67...",
  "offset": 0, "nbytes": 884736 }
```

`offset` is relative to `data_offset`. Roles: `gemm_b`, `bias`, `layernorm`,
`embedding`.

**`layout_hash` is a sha256 over the canonical JSON of the layout object.** A
file packed for different tile dimensions is not merely suboptimal for a kernel
expecting others — it is *silently wrong*. The loader compares and refuses.

## Pre-tiled `gemm_b` layout

`kind: "block_panel"` stores a `[K, N]` operand as a linear sequence of
`[tile_k, tile_n]` tiles in `(kb, nb)` order — `k` major — and, when `inner` is
`"s,t"`, permutes each tile internally into the MAC intrinsic's sub-tile order.

This absorbs **both** re-layouts the M2 design performed at runtime:

| stage | M2 did this at runtime | now |
|---|---|---|
| L3→L2 | `TensorTiler2D.step_tiler((K,N),(k,n),…)` gathering a tile out of row-major DDR | a contiguous read |
| L2→L1 | `dims_to_stream=[(k//s, s*n), (n//t, t), (s, n), (t, 1)]` | baked into the file |

Consequence — the access-pattern dimensions become *tile counts*, not extents:

| operand | `[K, N]` | k-blocks | n-blocks | max BD dim |
|---|---|---|---|---|
| `qkv` | `[384, 1152]` | 6 | 24 | 24 |
| `attn_out` | `[384, 384]` | 6 | 8 | 8 |
| `ffn_up` | `[384, 1536]` | 6 | 32 | 32 |
| `ffn_down` | `[1536, 384]` | 24 | 8 | **24** |

`ffn_down` needed `K = 1536` as a BD dimension **in the single-core design**. It
needs 24 now. The whole-array design already needed only 24 — its `step_tiler`
keeps k-blocks as their own dimension. M5 measured both descriptors side by side
and found identical dimension counts, identical sizes, and identical total
length; only the strides differ.

### Choosing `tile_n`

`mac_s = mac_t = 8` for **both** bf16 configurations on npu2 — `mac_dims` are
`(4,8,8)` plain and `(8,8,8)` under bfp16 emulation, and only `s` and `t` touch
the B layout. **So the bf16-vs-bfp16 decision does not force a repack.**

`tile_n = 48`, not M2's winning 32. The whole-array design requires
`N % (tile_n · n_cols) == 0`, and MiniLM's `N` dims are 384 / 1152 / 1536:

| n_cols | gcd(N/n_cols) | legal `tile_n` (multiples of 8) |
|---|---|---|
| 4 | 96 | 8, 16, 24, 32, 48, 96 |
| 8 | 48 | 8, 16, 24, **48** |

**M2's `tile_n = 32` is illegal at 8 columns** — `1152 / (32·8) = 4.5`. 48 is the
largest legal choice, needs **zero padding** on all three shapes, and fits L1:
`2·(64·64·2 + 64·48·2 + 64·48·4) = 53,248 < 65,536`.

This is arithmetic against documented constraints, **not yet a hardware result**.
M5 compiles and traces it.

## Offline fusions baked in

All five from [`README.md`](README.md), all pure weight rewrites:

1. **Q, K, V fused** into one `[384, 1152]` B matrix + `[1152]` bias.
2. **Transposed to `[K, N]`** — `nn.Linear` stores `[out, in]`.
3. **GEMM operands to bf16**, round-to-nearest-even. LayerNorm γ/β and every
   bias stay **fp32** — 384 floats each and numerically sensitive.
4. **`1/√head_dim` folded into the Q block** of the fused weight and bias, so
   the attention kernel has no scale multiply. Measured free (below).
5. **`position_embeddings` pre-sliced to 256** — MiniLM's real
   `max_seq_length`, not the 512 in `config.json`.

The **embedding table is stored un-tiled and fp32**: it is gathered, never
multiplied, so tiling would only hurt, and fp32 removes a source of numerical
drift at the input. It is 46.9 MB of the 68.8 MB file.

The **pooler is not stored at all** — 147,840 params sentence-transformers never
calls.

## Verification (the M4 gate)

`tools/verify_npue.py`, five checks:

| check | result |
|---|---|
| **A** header 64 B, magic, version, every tensor 4096-aligned | pass; overhead **0.223%** (150 KB on 68.8 MB) |
| **B** de-tile == bf16 of the fused source, **bit-exact** | **0 differing of 10,616,832** bf16 elements |
| **C** a wrong `layout_hash` must raise | `tile_n 48 → 64` refused |
| **D** the encoder run off packed weights vs the M3 goldens | `1-cos = 1.17e-05`, **0.92×** M3's end-to-end bf16 |
| **E** what the scale fold costs, in isolation | 1.17e-05 folded vs 1.28e-05 at runtime — **free** |

D is the check the others exist to support: it is the only one that catches a
fusion which round-trips perfectly but is mathematically wrong — the scale folded
into K instead of Q, a transpose applied twice, Q/K/V concatenated in the wrong
order.

## Sizes

| | |
|---|---|
| encoder weights, bf16 | 21.3 MB |
| embedding table, fp32 | 46.9 MB |
| alignment + JSON | 150 KB (0.223%) |
| **total** | **68.79 MB** |

## Not yet in v1

- Tokenizer vocabulary embedded in the file (GGUF's good idea; M7).
- int8/W8A8 tensors — deliberately deferred.
- A `[M, K]` activation layout: activations are runtime data, not weights.
