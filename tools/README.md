# tools/ — build-time tooling

Offline weight transformation. Runs once, before anything touches the NPU.
Built in M4 ([`tasks/0006`](../tasks/0006-m4-npue-pretiling/TASK.md)).

numpy only, so everything here runs in the **iron** env. Nothing here ships —
Python is build-time and prototyping only.

| | |
|---|---|
| `npue.py` | The `.npue` container: header, JSON directory, tiling, reader, writer. **This is the reference the C++ loader in M7 must match.** |
| `pack_npue.py` | HuggingFace checkpoint → pre-tiled, pre-fused `.npue`. |
| `verify_npue.py` | **The M4 gate.** Spec conformance, bit-exact round-trip, the stale-layout guard, and the encoder run off packed weights against the M3 goldens. |

Spec: [`docs/04-model/npue-format.md`](../docs/04-model/npue-format.md).

## Why the weights are transformed offline

The runtime must not transpose, convert dtypes, concatenate or re-tile — all are
pure functions of the weights. What is left at load time is `mmap` and pointer
arithmetic.

> **Corrected by M5** ([`tasks/0007`](../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md)).
> This directory originally also claimed pre-tiling was the only way to express
> `ffn_down` (the 10-bit, max-1023 DMA BD size field) and the main performance
> lever. On hardware, **neither holds for the whole-array design**: it already
> expresses `ffn_down` fine, and pre-tiling measures as a ±9% end-to-end wash
> with worse per-core stability. The BD failure was a property of the
> single-core design. The reasons above — no runtime transposes, conversions or
> concatenations — are real and are why the container stays.

## Run

```powershell
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\pack_npue.py
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\verify_npue.py
```

The `.npue` (68.79 MB) is gitignored — a deterministic derivative of the
sha256-pinned checkpoint, which travels inside the file as `source_sha256`.

## Retuning the tiling

The layout descriptor is **data, not code**. To try different tile dimensions,
repack rather than editing a loader:

```powershell
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\pack_npue.py --tile-n 32 --out models\minilm_n32.npue
```

`layout_hash` makes a stale file fail loudly instead of producing plausible
garbage embeddings.

> `tile_n = 48` is the default because M2's winning `tile_n = 32` is **illegal at
> 8 columns** for MiniLM's N dims (`1152/(32·8) = 4.5`). It would have run fine
> at 4 columns, which is where M2 measured.

## Status

**Gate passed.** Round-trip bit-exact — 0 of 10,616,832 bf16 elements differ —
and the encoder running off packed weights lands at 0.92× M3's end-to-end bf16
baseline, so the fusions cost nothing beyond the number format.

**On hardware (M5):** `tile_n = 48` confirmed at 8 columns for all four MiniLM
GEMMs, and baking the sub-tile order into the file is free. Pre-tiling itself
does not improve throughput — see the correction above.
