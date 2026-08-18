# 0011 — M5 GATE: the first encoder op on the NPU, validated against the goldens

- **Date** 2026-08-17
- **Milestone** M5
- **Status** done (**gate passed**)

## Goal

Close the loop M3 and M4 were built for. Every NPU run so far has been on
`iron.rand` data compared against a numpy reference computed from the same
random numbers. Run a **real encoder op** with **real weights** and check it
against the **HuggingFace-derived oracle**.

**Gate (from the roadmap):** *each encoder op validated against M3 goldens and
traced.*

## What was done

The op is layer 0's fused QKV projection, `[256, 384] × [384, 1152]` — M=256 is
MiniLM's real maximum sequence length, and the golden corpus is batch 4 × seq 64.

Nothing is re-derived along the way:

| input | source |
|---|---|
| A | golden `hf.emb.ln` — the genuine post-LayerNorm input to layer 0, verified against HuggingFace to 8.5e-08 |
| B | `layer.0.qkv` read straight out of the packed `.npue` — fused Q\|K\|V, transposed to `[K,N]`, bf16, `1/√32` already folded into Q |
| bias | `layer.0.qkv.bias` from the same file, fp32 |

Both sources carry the checkpoint sha256 and the script refuses to run if they
disagree.

### The comparison had to account for the fold

M3 deliberately left the attention scale unfolded so that M4's fold would be
*provable* ([0005](../0005-m3-python-reference/TASK.md), carry-forward 3). The
packed weights fold it. So the golden `L0.qkv` and the NPU result differ by
`1/√32` on the Q third by construction.

Unfolding the weights would have tested a different program than the one we
ship. Instead the **golden** is scaled into the space the packed pipeline
produces. That makes this a test of the fold as well as of the GEMM: folding
into K or V, or using the wrong constant, fails it.

To make a failure say *where*, the three blocks are reported separately.

The bias add stays on the host — it is elementwise, it is fp32 in `.npue` by
design, and M5 has not put an eltwise kernel on the array yet.

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

python experiments\m5-pretiled-gemm\validate_layer0_qkv.py
python experiments\m5-pretiled-gemm\validate_layer0_qkv.py --emulate-bfp16 `
    --out experiments\m5-pretiled-gemm\artifacts\validate_layer0_qkv_bfp16.json
```

## Result

```
layer 0 QKV on the NPU: [256,384] x [384,1152]  4 cols, tile (64,64,48)
  checkpoint 53aa51172d142c89...  (both sources agree)
  weights straight from all-MiniLM-L6-v2.npue, scale folded into Q
```

| quantity | **bf16** | bfp16 |
|---|---|---|
| rel. Frobenius vs golden `L0.qkv` | **1.507e-03** | 9.898e-03 |
| max abs difference | 1.722e-02 | 9.014e-02 |
| worst per-row `1 − cos` | **2.613e-06** | 7.203e-05 |
| verdict | **PASS** (tol 5e-3) | **PASS** (tol 2e-2) |

Per block, bf16 / bfp16:

| block | bf16 | bfp16 |
|---|---|---|
| Q | 1.380e-03 | 7.837e-03 |
| K | 1.440e-03 | 9.574e-03 |
| V | 2.009e-03 | 1.279e-02 |

**Three things worth reading out of that:**

1. **The whole chain works end to end on real data** — HuggingFace → M3 goldens
   → M4 `.npue` → NPU → oracle. This is the first time that has been true.
2. **The scale fold is confirmed correct.** All three blocks sit at the same
   error level. A fold applied to K or V, or with the wrong constant, would
   leave one block wrong by O(1), not by 4e-4.
3. **bfp16 reproduces [0008](../0008-m5-bfp16-real-data/TASK.md) independently.**
   That task measured this same op on real data at 9.02e-03 through a different
   script; this one gets 9.898e-03. Two paths, same number.

The bf16 result also sits comfortably under M3's simulated 2.4e-3 for `L0.ln2`
over the whole layer, which is the right ordering — one GEMM should be cleaner
than the layer that contains it.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| A correct bfp16 run reported `FAIL` | The tolerance was hard-coded for bf16, and bfp16's real level is ~1e-2 | Tolerance is per format: 5e-3 bf16, 2e-2 bfp16. Conflating them is exactly how a correct run gets thrown away |
| `TypeError: unsupported format string passed to dict.__format__` | Made the tolerance a dict and missed one interpolation site | Trivial, recorded because it fired only on the success path |

## Artifacts

- `experiments/m5-pretiled-gemm/validate_layer0_qkv.py`
- `artifacts/validate_layer0_qkv.json`, `artifacts/validate_layer0_qkv_bfp16.json`
- `artifacts/trace_layer0qkv_4c*.txt` and the matching physical MLIR

## Next

The pattern generalises — the remaining per-layer GEMMs (`attn_out`, `ffn_up`,
`ffn_down`) need only their golden tap and the matching `.npue` tensor, and all
three are already packed and already known to compile at these tile dimensions.

What is genuinely new work, in the order the roadmap wants it:

1. **The elementwise and reduction ops**: LayerNorm (fp32, and `docs/04-model`
   requires it stay fp32), softmax (fp32, max-subtracted), GELU (exact erf), and
   the bias adds. None has been on the array yet.
2. **Attention**, where `head_dim = 32` is the only awkward dimension in the
   model and QK\<sup>T\</sup>/A·V are the only GEMMs that see it.
3. **Then M6** — the full encode in Python, which is the bankable milestone and
   which turns [0010](../0010-m5-b-reuse-and-cost-model/TASK.md)'s projections
   into measurements.

Carried forward unchanged: the 150 µs fixed dispatch cost is the largest lever
([0010](../0010-m5-b-reuse-and-cost-model/TASK.md)), and every op added as a
separate dispatch pays it again — which is the argument for fusing whole layers
before the op count grows.
