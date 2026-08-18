# 0012 — M5: all four per-layer GEMMs validated, and where bfp16 actually hurts

- **Date** 2026-08-17
- **Milestone** M5
- **Status** done — **GEMM half of M5 closed; bfp16 damage localised to `ffn_down`**

## Goal

Generalise [0011](../0011-m5-first-op-validated/TASK.md)'s single-op check to
**every GEMM an encoder layer performs**, and see whether the format behaves the
same across all four.

## What was done

`validate_layer0_qkv.py` became `validate_layer_gemms.py`, covering layer 0's
four GEMMs in execution order. Each takes its activation from the golden tap
that feeds it and its weight straight from the packed `.npue`:

| GEMM | activation tap | weight | output tap |
|---|---|---|---|
| `qkv` | `emb.ln` | `layer.0.qkv` | `L0.qkv` |
| `attn_out` | `L0.ctx` | `layer.0.attn_out` | `L0.attn_proj` |
| `ffn_up` | `L0.ln1` | `layer.0.ffn_up` | `L0.ffn_up` |
| `ffn_down` | `L0.gelu` | `layer.0.ffn_down` | `L0.ffn_down` |

**Inputs come from the tap file, not the boundary file, and that needs saying.**
`L0.ctx` and `L0.gelu` have no HuggingFace counterpart — HF exposes no hook for
the attention interior or the FFN interior. The taps are our own reference's
values, trustworthy because `check_reference.py` proves that reference agrees
with HF at every point HF *does* expose (≤ 9.9e-07 at every layer boundary).
Using taps uniformly also keeps the four checks comparable.

Only `qkv` carries the `1/√32` fold, handled as in
[0011](../0011-m5-first-op-validated/TASK.md) by scaling the golden rather than
unfolding the weights.

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

python experiments\m5-pretiled-gemm\validate_layer_gemms.py
python experiments\m5-pretiled-gemm\validate_layer_gemms.py --emulate-bfp16
```

## Result

**bf16 — all four pass, comfortably.**

| GEMM | `[M,K,N]` | rel_fro | max_abs | worst `1−cos` |
|---|---|---|---|---|
| `qkv` | `[256, 384, 1152]` | 1.507e-03 | 1.722e-02 | 2.613e-06 |
| `attn_out` | `[256, 384, 384]` | 2.398e-03 | 1.137e-02 | 5.040e-06 |
| `ffn_up` | `[256, 384, 1536]` | 1.379e-03 | 2.905e-02 | 1.628e-06 |
| `ffn_down` | `[256, 1536, 384]` | 2.104e-03 | 6.962e-01 | 2.417e-05 |

**bfp16 — all four also pass, but not uniformly.**

| GEMM | rel_fro | max_abs | worst `1−cos` | `1−cos` vs bf16 |
|---|---|---|---|---|
| `qkv` | 9.898e-03 | 9.014e-02 | 7.203e-05 | 28× |
| `attn_out` | 1.471e-02 | 4.965e-02 | 2.595e-04 | 51× |
| `ffn_up` | 1.047e-02 | 1.952e-01 | 1.214e-04 | 75× |
| **`ffn_down`** | 1.315e-02 | 2.374e+00 | **8.856e-03** | **366×** |

The `rel_fro` column hides it — all four sit near 1e-2. **Direction error
does not.** `ffn_down` is 34–120× worse than the other three on `1−cos`, and
`1−cos` is what a cosine-ranked embedding model is actually judged on.

### Why `ffn_down`

Two things distinguish it, and both point the same way:

**Its input is post-GELU, which is bad for block floating point.** Within-block
statistics of each GEMM's activation input, blocks of 8 along K:

| GEMM | input | median blk max/min | **p90** | fraction \|x\| < 1e-3 |
|---|---|---|---|---|
| `qkv` | `emb.ln` | 18.2 | 133.1 | 0.3% |
| `attn_out` | `L0.ctx` | 18.6 | 113.8 | 0.5% |
| `ffn_up` | `L0.ln1` | 19.1 | 139.7 | 0.2% |
| **`ffn_down`** | `L0.gelu` | 16.5 | **251.0** | **1.2%** |

The *median* is unremarkable — slightly better than the others. **It is the tail
that matters**: GELU squashes negative pre-activations toward zero, so blocks
that straddle the activation threshold contain values spanning two orders of
magnitude, and one shared exponent cannot serve both. p90 is roughly double
every other input.

**Its reduction is 4× longer.** K=1536 against 384, so four times as many blocks
contribute error to each output element.

Note this refines, rather than contradicts,
[0008](../0008-m5-bfp16-real-data/TASK.md): that task measured `qkv` and found
real activations no worse than uniform. It is still true for `qkv`. The
distribution that hurts block FP in this model is not post-LayerNorm — it is
**post-GELU**, and only `ffn_down` sees it.

### The obvious mixed-precision idea, priced

Run `ffn_down` in bf16 and the rest in bfp16. Per token, the four GEMMs are
442k / 147k / 590k / 590k MACs, so `ffn_down` is **33% of layer FLOPs**. Amdahl:

```
speedup = 1 / (0.67/5.5 + 0.33/1) = 2.21x     against 5.5x for all-bfp16
```

**So it costs more than half the benefit.** Worth doing only if `ffn_down` alone
turns out to be what breaks the M8 MTEB budget — which is now a specific,
testable question rather than a vague worry.

## Problems hit

Nothing new. The tolerance-per-format fix from
[0011](../0011-m5-first-op-validated/TASK.md) earned itself immediately: every
bfp16 row here would have been reported as a failure under the bf16 bar.

## Artifacts

- `experiments/m5-pretiled-gemm/validate_layer_gemms.py` (renamed from
  `validate_layer0_qkv.py`, generalised)
- `artifacts/validate_layer_gemms_{bf16,bfp16}.json`
- `artifacts/trace_layer0_{qkv,attn_out,ffn_up,ffn_down}_4c*.txt`

## Next

**The GEMM half of M5 is done.** All four shapes compile, run, trace, and match
the oracle at both precisions. What remains is genuinely new work:

1. **Elementwise and reduction ops on the array**: LayerNorm (must stay fp32 —
   `docs/04-model`), softmax (fp32, max-subtracted), exact-erf GELU, bias adds.
   None has been on a core yet, and `research/notes/0001` warns that scalar
   float in a kernel costs 1,617×.
2. **Attention**, where `head_dim = 32` is the model's only awkward dimension.
3. **Then M6.**

Carried forward: every op added as a separate dispatch pays the 150 µs again
([0010](../0010-m5-b-reuse-and-cost-model/TASK.md)). With 4 GEMMs already and
~6 elementwise ops per layer to come, fusing whole layers should happen before
the op count grows, not after.
