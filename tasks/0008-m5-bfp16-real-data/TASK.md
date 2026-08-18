# 0008 — M5: bfp16 on real data, and a silent measurement trap

- **Date** 2026-08-17
- **Milestone** M5
- **Status** done — **M3's distribution finding refuted; a result-corrupting
  harness trap found and documented**

## Goal

Answer the question that has been carried forward since M2 and flagged again in
[0005](../0005-m3-python-reference/TASK.md) and
[0007](../0007-m5-pretiled-gemm-on-npu/TASK.md): **what does
`--emulate-bf16-mmul-with-bfp16` cost on real activations, on hardware?**

M2 measured 1.040e-02 on uniform [0,1). M3 predicted, in simulation, that real
data would be **6.0× worse**. Nothing had ever put real activations through the
device.

## What was done

### 1. The experiment

`experiments/m5-pretiled-gemm/bfp16_real_activations.py`. One GEMM — layer 0
QKV, 256×384×1152, MiniLM's real max sequence length — with four input pairs:

| A | B | why |
|---|---|---|
| uniform [0,1) | uniform [0,1) | M2's conditions, the anchor |
| real activations | real weights | what the model actually does |
| real activations | uniform | which operand carries the damage |
| uniform | real weights | the other half of that |

A is the M3 golden `hf.emb.ln` — the genuine post-LayerNorm input to layer 0,
verified against HuggingFace to 8.5e-08. B is the fused Q/K/V weight exactly as
M4 packs it.

**Every pair is run twice: flag off and flag on.** The flag-off run is the
control — bf16 in with fp32 accumulate must give ~1e-7 regardless of
distribution. That control is what caught the trap below.

### 2. The trap: two designs in one process silently corrupts every result after

The first version ran everything in one process. Its control column read
**6.66e+01** where it must read ~1e-7. Nothing raised; the numbers were simply
wrong, and they looked like a spectacular finding.

Bisected:

```
run0 emulate=False relfro=1.723e-07     <- correct
run1 emulate=True  relfro=9.015e-03     <- correct
run2 emulate=False relfro=2.964e+01     <- garbage
run3 emulate=True  relfro=3.272e+01     <- garbage
```

Then, to find the precise rule — never returning to an earlier design:

```
run0 emulate=False relfro=1.723e-07
run1 emulate=True  relfro=9.015e-03
run2 emulate=True  relfro=6.833e+01     <- SAME design as run1, garbage
run3 emulate=True  relfro=2.646e+01
```

**Once two different compiled designs have been dispatched in one process
without tracing, every dispatch after them returns garbage — including repeats
of a design that had just produced a correct answer.** Three dispatches of a
single design are fine indefinitely (verified: uniform / real / uniform again,
all ~1.8e-07).

Traced runs appear immune — every traced run in
[0004](../0004-m2-multicore-gemm/TASK.md) and
[0007](../0007-m5-pretiled-gemm-on-npu/TASK.md) reports a correct `rel_fro`
across many designs in one process, most likely because a fresh `TraceConfig`
per call forces a genuine recompile and reload. That is an observation, not a
mechanism, and it is not something to rely on.

**Fix: one design per process.** Every measurement in this experiment now runs
in a fresh interpreter.

### 3. The answer

Controls all pass (1.4–1.9e-07), so the harness is sound:

| inputs | bf16+f32 | **bfp16** | A blk range | B blk range |
|---|---|---|---|---|
| uniform × uniform (M2's conditions) | 1.88e-07 | **1.06e-02** | 10.8 | 10.7 |
| real act × real weight (**the model**) | 1.72e-07 | **9.02e-03** | 18.2 | 16.9 |
| real act × uniform | 1.73e-07 | 2.28e-01 | 18.2 | 10.7 |
| uniform × real weight | 1.42e-07 | 1.95e-01 | 10.8 | 16.9 |

**Real data is 0.85× the uniform error — indistinguishable, if anything
marginally better.** M3's simulation predicted 6.0× worse. **Refuted.**

Note the mechanism M3 identified *is* present: the median within-block dynamic
range really is 18.2 on real activations against 10.8 on uniform, exactly as the
outlier argument in `docs/04-model` predicts. It simply does not translate into
error on this datapath. The model was wrong about the consequence, not about the
input statistics.

### 4. The two mixed rows are an artifact, not a finding

`real × uniform` and `uniform × real` are 20× worse, and that is **not** evidence
about block floating point. Uniform [0,1) is strictly positive with mean 0.5;
real activations and real weights are both roughly zero-mean. Pairing a
zero-mean operand with an all-positive one makes the dot product a sum with
massive cancellation, so the result is small while the individual products are
not — and the same absolute quantisation error becomes a much larger *relative*
error. Both physically meaningful pairings (uniform×uniform, real×real) agree.

They are reported because they were run, and because someone reading only the
table would otherwise draw the wrong conclusion from them.

### 5. Recalibrating the M3 end-to-end estimate

M3's end-to-end numbers came from a block-float model whose mantissa width was
fitted against the *uniform* hardware measurement, then applied to real
activations. Now that the real-data hardware number exists, the fit is anchored
there instead (`reference/precision_study.py`, part 1c):

```
  bits/elem   sim rel_fro   ratio vs hw
          5     5.052e-02         5.60x
          6     2.503e-02         2.77x
          7     1.243e-02         1.38x     <- best
          8     6.484e-03         0.72x
```

**7 bits/element**, not 5. The end-to-end consequence:

| metric | M3 (5-bit, uniform-fitted) | **corrected (7-bit, real-fitted)** |
|---|---|---|
| worst `1 − cos` vs HuggingFace | 1.823e-02 | **1.773e-03** |
| max sentence-similarity shift | 1.409e-02 | **6.492e-03** |
| `last_hidden_state` rel_fro | 2.164e-01 | 5.987e-02 |

**bfp16's end-to-end cost is ~10× smaller than M3 claimed.** It is still ~140×
worse than bf16 (`1−cos` 1.3e-05), so bf16 remains the safe default — but a
6.5e-03 similarity shift for 5.5× throughput is a genuine tradeoff for M8 to
decide, not the disqualification M3 made it look like.

The fit is still only good to ~1.4×, so these are estimates. A real answer needs
the full encoder on the NPU (M6) or MTEB (M8).

### 6. Re-checking M5's pre-tiling conclusion, which was measured unsafely

[0007](../0007-m5-pretiled-gemm-on-npu/TASK.md)'s end-to-end table ran 8 designs
in one process with tracing off — exactly the condition that corrupts results.
Timing is plausibly unaffected (the kernel does the same work either way), but
that cannot be assumed, so it was re-run with one process per measurement.

First isolated attempt, one run each:

| shape | rowmajor | pretiled | change |
|---|---|---|---|
| qkv | 1.62 | 1.65 | +2.0% |
| proj | 0.73 | 0.72 | −2.7% |
| ffn_up | 1.61 | 1.82 | **+12.8%** |
| ffn_down | 1.93 | 2.17 | **+12.7%** |

That looked like pre-tiling finally winning on the two large-B shapes. It is
noise. With three isolated runs per variant:

| shape | rowmajor mean (spread) | pretiled mean (spread) | ratio |
|---|---|---|---|
| `ffn_down` | 2.133 (**1.8%**) | 2.149 (**16.5%**) | +0.7% |
| `ffn_up` | 1.876 (**3.3%**) | 1.826 (**9.4%**) | −2.6% |

**0007's conclusion stands: pre-tiling is a wash on throughput.** And the
stability gap it found per-core reproduces in wall clock — pre-tiled spreads
9–17% where row-major spreads 2–3%. That is now the most robust thing known
about pre-tiling: it does not make the design faster, it makes it less
predictable.

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

# the measurement (spawns one child process per configuration)
python experiments\m5-pretiled-gemm\bfp16_real_activations.py

# recalibrate the M3 model against the real-data hardware number (CPU only)
& "C:\Users\vegar\.conda\envs\iron\python.exe" reference\precision_study.py

# reproduce the corruption trap
python -c "
import sys; sys.path.insert(0,'experiments/m5-pretiled-gemm')
import aie.iron as iron
from aie.iron.device import from_name
iron.set_current_device(from_name('npu2', n_cols=None))
import bfp16_real_activations as X
M,K,N=256,384,1152
a16,b16,_ = X.build_set('real_real', 0, M,K,N)
for i,em in enumerate([False,True,True,True]):
    print(i, em, X.run(a16,b16,M,K,N,64,64,48,4,em))
"
```

The isolated bench sweeps were driven from PowerShell, one `python -c` per
measurement; results are in `artifacts/bench_all_shapes_c8_isolated.json` and
`artifacts/bench_repeats_c8_isolated.json`.

## Result

| claim | source | verdict |
|---|---|---|
| bfp16 is ~6× worse on real activations than uniform | 0005 (simulation) | **Refuted.** Hardware: 0.85×, indistinguishable |
| Post-LN outliers widen the within-block dynamic range | 0005 | **Confirmed** (18.2 vs 10.8) — but it does not produce error here |
| bfp16 costs `1−cos ≈ 1.8e-02` end to end | 0005 | **Corrected to ≈1.8e-03** once the model is fitted on real data |
| bf16 + fp32 accumulate is effectively free | 0003, 0005 | **Confirmed again** — controls at 1.4–1.9e-07 on every distribution |
| Pre-tiling is a throughput wash | 0007 | **Confirmed** under process isolation with repeats |

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| Control column read 6.66e+01 where bf16+fp32 must read ~1e-7 | Two different compiled designs dispatched in one process without tracing corrupts every subsequent dispatch, **including repeats of a design that just worked**. Silent — nothing raises | One design per process. The control is the only reason this was caught rather than published as a finding |
| A one-shot isolated bench showed pre-tiling +12.8% on two shapes | n=1 on a wall-clock measurement whose spread is 9–17% | Three repeats per variant. **A single wall-clock number is not a measurement on the pre-tiled path either** |
| `real × uniform` and `uniform × real` both ~20× worse, easy to read as a bfp16 property | Pairing a zero-mean operand with a strictly-positive one creates cancellation, inflating *relative* error | Reported with the explanation rather than dropped |
| A rewrite of the script left four `print(f"` literals split across lines | Escaped newlines in a generated edit became real newlines | Repaired; the script now asserts nothing and simply parses. Recorded because the same edit pattern will do it again |

## Artifacts

- `experiments/m5-pretiled-gemm/bfp16_real_activations.py`
- `experiments/m5-pretiled-gemm/artifacts/bfp16_real_activations.json`
- `experiments/m5-pretiled-gemm/artifacts/bench_all_shapes_c8_isolated.json`
- `experiments/m5-pretiled-gemm/artifacts/bench_repeats_c8_isolated.json`
- `reference/goldens/precision_study.json` — regenerated with the real-data fit

## Next

1. **bfp16 is back on the table.** M3 priced it out on a simulation artifact.
   The real question for M8 is whether a 6.5e-03 similarity shift costs more
   than 0.3 MTEB points, and that needs MTEB, not more GEMMs.
2. **Audit every multi-design measurement made without tracing.** M2's `--bench`
   sweep over M is the main one. Traced results look safe, but "look safe" is
   not the standard this project uses.
3. **Still untouched: B reuse across row blocks**, which
   [0007](../0007-m5-pretiled-gemm-on-npu/TASK.md) identified as the real
   remaining lever. Nothing in this task moved it.
4. The pre-tiled instability now has two independent confirmations (per-core
   traced, and wall clock) and still no mechanism.

---

## Correction — the trap in section 2 was misdiagnosed

Added 2026-08-17, same day. See
[0009](../0009-m5-sync-misdiagnosis/TASK.md).

**Section 2 above is wrong.** There is no "two designs in one process" rule. The
corruption was **our own missing host→device sync**: this task wrote inputs with
`A.numpy()[:] = values`, which touches only the host buffer.
`Tensor.numpy()` syncs *from* the device and hands back host memory; writing into
it never syncs back. `A[:] = values` syncs both ways.

Only the first dispatch after a load is correct either way, which is why the
symptom looked like a device-state problem. What killed the theory was the
sequence `A A A B B B`: corruption arrived at dispatch 1, before any second
design existed, and loading a new design appeared to cure it.

Also withdrawn from this task:

- **"Traced runs appear immune."** False — tested directly, traced runs with
  `.numpy()` writes are equally broken.
- **"Audit every multi-design untraced measurement."** Too wide. The real audit
  is anything that *writes chosen values* to a device tensor. Runs on
  `iron.rand` data were never at risk, because nothing was written.

**Everything measured in this task survives.** Each measurement ran in its own
process with exactly one dispatch — precisely the case where a `.numpy()` write
does reach the device. Re-run with `A[:] =` plus asserts that the device holds
the intended values, the four rows are **numerically identical**: 1.06e-02,
9.02e-03, 2.28e-01, 1.95e-01. Real data is 0.85× uniform; M3's 6.0× prediction
stays refuted; the recalibration to 7 bits and the corrected end-to-end figures
stand.

The process isolation adopted here was the right remedy for the wrong reason.
