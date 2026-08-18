# 0016 — M5: AIE float vector arithmetic is full fp32

- **Date** 2026-08-17
- **Milestone** M5
- **Status** done — **refutes [0015](../0015-m5-gelu-polynomial/TASK.md)'s inference**

## Goal

[0015](../0015-m5-gelu-polynomial/TASK.md) inferred, from a GELU polynomial
diverging 3.886e-03 from its own numpy fp32 model, that `aie::vector<float>`
arithmetic on AIE2P is effectively bf16. That inference had large consequences —
`docs/04-model` requires **both LayerNorm and softmax in fp32** — so it needed a
direct measurement rather than another application-level guess.

## Method

```
out = (1.0f + eps) - 1.0f
```

with `eps` an exact power of two supplied per lane, `2^-1 … 2^-24`.

- IEEE fp32 has a 24-bit mantissa, so this returns `eps` down to `2^-23`.
- bf16 has 8, so it returns 0 as soon as `eps < 2^-8`.

Powers of two are exactly representable in bf16, so `eps` survives both the
input and the output conversion untouched — whatever comes back is what the
*arithmetic* did, not what the format did. The script asserts that rather than
assuming it.

The subtraction is what makes it observable: without it the difference sits
below the bf16 output grid and cannot be seen.

Run twice, because the add path is only half the question — every Horner step in
a polynomial kernel is `aie::mul(...).to_vector<float>()`, which goes through an
accumulator:

| variant | expression |
|---|---|
| `add` | `(1.0f + eps) - 1.0f` |
| `mul` | `((1.0f + eps) * 1.0f) - 1.0f` |

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
python experiments\m5-eltwise\fp32_probe.py add
python experiments\m5-eltwise\fp32_probe.py mul
```

## Result

Both paths, identically:

```
  eps 1.192e-07  2^-23   returned 1.192e-07   exact
  eps 5.960e-08  2^-24   returned 0.000e+00   no

  deepest eps that survived : 2^-23
  implied mantissa bits     : ~24
  VERDICT: aie::vector<float> carries ~24 mantissa bits -> fp32
```

**`aie::vector<float>` is full IEEE fp32, on both add and multiply.**
[0015](../0015-m5-gelu-polynomial/TASK.md)'s inference is refuted, and with it
its consequences: **fp32 LayerNorm and fp32 softmax remain achievable**, and
`docs/04-model`'s requirement stands.

## What this leaves open

The GELU kernel's 3.886e-03 divergence from a numpy fp32 model of the same
polynomial is now known *not* to be arithmetic precision. Untested suspects:
`aie::abs`, `aie::min`, `aie::max` on float, or the rounding mode of the
fp32 → bf16 conversion. Truncation instead of round-to-nearest would fit the
magnitude, but that is a hypothesis, not a finding, and it is not chased here.

The kernel is unaffected and remains in use: 4.312e-03, 3.1× better than
`aie::tanh`.

## Problems hit

None. The experiment worked first time, which is the point of making it small.

## Artifacts

- `experiments/m5-eltwise/kernels/fp32_probe.cc` (two entry points)
- `experiments/m5-eltwise/fp32_probe.py`
- `artifacts/fp32_probe_{add,mul}.json`

## Next

Nothing blocks LayerNorm or softmax on precision grounds. Both can be written in
fp32 as `docs/04-model` requires.

**Method note worth keeping.** This is the second time in one milestone that a
plausible mechanism explaining several independent observations turned out to be
wrong — the first was [0009](../0009-m5-sync-misdiagnosis/TASK.md). Both times
the resolution was a *direct* measurement of the suspected component in
isolation, and both times it took one short experiment. Reaching for that sooner
is cheaper than another round of inference from application results.
