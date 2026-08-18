# 0009 — M5: the "NPU corruption" was our own missing host→device sync

- **Date** 2026-08-17
- **Milestone** M5
- **Status** done — **[0008](../0008-m5-bfp16-real-data/TASK.md)'s trap section
  and note 0003 are retracted; their measurements survive**

## Goal

Check whether the silent corruption reported in
[0008](../0008-m5-bfp16-real-data/TASK.md) — *"two compiled designs in one
process corrupt every dispatch after them"* — is really what it looked like,
before any of it is passed to a third party.

The prompt was practical: the finding looked like it explained
[ROCm/FastFlowLM#647](https://github.com/ROCm/FastFlowLM/issues/647), and the
next step would have been posting that analysis publicly.

## What was done

### 1. A sequence chosen to falsify, not to confirm

Everything measured in 0008 fitted the two-designs theory. So instead of
gathering another confirming case, run the one sequence the theory cannot
survive — three dispatches of design A, then three of design B:

```
sequence A A A B B B
  dispatch0  design=A  relfro=1.723e-07  ok
  dispatch1  design=A  relfro=6.463e+01  GARBAGE
  dispatch2  design=A  relfro=6.641e+01  GARBAGE
  dispatch3  design=B  relfro=9.015e-03  ok
  dispatch4  design=B  relfro=6.928e+01  GARBAGE
  dispatch5  design=B  relfro=6.218e+01  GARBAGE
```

**Corruption arrives at dispatch 1, before any second design exists.** And
loading a new design at dispatch 3 appears to *cure* it. The theory dies on both
counts.

The real pattern: **only the first dispatch after a design is loaded is
correct.** Confirmed on a single design, 8 dispatches, four separate processes —
first bad dispatch was index 1 every time.

### 2. The clue in the wrong answers

The corrupted values were not random. Re-running an earlier three-input test
gave, reproducibly across processes:

```
  uniform   relfro=1.884e-07  ok
  real      relfro=6.383e+01  GARBAGE
  uniform2  relfro=6.086e-02  GARBAGE
```

`uniform2` is wrong by only ~6e-2. For two independent uniform [0,1) matrices,
`A@B` is dominated by the mean — every entry is ≈ K/4 with small variance — so
computing with the *wrong* uniform matrix is only a few percent off. Whereas
real, zero-mean activations against stale uniform data are wrong by 60×.

That is the signature of **stale input buffers**, not of a corrupted datapath.

### 3. The cause

`aie/utils/hostruntime/tensor_class.py`:

| | |
|---|---|
| `Tensor.numpy()` (line 307) | syncs **from** device, returns the host buffer |
| `Tensor.__setitem__` (line 182) | syncs from device, writes, syncs **to** device |

Our code did `A.numpy()[:] = values`. That writes the host buffer and **never
syncs to the device**. Every experiment that put chosen values on the NPU had
this bug.

### 4. Confirmed by fixing it

```
mode=numpy: 6 dispatches, one design, identical inputs
  dispatch0  relfro_vs_intended=1.723e-07  ok
  dispatch1  relfro_vs_intended=4.998e+01  WRONG
  dispatch2  relfro_vs_intended=5.032e+01  WRONG
  dispatch3  relfro_vs_intended=3.811e+01  WRONG
  dispatch4  relfro_vs_intended=1.720e+01  WRONG
  dispatch5  relfro_vs_intended=6.744e+01  WRONG

mode=setitem: 6 dispatches, one design, identical inputs
  dispatch0..5  relfro_vs_intended=1.723e-07  ok   (all six)
```

`A[:] = values` is correct on every dispatch. **It was our bug, not the driver's,
not the compiler's, not FastFlowLM's.**

### 5. Why the read-back check did not catch it

The correctness check in 0008 was `ref = A.numpy() @ B.numpy()`. `A.numpy()`
re-syncs from the device, so `ref` was computed from whatever the device
actually used — and agreed with `got` perfectly while measuring nothing about
the values we meant to send.

`A_matches_intended=True` throughout, because `.numpy()` returns the host buffer
we had just written. **Both the check and the read-back lied, consistently.**

The fix is to reference against the values *intended*, and to assert the device
agrees:

```python
ref = A_np @ B_np
assert np.array_equal(A.numpy(), A_np), "A did not reach the device"
```

### 6. Tracing is not a workaround

0008 speculated that traced runs were immune. Tested directly with `.numpy()`
writes:

```
tracing=ON, 4 dispatches:
  dispatch0  relfro_vs_intended=1.723e-07  ok
  dispatch1  relfro_vs_intended=nan        WRONG
  dispatch2  relfro_vs_intended=nan        WRONG
  dispatch3  relfro_vs_intended=7.082e+01  WRONG
```

**Also broken.** That speculation is withdrawn.

## What survives, and why

### [0008](../0008-m5-bfp16-real-data/TASK.md)'s bfp16 result: **valid**

Every measurement there ran in its own process with exactly one dispatch — which
is precisely the case where a `.numpy()` write does reach the device. Re-run with
`A[:] =` plus asserts that the device holds the intended values:

| inputs | bf16+f32 | bfp16 |
|---|---|---|
| uniform × uniform | 1.88e-07 | 1.06e-02 |
| real act × real weight | 1.72e-07 | 9.02e-03 |
| real act × uniform | 1.73e-07 | 2.28e-01 |
| uniform × real weight | 1.42e-07 | 1.95e-01 |

**Numbers identical to 0008.** The headline stands: real data is **0.85×** the
uniform error, refuting M3's predicted 6.0×. The process isolation was adopted
for the wrong reason and happened to be exactly the right remedy.

### [0007](../0007-m5-pretiled-gemm-on-npu/TASK.md)'s pre-tiling results: **valid**

`gemm_pretiled.py` wrote its pre-tiled B via `.numpy()` too, so this needed
checking. It is saved by its own correctness check: the reference is built from
a host copy taken *before* the tiled write, so a write that failed to land would
show as `rel_fro` of order 1, not 1.04e-02. Every repeat in 0007 reported
`relfro=1.04e-02 PASS`.

Re-run after switching to `B[:] =`:

```
  rowmajor         over 3 runs: mean  140.8  range 140.8-140.9  spread  0.1%
  pretiled[k,n|st] over 3 runs: mean  116.2  range 107.8-123.7  spread 13.7%
```

Unchanged, including the stability gap.

### Retracted

- **note 0003's "two designs per process" rule** — rewritten.
- **CLAUDE.md trap 6b** — replaced.
- **0008's claim that traced runs are immune** — false.
- **0008's suggestion to "audit every multi-design untraced measurement"** — the
  right audit is narrower: anything that *writes chosen values* to a device
  tensor. Measurements on `iron.rand` data were never affected, because nothing
  was written.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| A theory that fitted every observation, and was wrong | Only confirming sequences had been run | Construct the sequence that would falsify it. `A A A B B B` took two minutes and killed it |
| The correctness check passed while measuring nothing | Reference computed from a device read-back, so it agreed with whatever the device used | Reference against intended values; assert the device matches |
| Read-back "proved" the write landed | `.numpy()` returns the host buffer that was just written | The same assert, but only meaningful *after* a dispatch |
| Nearly published the misdiagnosis to a third-party bug tracker | Confidence from a plausible symptom match | Checked our own stack before attributing anything to someone else's |

## Artifacts

Probes live in the scratchpad rather than the repo — they exist to answer one
question each and their outputs are quoted above in full:
`trigger_probe.py`, `single_design_probe.py`, `recheck_earlier.py`,
`api_probe.py`, `trace_sync_probe.py`.

Committed changes: `experiments/m5-pretiled-gemm/bfp16_real_activations.py` and
`gemm_pretiled.py` now write through `Tensor.__setitem__`;
`research/notes/0003` rewritten; `FASTFLOW_ISSUE.md` added.

## Next

1. **Grep for `.numpy()[` on device tensors across the repo** before trusting any
   future measurement that sets input values.
2. Unchanged from [0007](../0007-m5-pretiled-gemm-on-npu/TASK.md): **B reuse
   across row blocks** is still the untouched lever.
3. The pre-tiled instability (13.7% spread here, reproduced again) still has no
   mechanism — and is now known *not* to be a sync artifact, since it survives
   the fix.
