# Note 0003 — `Tensor.numpy()` writes never reach the device

**Found:** 2026-08-17 · **Corrected the same day**, see
[`tasks/0009`](../../tasks/0009-m5-sync-misdiagnosis/TASK.md)
**Severity:** silent wrong answers, no exception, plausible-looking output

> **This note originally claimed something else** — that dispatching two
> different compiled designs in one process corrupts the NPU. **That was wrong.**
> It was this bug, seen from the outside. The retraction is kept below because
> the misdiagnosis is the instructive part.

## The rule

**To put values on the device, write through the tensor, not through its
numpy view.**

```python
A = iron.rand((M, K), dtype=bfloat16, device="npu")

A.numpy()[:] = my_values     # WRONG -- host only. Silently ignored by the NPU.
A[:] = my_values             # RIGHT -- syncs host AND device.
```

`Tensor.numpy()` syncs *from* the device and hands back the host buffer
(`tensor_class.py:307`). Writing into that array never syncs back.
`Tensor.__setitem__` syncs both ways (`tensor_class.py:182`).

## Why it is so easy to ship

**The first dispatch in a process comes out correct either way.** Every dispatch
after it silently uses stale device data.

```
mode=numpy: 6 dispatches, one design, identical inputs
  dispatch0  relfro_vs_intended=1.723e-07  ok
  dispatch1  relfro_vs_intended=4.998e+01  WRONG
  dispatch2  relfro_vs_intended=5.032e+01  WRONG
  ...
mode=setitem: 6 dispatches, one design, identical inputs
  dispatch0..5  relfro_vs_intended=1.723e-07  ok   (all six)
```

Worse, `A.numpy()` **keeps reporting the values you wrote**, because it returns
the host buffer you just modified. So a read-back check passes while the device
is running on something else entirely.

## The trap inside the trap: never validate against a read-back

The natural correctness check is the one that hides this bug:

```python
got = C.numpy()
ref = A.numpy() @ B.numpy()      # WRONG -- re-syncs from the device
```

If the write never landed, `ref` is computed from whatever the device actually
used, so `got` and `ref` agree and the check passes while measuring nothing.
Compare against **the values you intended to send**:

```python
ref = A_np @ B_np                                  # what we meant to compute
assert np.array_equal(A.numpy(), A_np), "A did not reach the device"
```

## Retracted: the "two designs per process" theory

The symptom was first seen as *"two different compiled designs in one process
corrupt everything after them"*, and that framing got as far as a committed note,
a task log, and a rule in CLAUDE.md. Evidence that looked conclusive:

```
run0 emulate=False relfro=1.723e-07   correct
run1 emulate=True  relfro=9.015e-03   correct
run2 emulate=True  relfro=6.833e+01   garbage -- SAME design as run1
```

It even reproduced the shape of a real bug report (below). What killed it was a
sequence that the theory could not survive:

```
sequence A A A B B B
  dispatch0  design=A  relfro=1.723e-07  ok
  dispatch1  design=A  relfro=6.463e+01  GARBAGE   <- second design not involved
  dispatch3  design=B  relfro=9.015e-03  ok        <- new design "fixes" it
```

Corruption arrived before any second design, and loading a new design appeared
to cure it. Both facts are explained by "only the first dispatch after a fresh
load carries the host write" and neither by the design-count theory.

**Lesson.** The first theory fitted every observation available at the time and
was still wrong. What broke it was constructing a sequence chosen to *falsify*
it rather than one more that could confirm it.

## Relation to FastFlowLM issue #647

[ROCm/FastFlowLM#647](https://github.com/ROCm/FastFlowLM/issues/647) reports
non-deterministic embeddings on the same SKU: identical requests returning
different vectors, the first request after startup reproducible across restarts,
and short-text embeddings permanently shifting once a longer document is
processed.

**This note does not diagnose that issue** — the bug here is our own API misuse
in Python, and FastFlowLM is C++ against XRT directly. But the *symptom class* is
worth passing on, because a missing host→device sync on an input buffer produces
exactly that fingerprint: first request correct, later ones stale, values that
change when something else perturbs the buffer state. See
a draft comment kept in the working repository.

## Related

- [`tasks/0009`](../../tasks/0009-m5-sync-misdiagnosis/TASK.md) — the misdiagnosis
- [`tasks/0008`](../../tasks/0008-m5-bfp16-real-data/TASK.md) — where it surfaced
- [note 0002](0002-iron-silent-arch-fallback.md) — the other silent-wrong-answer trap
