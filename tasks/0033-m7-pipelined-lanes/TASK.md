# 0033 — M7: pipelined lanes. 611 → 833 seq/s — CPU wall parity PASSED

- **Date** 2026-08-18
- **Milestone** M7
- **Status** done — **833.4 seq/s** (10-group bench, batch 2×128,
  `--threads 24 --pipeline 2`), `1-cos` **1.086e-05** with all lanes
  **bit-identical**. Against the CPU's 710 seq/s: **1.17× faster in wall
  clock**, at ~5.3 cores → still **2.7× better per core**.

## Goal

Expert §6a, deferred twice with cause, now unblocked by 0032: with zero
design switches and one hw_context, overlap one encode's HOST work with
another encode's NPU work. The two-process experiment (0024/0025) had already
measured 1.46× of exactly this overlap — from outside the process, paying two
contexts. Do it inside, paying nothing.

## What was built

Two (or N, `--pipeline N`) Encoder lanes over the SAME unified design:

- Each lane owns its **A and C buffer slots** on the shared design
  (`Design::stage_alloc` — a staged slot with no data) and converts into its
  own slot **outside** the lock.
- Every NPU interaction — `bind_instr` + `bind` + syncs + dispatch — happens
  under **one mutex**. The array serializes dispatches anyway (note 0004);
  the lock only makes explicit what the hardware enforces, and it
  self-organizes the lanes into anti-phase: while lane A holds the NPU, lane
  B does host work, and vice versa.
- Weights and LN/GELU/softmax parameters are the design's, shared read-only.
- Thread budget splits: `--threads 24 --pipeline 2` = 12 host threads per
  lane.

**Fail-closed validation:** the lanes run the same input through
deterministic math, so their outputs must be **bit-identical** — memcmp over
all 3,145,728 floats, before the golden check. Any cross-lane buffer
corruption fails loudly. PASS, and `1-cos` 1.086e-05 unchanged.

## Measured (batch 128 per lane, seq 64)

| config | seq/s | cores | notes |
|---|---|---|---|
| single lane (0032) | 604–618 | ~3.5 | |
| `--pipeline 2 --threads 16` | 791–818 | 4.3–5.5 | 8+8 threads |
| `--pipeline 2 --threads 24` | **832.6–836.2** | ~5.3 | **production setting** |
| `--pipeline 2 --threads 32` | 817.9 | 8.3 | oversubscribed, worse |
| `--pipeline 3 --threads 24` | 836.7 | 6.8 | **plateau — see below** |

Stability: 833.4 over a 10-group bench. The 2-lane breakdown:

```
  NPU dispatch+wait (serialized)  156.6 ms  51.0%   48 dispatches/group
  p1 host work                    149.1 ms  48.6%  (conv 23  bias 77  attn 29  elt 20)
  p2 host work                    150.7 ms  49.1%
```

Wall 307 ms against a serial sum of 157 + 149 + 151 = 457 ms → **1.49×
overlap** — almost exactly the 1.46× the two-process probe measured, now
without the second process.

## Why 3 lanes buys nothing: the readback is DRAM-bound

Adding a third lane kept 836 seq/s and inflated per-lane bias/readback from
~78 to ~135 ms — the C-readback path saturates memory, and more lanes just
queue on DRAM. A streaming-load (`movntdqa`) variant of the bias pass changed
nothing (the bo memory is cacheable; NT loads only help on WC memory), which
localizes the cost: **~340 MB per encode of cold C reads + out writes is the
host-side wall**. The fixes are at the source, not in the loop:

1. **bf16-C epilogue** (0030 §6c route): halves the C bytes.
2. **Device-resident intermediates** (§6b): the up→down hop never visits the
   host at all.
3. Both feed the pipelined fused design.

Also structural, worth recording: a lane cannot overlap its OWN dispatch (the
next op depends on it), so a lane's floor is `host + own-NPU` ≈ 150 + 78 =
228 ms. Two lanes measured 307 for two encodes — the pipeline is already
within ~35% of its per-lane bound, and the NPU sits at 51% occupancy. Getting
further means shrinking the lane itself, not adding lanes.

## The CPU baseline, re-measured the same day

The 710 seq/s comparator is from 0018. Re-measured now
(`bench_cpu_baseline.py`, best of 5, 12 torch threads, NPU idle):

```
  batch   4:   277.4 seq/s
  batch  32:   589.3 seq/s
  batch 128:   662.9 seq/s
```

662.9 today against 710 historically -- machine state varies. The claim is
made against the CONSERVATIVE (higher) number: **833 vs 710 = 1.17×**;
against today's measurement it is 1.26×. Parity holds either way.

## Where the project stands after today

| | this morning | now |
|---|---|---|
| seq/s at batch 128 | 296 | **833 (pipelined), 618 (single lane)** |
| vs CPU 710 wall | 0.42× | **1.17×** — **parity passed** |
| `1-cos` vs HuggingFace | 2.469e-04 | **1.086e-05** |
| design switches per encode | 49 | **0** |
| cores busy | 2.5 | 5.3 (pipelined) / ~3.5 (single) |
| per-core advantage | 3.2× | 2.7× (pipelined) / 3.0× (single) |

## Exact commands

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\runtime
cmake --build build --config Release
.\build\npuembed.exe .. --artifacts artifacts_b128il --threads 24 --pipeline 2            # validate (bitwise lanes + golden)
.\build\npuembed.exe .. --artifacts artifacts_b128il --threads 24 --pipeline 2 --bench 10 # measure
```

## Next

1. **Shrink the lane**: bf16-C epilogue and/or device-resident up→down —
   the ~77 ms bias/readback and 23 ms conversions are the lane's biggest
   removable items, and DRAM saturation says remove bytes, not add threads.
2. **B-reuse** (acquire-and-hold): NPU occupancy is 51%; GEMM wait 78 ms per
   encode still re-streams B 32×. Worth ~20 ms per encode — which in the
   pipelined regime converts directly to throughput only once the host lane
   shrinks below the NPU lane.
3. **Energy measurement** — now genuinely urgent: 5.3 cores at 833 seq/s vs
   12 cores at 710 needs the energy number to close the project's own
   argument.
4. The **pipelined fused design** remains the correct endgame for both 1 and 2.
