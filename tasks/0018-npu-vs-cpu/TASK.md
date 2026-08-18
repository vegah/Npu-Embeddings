# 0018 — NPU vs CPU: the engine wins, the plumbing loses

- **Date** 2026-08-17
- **Milestone** M8 (partial — the speed half)
- **Status** done — **first honest head-to-head; result is not flattering and
  the reason is precise**

## Goal

Measure NPU against CPU, now that M6 produces a validated encode.

## Method and its limits

Wall clock on both sides. `docs/05-measurement` permits that for end-to-end
throughput and host cost, and forbids it as a kernel-cycle claim — nothing here
is presented as a statement about kernel quality.

**The NPU was quiesced.** `flm` (PID 8944, running since the 15th) was killed
before measuring; it holds the device and would have silently inflated every
number. `LemonadeServer` remained but is idle and confirmed not to be using the
NPU. The script prints what it finds still running so a reader can judge.

The CPU side gets all 12 threads — numpy on a multithreaded BLAS for the GEMMs,
and torch for the encode. That is the fair comparison: it is what someone would
actually run instead of this.

## Result 1 — the GEMM engine, apples to apples

Four MiniLM GEMM shapes, bf16 on the NPU at 8 columns, fp32 numpy on the CPU:

| M | CPU, all four | NPU, all four | |
|---|---|---|---|
| 256 | 3,276 µs | 769 µs | **4.26× NPU** |
| 1024 | 6,367 µs | 1,424 µs | **4.47× NPU** |
| 4096 | 20,145 µs | 4,234 µs | **4.76× NPU** |

Per shape at M=4096 the NPU reaches 2.4–3.9 TFLOP/s against the CPU's
0.54–0.84. **The engine does what it is supposed to do**, and the advantage
grows with batch, exactly as the cost model in
[0010](../0010-m5-b-reuse-and-cost-model/TASK.md) predicts.

## Result 2 — a full encode, and here we lose badly

`sentence-transformers` on CPU, same model, same corpus, seq 64:

| batch | CPU | our NPU path |
|---|---|---|
| 4 | **267.6 seq/s** | **12.6 seq/s** |
| 32 | 548.1 seq/s | — |
| 128 | 710.0 seq/s | — |

**The CPU is 21× faster than our encode.** That is the honest number.

## Why, precisely

The breakdown is what makes this useful rather than merely discouraging:

```
  wall clock for 4 sequences at seq 64: 316.8 ms  ->  12.6 seq/s
    NPU dispatches (24 GEMM + 6 GELU)         252.6 ms  79.7%
    host (LayerNorm, softmax, attention, ...)  64.2 ms  20.3%
    of the NPU time, ~4.5 ms is the 30 x 150 us fixed dispatch cost
```

**252.6 ms across 30 dispatches is 8.4 ms each, against a hardware dispatch cost
of 150 µs — 56× more.** And Result 1 measured those same 24 GEMMs at roughly
4.6 ms *in total*. So of the 252.6 ms, about **248 ms is Python glue**: device
tensor allocation per call, bf16 conversion, host↔device copies, and JIT
dispatch bookkeeping, all inside `NpuGemm.__call__`.

The 64.2 ms of host time is not a CPU baseline either — it is the M3 *reference*
implementation, which computes LayerNorm in fp64 and runs 576 tiny attention
GEMMs through a Python loop because it was written to be obviously correct, not
fast.

**So this is a correctness artifact being timed, not a performance artifact.**
Neither side of the 316.8 ms is optimised, and reporting 12.6 seq/s as an NPU
result would be misleading. The two numbers that *are* real today are the GEMM
engine comparison above and the cost model from
[0010](../0010-m5-b-reuse-and-cost-model/TASK.md).

## What the model says is reachable

From [0010](../0010-m5-b-reuse-and-cost-model/TASK.md)'s validated
`t = 150 µs + traffic / 33 GB/s`, and the measured 3.79 TFLOP/s at M=4096, a
full 6-layer encode at seq 128 projects to **~1,300 seq/s** — against the CPU's
measured **710 seq/s** at batch 128. Roughly **1.8×**, before any fusion, and
[0010](../0010-m5-b-reuse-and-cost-model/TASK.md) prices B reuse at a further
1.68×.

That is a real but modest margin, and it is worth saying plainly: **on this
hardware, at this model size, the NPU's win over a 12-thread CPU is measured in
small single digits, not orders of magnitude.** The case for it rests on power
and on leaving the CPU free — neither of which this task measured.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| First instinct was to report 12.6 seq/s vs 267.6 as "NPU vs CPU" | It is neither. 78% of it is Python glue and the host side is deliberately unoptimised reference code | Decompose before comparing. The breakdown turned a discouraging number into a specific target |
| A killed process is a measurement prerequisite, not a detail | `flm` had been holding the NPU since the 15th | The bench script now prints any surviving NPU user and warns if `flm` is among them |

## Artifacts

- `experiments/m8-npu-vs-cpu/bench_gemm.py`, `bench_cpu_baseline.py`
- `artifacts/bench_gemm.json`, `artifacts/bench_cpu_baseline.json`
- `reference/encode_npu.py` gained a wall-clock breakdown

## Next

The ranking of work changed, and this is the clearest evidence yet for it:

1. **The Python dispatch glue is the single biggest cost in the encode** —
   8.4 ms against 150 µs of hardware. That is M7's job, and it is now measured
   rather than assumed.
2. **Fuse layers.** 30 dispatches per encode is the shape of the problem.
3. **The host ops need real implementations** (LayerNorm, softmax, attention),
   either on the array or as non-reference numpy. 64 ms for what should be a
   few ms.
4. **Power and CPU-occupancy are unmeasured** and are half the actual argument
   for an NPU. `docs/04-model` sets "<5 W package, CPU stays free" as a
   requirement alongside throughput.
