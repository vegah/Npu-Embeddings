# 0019 — Offload: the NPU path costs 159× less CPU

- **Date** 2026-08-17
- **Milestone** M8 (the half that actually justifies the project)
- **Status** done

## Goal

[0018](../0018-npu-vs-cpu/TASK.md) measured throughput and closed by noting that
power and CPU-occupancy were unmeasured and are "half the actual argument for an
NPU". This measures the offload half.

The framing matters: **throughput parity with the CPU is already worth it** if
the work leaves the CPU and costs less energy. `docs/04-model` says so directly —
*"<5 W package, CPU stays free"* sits alongside the throughput tiers as a
requirement, not a nice-to-have. Everything above parity is bonus.

## Method, and what could not be measured

**Package power is not readable on this machine.** The Windows `Power Meter` and
`Energy Meter` counter sets exist but expose no instances, so there is no energy
figure to be had without external hardware. Said plainly rather than
substituted for.

**CPU time consumed** is the stand-in, and for the offload question it is not a
proxy at all — it *is* the measurement. `time.process_time()` sums user and
kernel CPU across every thread, so a 12-thread BLAS burning 12 cores for one
second reads as 12 CPU-seconds. As an energy proxy it is reasonable but not
exact: on a fixed part, CPU-seconds at full clock is roughly proportional to CPU
energy.

Four MiniLM GEMM shapes at M=4096, 8 columns, **200 iterations**. The first
attempt used 20 and landed exactly on the Windows `process_time` tick of 15.6 ms
— readings of `0.0 / 15.6 / 31.2` — which would have made the ratio a bound
rather than a measurement. At 200 iterations the NPU figure sits **7× above the
timer floor**.

NPU quiesced: `flm` killed in [0018](../0018-npu-vs-cpu/TASK.md), `LemonadeServer`
idle and confirmed not using the device.

## Result

| shape | path | wall ms | CPU ms | cores busy |
|---|---|---|---|---|
| `qkv` | CPU | 1137.0 | 19078.1 | 16.78 |
| | **NPU** | **295.0** | **46.9** | **0.16** |
| `attn_out` | CPU | 411.5 | 4546.9 | 11.05 |
| | **NPU** | **193.8** | **125.0** | **0.64** |
| `ffn_up` | CPU | 1467.4 | 22625.0 | 15.42 |
| | **NPU** | **438.7** | **140.6** | **0.32** |
| `ffn_down` | CPU | 1196.5 | 20765.6 | 17.36 |
| | **NPU** | **348.4** | **109.4** | **0.31** |

Per encode-layer-equivalent — all four GEMMs once:

| | wall | CPU consumed |
|---|---|---|
| CPU path | 21.06 ms | **335.08 ms** |
| **NPU path** | **6.38 ms** | **2.11 ms** |

**159× less CPU, for 3.3× less wall time.**

The CPU path occupies **11–17 hardware threads**. The NPU path occupies about a
**third of one core** — the machine is, for practical purposes, free while it
runs.

## What this means, and what it does not

**It means the project clears its own bar comfortably.** The criterion is
parity; we have 3.3× on wall clock *and* the CPU handed back. Even if every
remaining optimisation failed, an NPU path that merely matched CPU throughput
while consuming 1/159th of the CPU would be worth shipping.

**It does not mean 159× less energy.** CPU-seconds is not joules. The NPU draws
power of its own, and this measures neither. What can be said is that the CPU
side keeps 11–17 threads at full clock for 21 ms per layer-equivalent, and the
NPU side does not — and on a mobile part that difference is the reason the
silicon exists. A real energy number needs external instrumentation.

**One caveat on the NPU figure.** Weights and activations are written to the
device once, before the timing loop, so the 2.11 ms is dispatch cost and not
data movement. That is representative of the *target* design — one resident
`.xclbin` with weights staged, per F1 — but not of
[0017](../0017-m6-full-encode/TASK.md)'s current encode, which reallocates and
converts per call and pays 8.4 ms of Python for each dispatch
([0018](../0018-npu-vs-cpu/TASK.md)).

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| First run gave CPU readings of exactly `0.0`, `15.6`, `31.2` ms | Windows `process_time()` ticks at 15.6 ms, and 20 iterations put the NPU's CPU cost on the floor | 200 iterations, and the script now prints the floor alongside the result so the reader can see the margin |
| Wanted an energy number and there is none | `Power Meter` / `Energy Meter` counter sets have no instances on this machine | Not solved. Reported as unmeasured rather than approximated |

## Artifacts

- `experiments/m8-npu-vs-cpu/bench_offload.py`
- `artifacts/bench_offload.json`

## Next

1. **Get a real energy number.** External power measurement, or a part that
   exposes the Energy Metering Interface. This is the one claim in
   `docs/04-model` still entirely unevidenced.
2. **Re-measure offload on the fused design** once M7 exists, since the current
   encode's Python glue burns CPU that the target design will not.
3. The throughput work from [0018](../0018-npu-vs-cpu/TASK.md) still stands, but
   its urgency drops: with parity as the bar, we are already past it.
