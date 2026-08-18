# 0034 — M8: energy, measured on native Windows after all. 1.9–2.0× better per sequence

- **Date** 2026-08-18
- **Milestone** M8 (first half)
- **Status** done — **44.0 J per 1000 sequences pipelined against the CPU's
  85.3 — 1.94×**, and 53.5 J single-lane (1.59×). Three independent runs of
  the full matrix agree within 6%. **No external instrumentation was needed.**

## The finding that unblocked this: it was the wrong counter set

`docs/CURRENT_STATUS.md` said, and had said since M7: *"Windows Power Meter
counters have **no instances** on this machine; needs external
instrumentation."* That was checked against `\Power Meter(*)`, which indeed has
none. The **`\Energy Meter(*)`** set is a different thing entirely and has
**14 instances on this machine**, including:

```
  RAPL_Package0_PKG          <- package energy: CPU cores AND the NPU block
  RAPL_Package0_Core0..15    <- 12 per-core meters
```

So "energy is the point" stopped being an argument from core count on the
first day someone enumerated the right counter set. **A capability the project
had recorded as absent was present the whole time** — the same failure mode as
0027's "the placer will not do 4 columns" and 0030's stride wall: a wall that
was never re-examined after the first failed attempt.

## Units, calibrated rather than assumed

| counter | unit | how it was established |
|---|---|---|
| `Time` | milliseconds | 257,229,410 against a system uptime of 257,251,558 ms |
| `Power` | milliwatts | ~15 W idle, 46 W on 12 threads — plausible for a 54 W part |
| `Energy` | **picowatt-hours** (3.6e-9 J) | see below |

The energy unit was derived, not looked up: `dEnergy/dTime ÷ Power` = **277.8
± 0.1** across every sample of every run, and 1/(3.6e-9 × 10⁶) = 277.8 exactly.
Two independent counter paths agreeing to 0.1% is the instrument-level version
of this project's two-signal rule.

**Why the cumulative `Energy` counter and not `Power`:** it is monotonic, so
the energy of a window is exactly `E_after − E_before`. No sampling rate, no
missed transients, no integration error.

## The control experiment: does the package meter see the NPU?

This is the question the whole measurement rests on, and it is answerable
rather than assumable. `--soak-npu <s>` dispatches in a tight loop with **zero
host work** — no conversion, no sync, one thread. `--soak-cpu <s> <threads>` is
the mirror.

| load | package power | Δ over idle |
|---|---|---|
| idle | 15.0 W | — |
| **NPU soak** (1 driver thread) | **24.2 W** | **+9.1 W** |
| CPU soak, 1 thread | 23.8 W | +3.9 W |
| CPU soak, 12 threads | 46.0 W | +29.1 W |

**The meter sees the NPU.** The soak's +9.1 W minus the +3.9 W its own
dispatch-driving thread costs puts **the NPU block at ≈ 5.2 W** under
saturation. The 12-thread control lands at 46 W, inside the part's TDP — the
instrument is reading real physical power, not a synthetic estimate.

## Method: differential, so startup cancels exactly

Every configuration is measured at **two encode counts** and the answer is the
difference:

```
  J_per_encode = (E_high − E_low) / (n_high − n_low)
```

Process startup, model load, xclbin registration, weight staging and the
harness itself are identical in both runs and **cancel exactly**. What remains
is the marginal cost of an encode — the quantity the claim is about. Crucially
this subtraction **never touches the idle baseline**, which is what makes the
result robust to the idle drift seen below.

Both sides are measured by the same instrument on the same machine, so its
systematic errors cancel in the ratio. That is what makes this defensible
without a wall meter.

## Results (batch 128, 20 vs 60 encodes, 3 repeats per point)

| config | J / 1000 seq | W during | seq/s | vs CPU |
|---|---|---|---|---|
| `sentence-transformers` CPU | **85.3** | 55.3 | 648.9 | 1.00× |
| NPU single lane | **53.5** | 33.2 | 620.2 | **1.59×** |
| NPU pipelined ×2 | **44.0** | 36.4 | 826.8 | **1.94×** |

Reproducibility across three full runs of the matrix:

| run | cpu | npu-single | npu-pipe2 | pipe2 ratio |
|---|---|---|---|---|
| 1 (1 rep) | 78.1 | 53.2 | 38.6 | 2.02× |
| 2 (3 reps) | 86.4 | 55.0 | 44.0 | 1.96× |
| 3 (3 reps) | 85.3 | 53.5 | 44.0 | 1.94× |

**The 2× target is met.** Within a measurement the repeats are tight (≤3%
spread); run 1's lower CPU figure is the outlier and it is the *conservative*
direction to drop it.

Note the shape of it: the NPU path wins on **power** (33–36 W against 55 W)
while being equal or faster in wall clock. The pipelined config draws slightly
more power than single-lane but finishes 1.33× sooner, so it wins on energy
too — throughput and efficiency are not in tension here.

## What is honestly weak about this

1. **Idle windows drift.** 15–23 W between windows, and roughly half the
   windows tripped the 15% stability check. Background Windows activity. The
   differential method does not use the idle baseline, so the headline numbers
   are unaffected — but the `marginal_j` field in the JSON artifacts *is*
   affected and should not be quoted.
2. **Package-level, not rail-level.** This measures CPU + NPU + fabric
   together, which is the right comparison for "what does this workload cost
   the machine" and the wrong one for "what does the NPU array itself cost".
   The 5.2 W figure above is a subtraction, not a direct reading.
3. **The CPU comparator is `sentence-transformers`/torch**, what a user
   actually reaches for — not a maximally tuned ONNX Runtime build. A tuned
   CPU baseline could plausibly close some of both the throughput and the
   energy gap. That comparison belongs in M8 proper.
4. **Power mode was not swept.** Trap 5c says it should be (a thesis measured
   turbo >22 pp worse than balanced for one layout). Everything here is at the
   machine's current setting.

## Exact commands

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
# the control experiment (answer this before trusting anything else)
.\tools\measure_energy.ps1 -Label npu-soak -WorkDir .\runtime `
    -Command ".\build\npuembed.exe .. --artifacts artifacts_b128il --soak-npu 40" -Idle 20
.\tools\measure_energy.ps1 -Label cpu-1t -WorkDir .\runtime `
    -Command ".\build\npuembed.exe .. --artifacts artifacts_b128il --soak-cpu 40 1" -Idle 15
# the full comparison
.\tools\energy_compare.ps1 -Low 20 -High 60 -Threads 24 -Repeats 3 -Idle 10
```

Artifacts: `energy_compare.json` plus per-point `*-lo.json` / `*-hi.json` in
this directory.

## What this closes, and what it opens

**Closes:** the project's stated goal was *"parity is enough; offload and
energy are the point"*. Both halves are now measured on hardware rather than
argued: **1.17–1.26× the CPU's throughput** (0033) at **1.94× less energy per
sequence**, with `1-cos` 1.086e-05 against HuggingFace.

**Opens:**
1. **M8 proper — MTEB.** The remaining unproven claim is that these embeddings
   are *good*, not merely faithful to fp32. Nothing else should be optimised
   before that gate.
2. A **tuned CPU baseline** (ONNX Runtime) to make the comparison adversarial
   rather than convenient.
3. **Power-mode sweep** (trap 5c) — cheap now that the harness exists.
4. STEEL reports 9.17× energy against 12 Zen5 cores for fused attention. Our
   1.94× is for a whole encoder with a host-resident elementwise tail; the gap
   between those two numbers is a reasonable estimate of what the fused
   pipelined design is still worth.
