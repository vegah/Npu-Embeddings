# 0032 — M7: the one-xclbin production encode. 305 → 611 seq/s, and the best accuracy the project has produced

- **Date** 2026-08-18
- **Milestone** M7
- **Status** done — **604.5 / 618.4 seq/s at batch 128** (two runs), `1-cos`
  **1.086e-05**, 24 dispatches, **zero design switches**, one hw_context.
  Wall against the CPU's 710: **0.86×**, from 0.42× this morning.

## Goal

0031 measured the switch bill at ~115 ms of a 420 ms encode. Remove it: every
NPU operation becomes an instruction stream over ONE static design
(mechanisms proven in 0029/0030), and every operation that cannot join that
design earns its place on the NPU or leaves.

## What happened, in order — because the plan changed twice on measurement

### 1. The spatial-split unified design died on program memory

The planned form — GEMM on columns 0–3, an opcode-switched GELU+LN+softmax
worker on columns 4–7 (`experiments/m7-unified/unified_design.py`,
`kernels/eltwise_universal.cc`) — got through three real walls:

- **Idle-half endpoints**: a stream that never fills the other half's fifos
  fails `Program.resolve()` ("prod was not created"). Fixed by creating the
  handles, pinning their shim tiles explicitly, and registering them in
  `rt._fifos` so the tiles materialize — endpoints without DMA tasks.
- **Include-order collision**: the shipped `aie_kernels` tree has its own
  `softmax.cc`, and quote-include falls back to `-I` order — the universal TU
  silently included AMD's softmax instead of ours. Our kernels dir must come
  FIRST.
- **L1**: the 0x4000 stack the il4 softmax needs plus double-buffered 12 KB
  objects both ways is 68.6 KB. Output side went single-buffered.

…and then hit a wall with no door at this size: **`Overflow of program
memory`**. A core's PM is 16 KB; the il4 softmax core ELF alone measures
12,192 B and LayerNorm-il4 8,096 B (llvm-size on the production ELFs). Three
kernels in one worker do not fit. The experiment files are kept, and the
endpoint-pinning technique is what any future multi-stream design will use.

### 2. The pivot: eltwise ops must EARN the dispatch — and none of them do

0031's numbers said a LayerNorm call is 725 µs of kernel inside ~3 ms of
switch + conversion. A threaded fp32 AVX2 LayerNorm on the host prices at
~0.3 ms/call. So instead of forcing LN into a worker, it left the NPU —
then softmax and GELU followed, because the same arithmetic held for both
(`--host-ln`, `--host-sm`, `--host-gelu` in `main.cpp`; softmax uses the
project's own exp2 polynomial ported to AVX2, GELU the project's own
degree-8 even-correction polynomial — the SAME coefficients the NPU kernels
run, evaluated in fp32):

| step | seq/s | `1-cos` | host cost of the moved op |
|---|---|---|---|
| 0031 baseline (all-NPU eltwise) | 305 | 2.469e-04 | — |
| + host LN | 345.5 | **1.896e-05** | 3.6 ms /encode |
| + host softmax | 396.1 | 2.198e-05 | 3.3 ms |
| + host GELU | 449.6 | **1.086e-05** | 8.4 ms |

Every move was **faster AND more accurate**: fp32 on the host removes both
the bf16 round trip and the switch. 1.086e-05 is exactly the value M6
predicted for a host-GELU pipeline (1.09e-05, tasks/0017). The host eltwise
total is ~15 ms — the NPU path for the same three ops was ~120 ms.

This is TileFuse's small-op warning and 0026's closing verdict, finally acted
on: at h = 384, elementwise ops are 1:3h of the FLOPs and cannot amortise a
2.3 ms design switch. **The NPU earns its keep on MACs; the memory-bound glue
belongs to the host** — until a fused/pipelined design brings it back on-array
with zero marginal dispatches.

### 3. With the NPU pure GEMM, the proven RTP mechanism finishes it

24 GEMM dispatches, four shapes — exactly what 0030 §1 proved shares one
static image. `tools/export_gemm_rtp.py` builds all four at batch 128 /
8 columns with `rtp=True`, verifies the four xclbins are byte-identical
modulo UUIDs (67–70 differing bytes, matching 0029's 67-byte footprint), and
exports ONE xclbin + four instruction streams. The runtime (`gemm_rtp` kind)
loads it as one Design / one hw_context, binds a stream + weight slot per
dispatch, and syncs partially (the shared buffers are max-sized; ffn_up's C
is 50 MB and qkv only touches 6.3 MB of A).

```
  dispatch + wait   75.4 ms   24 dispatches   3,142 us each
```

3,142 µs ≈ the measured ALONE cost of these GEMMs (0031 probes: 3,104 /
1,326 / 3,953 / ~4,000). **The switch bill is zero**, measured, not argued.

## Result

```
  wall  207-212 ms  ->  604.5 / 618.4 seq/s     (batch 128, --threads 16)
  1-cos vs HuggingFace   1.086e-05              (identical to the legacy
                                                 path with host eltwise --
                                                 the GEMM streams are exact)
  CPU (0018 baseline)    710 seq/s on 12 threads -> wall ratio 0.86x
  cores busy             ~3.5 (noisy, 3.4-6.2 across runs)
```

Day total: **296 → 611 seq/s (2.06×)** and `1-cos` **2.469e-04 → 1.086e-05
(23×)**. M7's overall run: 42.4 → 611, **14.4×**.

Where the remaining ~207 ms lives: GEMM wait 75, read-out+bias 38,
"everything else" 47 (residual copies, mean-pool, sequencing), attention
23–25, conversions 17, host eltwise 15, syncs 8.

## Failures and traps hit (the valuable part)

1. **PM overflow** ends the three-op universal worker; measured ELF sizes
   above. Any future opcode worker gets TWO ops max at these kernel sizes.
2. **Idle-half fifo endpoints** (see §1) — the technique that makes
   multi-stream heterogeneous designs resolvable at all.
3. **Quote-include falls back to -I order** — AMD's `softmax.cc` shadowed
   ours. Symptom: `use of undeclared identifier softmax_il4_impl`.
4. **The Bash-tool heredoc eats one backslash level** (session tooling, not
   the repo): patches containing `\n` in C string literals must be applied
   from script files, never inline heredocs. Cost three failed edits.
5. **Aggregate init breaks silently when fields are inserted mid-struct** —
   `Encoder enc{...}` assigned a vector to a bool. Caught at compile.
6. **The bench's per-design loops double-counted** the unified Design seven
   times (241% of wall) — references aliasing one object. Dedupe by pointer.

## Exact commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
# the one-xclbin GEMM export (into an existing artifact set)
python tools\export_gemm_rtp.py --batch 128 --cols 8 --out runtime\artifacts_b128il
cd runtime; cmake --build build --config Release
# unified mode is auto-detected from artifacts/<set>/gemm_rtp/design.json
.\build\npuembed.exe .. --artifacts artifacts_b128il --threads 16            # validate
.\build\npuembed.exe .. --artifacts artifacts_b128il --threads 16 --bench 5  # measure
# legacy A/B (7 xclbins) still works: --artifacts artifacts_b128e8, and the
# host flags compose with it: --host-ln --host-sm --host-gelu
```

## Next

1. **Micro-batch pipelining** (expert §6a): with one context and 24
   dispatches the encode is 75 ms NPU + ~130 ms host in strict alternation;
   splitting the batch in two and overlapping host work of one half with NPU
   work of the other can hide most of the smaller side. No switches to pay
   anymore — the blocker 0030 deferred it on is gone.
2. **B/A reuse in L2/L1** (`acquire`-and-hold, `repeat_count`): GEMM wait is
   now the largest NPU item at 75 ms and its B operand is re-streamed 32× per
   call at batch 128.
3. **Read-out+bias (38 ms) and "everything else" (47 ms)**: the residual
   memcpy, single-threaded mean-pool, and the bias pass are now visible.
4. **The pipelined fused design** remains the endgame for bringing eltwise
   back on-array with zero marginal dispatches (STEEL pattern) — now with a
   measured bar to clear: it must beat 15 ms of host time for the whole
   eltwise family, at zero switch cost.
5. **Energy** (AGT methodology from STEEL) — "offload and energy are the
   point," and the host now does ~3.5 cores of work at peak throughput; the
   claim needs numbers.
