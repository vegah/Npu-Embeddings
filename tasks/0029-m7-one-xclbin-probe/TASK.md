# 0029 — M7: one xclbin, many instruction streams — step 0 confirms it

- **Date** 2026-08-17
- **Milestone** M7
- **Status** done — **the design switch is eliminable, measured.** Two
  instruction streams through one hw_context alternate at alone-cost.

## Why this experiment

External review of the whole codebase pointed at the largest unexploited lever,
and it was already sitting in our own research index: Rösti's *"one static
configuration; only shim DMAs and runtime scalars change"* architecture
(INDEX.md F1, 3.5× measured on the same
class of machine).

The reasoning against our own results:

1. [note 0004](../../research/notes/0004-context-switch-cost.md): the switch
   costs ~25 µs + 7.2 µs per lock, is paid **unconditionally** (A↔A′ with the
   identical xclbin costs the same as A↔B), and follows **hw_context changes**.
   Within one context, repeat-dispatch costs only the dispatch.
2. In mlir-aie, `final.xclbin` is the *static* configuration (core ELFs,
   mem-tile/core BDs, locks, routing) while `insts.bin` is the *runtime
   sequence* the command processor replays per dispatch — and the shim BDs are
   programmed **there**, not in the xclbin. Our runtime already passes
   `bo_instr` as an ordinary per-call kernel argument.
3. Therefore: operations that share a static design are just **different
   instruction streams over one context** — and the switch should vanish,
   not amortise.

Crucially, `--probe-ctx` in [0024](../0024-m7-dispatch-cost-anatomy/TASK.md)
tested the same xclbin in **two contexts**. It never tested two instruction
streams in **one** context. That is the gap this task closes.

## Step 0a: is the static configuration really sequence-independent?

`build_passthrough.py --order {fwd,rev}` builds the same design twice with the
runtime sequence as the *only* difference: identical workers, fifos and taps,
with the fill/drain tasks issued in reversed order. Both at batch 64, 2 columns.

- `insts.bin`: **different** (2288 bytes each, different sha256) — so the two
  builds really are distinct sequences, and the cache matcher did not hand back
  the same directory twice.
- `final.xclbin`: **67 of 25,977 bytes differ — every one of them identity
  metadata.** The xclbin UUID at 0x1a0 (16 bytes), the same UUID as a hex
  string inside the JSON build metadata (`"XclBinUUID":"…"`, 32 bytes), a
  second partition UUID at 0x50c0, and four single-byte fragments of the same
  strings. **Zero configuration bytes differ.**

So the static configuration is sequence-independent, byte for byte. (Plain
`sha256` comparison of xclbins can never pass — the UUID is regenerated every
build — which is worth knowing before anyone "refutes" this with a hash.)

## Step 0b: the functional and timing probe

`npu::Design` gained `load_instr(path)` / `bind_instr(slot)` — additional
instruction-stream BOs in the *same* context, selected per dispatch, exactly
parallel to `stage()`/`bind()` for weights. `--probe-insts <dirA> <dirB>` loads
dirA's xclbin once and both instruction streams.

**Correctness first** — completion status is not data. With a ramp staged in:

```
      stream 0 output: exact (0 of 6291456 wrong)
      stream 1 output: exact (0 of 6291456 wrong)
```

A foreign build's instruction stream, dispatched through another build's
xclbin, reproduces all 6.3 M elements exactly.

**Timing** (100 repeats; paired rows are two dispatches, shown halved):

| | µs per dispatch |
|---|---|
| A alone (stream 0) | 503 |
| B alone (stream 1, same context) | 490 |
| **A ↔ B, ONE context, two streams** | **500** |
| A ↔ A′, TWO contexts (control) | 976 |

**Alternating two instruction streams in one context costs alone-price.** The
switch is not amortised — it is *gone*.

And the control row closes the loop on 0024's model with satisfying precision:
the two-context penalty is 976 − 497 ≈ **479 µs**, against the model's
prediction for this design of 25 + 7.2 × 64 locks = **486 µs**. Two independent
measurements, 1.5% apart.

## What this changes

Today an extra NPU operation costs ~0.6–2.4 ms, because it is a design switch.
In the one-xclbin architecture it costs ~150–500 µs, because it is a dispatch.
Consequences, in order of importance:

1. **The 49 switches per encode (~60 ms at batch 128) are removable without
   any fusion**, by unifying designs into shared static configurations.
2. **The "data movement and switching are one budget" coupling from
   [note 0004](../../research/notes/0004-context-switch-cost.md) dissolves.**
   Descriptors added to feed the array better are no longer re-paid per switch,
   so dataflow optimisation is unshackled.
3. **The width story changes again.** `--cols 2` won *because* switches
   dominated; with switches gone, the 8-column designs — 1.4–1.9× faster in
   isolation — should win outright. Re-measure.
4. **Host-side glue re-prices.** Residual adds, bias, attention lime were "not
   worth a switch"; at dispatch prices several become worth moving.

## What stands between step 0 and the full encode

The four GEMM designs share tile geometry (m,k,n = 64,64,48) but their core
ELFs differ: `core_fn` compiles `n_tiles_per_core` and `K//k` into the binary.
Rösti's answer — and mlir-aie has the mechanism (**RTPs**, runtime parameters
writable from the runtime sequence) — is to hoist those two loop counts into
runtime scalars. Then one ELF serves all four shapes, the static design
unifies, and each GEMM becomes an instruction stream + two RTP writes.

Same story for eltwise: one kernel with an opcode RTP would make GELU, LayerNorm
and softmax three streams over one static design.

Not yet done, and two honest risks going in: RTP maturity in IRON at our
version, and the loop bounds becoming runtime values (Peano may pipeline
`range_(rtp)` worse than a compile-time constant — measure the kernel before
and after).

## Artifacts

- `experiments/m7-switch-cost/build_passthrough.py` — `--order {fwd,rev}`
- `runtime/include/npu_device.hpp`, `runtime/src/npu_device.cpp` —
  `load_instr()` / `bind_instr()`
- `runtime/src/main.cpp` — `--probe-insts`, with the exactness gate
- `runtime/artifacts_seq{A,B}/passthrough/` — the two builds (gitignored;
  rebuild with the commands above)

## Next

1. **RTP-ify the GEMM kernel** (loop counts as runtime scalars), unify the four
   GEMM designs into one xclbin + four instruction streams, and re-run this
   probe on the real GEMM pair.
2. Fold eltwise into the same static design on its own columns.
3. Re-measure the width crossover with switches gone.
