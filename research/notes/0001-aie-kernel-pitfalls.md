# 0001 — AIE kernel pitfalls found by disassembly

*First observed during [M1](../../tasks/0002-m1-hello-npu/TASK.md), 2026-08-16.*

A running list of things that silently destroy AIE kernel performance and are
**invisible in the source** but obvious in the emitted object. Check these before
blaming the design.

```powershell
$P   = $env:PEANO_INSTALL_DIR
$obj = "<jit-cache-dir>\<kernel>.o"

& "$P\bin\llvm-nm.exe" $obj | Select-String '__div|__mul|__udiv|__float'   # must be empty
& "$P\bin\llvm-nm.exe" --print-size $obj                                    # spills / size
& "$P\bin\llvm-objdump.exe" -d --no-show-raw-insn $obj                      # loop body
```

## 1. Scalar float arithmetic becomes a function call

**Measured cost: 1,617×.** The scalar SAXPY variant ran 541,662 cycles against 335 for
the vectorised one — 132 cycles per element.

The cause was not narrowness. `llvm-nm` showed `U __mulsf3`, and the loop contained a
call instruction per element:

```
      6c:  lda.s16  r0, [p7, dj0]
      70:  lda.s16  r3, [p0, dj0]
      76:  jl  #0x0                  <-- call to __mulsf3, once per element
```

**Rule:** never use scalar `float`/`bfloat16` arithmetic in a kernel body. Use
`aie::vector` / `aie::accum` and the `aie_api` operations. A stray scalar `a * x` in a
tail/epilogue is enough to dominate the kernel.

**Symptom to watch for:** an undefined symbol beginning `__` in `llvm-nm` output.
`__mulsf3`, `__divsf3`, `__floatsisf` and friends are all software-float routines.

## 2. Integer division emits `__divsi3`

Called out by AMD's own `skills/aie-kernel-opt/SKILL.md` as a priority lever. AIE has
no integer divide. Any `/` or `%` by a non-power-of-two runtime value becomes a library
call in the inner loop.

**Rule:** make divisors compile-time constants (ideally powers of two) so the compiler
turns them into shifts. Our M1 kernel is clean — verified no `__div` symbols.

## 3. `nop` padding means the loop is issue-limited, not compute-limited

The M1 vectorised loop body is 5 VLIW bundles carrying only 8 real vector ops across
~25 slots:

```
.LBB0_1:
  90:  nopa ; vldb x4       ; nops ; nopxm ; nopv
  a0:  nopa ; vldb x5       ; nops ; nopxm ; vadd.f
  b0:  nopa ; nopb          ; vst  ; nopxm ; nopv
  c0:  vlda ; nopb          ; vst  ; nopxm ; vmul.f
  d0:  vlda ; nopb          ; nops ; nopxm ; nopv
```

This is what `get_vector_time()` reported as **0.382** — 62% of the window is not
issuing vector work.

For a memory-bound elementwise kernel that is expected and not worth fixing. For GEMM
it is the pathology to avoid: it is exactly why Rösti
([2504.03083](https://arxiv.org/abs/2504.03083)) computes **four independent output tiles in
four accumulator registers**, so four back-to-back VMACs fill the slots instead of
stalling on a RAW hazard against a single accumulator (4-cycle result latency).

**Rule:** in a GEMM microkernel, count the `nop`s in the ZOL body. Their presence is
the signal to add accumulator independence.

## 4. Cross-checking cycles without hardware

The trace and the disassembly must agree. For M1:

```
loop count from  `add.nc lc, r2, #-0x3`  with r2 = 0x40  ->  61
5 bundles x 61 iterations                                ->  305 cycles
+ prologue/epilogue                                      ->  ~335
measured                                                     335
```

Because AIE cores never stall (no cache, no OoO, no branch prediction, fixed
latencies), bundle count × iterations *is* the execution time for a compute-bound
kernel. When the two disagree, the kernel is waiting on data — which is itself the
finding.

Note the `-0x3`: three iterations are peeled into the prologue/epilogue for software
pipelining, so the ZOL trip count is `N/lanes - 3`, not `N/lanes`. Don't be surprised
by the off-by-three.

## 5. `trace.txt` can be silently empty

`trace_512x512x512.txt` in the user's earlier experiments is **0 bytes** — the run
completed, produced no trace, and said nothing. Almost certainly trace-buffer overflow
(`trace_size=524288` was not enough for 512³).

**Rule:** always assert `trace.txt` is non-empty before trusting any conclusion. Our
`experiments/m1-hello-npu/saxpy.py` checks this and warns explicitly. A missing
measurement must never be mistaken for a good one.
