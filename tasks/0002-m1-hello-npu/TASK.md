# 0002 — M1: Hello NPU (first design built, run, and traced)

- **Date** 2026-08-16
- **Milestone** M1
- **Status** done — **gate passed**

## Goal

Prove the entire native-Windows chain works end to end:

```
IRON design -> Peano kernel -> aiecc -> xclbin -> NPU -> hardware trace -> cycle counts
```

**Gate:** a non-empty `trace.txt` that becomes real cycle counts. Everything after M1
is built on this chain, so it is proven first on the simplest kernel that still
exercises all of it.

## Context

From [0001](../0001-scaffold-and-research-index/TASK.md): the environment is verified,
and `C:\Users\vegar\.npu\cache\` already held 96 designs plus trace JSON from the
user's earlier tracing experiments — strong evidence the flow had worked before, but
not evidence that *we* could drive it.

The user confirmed those earlier traces were exploratory learning, and granted
read-anywhere / write-only-in-this-project.

## What was done

### 1. Validated the measurement pipeline against known-good data first

Before generating anything new, ran `get_trace_summary.py` over the user's existing
traces. This is a free dry run: if our parse path is wrong, we find out against data
whose provenance we trust.

Result: real cycle counts came out (SAXPY 154 and 335 cycles; the matmul sweep 461 to
2790 cycles). Pipeline confirmed.

**Correction recorded:** those matmul traces are **`np.int16`**, not bf16
(`element_type=np.int16`, tiles `_TILE_M=32, _TILE_K=64, _TILE_N=64`). An initial
reading assumed bf16 and started computing bf16 efficiency from them — wrong, and
abandoned. Filename-implied shapes are not a substitute for reading the design.

### 2. Analysed the user's acquire ablation

`matrix_multiplication_single_core.py` vs `...2.py` differ in **two** ways at once:

```diff
-_TILE_K = 64 ;  _TILE_N = 64
+_TILE_K = 32 ;  _TILE_N = 128

-for _ in range_(K // k):
-    elem_in_a = of_a.acquire(1); elem_in_b = of_b.acquire(1)
-    matmul(elem_in_a, elem_in_b, elem_out)
-    of_a.release(1); of_b.release(1)
+for _ in range_(K // k // 2):
+    elem_in_a = of_a.acquire(2); elem_in_b = of_b.acquire(2)
+    matmul(elem_in_a[0], elem_in_b[0], elem_out)
+    matmul(elem_in_a[1], elem_in_b[1], elem_out)
+    of_a.release(2); of_b.release(2)
```

Measured:

| variant | invocations | min | avg | max |
|---|---|---|---|---|
| `trace_without_acquire` (acquire 1) | 126 | 427 | **1851** | 2209 |
| `trace_acquire` (acquire 2) | 128 | 1697 | **2042** | 2225 |

Both do identical MACs per call (`32·64·64 = 32·32·128 = 131072`), so this is
comparable — and average cycles got **~10% worse**.

**The comparison is confounded** (tile shape *and* acquire batching changed together)
so it cannot be attributed. Two observations:

1. Tile shape is the likelier cause. At int16 the mac dims are `(r,s,t) = (4,4,8)`, so
   `(32,64,64)` gives an inner `colA` loop of 16 iterations vs only 8 for
   `(32,32,128)`. A deeper inner loop amortises prologue/epilogue better.
2. **This measurement cannot see what the acquire change was meant to improve.**
   `acquire`/`release` happen *outside* the `event0`/`event1` window, so lock traffic
   is invisible to the cycle summary. It would show up in the Perfetto timeline and
   port events instead.

The `min 427` against `avg 1851` in the baseline smells like truncated invocations at
the trace-buffer boundary rather than a real fast path.

### 3. Built our own M1 design

Copied the stock SAXPY example into `experiments/m1-hello-npu/` rather than running it
in place, so artifacts land in this repo and `C:\dev\mlir-aie` stays untouched.
Changes from upstream, each for a reason:

- `vec_size` passed as `-DSAXPY_VEC_SIZE` instead of hardcoded — upstream hardcodes
  4096 and **silently produces wrong results** if the tensor size differs.
- `trace_to_json()` runs automatically (upstream leaves it commented out, so you get a
  raw `trace.txt` and no cycle counts unless you know to run `parse.py` by hand).
- Prints the cycle summary and vector-time fraction — the actual deliverable.
- Copies `input_with_addresses.mlir` out of the JIT cache next to the trace, so the run
  stays reproducible after a cache clear.
- `--scalar` selects the non-vectorised kernel as an ablation.

## Commands

```powershell
# Every command below assumes this first, in every new shell:
cd C:\dev\mlir-aie; . .\iron_env.ps1

# Environment sanity
python -c "import pyxrt; d=pyxrt.device(0); print(d.get_info(pyxrt.xrt_info_device.name))"
# -> NPU Strix

# Validate the pipeline against pre-existing traces
python C:\dev\mlir-aie\python\utils\trace\get_trace_summary.py `
    --input C:\dev\mlir-aie\programming_examples\getting_started\01_SAXPY\trace64.json

# The M1 run itself
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\experiments\m1-hello-npu
python saxpy.py
python saxpy.py --scalar

# Second measurement signal: static analysis of the emitted object
$P   = "C:\dev\mlir-aie\ironenv\Lib\site-packages\llvm-aie"
$obj = "C:\Users\vegar\.npu\cache\9dd14dbd8b5ce31dea33b601\saxpy.o"
& "$P\bin\llvm-objdump.exe" -d --no-show-raw-insn $obj
& "$P\bin\llvm-nm.exe" --print-size $obj
& "$P\bin\llvm-nm.exe" $obj | Select-String '__div'      # must be empty
```

## Result — GATE PASSED

| variant | cycles | vector fraction | trace.txt | correctness |
|---|---|---|---|---|
| **vector** | **335** | **0.382** | 478 B | `max_abs_err = 0` |
| **scalar** | **541,662** | 0.0076 | 17,758 B | `max_abs_err = 0` |

**1,617× speedup from vectorisation.** Both bit-exact against the NumPy reference
(bf16 `arange` values are exactly representable, and accumulation is fp32).

**Cross-check between our two measurement signals — they agree.**

The vectorised loop `.LBB0_1` → `.L_LEnd0` is a **5-bundle zero-overhead loop**, and
`add.nc lc, r2, #-0x3` with `r2 = 0x40` sets the loop count to **61** (3 iterations
peeled for software pipelining):

```
61 iterations x 5 bundles = 305 cycles
    + ~30 cycles prologue/epilogue
    = ~335   vs measured 335
```

Two fully independent methods, same answer. The doctrine in
[`docs/05-measurement/`](../../docs/05-measurement/README.md) is validated.

**Reproducibility cross-check:** our 335 cycles exactly reproduces the user's
`trace64.json` from 2026-08-15, generated independently a day earlier.

### Why scalar is 1,617× slower — the actual reason

Not merely "scalar is narrower". `llvm-nm` shows an **undefined reference to
`__mulsf3`**, and the scalar loop contains `jl #0x0` — a **function call per element**
into a software floating-point multiply routine.

```
saxpy_scalar loop @0x50:
      6c:  lda.s16  r0, [p7, dj0]
      70:  lda.s16  r3, [p0, dj0]
      76:  jl  #0x0                  <-- call to __mulsf3, PER ELEMENT
```

That is 132 cycles/element versus 0.082 vectorised.

> **Rule for our kernels: never use scalar float arithmetic in an AIE kernel body.**
> It lowers to library calls. This sits alongside "no integer division / `__divsi3`"
> as a thing to check in the disassembly.

### Instruction mix (whole object, both kernels)

```
vldb 8   vlda 8   vst 8   vmul 4   vadd 12   vbcst 1   vmov 17
nopa 10  nopb 12  nops 10  nopv 9  nopx 15
```

The 5-bundle loop body carries only 8 real vector ops across ~25 VLIW slots, padded
with `nopa/nopb/nops/nopxm/nopv`. That is what the 0.382 vector fraction is reporting:
**the loop is issue-limited, not compute-limited.** For a memory-bound elementwise op
that is expected and not worth fixing — but it is precisely the pathology we must
avoid in GEMM, where Rösti's four-independent-accumulator technique exists to fill
those slots.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| Nearly computed bf16 efficiency from the pre-existing matmul traces | Assumed dtype from filenames instead of reading the design; they are `int16` | Read `matrix_multiplication_single_core.py`. **Filenames are not metadata.** |
| Acquire ablation appeared to make things worse | Two variables changed at once, and the metric can't see lock traffic anyway | Reported as confounded and unattributable rather than guessing. Not solved — would need a single-variable rerun plus port events |
| Running the stock example would write into `C:\dev\mlir-aie` | It emits `trace.txt` and copies MLIR into its own directory | Made our own copy under `experiments/m1-hello-npu/` writing to `artifacts/` |

## Artifacts

In `experiments/m1-hello-npu/artifacts/`:

- `trace_vector_4096.{txt,json}` + `input_with_addresses_vector_4096.mlir`
- `trace_scalar_4096.{txt,json}` + `input_with_addresses_scalar_4096.mlir`

JIT cache dirs (outside the repo, regenerable):
`9dd14dbd8b5ce31dea33b601` (vector), `aa0f206cb2620615e9b0046d` (scalar).

## Next

**M2 — bf16 GEMM at MiniLM shapes, traced.**

Carry forward:

1. **Use `whole_array`, not the single-core example** — MiniLM needs the full 8×4 array.
2. **Set `element_type` to bf16 explicitly.** The getting-started matmul defaults to
   int16; our numbers must be bf16 to be comparable to the 14.7 TOPS ceiling.
3. **A/B `--emulate-bf16-mmul-with-bfp16`** (`4×8×8` vs `8×8×8` geometry) as a
   single-variable change.
4. **Change one variable at a time** — the lesson from the acquire ablation above.
5. **Check the disassembly for `__mulsf3` and `__divsi3`** as a standing health check.
6. Target shapes: 384, 1152, 1536 on 64×64×64 tiles.
7. Watch for mlir-aie #2411 (Peano bf16 GEMM shape failures on Strix).
8. Sanity-check the trace buffer: the user's `trace_512x512x512.txt` is **0 bytes** —
   a trace that silently produced nothing, most likely buffer overflow at 512³.
   Size `trace_size` deliberately and **always assert the file is non-empty**.
