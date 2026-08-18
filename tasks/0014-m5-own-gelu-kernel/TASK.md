# 0014 — M5: our own GELU kernel, and the error is `aie::tanh` itself

- **Date** 2026-08-17
- **Milestone** M5
- **Status** done — **own-kernel path works; hypothesis refuted; error source
  isolated**

## Goal

[0013](../0013-m5-first-eltwise-kernel/TASK.md) measured IRON's GELU at
**1.332e-02** against our exact-erf golden, where bf16 output rounding alone
costs 1.689e-03, and concluded we should write our own. This is that attempt.

## The hypothesis, and why it looked solid

The shipped `aie_kernels/aie2p/gelu.cc` is **not a LUT** — it is the same tanh
polynomial we would write. But every intermediate is bf16:

```cpp
aie::vector<bfloat16,16> x2 = aie::mul(x, x);      // bf16
aie::vector<bfloat16,16> x3 = aie::mul(x, x2);     // bf16
... x3*beta, +x, *sqrt(2/pi), +1, *0.5, *x         // all bf16
```

bf16 carries ~8 mantissa bits. Cubing in bf16 and chaining seven more rounded
operations plausibly compounds to ~1e-2. It even stores `sqrt(2/pi)` as a
bfloat16 constant. The fix looked obvious: keep the polynomial in fp32 and round
once, on the store.

## What was done

`experiments/m5-eltwise/kernels/gelu_fp32.cc`, built through
`ExternalFunction(source_file=...)` — the same mechanism IRON uses for its own
kernels.

Three things had to be learned to get it to compile and run at all, and two of
them are traps.

### Trap 1 — `vector::to_vector<T>()` does not exist

Element-type conversion is an **accumulator** method. Every shipped kernel calls
`.to_vector<float>()` on the result of `aie::mul`, never on a vector. Widening
bf16 → fp32 therefore goes through a multiply by 1.0, which is exact:

```cpp
aie::vector<float,16> x = aie::mul(xb, vone_bf).to_vector<float>();
```

### Trap 2 — `aie::tanh<float>` compiles to an empty function, silently

`aie::tanh` is declared `template <typename TR = bfloat16, unsigned Elems>`, and
all four shipped kernels that use it (gelu, silu, swiglu, softmax) instantiate
`<bfloat16>`. Asking for `<float>`:

- compiles with **no error or warning**
- produces a function whose body is **gone**
- and the core **hangs**: `ERT_CMD_STATE_TIMEOUT`

The object told the story immediately. `llvm-objdump -t`:

```
ours: 00000000 g F .text.gelu_fp32_bf16   00000000 gelu_fp32_bf16      <- 0 bytes
IRON: 00000000 g F .text.gelu_bf16        00000120 gelu_bf16           <- 288 bytes
```

and `-d` showed the whole body reduced to a return:

```
0: ... ret lr; ...
18: event #0
1c: event #1
```

### The bisect that made this cheap

Rather than guess whether our C++ or our *build setup* was at fault, IRON's own
`gelu.cc` was copied into our kernel directory and compiled through **our**
`ExternalFunction` path, renamed to avoid a symbol clash.

It produced a 1552-byte object against IRON's own 1540, ran on hardware, and
returned **exactly the same numbers as IRON's kernel**. Our build setup was
therefore correct, and the fault was in our code. That took one run.

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

python experiments\m5-eltwise\gelu_kernel.py --kernel iron      # baseline
python experiments\m5-eltwise\gelu_kernel.py --kernel control   # IRON's source, our build
python experiments\m5-eltwise\gelu_kernel.py --kernel ours      # fp32 polynomial

# the diagnostic that found the empty function
$od="C:\dev\mlir-aie\ironenv\Lib\site-packages\llvm-aie\bin\llvm-objdump.exe"
& $od -t <cache>\gelu_fp32_bf16.o
& $od -d <cache>\gelu_fp32_bf16.o
```

## Result

| kernel | object | rel_fro vs exact-erf golden | vs fp32 tanh reference |
|---|---|---|---|
| IRON `gelu_bf16` | 1540 B | 1.332e-02 | 1.318e-02 |
| control (IRON source, our build) | 1552 B | 1.332e-02 | 1.318e-02 |
| **ours, fp32 polynomial** | 2148 B | **1.316e-02** | **1.301e-02** |
| — bf16 output rounding alone | — | 1.689e-03 | — |
| — tanh-vs-erf formula alone | — | 1.975e-03 | — |

**The hypothesis is refuted.** Moving the entire polynomial to fp32 improved
1.332e-02 → 1.316e-02, about **1%**. The bf16 intermediates were never the
problem, despite being a completely plausible cause.

### Where the error actually is

The isolation now leaves only one candidate. In our kernel:

- the polynomial (`x`, `x²`, `x³`, `inner`, `arg`) is fp32 and effectively exact
- the final store to bf16 costs **1.689e-03**
- the tanh output is rounded to bf16 once, worth roughly another 2e-03
- the formula itself accounts for **1.975e-03**

and yet the measured gap to a true fp32 tanh-GELU is **1.301e-02**, some 6.6×
larger than everything above combined. **`aie::tanh` is itself only accurate to
about 1%**, and it dominates regardless of what precision is fed to it.

That is a hardware-library property, not something a caller can fix. It also
retroactively explains [0013](../0013-m5-first-eltwise-kernel/TASK.md): the
answer there was right (do not use the built-in) but the reason given was wrong
(the LUT / the formula).

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| Kernel compiles, then `ERT_CMD_STATE_TIMEOUT` | `aie::tanh<float>` is not a supported instantiation; it silently emits an empty function | Use `aie::tanh<bfloat16>`. **Check the object size** — a 0-byte symbol is the tell, and `llvm-objdump -t` finds it in seconds |
| `no member named 'to_vector' in aie::vector<float,16>` | `to_vector<T>()` is an accumulator method, not a vector method | Widen via `aie::mul(v, 1.0).to_vector<float>()` |
| A plausible hypothesis consumed the whole attempt | bf16 intermediates *looked* like the obvious cause, and the shipped source made the case for us | The control run is what settled it. Worth doing earlier next time: measure the suspected component in isolation before rewriting around it |

## Artifacts

- `experiments/m5-eltwise/kernels/gelu_fp32.cc` — kept: it is correct, marginally
  better, and it is the scaffold the polynomial version will slot into
- `experiments/m5-eltwise/kernels/gelu_control.cc` — IRON's source under our
  build, kept as a permanent A/B control for future kernels
- `artifacts/gelu_kernel_{iron,control,ours}.json`

## Next

**GELU has to avoid `aie::tanh` entirely.** The route is a minimax polynomial
evaluated in fp32 with nothing but `mul` and `add` — no transcendental call at
all. GELU is smooth, odd-symmetric about its inflection, and saturates linearly;
a modest polynomial with range reduction should reach ~1e-4, far below the bf16
floor. Design and validate it on CPU first, where iteration is free, and only
then port.

Two things to carry into that work:

1. **`aie::exp2` and `aie::invsqrt` are the same family** and should be assumed
   equally coarse until measured. `invsqrt` is what LayerNorm needs, so this
   question will come back immediately.
2. **Check every new kernel's object size before trusting a result.** A silently
   empty function is now a known failure mode on this toolchain.
