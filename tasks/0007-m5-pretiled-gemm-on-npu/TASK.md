# 0007 — M5 (first): pre-tiled GEMM on the NPU, and two M4 claims tested

- **Date** 2026-08-17
- **Milestone** M5
- **Status** done — **`tile_n=48` validated on hardware; M4's BD claim and its
  performance claim both refuted**

## Goal

Take M4's `.npue` layout to the hardware and answer the two things M4 left as
arithmetic rather than measurement:

1. Does `tile_n = 48` actually compile, run and trace at 8 columns?
2. Does pre-tiling do what M4 said — make `ffn_down` expressible, and move the
   data-movement numbers?

## Context

[0006](../0006-m4-npue-pretiling/TASK.md) built the container and verified it
offline: round-trip bit-exact, fusions free. But it made two claims it could not
test, and this task tests both. One survives. One does not, and neither does the
premise underneath it.

## What was done

### 1. Reproduce the failure first

Before writing the design, re-ran M2's **single-core** `ffn_down` to get the real
error text rather than a remembered one:

```
error: 'aie.dma_bd' op Size 1 exceeds the [0:1023] range.
  aie.dma_bd(%arg1 : memref<1536x384xbf16>, 0, 589824,
    [<size=1, stride=0>, <size=12, stride=32>,
     <size=1536, stride=384>, <size=32, stride=1>])
```

`<size=1536, stride=384>` is K itself. That is a genuine failure and it is
exactly what [0003](../0003-m2-bf16-gemm/TASK.md) recorded.

### 2. The design

`experiments/m5-pretiled-gemm/gemm_pretiled.py`, derived from our M2 whole-array
design. Three changes, each marked `# NPUE-M5:`:

1. B's L3→L2 transfer uses an explicit `TensorAccessPattern` over **tile
   indices** instead of `TensorTiler2D.step_tiler` over the tensor.
2. B's L2→L1 `forward` drops its `dims_to_stream`, because M4 bakes the (s,t)
   sub-tile order into the file.
3. The host builds B with `tools/npue.tile_b` — the *same* function that packs
   the shipped `.npue`, so this exercises the real layout, not a lookalike.

Both variants are checked against a fp64 reference on every run, and the tiling
permutation is asserted invertible on exactly the bytes handed to the device
before any timing is believed.

### 3. **The BD premise does not hold for the whole-array design**

The very first `--baseline` run was supposed to show row-major `ffn_down`
failing. It compiled and ran:

```
  -- M2 row-major B path --
  cols= 4 rowmajor cores=16  avg=1396.9 cyc  per-core=140.7 MACs/cyc  PASS
```

Checked directly against the unmodified M2 design, in case our copy had drifted:

```powershell
python experiments\m2-bf16-gemm\gemm_whole_array.py --cols 4 -M 512 -K 1536 -N 384 -n 32 --emulate-bfp16
  cols= 4  cores=16  avg= 975.9 cyc  per-core= 134.3 MACs/cyc  relfro=1.04e-02 PASS
```

**M2's whole-array design runs `ffn_down` fine.** The BD failure is a property of
the **single-core** design only. There, B is walked as one full column strip, so
the k-blocks collapse into a single `size=1536`. The whole-array design's
`step_tiler` keeps k-blocks as their own dimension — `<size=24, stride=24576>` —
and K never appears as a size at all.

So the claim inherited from 0003 into 0006 and CLAUDE.md — *pre-tiling is the
only way to express `ffn_down`* — is **true of the single-core design and false
of the design we actually intend to ship**. Corrected below.

### 4. Pre-tiling is not faster. It is slower, and unstable

Two runs of the identical pre-tiled config differed by 4.7%, so nothing could be
concluded from single runs. Added `--repeat` and measured spread.

`ffn_down`, M=512, 4 columns, n=48, per-core MACs/cycle over 5 runs:

| variant | mean | range | spread |
|---|---|---|---|
| rowmajor (M2 path) | **140.9** | 140.7–141.0 | **0.2%** |
| pretiled `k,n` | 118.9 | 102.7–126.0 | 19.6% |
| pretiled `n,k` | 118.5 | 105.7–137.0 | 26.4% |

### 5. Locality is not the cause

The obvious hypothesis was tile order: in `k,n` order the tiles a core walks
consecutively sit `(N/n)·k·n` apart — 48 KB — so the DMA scatters over ~1.1 MB,
while `n,k` makes that walk contiguous. Implemented both.

**No difference** (118.9 vs 118.5). The hypothesis was wrong.

### 6. Isolating which of the two changes costs

Added `--inner rowmaj`, which keeps the pre-tiled L3→L2 access pattern but
leaves the tile interior row-major and restores `dims_to_stream` on the forward.
Numerically identical; only *where* the sub-tile reorder happens differs.

| variant | mean | spread |
|---|---|---|
| rowmajor | **141.0** | 0.0% |
| pretiled, sub-tile order in the file | 125.7 | 3.8% |
| pretiled, sub-tile order still in `dims_to_stream` | 126.4 | 13.6% |

Both pre-tiled variants land at ~126. **Moving the sub-tile reorder offline is
free; the cost is entirely in the L3→L2 access pattern.**

### 7. The two descriptors, side by side

From the stored physical MLIR, the B shim BD for the same shape:

```
rowmajor  [<size=2, stride=  192>, <size=24, stride=24576>, <size=64, stride=384>, <size=48, stride=1>]
pretiled  [<size=2, stride=12288>, <size=24, stride=24576>, <size=64, stride= 48>, <size=48, stride=1>]
```

**Identical dimension count, identical sizes, identical total length (73,728).**
The k-block stride is even the same 24576 by coincidence (`(N/n)·k·n = k·N`). The
only real difference is the innermost pair: pre-tiled reads **3072 contiguous
elements** per tile; row-major reads 64 runs of 48 with a 384-element stride.

The contiguous one is the slower one. We do not have a mechanism for that, and
this task does not propose one — the measurement is the result.

### 8. Where the instability comes from — a partial answer

Traced all four MiniLM GEMMs, M=512, 4 columns, n=48, 3 runs each:

| shape | B elements | rowmajor | pretiled mean | pretiled best | pretiled spread |
|---|---|---|---|---|---|
| `proj` `[384,384]` | 147K | 149.0 (0.5%) | **148.7** | 149.0 | **0.3%** |
| `qkv` `[384,1152]` | 442K | 148.9 (0.1%) | 139.7 | 149.0 | 11.7% |
| `ffn_down` `[1536,384]` | 590K | 140.9 (0.1%) | 122.2 | 132.2 | 15.7% |
| `ffn_up` `[384,1536]` | 590K | 149.9 (0.2%) | 136.5 | 149.0 | 22.1% |

Two things fall out:

- **Pre-tiled's *best* runs match row-major** (149.0 on three of four shapes). It
  is not intrinsically slower — it intermittently stalls.
- **The instability scales with B size.** The smallest B is stable and matches
  row-major exactly; the two largest are the worst. Row-major is stable at every
  size.

### 9. End-to-end, which is what M4 actually claimed

Per-core cycles measure a window that includes DMA stalls, but "the array is
starved" is a claim about the whole dispatch. Wall clock is legitimate for that
under `docs/05-measurement`, labelled as such. All four shapes, M=512, **8
columns**, 100 iterations:

| shape | rowmajor TFLOP/s | pretiled TFLOP/s | change |
|---|---|---|---|
| `qkv` | 1.72 | 1.65 | −4% |
| `proj` | 0.70 | 0.64 | −9% |
| `ffn_up` | 1.74 | 1.78 | +2% |
| `ffn_down` | 1.94 | 1.99 | +3% |

**A wash.** No consistent direction, all within ±9%. Pre-tiling does not move the
data-movement numbers on this design.

### 10. What *did* survive: `tile_n = 48`

All four MiniLM GEMMs now run at **8 columns**, which is the configuration M2
identified as worth 1.44× and the one we intend to ship. M2's `tile_n = 32`
cannot:

| shape | N | `N % (32·8) == 0` | `N % (48·8) == 0` |
|---|---|---|---|
| `qkv` | 1152 | **False** | True |
| `proj` | 384 | **False** | True |
| `ffn_up` | 1536 | True | True |
| `ffn_down` | 384 | **False** | True |

So M4's tile choice is load-bearing — just for the divisibility constraint, not
the BD limit. And at 4 columns traced, n=48 gives **148.9–149.9 MACs/cycle**
(58–59% of the 256 peak) on three of four shapes, against M2's 141.7 at n=32.

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

# reproduce the single-core BD failure (this one really does fail)
python experiments\m2-bf16-gemm\gemm_single_core.py --preset ffn_down -n 32 --emulate-bfp16

# the same shape on the UNMODIFIED M2 whole-array design -- this one works
python experiments\m2-bf16-gemm\gemm_whole_array.py --cols 4 -M 512 -K 1536 -N 384 -n 32 --emulate-bfp16

# rowmajor vs pretiled, 5 runs each
python experiments\m5-pretiled-gemm\gemm_pretiled.py --preset ffn_down -M 512 --cols 4 -n 48 `
    --emulate-bfp16 --baseline --repeat 5 --out experiments\m5-pretiled-gemm\artifacts\ffn_down_M512_n48_c4.json

# tile order: k,n vs n,k
python experiments\m5-pretiled-gemm\gemm_pretiled.py --preset ffn_down -M 512 --cols 4 -n 48 `
    --emulate-bfp16 --baseline --orders "k,n;n,k" --repeat 5 `
    --out experiments\m5-pretiled-gemm\artifacts\ffn_down_M512_n48_c4_orders.json

# isolate the L3->L2 change from the sub-tile reorder
python experiments\m5-pretiled-gemm\gemm_pretiled.py --preset ffn_down -M 512 --cols 4 -n 48 `
    --emulate-bfp16 --baseline --inner both --repeat 3 `
    --out experiments\m5-pretiled-gemm\artifacts\ffn_down_isolation.json

# all four shapes, traced at 4 columns
python experiments\m5-pretiled-gemm\gemm_pretiled.py --all-shapes -M 512 --cols 4 -n 48 `
    --emulate-bfp16 --baseline --repeat 3 `
    --out experiments\m5-pretiled-gemm\artifacts\traced_all_shapes_c4.json

# all four shapes, wall-clock end-to-end at 8 columns
python experiments\m5-pretiled-gemm\gemm_pretiled.py --all-shapes -M 512 --cols 8 -n 48 `
    --emulate-bfp16 --baseline --bench --bench-iters 100 `
    --out experiments\m5-pretiled-gemm\artifacts\bench_all_shapes_c8.json

# every M5 trace JSON regenerates byte-identically from the stored .txt + .mlir
python experiments\m2-bf16-gemm\regen_trace_json.py --artifacts experiments\m5-pretiled-gemm\artifacts --check
```

## Result

Restated as claims, with verdicts:

| claim | source | verdict |
|---|---|---|
| `ffn_down` cannot be expressed without pre-tiling | 0003 → 0006 → CLAUDE.md | **False for the whole-array design.** True only for the single-core design |
| Pre-tiling is the main performance lever | 0004 → 0006 | **Unsupported.** Per-core equal at best, −11% mean; end-to-end a ±9% wash |
| `tile_n = 48` is required and workable at 8 columns | 0006 | **Confirmed on hardware.** All four shapes run; n=32 is impossible for three of them |
| Baking the sub-tile order into the file is free | 0006 | **Confirmed.** 125.7 vs 126.4, indistinguishable |
| Correctness is unaffected | — | `rel_fro = 1.04e-02` on every run, matching M2 exactly |

The honest summary is that **M4 built a correct container for the wrong stated
reasons.** The `.npue` remains worth having — the fusions, bf16 conversion,
fp32 biases, `mmap`, and the sha256 pin are all still real and all still free.
But it is not a performance win on this design, and it was never load-bearing
for `ffn_down` outside the single-core case.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| The `--baseline` control, written to demonstrate a failure, **passed** | The BD limit was inherited from the single-core design and never re-tested on the whole array | Tested it. The premise was wrong; the docs are corrected rather than quietly dropped |
| `TypeError: expected str, bytes or os.PathLike object, not NoneType` in `trace_to_json`, only on the **second** run of a config | `physical_mlir_path` is set by the JIT **only when it actually compiles**. On a cache hit nothing compiles and the attribute stays `None` | Fall back to the `mlir_*.mlir` copy we already keep for offline regeneration. This trap makes any repeat-run measurement loop fail on its second iteration |
| `OSError: [Errno 22] Invalid argument` from the trace writer, **after** the kernel had already run | The variant label `pretiled[k,n\|st]` was used as the filename, and `\|` and `[` are invalid on Windows | Separate display label from filesystem slug. The cost is a full compile+run wasted before the error surfaces |
| `ValueError: tensor does not divide evenly into tile groups in dimension 0` | M2 hard-codes `tb_n_rows = tb_max_n_rows // 2 = 2` and only ever ran M=512, which has exactly 2 row blocks. MiniLM's real single-sequence shape M=256 has **one** | `tb_n_rows = min(tb_max_n_rows // 2, M // m // n_aie_rows)` |
| First pretiled-vs-rowmajor comparison looked decisive from one run each | The pre-tiled path varies up to 22% run to run; row-major varies 0.1% | `--repeat`, and report mean/range/spread. **A single per-core number is not a measurement on the pre-tiled path** |
| One run reported `EMPTY TRACE` | Intermittent; not reproduced | Not solved. Worked around by repeating; recorded because it means a single silent run cannot be trusted |
| Locality hypothesis (`n,k` tile order) explained nothing | The instability is not about stride distance | Kept both orders in the tool — the negative result is the finding |

## Artifacts

`experiments/m5-pretiled-gemm/`:

- `gemm_pretiled.py` — the design and driver
- `artifacts/trace_*.txt`, `mlir_*.mlir` — committed, 7.9 MB
- `artifacts/{ffn_down_M512_n48_c4,ffn_down_M512_n48_c4_orders,ffn_down_isolation,traced_all_shapes_c4,bench_all_shapes_c8}.json`
- `artifacts/trace_*.json` — 63 MB, gitignored; all 14 verified to regenerate
  byte-identically with `regen_trace_json.py --artifacts ... --check`

Two files were renamed from an earlier `pretiled[k,n]` / `pretiled[n,k]` naming
to `pretiled_kn_st_r0` / `pretiled_nk_st`; contents are untouched.

## Next

1. **Do not spend more on B pre-tiling.** Two shapes at two column counts, three
   variants, and an isolation experiment all say the same thing. The lever is
   elsewhere.
2. **The real lever is still unexplored: B reuse across row blocks.** M2's
   bandwidth estimate had B re-streamed 16× at M=4096, and this design still
   re-fills B once per row block — visible in the runtime sequence as
   `rt.fill(B_l3l2_fifos[col] ...)` inside the `tile_row` loop. That is a
   dataflow change, not a layout change, and it is the untested half of
   [0004](../0004-m2-multicore-gemm/TASK.md)'s recommendation.
3. **Larger L2 megatiles** ([2602.06063](https://arxiv.org/abs/2602.06063) got
   5.9 → 13.7 TOPS from megatile size alone) remain untried.
4. **Explain the pre-tiled instability, or stop caring.** Best-case runs match
   row-major exactly, so it is a stall, not a throughput ceiling. If B reuse
   removes the re-streaming, it may vanish on its own.
5. **Still outstanding from [0005](../0005-m3-python-reference/TASK.md): confirm
   the 5-bit bfp16 fit on hardware with real activations.** Nothing here touched
   it, and it is still the highest-value single measurement available — the
   `rel_fro = 1.04e-02` seen on every run above is the uniform-input figure again.

---

## Follow-up from [0008](../0008-m5-bfp16-real-data/TASK.md)

Added 2026-08-17.

**Item 5 above is done.** Hardware says the distribution barely matters (0.85×),
refuting the M3 simulation. See [0008](../0008-m5-bfp16-real-data/TASK.md).

**The end-to-end bench table in this task was measured unsafely.** It ran eight
compiled designs in one process with tracing off, which
[note 0003](../../research/notes/0003-two-designs-per-process.md) has since shown
corrupts every dispatch after the second design. Timing is plausibly unaffected —
the kernel does the same work either way — but that cannot be assumed.

It was re-run with one process per measurement and three repeats:

| shape | rowmajor mean (spread) | pretiled mean (spread) | ratio |
|---|---|---|---|
| `ffn_down` | 2.133 (1.8%) | 2.149 (16.5%) | +0.7% |
| `ffn_up` | 1.876 (3.3%) | 1.826 (9.4%) | −2.6% |

**The conclusion is unchanged: pre-tiling is a throughput wash.** The absolute
TFLOP/s differ from the table above, so use these numbers, not those.

One near-miss worth recording: the *first* isolated re-run, one measurement per
variant, showed pre-tiling at **+12.8%** and **+12.7%** on exactly these two
shapes — which would have reversed this task's conclusion. Three repeats erased
it. A single wall-clock number is not a measurement on the pre-tiled path either.

The stability finding is now the most robust thing known about pre-tiling: it
reproduces independently in traced per-core cycles **and** in wall clock.
