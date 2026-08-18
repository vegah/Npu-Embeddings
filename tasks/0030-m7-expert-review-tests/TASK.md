# 0030 — M7: testing the expert review, end to end

- **Date** 2026-08-17
- **Milestone** M7
- **Status** done — 7 of 10 claims resolved on hardware, 1 refuted by an
  existing measurement, 2 deferred with a measured pricing argument.
  **Encode: 251.3 → 298.5 seq/s. Accuracy: 3.397e-04 → 2.469e-04.**

## Goal

An external reviewer read the whole codebase and made a set of falsifiable
claims (reproduced in context in
[note 0005](../../research/notes/0005-expert-review-tests.md), which is the
living scoreboard for this task). Test all of them, updating the note as each
lands.

## The scoreboard (final)

| § | claim | verdict |
|---|---|---|
| §1 step 0 | insts streams over one context switch for free | **CONFIRMED** ([`0029`](../0029-m7-one-xclbin-probe/TASK.md)) |
| §1 step 1 | RTP loop bounds unify GEMM shapes into one xclbin | **CONFIRMED** — exact results, zero switch cost, RTP overhead +1.6% |
| §2 | GELU fusable into ffn_up (K-augmented bias + epilogue) | **CONFIRMED** — and 10× *more accurate* fused (3.167e-04) |
| §3 | LN/softmax reach 8 columns via split/join + param broadcast | **CONFIRMED** — 251 → 298.5 seq/s |
| §4 | Post-attention block fusion | mechanisms proven; naive form priced to a wash; pipelined form is milestone-scale |
| §5a | exp2_poly bug is the worker stack | **CONFIRMED** — and poly softmax is now production, 4× better |
| §5b | stride wall is the C-drain row-block stride 256·N | **CONFIRMED** — 4/4 prediction, boundary inclusive, bge-large clears |
| §6a | micro-batch pipelining after §1 | deferred with cause (needs §4's architecture) |
| §6b | device-resident intermediates | deferred with cause (same; ~70 ms prize) |
| §6c | bf16 GEMM output at identical numerics | **REFUTED as stated** (M2 trap 2); true via a §2-style epilogue |

## What changed in production

- **softmax runs `exp2_poly`** (stack 0x2000): NPU-vs-golden 1.744e-02 →
  4.278e-03, and the full encode 3.397e-04 → **2.469e-04** — the project's
  best accuracy.
- **All three eltwise designs at 8 columns** (LayerNorm's params broadcast from
  the mem tile): 251.3 → **298.5 seq/s**.
- **`purge_ambiguous()`** in the exporter: the fifth fail-open (bfp16 GEMM
  builds shadowing bf16 ones — mtime picked them; encode silently regressed to
  *exactly* 0026's bfp16 number) is closed by deleting ambiguous cache
  candidates before every build.
- **`gemm_pretiled` gained `rtp=` and `epilogue=`** — both proven, neither yet
  wired into the exported production set.
- **`tb_n_rows` caps to 1 above the 2²⁰ stride limit** — h ≥ 1536 designs now
  build (buildability, not yet correctness; no goldens exist there).

## Three cache/identity traps found during the work

1. **The RTP initializer is part of the static image** — shape-dependent
   `initial_value` was exactly the 8 bytes keeping two shapes' xclbins from
   identity. Zero it; the sequence writes real values.
2. **The JIT does not hash `initial_value`** — changing it served stale builds.
3. **bfp16 emulation is invisible to every marker the matcher reads** — only
   the kernel-hash prefix differs, and its expected value is not computable.
   Hence purge-before-build.

## Where the numbers stand after this task

| | before review | after |
|---|---|---|
| seq/s at batch 128 | 251.3 | **298.5** |
| `1-cos` vs HuggingFace | 3.397e-04 | **2.469e-04** |
| vs CPU (710) wall | 0.35× | **0.42×** |

## Next

1. **The pipelined §4 design** — GEMM columns streaming into eltwise columns
   through the mem tiles, one dispatch per layer block. Every mechanism it
   needs is now individually proven. This is the remaining large lever.
2. Wire `rtp=` and `epilogue="gelu"` into the exported production set.
3. bge-large goldens, now that h = 1024 clears the stride wall.
