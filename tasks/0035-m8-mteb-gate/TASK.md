# 0035 — M8 GATE PASSED: MTEB says the NPU embeddings are indistinguishable from the CPU's

- **Date** 2026-08-18
- **Milestone** M8 (second half — the accuracy gate)
- **Status** **PASSED.** Five MTEB tasks, mean delta **+0.04 points**, worst
  single task **−0.01**. The gate was |mean| ≤ 0.5 and no task worse than −0.5.

## Why this task exists

Every accuracy number the project had produced was a **fidelity** measure: how
close the NPU embedding sits to the fp32 oracle on four fixed sentences
(`1-cos` 1.086e-05). Necessary, and not sufficient. The claim the whole project
rests on is that the embeddings are still **good** — that a downstream task
cannot tell the difference. Only a benchmark can say that, and until it did,
the bf16-vs-bfp16 decision and every optimisation resting on "accuracy
unchanged" were unproven.

## The bridge that made it runnable

MTEB needs `encode(list[str]) -> ndarray`. The C++ runtime has no tokenizer
(WordPiece is still unwritten) and CLAUDE.md rule 5 forbids Python at runtime.
Both hold if the boundary stays where this project always puts it — **data
crosses as files**:

```
  text --[HF tokenizer + embedding lookup, npu_encoder.py]--> emb_sum.f32 + masks
       --[npuembed.exe --encode-file]--> out.f32 --> embeddings
```

New pieces:

- **`npuembed --encode-file <dir>`** — reads `emb_sum.f32`, `add_mask.f32`,
  `attention_mask.f32` for an arbitrary number of sequences, encodes in
  design-sized chunks (padding the tail with zero rows, which are masked out of
  their own pooling and cannot reach any other row), mean-pools, L2-normalises,
  writes `out.f32`.
- **`experiments/m8-npu-vs-cpu/npu_encoder.py`** — an mteb 2.x encoder that
  tokenizes, does the three-table embedding sum in fp32 (exactly
  `reference/encoder.py`'s `embed()`), and shells out to the runtime.
- **`experiments/m8-npu-vs-cpu/run_mteb.py`** — runs both sides over the same
  tasks and applies the gate.

Bridge self-test before any benchmark: the golden corpus through the bridge
matches `sentence-transformers` at seq 64 to **1.078e-05** worst 1-cos — the
same figure the golden path reports, through a completely different entry
point.

## The comparison is paired, which is what makes the gate meaningful

Both sides run the same tasks, same mteb 2.19.5, same checkpoint, and — this
one matters — **the same sequence length**. The compiled designs are seq 64, so
`SentenceTransformer.max_seq_length` is set to 64 too. Giving the CPU its
default 256 would hand it strictly more information and quietly make the
comparison dishonest.

Consequence: the absolute scores below are "MiniLM at seq 64" and sit under the
published seq-256 leaderboard numbers by construction. **The claim is the
delta**, and a gate on the delta cannot be gamed by picking an easy task.

## Results

Three STS tasks (is semantic similarity preserved?), one classification and one
clustering task (is the embedding *space* still shaped the same?):

| task | CPU | NPU | delta (points) |
|---|---:|---:|---:|
| STSBenchmark | 82.03 | 82.04 | **+0.01** |
| SICK-R | 77.58 | 77.59 | **+0.01** |
| STS12 | 72.37 | 72.36 | **−0.01** |
| Banking77Classification | 80.05 | 80.05 | **+0.00** |
| TwentyNewsgroupsClustering | 45.81 | 45.98 | **+0.17** |
| **mean** | | | **+0.04** |

**PASS.** Four of five tasks are within ±0.01 points — below the noise floor of
the metrics themselves. The clustering task's +0.17 is k-means seed sensitivity,
not a real gain; it is in our favour and still reported as noise.

The bf16 datapath with fp32 accumulation, our own polynomial GELU, exp2 and
LayerNorm, host fp32 elementwise, and a pre-tiled `.npue` weight layout together
cost **nothing measurable in downstream quality**.

## What this decides

1. **The bf16 + fp32-accumulate configuration is confirmed as the production
   default**, on evidence rather than on the tolerance argument.
2. **`--emulate-bfp16` stays rejected.** It failed the fidelity tolerance in
   0026 (3.470e-03 against 2e-03) and there is now no quality headroom argument
   left for it: the current path costs 0.04 points, so bfp16's only case would
   be speed it does not deliver either (+2.2% end to end).
3. Every optimisation in M7 that claimed "accuracy unchanged" is now backed by
   a downstream measurement, not only by `1-cos`.

## Honest limits

- **Five tasks, not the full MTEB suite.** Retrieval is deliberately absent —
  it is the slowest family by an order of magnitude and STS + clustering
  already exercise the same geometry. A full-suite run is a day of compute, not
  a session.
- **Seq 64.** Both sides equally, but it means these numbers are not comparable
  to published leaderboards.
- The bridge runs one subprocess per `encode()` call, so ~2–3 s of process
  startup is included in the per-task times below; they are not a throughput
  measurement (0033 is).

## Exact commands

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
& ".\.venv-ref\Scripts\python.exe" -m pip install mteb          # once
& ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\npu_encoder.py   # bridge self-test
& ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\run_mteb.py      # the gate
```

Artifact: `experiments/m8-npu-vs-cpu/artifacts/mteb_results.json`.

## Where the project stands

Every claim the project set out to make is now measured on hardware:

| claim | status |
|---|---|
| Runs the model on the NPU, C++, no Python | ✅ 0022–0023 |
| Faithful to fp32 | ✅ `1-cos` 1.086e-05 |
| **Downstream quality preserved** | ✅ **this task, +0.04 points** |
| Faster than the CPU | ✅ 833 vs 663–710 seq/s (0033) |
| Lower energy | ✅ 1.94× per sequence (0034) |
| Fewer cores | ✅ ~5.3 vs 12 |

**Remaining for a standalone product:** the WordPiece tokenizer (~500–700 LOC,
no research risk — it is the last thing standing between this and text in →
vector out in one process).
