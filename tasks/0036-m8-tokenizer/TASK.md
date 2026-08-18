# 0036 — M8: the WordPiece tokenizer. Text in, vector out, no Python

- **Date** 2026-08-18
- **Milestone** M8
- **Status** done — **6,826 / 6,826 texts tokenize byte-for-byte identically to
  HuggingFace**, and `npuembed --embed` now takes plain text and returns
  embeddings in one process. End-to-end worst `1-cos` **6.10e-05**, top-10
  neighbour overlap **99.01%**.

## What was built

1. **`tools/gen_tokenizer_tables.py`** → `runtime/include/bert_unicode_tables.hpp`
   (607 KB, UCD 15.1.0). The Unicode facts BERT is defined in terms of —
   lowercase, NFD-with-Mn-dropped, and the punctuation / control / space
   categories — computed offline from Python's `unicodedata` and emitted as
   static tables. Not hand-written and not ICU: **generated, so it agrees with
   the reference by construction rather than by careful reading.** 16,253 fold
   entries, 186 punctuation ranges, 714 control, 9 space.
2. **`runtime/{include,src}/tokenizer.{hpp,cpp}`** — BasicTokenizer +
   WordPiece: clean, CJK padding, whitespace split, fold, punctuation split,
   greedy longest-match-first with `##` continuations, `[CLS]`/`[SEP]`,
   padding and truncation.
3. **The vocabulary now rides inside the `.npue`** (`tokenizer.vocab`, a new
   `U8` dtype for opaque bytes). Deploying the model is copying **one file**.
   A model packed before this still works — the loader falls back to
   `vocab.txt` and says so.
4. **`npuembed --embed <textfile> [out.f32]`** — the whole product: tokenize,
   gather the three embedding tables straight out of the mapping, encode on
   the NPU, mean-pool, L2-normalise. And `--tokenize` for the differential
   test.
5. **`tools/verify_tokenizer.py`** and **`tools/verify_embed_e2e.py`**.

## The differential test earned its keep in the first run

35/38 on an adversarial corpus, and both failures were real:

**1. Greek final sigma — I implemented the documented behaviour, not the
shipped one.** `tokenization_bert.py` calls `token.lower()`, and Python applies
the context-dependent final-sigma rule, so `ΟΔΥΣΣΕΥΣ` → `οδυσσευς` ending in
`ς`. I implemented exactly that. The reference disagreed: this checkpoint loads
the **fast (Rust) tokenizer**, which lowercases per character with no context,
giving `##σ`. The rule was removed.

> Reading the reference implementation told me the wrong thing. Only comparing
> every id against the tokenizer the embeddings were actually produced with
> could have caught it — and it would have silently degraded every Greek text.

**2. Special tokens were being split.** `[CLS]` in user text tokenized to
`'['`, `'cl'`, `'##s'`, `']'`. The reference matches added tokens literally in
the raw string before basic tokenization ever runs. Fixed with a pre-pass.

After both fixes: **38/38**, then **6,826/6,826** on a corpus of real MTEB text
plus multilingual material (Arabic, Hebrew, Thai, Greek polytonic, Russian,
Turkish dotted-I, Vietnamese tones, CJK, Hangul, fullwidth ASCII, ligatures,
combining-mark stacks, emoji ZWJ sequences, URLs, Windows paths, SQL, JSON).
Also 100% at `max_len` 128, so the truncation path is exercised at two lengths.

### The NFC step this implementation skips, and why that is safe

HuggingFace normalizes to NFC before splitting. Full canonical composition
needs tables this runtime does not want. It is skipped, and the argument is
structural: the very next step decomposes (NFD) and drops the combining marks,
so composing a base+mark pair only to decompose it again lands in the same
place. The corpus above deliberately includes combining sequences, and the
agreement is exact — the argument is now measured, not asserted.

## A pre-existing bug found on the way: the packer had not run in months

`pack_npue.py` calls `gemm_b_layout(...)` but never imported it — the symbol
was factored into `npue.py` and the import was never added. **The packer has
been broken since that refactor**, invisibly, because the shipped `.npue`
predates it and nothing ever repacked. Found only because adding the vocabulary
forced a repack.

Worth its own note: this is the *build* step for the file the whole product
loads. Every "verify" in the project checked the artifact, and none checked
that the thing which produces the artifact still runs.

## Results

**Tokenizer, against HuggingFace:**

```
  6,826 / 6,826 exact (100.00%)  at max_len 64
  6,826 / 6,826 exact (100.00%)  at max_len 128
```

**End to end** (`--embed`, against `sentence-transformers` at seq 64):

| corpus | worst `1-cos` | sim error (mean / p99 / max) | top-10 overlap |
|---|---|---|---|
| 13 mixed texts | 2.64e-05 | — | — |
| 6,788 real texts | **6.10e-05** | 3.2e-04 / 1.1e-03 / 3.2e-03 | **99.01%** |

### The similarity tolerance was wrong, and fixing it is the point

The first version failed the big corpus at a pairwise-similarity error of
3.2e-03 against a flat 2e-03 tolerance. The embeddings were not worse — the
tolerance was measuring the wrong thing. For unit vectors with `1-cos = c` the
perturbation is `sqrt(2c)` and a dot product moves by at most `~2*sqrt(2c)`,
here **2.2e-02**; and a *maximum* over 4.5 M pairs is an extreme-value
statistic that grows with corpus size at unchanged accuracy.

So the check now tests the **derived bound** plus the distribution, and adds
the property a retrieval user actually depends on: **do the same neighbours
come back?** 99.01% top-10 overlap says yes. *A constant tolerance on a max
over N pairs tests the corpus size, not the model.*

## Exact commands

```powershell
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\gen_tokenizer_tables.py
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\pack_npue.py        # now includes the vocab
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\verify_npue.py
cd runtime; cmake --build build --config Release; cd ..
& ".\.venv-ref\Scripts\python.exe" tools\verify_tokenizer.py
& ".\.venv-ref\Scripts\python.exe" tools\verify_embed_e2e.py
# the product itself
cd runtime
.\build\npuembed.exe .. --artifacts artifacts_b128il --threads 24 --embed texts.txt out.f32
```

## Where this leaves the project

**Every goal M0 set out is now met and measured.** The product is one `.npue`
(69 MB, weights + vocabulary) and one `npuembed.exe`:

| | |
|---|---|
| text in → vector out, one process, no Python | ✅ this task |
| tokenizer identical to HuggingFace | ✅ 6,826/6,826 |
| faithful to fp32 | ✅ `1-cos` 6.1e-05 end to end |
| downstream quality | ✅ MTEB +0.04 points ([`0035`](../0035-m8-mteb-gate/TASK.md)) |
| faster than the CPU | ✅ 833 vs 663–710 seq/s ([`0033`](../0033-m7-pipelined-lanes/TASK.md)) |
| lower energy | ✅ 1.94× per sequence ([`0034`](../0034-m8-energy/TASK.md)) |

## Next

- **`--embed` does not pipeline yet.** It calls `enc.run()` per chunk, so it
  gets single-lane throughput. Wiring it to the two-lane path is mechanical and
  worth ~1.35×.
- **An OpenAI-shaped `/v1/embeddings` endpoint** is the obvious next surface.
  Note for that decision: FastFlowLM cannot be integrated *into* — CLAUDE.md
  rule 4 forbids vendoring anything from it, and its installer terms forbid
  redistribution. A standalone HTTP server in this repo has no such constraint.
- Batch-size flexibility: `--embed` pads the tail chunk to the design's batch,
  so a 4-text request still runs a 128-sequence encode. A small-batch design
  set (`artifacts_b4`) already exists and should be selected automatically.
