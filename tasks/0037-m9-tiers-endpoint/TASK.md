# 0037 — M9: batch tiers and an OpenAI-shaped /v1/embeddings endpoint

- **Date** 2026-08-18
- **Milestone** M9 (product surface)
- **Status** done — one xclbin now carries **16 instruction streams**
  (4 shapes × 4 batch tiers), so a request is *right-sized* instead of padded:
  a 1-text request went **~210 ms → 15 ms**. `npuembed --serve` speaks the
  OpenAI embeddings API and is verified against the **official `openai`
  client**.

## 1. Batch tiers: one xclbin serves every batch size

0032 proved four *shapes* share a static design. Batch is the same kind of
variable — M enters only through `n_tiles_per_core`, which is a runtime
parameter — so the same argument should extend to it. It does
(`experiments/m7-switch-cost/batch_share_probe.py`):

```
  batch    4 (M=  256)  insts   5,072 B
  batch   16 (M= 1024)  insts  10,704 B
  batch  128 (M= 8192)  insts  69,392 B
  batch 4 vs 16:  67 differing bytes -- SHARED (UUID-only)
  batch 4 vs 128: 69 differing bytes -- SHARED
```

`tools/export_gemm_rtp.py --batches 4,16,32,128` now builds all sixteen and
checks **every** pair against the reference before emitting — 15 identity
checks, all 64–69 bytes. If any diverged the export refuses, because shipping
an artifact that claims one context while needing two is worse than failing.

**Requests are planned greedily against the tier ladder.** 64 texts with tiers
{4,16,32,128} becomes 32+32 — both exact — rather than one half-padded 128.
It also produces more, smaller jobs, which balances the lanes better.

| request size | seq/s | wall |
|---|---|---|
| 1 | 66 | **15 ms** (was ~210 ms padded to 128) |
| 8 | 250 | 32 ms |
| 64 | 348 | 184 ms |
| 100 | 460 | 218 ms |
| 512 | **918** | 558 ms |

Full-encode benchmark unchanged at **840.7 seq/s**, `1-cos` 1.086e-05, lanes
bitwise identical.

## 2. The endpoint

`npuembed --serve [port] [--bind addr]`, built on `include/http.hpp` — a few
hundred lines of Winsock and just enough JSON. No dependency was added; the
deployment story stays *copy two files and run*.

- `POST /v1/embeddings` — `input` as a string or array of strings,
  `encoding_format` `float` or `base64`, OpenAI-shaped response with
  `usage.prompt_tokens` counted from the real tokenizer.
- `GET /v1/models`, `GET /health`.
- Refuses rather than guesses: no `input` → 400, empty array → 400, unknown
  `encoding_format` → 400, token-id arrays → 400 (with a message saying so),
  >2048 inputs → 413, unknown path → 404.

**Requests are served one at a time, deliberately.** The NPU serializes
dispatches anyway (note 0004) and the lanes already parallelise *inside* a
request, so concurrent handling would add contention and lock complexity to
buy nothing. Throughput comes from batching within a request — which is
exactly what an embeddings client does.

`--embed` and `--serve` share one `EmbedService`, so the endpoint cannot drift
from the path the tests measure. Verified: **max abs diff 0.000e+00** between
them.

### Verified against the real client, not against a guess

`tools/verify_endpoint.py` drives it with the official `openai` package
(3.2.0): `models.list()`, single string, batch with index-order checking,
base64, semantics, cross-check against `--embed`, and four error cases.

One thing worth recording: **the OpenAI client requests base64 by default**
and decodes it itself. So the plain `embeddings.create(...)` calls were
already exercising the base64 path. Asking for base64 *explicitly* makes the
client hand back the raw string by design — which is what a first version of
the test mistook for a bug.

## 3. A real bug the endpoint test caught, and why it was silent

The first full run failed on two lines:

```
  vs --embed      max abs diff 2.504e-01     <-- not rounding
  unrelated cos   +0.6488                    <-- stock market vs blueberry jam
```

`experiments`-style isolation (1 lane vs 2 lanes vs the reference) localised
it immediately:

```
  1 lane    worst 1-cos vs reference: 1.324e-05
  2 lanes   worst 1-cos vs reference: 1.056e+00   -- on texts 4..7 only
```

**The extra lanes never inherited the tier table.** `enc.tiers` /
`enc.tier_slots` were populated on lane 0 only, so a lane's `use_tier()` was a
no-op and its `is_qkv..is_fd` kept the **pre-0037 flat slot contract**
(0,1,2,3). Under the 16-stream export those indices point at entirely
different streams — `qkv@b128`, then the b4 quartet — so lane 2 ran the wrong
GEMMs on whichever chunk it happened to take.

Silent because: every field it *did* inherit was correct, the old default was
a *valid* slot range rather than an out-of-bounds one, and the single-lane
path (which every existing test used) was unaffected. The pipelined
bitwise-lane check would have caught it, but it only runs under
`--pipeline N` without `--bench`, and that combination had not been re-run
since the tier change.

**Fixes:** lanes copy `tiers`/`tier_slots` and call `use_tier`, and the
construction now **fails closed** — a lane whose stream policy differs from
lane 0 throws rather than running.

> Two lessons, both already in this project's vocabulary. A default that is
> *valid but wrong* is the worst kind (this is the sixth fail-open). And the
> test threshold that "looked too strict" — `unrelated cos < 0.4` — was
> correct: 0.6488 was the bug, and after the fix it reads **0.0881**.

## Exact commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
python experiments\m7-switch-cost\batch_share_probe.py            # the hypothesis
python tools\export_gemm_rtp.py --batch 128 --batches 4,16,32,128 `
       --cols 8 --out runtime\artifacts_b128il
cd runtime; cmake --build build --config Release

# serve
.\build\npuembed.exe .. --artifacts artifacts_b128il --threads 24 --pipeline 2 --serve 8420
# verify (another shell)
cd ..; & ".\.venv-ref\Scripts\python.exe" tools\verify_endpoint.py --port 8420
```

```bash
curl http://127.0.0.1:8420/v1/embeddings -H "Content-Type: application/json" \
  -d '{"input":["hello world","another text"],"model":"all-MiniLM-L6-v2-npu"}'
```

## Next

- **README and a public repo.** The docs cite the indexed papers heavily and
  quote them closely; a public version needs `research/papers/` reduced to
  citations plus our own conclusions, and `OthersResarch/` must not ship at
  all.
- Concurrency: one request at a time is right for a single NPU, but a queue
  with a deadline would let a server coalesce several small requests into one
  tier — the same batching win, applied across clients.
- `--serve` binds localhost and speaks no TLS. That is a deliberate scope
  line, and it belongs in the README rather than in the code.
