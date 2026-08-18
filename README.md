# NpuEmbeddings

**Sentence embeddings on the AMD Ryzen AI NPU, from hand-written AIE kernels,
on native Windows, with a C++ runtime.**

Text in, vectors out — one executable, one model file, no Python at runtime.
It serves an OpenAI-shaped `/v1/embeddings` endpoint, so anything that already
talks to an embeddings API can point at it instead.

```bash
curl http://127.0.0.1:8420/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["hello world", "another sentence"]}'
```

---

## What this is

An implementation of **all-MiniLM-L6-v2** that runs its matrix multiplies on
the **XDNA2 NPU** in AMD Ryzen AI processors (Strix Point). The kernels are
written directly against the AI Engine array using
[MLIR-AIE / IRON](https://github.com/Xilinx/mlir-aie) — not through ONNX
Runtime, not through a vendor overlay — and the shipped runtime is C++ and XRT.

It is also a **learning project**, and that is visible in the repository. Every
result has a task log recording what was measured, on which command, and what
went wrong on the way. The failures are kept deliberately; several of them are
more useful than the successes.

| | |
|---|---|
| Model | all-MiniLM-L6-v2 — 6 layers, hidden 384, 384-dim embeddings |
| Sequence length | 64 tokens (fixed at build time) |
| Hardware | AMD Ryzen AI, XDNA2 / Strix Point (developed on a Ryzen AI 9 HX 370) |
| OS | Windows, native — no WSL |
| Precision | bf16 with fp32 accumulation on the NPU; fp32 on the host |
| License | Apache-2.0 |

**What actually runs where.** The 24 GEMMs per encode run on the NPU as
instruction streams over a single design. LayerNorm, softmax and GELU run on
the host in fp32 — that is not a shortcut but a measured decision: at hidden
384 those ops are too small to earn an NPU dispatch, and moving them to the
host made the model both *faster and more accurate*. Attention's per-head
matmuls are on the host too (`head_dim = 32` does not tile the array).

## What it solves

A sentence-embedding workload is a good fit for an NPU and a poor fit for the
CPU it shares a package with: it is a fixed, small model applied to a large
number of short inputs. The point is to **get that work off the CPU cores**
without losing throughput or quality.

Measured on a Ryzen AI 9 HX 370, against `sentence-transformers` on the same
machine at the same sequence length:

| | NPU (this project) | CPU (`sentence-transformers`, 12 threads) |
|---|---|---|
| Throughput, large batch | **833–918 seq/s** | 663–710 seq/s |
| CPU cores occupied | **~5.3** | 12 |
| Energy per 1000 sequences | **44.0 J** | 85.3 J |
| Latency, single text | 15 ms | — |

So: **1.2–1.4× the throughput, on under half the cores, at 1.94× less energy
per sequence.**

And the embeddings are the same embeddings:

| check | result |
|---|---|
| Worst `1 − cos` vs the HuggingFace reference | **1.086e-05** |
| MTEB, 5 tasks (STS, classification, clustering), delta vs CPU | **+0.04 points** |
| Tokenizer, vs HuggingFace over 6,826 texts | **6,826 / 6,826 exact** |
| Top-10 nearest-neighbour overlap vs reference | **99.0%** |

Every figure above is reproducible from this repository —
see [`tools/verify_*.py`](tools/) and the task logs.

### What it does not solve

Being explicit, because the honest limits are part of the point:

- **One model, one sequence length.** The designs are compiled for
  all-MiniLM-L6-v2 at seq 64. Longer inputs are truncated. bge-small is a
  drop-in weight swap by design, but is untested.
- **One machine architecture.** XDNA2 on native Windows. Not XDNA1, not Linux,
  not other NPUs.
- **The server is a localhost server.** No TLS, no authentication, one request
  at a time. Put a reverse proxy in front of it, or use it in-process.
- **The NPU is shared.** Other processes using it will affect your throughput.

## How to use it

### 1. Download

Grab the latest `npuembeddings-*-win-x64.zip` from
[**Releases**](../../releases) and unzip it anywhere — it is about 300 KB:

```
npuembed.exe              the runtime            0.6 MB
gemm_rtp/                 the compiled NPU design 0.6 MB
get-model.cmd             fetch the weights (first run only)
run-server.cmd            start the endpoint
embed.cmd                 embed a text file
manifest.json             sha256 of every file, and the verified figures
```

No installer, no Python, nothing written outside the folder.

### 2. Fetch the model, once

```cmd
get-model.cmd
```

**The weights are not redistributed here.** They are
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
Apache-2.0, and worth reading the model card for. `get-model.cmd` downloads
them from HuggingFace, **checks the checksum against the exact checkpoint this
build was verified against**, and packs them into `models/all-MiniLM-L6-v2.npue`
— weights pre-tiled into the order the NPU's DMA reads them, plus the tokenizer
vocabulary, in one file.

That packing runs inside `npuembed.exe`, so there is still no Python anywhere.
It is verified byte-identical to the reference packer
([`tools/verify_pack_parity.py`](tools/verify_pack_parity.py)) — two
implementations of one binary layout is a real risk, so it is tested rather
than trusted.

### 3. Run the endpoint

```cmd
run-server.cmd
```

```
serving http://127.0.0.1:8420/v1/embeddings   (model all-MiniLM-L6-v2-npu, seq 64)
```

Use it from the official OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8420/v1", api_key="not-needed")
r = client.embeddings.create(
    model="all-MiniLM-L6-v2-npu",
    input=["a man is playing a guitar", "someone plays guitar at a concert"],
)
print(len(r.data[0].embedding))     # 384
```

Supported: `POST /v1/embeddings` (`input` as a string or array of strings,
`encoding_format` `float` or `base64`), `GET /v1/models`, `GET /health`.
Requests carrying token-id arrays instead of text are rejected rather than
guessed at.

**Right-sizing.** The design carries instruction streams for several batch
sizes, so a 3-text request runs a 4-sequence encode rather than padding to
128. That is why a single text takes 15 ms and 512 texts take 558 ms.

### 4. Or embed a file directly

```cmd
embed.cmd texts.txt out.f32
```

One text per line in; `out.f32` is `[n_texts, 384]` little-endian fp32,
L2-normalised, in input order.

```python
import numpy as np
v = np.fromfile("out.f32", dtype=np.float32).reshape(-1, 384)
```

### Useful flags

```
npuembed.exe <root> --artifacts <dir> [options]

  --serve [port]        OpenAI-shaped HTTP endpoint (default 8420)
  --bind <addr>         interface to bind (default 127.0.0.1)
  --embed <in> [out]    embed a text file
  --tokenize <file>     token ids only, one line per input
  --threads N           host thread budget (default 1)
  --pipeline N          N concurrent encode lanes over the one design
```

## What else you need

### To run a release

| | |
|---|---|
| **A Ryzen AI machine** | XDNA2 NPU — Strix Point (Ryzen AI 300 series) or newer |
| **The AMD NPU driver + XRT** | Ships with the [AMD Ryzen AI Software](https://ryzenai.docs.amd.com/en/latest/inst.html) install. The runtime loads `xrt_coreutil.dll` from it. |
| **MSVC 2015–2022 redistributable** | [Microsoft download](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) — most machines already have it |

Check the NPU is present and idle:

```cmd
C:\Windows\System32\AMD\xrt-smi.exe examine
```

> `XILINX_XRT` must **not** be set in your environment. It breaks Windows XRT
> builds. Use `XRT_ROOT` if you need to point at an install.

### To build from source

You additionally need **MLIR-AIE (IRON)**, which is what compiles the AIE
kernels into the `.xclbin` the runtime loads, plus the Peano LLVM-AIE compiler
and a C++ toolchain. Installation is involved and is documented upstream — we
link to it rather than duplicating it.

**→ See [BUILD.md](BUILD.md)** for the full build, the model preparation steps,
and the verification suite.

### The model

Not shipped, by choice — see step 2 above. `get-model.cmd` fetches and prepares
it in one command, and building from source does the same thing through
[BUILD.md](BUILD.md)'s pipeline.

Redistributing it would have been permitted, since the model is Apache-2.0. It
would also have been the worse deal: a 66 MB blob in a stranger's zip that
nobody can practically check against the original, going stale the moment
upstream changes, and hiding the model card. A checksummed download from the
source is both smaller and more honest.

## How it works, briefly

```
text ──► WordPiece tokenizer ──► embedding gather ──┐
                                                     │  (C++, host)
        ┌────────────────────────────────────────────┘
        ▼
   6 × encoder layer
     ├── QKV / attn-out / FFN-up / FFN-down GEMMs ──► NPU  (bf16, fp32 accum)
     ├── attention per head ──────────────────────►  host (AVX2)
     └── LayerNorm / softmax / GELU ──────────────►  host (fp32, AVX2)
        │
        ▼
   mean pool ──► L2 normalise ──► 384-dim vector
```

Two design decisions carried most of the performance, and both came out of
measurement rather than intuition:

1. **One xclbin, many instruction streams.** Changing which design the NPU is
   configured for costs 2.2–2.6 ms — far more than a dispatch. By moving the
   only shape-dependent values into runtime parameters, all four GEMM shapes
   *and* all four batch sizes share one static design and one hardware context,
   so an encode pays **zero** switch cost. That alone took the encode from 305
   to 611 seq/s.
2. **Two encode lanes over one design.** The NPU serialises dispatches
   regardless, so one lane's host work overlaps another's NPU work — measured
   1.49× — for 833 seq/s.

The full reasoning, including the routes that did *not* work, is in
[`tasks/`](tasks/README.md).

## About this repository

This is the public subset of a larger working repository, assembled by
[`tools/sync_public_repo.py`](tools/sync_public_repo.py). What is not here is
an indexed literature review: summaries of the published work on AIE
programming that the project read as it went, and the papers themselves. Those
summaries were written to be complete enough to use instead of the papers,
which makes them exactly the thing not to republish.

Everything else ships, including the task logs that mention them. So when a
document refers to `research/papers/` or `OthersResarch/`, it is describing
that private material — the papers themselves are cited by arXiv id and linked
to arxiv.org wherever a claim depends on one.

## Repository layout

```
runtime/        the product: C++ + XRT, no Python
  include/      npue reader, XRT dispatch, tokenizer, http
  src/          main.cpp is the encoder, the server and the probes
experiments/    IRON designs and kernels (Python at build time only)
  m5-eltwise/   GELU, LayerNorm, softmax kernels (.cc) and their designs
  m5-pretiled-gemm/  the production GEMM design
tools/          build-time: pack the model, export designs, verify everything
docs/           how it works, and the measurement rules
tasks/          what happened, day by day, failures included
reference/      the numpy oracle everything is validated against
```

## Acknowledgements

Built on [MLIR-AIE / IRON](https://github.com/Xilinx/mlir-aie) and
[XRT](https://github.com/Xilinx/XRT).

Sixteen papers, theses and write-ups shaped this project's architecture, and
three of them changed the design outright.
[`docs/literature.md`](docs/literature.md) records **which source led to which
decision** — including the three places our own measurements ended up
disagreeing with published guidance.

### Licensing

This repository is **Apache-2.0**. Five files began as MLIR-AIE examples and
remain **Apache-2.0 WITH LLVM-exception**, keeping their original copyright
headers and stating what was changed — one of them is a near-verbatim copy of
AMD's GELU kernel, vendored on purpose as a control experiment.

[`THIRD-PARTY.md`](THIRD-PARTY.md) lists them, and it is *generated* by
[`tools/audit_third_party.py`](tools/audit_third_party.py) rather than
maintained by hand: attribution rots as files are rewritten, so the audit
measures the longest contiguous run of shared code against an mlir-aie
checkout and fails if anything substantial lacks a header. Everything else
here is original work.

This is an independent implementation and is not affiliated with or endorsed
by AMD.
