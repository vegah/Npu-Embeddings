# 0001 — Scaffold repo, index research, write foundational docs

- **Date** 2026-08-16
- **Milestone** M0
- **Status** done

## Goal

Turn an empty repository into one that explains itself: documentation skeleton,
a research index that makes the `OthersResarch/` PDFs never need re-reading, the
task-log system, and every verified fact about this machine's hardware and toolchain
written down. **No code.**

Gate: a new session can orient itself from `CLAUDE.md` + `docs/` without re-deriving
anything.

## Context

Starting state: the repo contained only `LICENSE` (Apache-2.0), `.gitignore` (C/C++),
`.gitattributes`, and `OthersResarch/` with 7 unread arXiv PDFs. Single commit
`773b63b Initial commit`.

Planning had already produced the decisions recorded in `docs/00-overview.md`:
native-Windows/Peano, IRON Python at build time only, bf16 first, one validated kernel
first, standalone Apache-2.0 tool, all-MiniLM-L6-v2 as the first model.

## What was done

**1. Environment verification.** Rather than trusting the plan's assumptions, every
claim was checked on the machine. Highlights and corrections:

- The NPU is a **Ryzen AI 9 HX 370 (Strix Point, XDNA2/AIE2P/npu2)** — 8 columns × 4
  rows = 32 tiles, *not* the 4-column NPU1 most papers use.
- **XRT 2.21.0 is installed at `C:\Xilinx\XRT`**, not `C:\Program Files\Xilinx\XRT`.
  An initial probe missed it and wrongly concluded no SDK was present. It is the full
  prebuilt Windows SDK (headers, `xrt_coreutil.lib`, `aiebu_static.lib`,
  `xclbinutil.exe`, `nputrace_tool/`, `pyxrt.pyd`), so no import-library hack is needed.
- **`xchesscc` is absent** (no Vitis) — Peano-only confirmed.
- **Native bfp16 is chess-gated and unavailable**; an earlier note claiming bfp16 was
  reachable was wrong. The Peano-compatible substitute is
  `--emulate-bf16-mmul-with-bfp16`, which upgrades bf16 mmul geometry `4×8×8` → `8×8×8`.
- **The native-Windows flow already works here** — 96 compiled designs in
  `C:\Users\vegar\.npu\cache\` and locally-generated trace JSON in `01_SAXPY/`.
  This retires the project's biggest risk before any code is written.
- MinGW provides **GNU Make 4.4.1** as `mingw32-make.exe`, but this does *not* rescue
  the example Makefiles (POSIX assumptions + WSL hooks).
- HuggingFace is reachable, so M3 can fetch the model itself.
- The `iron` conda env lacks transformers/safetensors/huggingface_hub/mteb → decided to
  create a separate `npuembed-ref` env in M3 rather than risk the working toolchain env.

**2. Research indexed.** All 7 PDFs read once and summarised into
`research/papers/<id>.md`, with `manifest.json` (sha256 + summary path) for new-file
detection and `INDEX.md` as the human view. Three cross-cutting findings (F1 dispatch
overhead, F2 batching/bandwidth, F3 attention isn't the bottleneck) were extracted and
now drive the architecture.

**3. Prior art recorded** in `research/prior-art.md`, including the two findings that
most affect strategy: FastFlowLM's model kernels are **closed prebuilt binaries** with
contradictory licence terms (so kernels cannot be contributed as source today), and
AMD's ONNX/VitisAI path has an **open hang bug on this exact SKU** (`RyzenAI-SW` #312).

**4. Docs written** — `00-overview`, `01-hardware`, `02-toolchain`, `03-kernels`
(stub + inventory), `04-model`, `05-measurement`.

**5. Decided against upgrading the mlir-aie checkout** — see "Decisions" below.

## Commands

```powershell
# Environment verification
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -products * -format json
C:\Windows\System32\AMD\xrt-smi.exe examine
Get-ChildItem "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC"
foreach ($t in @("cmake","ninja","clang","cl","git","python")) { (Get-Command $t -EA SilentlyContinue).Source }
foreach ($t in @("make.exe","mingw32-make.exe","gcc.exe","g++.exe")) { Test-Path "C:\msys64\mingw64\bin\$t" }
"C:/msys64/mingw64/bin/mingw32-make.exe" --version

# Model reachability + architecture confirmation
curl -sS -L https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/raw/main/config.json

# What's already in the iron env
C:\Users\vegar\.conda\envs\iron\python.exe -c "import importlib.util as u; ..."

# Paper hashes for the manifest
Get-ChildItem OthersResarch\*.pdf | ForEach-Object {
  "{0}|{1}|{2}" -f $_.Name, (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Length }

# mlir-aie checkout state + capturing local modifications
cd C:\dev\mlir-aie
git log --oneline -1 ; git describe --tags ; git status -sb ; git status --porcelain
git diff -- programming_examples/ > <repo>\tasks\0001-.../artifacts\mlir-aie-local-example-mods.diff
git status --porcelain           > <repo>\tasks\0001-.../artifacts\mlir-aie-working-tree-status.txt
```

## Result

Repo now contains:

```
CLAUDE.md
docs/{00-overview.md, 01-hardware/, 02-toolchain/, 03-kernels/, 04-model/, 05-measurement/}
research/{README.md, prior-art.md, papers/{INDEX.md, manifest.json, <7 summaries>.md}}
tasks/{README.md, 0001-scaffold-and-research-index/}
```

Key verified facts now written down: XDNA2/AIE2P 8×4=32 tiles, 64 KB L1, 512 KB L2×8;
bf16 256 MACs/cycle/core; attainable **14.7 TOPS bf16 / 38 TOPS int8** (not 50);
**~40–60 GB/s** NPU DRAM read bandwidth; DMA 4-byte min stride and 128-byte payload
divisibility; matmul geometry `4×8×8` native / `8×8×8` bfp16-emulated.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| Initial probe reported "XRT not installed" | Looked only in `C:\Program Files\Xilinx\XRT` | It is at `C:\Xilinx\XRT`; corrected in `docs/02-toolchain/` |
| Assumed bfp16 (512 MACs/cycle) was usable | Confused the *hardware* capability with *toolchain* access — native bfp16 kernels require Chess | Recorded as unavailable; `--emulate-bf16-mmul-with-bfp16` is the Peano path. **Worked around, not solved** |
| Assumed no `make` meant example Makefiles were dead | MinGW `make` exists | Still effectively dead — POSIX assumptions and WSL hooks. Use `run_example.py`. **Worked around** |
| Wanted upstream `main` docs | Local checkout is 6 weeks stale | Extracted the needed content without upgrading — see Decisions |

## Decisions

**Do not upgrade the `C:\dev\mlir-aie` checkout to `main`.** It sits at `v1.3.4`
(commit `ed23bba`), detached HEAD, with 33 dirty entries.

Rationale:

1. **The toolchain is the pip wheel, not the checkout.** `mlir-aie 1.3.4` and
   `llvm-aie 21.0.0` in `ironenv` supply every binary (`aiecc.exe`, Peano `clang++`).
   The checkout supplies only examples, `run_example.py`, `aie_kernels/` sources, the
   programming guide, and `skills/`. Moving the checkout to main while the wheel stays
   at 1.3.4 risks examples calling APIs the wheel lacks — the more likely failure mode
   than anything 1.3.4 is missing.
2. **Local modifications may be load-bearing** — `01_SAXPY/saxpy.{cc,py}` and
   `matrix_multiplication_single_core.py` are edited, sitting alongside the trace JSONs
   that prove the flow worked.
3. **The only known benefit was already harvested.** Upstream's improved
   `buildHostWinNative.md` switches to the prebuilt XRT SDK — and `C:\Xilinx\XRT`
   already matches that layout.
4. **M1 exists to establish a baseline.** Changing the toolchain first is backwards.

If upgraded later, do it as a deliberate task: bump wheel **and** checkout together,
stash the example edits, re-run M1, compare traces.

## Artifacts

- `artifacts/mlir-aie-local-example-mods.diff` (10,372 B) — the local example edits,
  captured because they exist nowhere else and are not under version control anywhere.
- `artifacts/mlir-aie-working-tree-status.txt` — full dirty-state listing.

### Notable discovery in that listing

`programming_examples/getting_started/03_matrix_multiplication_single_core/` already
contains a **traced GEMM shape sweep** done on this machine:

```
trace8x64x128.json    trace16x32x256.json   trace32x32x64.json   trace32x32x128.json
trace32x64x128.json   trace64x32x128.json   trace64.json         trace128x32x32.json
trace256x256.json     trace_256x256x256.txt trace_512x512x512.txt
input_with_addresses.mlir
```

Plus `01_SAXPY/` has `trace{,32,64,64_2,128,256,512}.json` and its
`input_with_addresses.mlir`.

**This is pre-existing M2 data.** Mine it before re-running anything: it shows how
single-core matmul already behaves across shapes on this exact silicon, and the
matching `input_with_addresses.mlir` files mean the traces are parseable today. Open
question for the user: were these exploratory, or tuned toward something specific?

## Next

**M1 — Hello NPU.** Run `01_SAXPY` and `vector_scalar_mul` natively, produce
`.xclbin` + insts, and get a trace out end-to-end through
`parse.py` → `get_trace_summary.py`. Gate: non-empty `trace.txt` becoming real cycle
counts.

Two things to fold in:
- Re-parse the existing `01_SAXPY` traces first — a zero-cost dry run of the
  measurement pipeline against known-good data, before generating anything new.
- Confirm whether the modified `saxpy.py`/`saxpy.cc` still run, since the diff is
  local and unversioned.
