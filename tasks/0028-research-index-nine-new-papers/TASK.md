# 0028 — Index nine new documents in `OthersResarch/`

- **Date** 2026-08-17
- **Milestone** M7 / M8 prep (research only — no code, no hardware runs)
- **Status** done

## Goal

The user dropped new files into `OthersResarch/`. Per `CLAUDE.md` rule 2 and
`research/README.md`: detect which are new, read only those,
write a summary per document that is a genuine substitute for the PDF, and update
`manifest.json` + `INDEX.md`.

Gate: `manifest.json` covers every PDF in `OthersResarch/`, every hash matches, every
summary file exists, and no future session ever needs to open these PDFs again.

## Commands run

**1. New-file detection.** The PowerShell snippet in `research/README.md` was replaced
with an equivalent shell run, because the shell is faster here and hashing is the same:

```bash
cd OthersResarch && sha256sum *.pdf
```

Cross-checked against `research/papers/manifest.json`. **All 7 previously indexed PDFs
still hash-match** — nothing was silently replaced. **9 PDFs were new.**

**2. Page counts, without opening anything.** `pdftoppm` is not installed, so the `Read`
tool's PDF path (`pdftoppm` → page images) fails outright:

```
pdftoppm is not installed. Install poppler-utils ... to enable PDF page rendering.
```

`PyMuPDF` (`fitz`) *is* present in the conda `iron` env — nothing else is
(`pypdf`, `PyPDF2`, `pdfminer`, `pdfplumber` all absent). So text extraction went through
a 10-line script written to the scratchpad:

```bash
"C:/Users/vegar/.conda/envs/iron/python.exe" "$SP/pdftxt.py" "<file>.pdf" "$SP/<file>.txt"
```

`pdftxt.py` is `fitz.open()` → `page.get_text("text")` per page with a
`===== PAGE n / N =====` separator, so page numbers survive into the extract and the
summaries can cite them. Extraction results:

| File | Pages | Chars |
|---|--:|--:|
| 2507.14403v1 | 19 | 47,207 |
| 25_11_13_steinert_msc_thesis | 41 | 90,132 |
| 2605.01124v1 | 32 | 117,658 |
| 2606.07586v2 | 6 | 27,893 |
| 2607.09385v1 | 6 | 31,959 |
| 3706628.3708870 (ARIES) | 11 | 76,921 |
| 3754598.3754612 (ICPP) | 10 | 57,999 |
| Exploration Of Real-Time Power Electronic Simulation | 55 | 80,508 |
| **getting_peak** | **42** | **1,083** |

**3. `getting_peak.pdf` needed a different route.** 1,083 characters over 42 pages means
no text layer. Its PDF metadata gave it away:

```
{'title': 'Getting peak TOPS on a Ryzen AI 7 350 NPU – Daniel Estévez',
 'producer': 'Microsoft: Print To PDF', 'author': 'Vegard Berget'}
```

It is a *Print To PDF* capture of a blog post, drawn as glyph paths. Two options: render
42 pages to PNG and read them visually (expensive), or read the live article. Chose the
article, then **verified the identity** by rendering four pages at 100 dpi with
`page.get_pixmap(dpi=100)` and reading them: page 3 (the AIE generation naming table and
the 13 May update about `aie2p` having 5 accumulators), page 7 (tile connectivity), page 8
(the VLIW functional-unit list), page 41 (the final TOPS calculation). All four matched
the article, including the exact numbers 539 cycles / 1.808 GHz / 2.501 s / 56.28 TOPS /
58.9824 theoretical / 95%. The summary records both the URL and the fact that the PDF
is textless, so nobody wastes time on it again.

**4. Summaries written**, one per document, to `research/papers/`.

**5. Manifest + index updated**, then validated mechanically:

```bash
"C:/Users/vegar/.conda/envs/iron/python.exe" -c "<validate manifest vs disk>"
# papers: 16
# problems: 0
```

The validator checks four things per entry — file exists, sha256 matches, byte count
matches, summary file exists — and then sweeps `OthersResarch/*.pdf` for anything
**not** in the manifest. Worth keeping; it is stronger than the README's two snippets
because it also catches a manifest entry whose summary was never written.

## Results

Nine summaries, relevance rated against *this* project:

| Summary | Rel. | The one thing it gives us |
|---|:--:|---|
| `icpp25-aie-mmm-models.md` | 5 | The XDNA2 constant table + register/local tile model, and the finding that **AMD's stock IRON matmul is Stationary C** — the weaker algorithm |
| [`2607.09385.md`](https://arxiv.org/abs/2607.09385) (STEEL) | 5 | **Mem-tile port budget: 6 in / 6 out, 48 total**; fused vs layer-by-layer = **22.8×** |
| [`2606.07586.md`](https://arxiv.org/abs/2606.07586) | 4 | AMD on our SKU: **15 dispatches/layer → 3** by merging pre- and post-attention blocks |
| `steinert-msc-tensor-contractions.md` | 4 | VLIW-level kernel craft at **92.8% of peak**; transposed-output costs 18.7 pts; **power mode swings 22 pts** |
| `destevez-peak-tops.md` | 4 | **95% of peak on 32 tiles**; `aie2p` has **5 accumulators, not 8** |
| [`2507.14403.md`](https://arxiv.org/abs/2507.14403) (NPUEval) | 3 | **Peano silently ignores `chess_*` pragmas**; SotA AIE kernels score 10–30% vectorisation |
| `fpga25-aries.md` | 3 | Scalar on-chip dataflow beats the vendor overlay **1.24×** before any optimisation |
| `clemson-msc-power-electronics-npu.md` | 3 | fp32 on our chip is **compute-bound 8:1** and is **emulated** |
| [`2605.01124.md`](https://arxiv.org/abs/2605.01124) (PEQC-MLIR) | 2 | A **real, test-invisible race** in the MLIR-AIE matmul's mem-tile lock protocol |

### Corrections these force to our own documents

1. **Trap 3's L1 budget is wrong by 1 KB.** `CLAUDE.md` says
   `2*(m*k*in + k*n*in + m*n*out) < 65536`. The ICPP paper and the Steinert thesis both
   state **1 KB of the 64 KB DMEM is reserved for the program stack**, leaving 63 KB.
   Also, that inequality is the *Stationary C* form; Stationary B's is
   `2mk·T_in + kn·T_in + 2mn·T_out`.
2. **`R_acc` is 40 × 256-bit = 5 × 2048-bit accumulators**, and the AIE-MLv2 documentation
   saying 8 is wrong for `aie2p`. Two independent sources (llvm-aie
   `AIE2PRegisterInfo.td#L466` via Estévez; the ICPP hardware table). With bf16's
   `C_v = 8`, that means **at most 5 independent MMAC accumulators per core** — a
   constraint we have never checked our kernels against.
3. **Power mode belongs in every measurement.** The Steinert thesis measures turbo
   performing **>22 percentage points worse** than balanced for one data layout. Our
   `docs/05-measurement/` rules do not mention power mode at all, and we have 22%
   unexplained run-to-run spread on record from
   [`0007`](../0007-m5-pretiled-gemm-on-npu/TASK.md).
4. **`chess_*` pragmas in our kernels are dead weight.** We are Peano-only. NPUEval shows
   Peano accepts and then discards them, and measured a kernel dropping 47% → 26%
   vectorisation because of exactly this.

### What did *not* change

Nothing in these nine contradicts a measured result of ours. The two that come closest
to our open questions both **support** our position: the Clemson thesis independently
finds fp32 compute-bound on the same silicon (our
[`0026`](../0026-m7-closing-on-cpu/TASK.md) passthrough-vs-GELU 15× gap), and Estévez's
95%-of-peak run rules out "the array is hard to saturate" as an explanation for our
GEMM's 58–59%.

## Problems hit

- **`Read` cannot open PDFs on this machine.** It shells out to `pdftoppm` (poppler),
  which is not installed. This will recur every time a PDF arrives. The workaround is
  PyMuPDF from the conda `iron` env; `pdftxt.py` is preserved in the scratchpad but is
  short enough to retype. *Do not* `pip install` poppler-adjacent things into the IRON
  env — see the environment rules in `CLAUDE.md`.
- **`strings` does not exist** in this Git Bash. First attempt at page counting via
  `strings file.pdf | grep -c '/Type /Page'` returned 0 for every file, which looks like
  a valid answer rather than a failure. Caught only because *every* file returned 0.
  Counting `/Type /Page` in raw bytes with Python also returns 0 for PDFs using
  compressed object streams (4 of 9 here). `fitz.open()` → `len(doc)` is the only
  reliable count.
- **Printing extracted text through Python's stdout fails on cp1252.** `UnicodeEncodeError`
  on `\ufb01` (the "fi" ligature). Write to a UTF-8 file and read the file instead of
  printing.
- **The ARIES extract exceeded the `Read` tool's 25k-token page cap** and had to be read
  in three ranges. Page-offset lookup by searching for the `===== PAGE n` markers and
  counting newlines made this cheap.

## Not indexed

`OthersResarch/github-scan-research.md` (22,947 bytes) is **not a paper** — it is a scan
of the mlir-aie/IRON GitHub ecosystem (Triton-XDNA, amd/IRON, mlir-air, xdna-driver,
Riallto, NPUEval, ARIES, TileFuse, FastFlowLM, open-xdna …) with a note that
**Triton-XDNA is native-Windows with prebuilt wheels for Python 3.10–3.14**. It belongs
alongside `research/prior-art.md` rather than in the paper
manifest, and it has not been merged there yet. `manifest.json`'s `$comment` now records
that non-PDF files in the inbox are out of scope for it.

## Next

Not done here, deliberately — these are code/measurement tasks, not research ones:

1. `grep` the kernel sources for `chess_` and delete what Peano ignores.
2. Check our GEMM kernel against `(m_r·n_r)·C_v ≤ R_acc = 40`.
3. Add power mode to the measurement protocol and re-check `0007`'s spread.
4. Correct trap 3's 65536 → 63 KB in `CLAUDE.md`.
5. The big one: **fuse encoder layers into single dispatches.** Four independent sources
   added today all point at it, and [`2606.07586`](https://arxiv.org/abs/2606.07586)
   gives the exact block boundaries to fuse along.
