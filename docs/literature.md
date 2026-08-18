# What we read, and what we did with it

This project did not start from scratch. Sixteen papers, theses and write-ups
shaped its architecture, and three of them changed the design outright. This
document records **which source led to which decision**, so a reader can follow
the reasoning back to its origin — and check whether we drew the right
conclusion.

It is a reading list with our own commentary, not a literature review. Each
entry says what the work is in a sentence, links to it, and then spends its
words on the part that is ours: what we took, what we ignored, and where our
own measurements later agreed or disagreed. For the papers themselves, follow
the links — they are the authoritative source and we are not trying to
substitute for them.

> Where a claim below is attributed to a source, it is that source's result,
> not ours. Where it says "we measured", it is in [`tasks/`](../tasks/README.md)
> with a command and an artifact.

---

## The three findings that drove the architecture

Each is supported by more than one independent source, which is why we trusted
them enough to build on.

### F1 — Per-dispatch and per-reconfiguration overhead dominates, not kernel throughput

The most consequential thing we learned before writing a kernel. Several
groups, independently:

- **TileFuse** ([2606.11357](https://arxiv.org/abs/2606.11357), UIUC) gives the
  most candid accounting of dispatch overhead we found, and reports the NPU
  *losing to the integrated GPU* at short prompts. A 256-token encode sits
  exactly in that regime — which is what made this a risk to the whole project
  rather than an optimisation note.
- **MLIR-AIR** ([2510.14871](https://arxiv.org/abs/2510.14871), AMD Research)
  reports 2.24× from fusing five dispatches into one.
- **AMD + Cornell** ([2606.07586](https://arxiv.org/abs/2606.07586)) state
  plainly that dispatch overhead can exceed kernel execution time, and cut 15
  dispatches per layer to 3 on *our exact SKU*.
- **STEEL** ([2607.09385](https://arxiv.org/abs/2607.09385), ETH/AMD) measures
  fused attention at 22.8× over the layer-by-layer equivalent, crediting one
  design load and never materialising intermediates off-chip.
- **ARIES** (FPGA '25, [doi](https://doi.org/10.1145/3706628.3708870)) is the
  cleanest control: on-chip dataflow beat a vendor operator-at-a-time overlay
  **1.24× using scalar, unvectorised code on fewer cores**. Kernel quality held
  constant and worse; structure alone won.
- **Estévez** ([blog](https://destevez.net/2026/05/getting-peak-tops-on-a-ryzen-ai-7-350-npu/))
  reached 95% of theoretical peak on all 32 tiles from IRON and Peano — with a
  design that dispatches once and moves no data. That closes off "the array is
  simply hard to saturate" as an excuse.

**What we did.** Batch, keep one resident design, and drive operations as
instruction streams rather than separate designs. Our own measurements sharpened
the finding: the expensive event is not the dispatch (~150 µs) but *changing
which design the array is configured for* (2.2–2.6 ms), which we traced to
descriptor count rather than array width
([note 0004](../research/notes/0004-context-switch-cost.md)). Removing all 49
switches per encode took us from 305 to 611 seq/s
([`tasks/0032`](../tasks/0032-m7-one-xclbin-production/TASK.md)).

### F2 — Batching is mandatory, and bandwidth is the objective

**Gemma3 / FastFlowLM** ([2602.06063](https://arxiv.org/abs/2602.06063), Clemson
and URI) gives the best public tile-size-to-throughput calibration for XDNA2
bf16 GEMM, and its practical lesson is that small M leaves most of the array's
throughput unused. Both it and **FastTPS**
([2607.11211](https://arxiv.org/abs/2607.11211), AMD) argue for optimising
*measured bandwidth utilisation* rather than TOPS.

**What we did.** Batching, and a cost model fitted to our own hardware rather
than borrowed. We are bandwidth- and dispatch-bound, never compute-bound: our
GEMM sits at 58–59% of MAC peak while the array waits for data
([`tasks/0010`](../tasks/0010-m5-b-reuse-and-cost-model/TASK.md)).

### F3 — bf16 with fp32 accumulation is safe for embedding quality

Two independent confirmations — **Rösti**
([2504.03083](https://arxiv.org/abs/2504.03083), UC Irvine, FCCM '25) at under
0.1% divergence from an fp32 CPU reference, and **FastTPS** at ~1e-3 max error
*improving* with sequence length.

**What we did.** bf16 in, fp32 accumulate, from the first kernel — and we set
our acceptance thresholds against those numbers instead of guessing. Our own
end-to-end result is `1-cos` 1.086e-05 against HuggingFace, and MTEB cannot
distinguish our embeddings from the CPU's
([`tasks/0035`](../tasks/0035-m8-mteb-gate/TASK.md)).

---

## The sources, and what each one gave us

### Directly about our hardware

**Zen-Attention** ([2508.17593](https://arxiv.org/abs/2508.17593)) — AMD, on our
exact SKU, and the only work we found that benchmarks BERT on XDNA2. Its most
useful result is negative: full attention folding bought about 1.4% end to end.
That told us where *not* to spend effort, and our own FLOP split agrees
(attention is ~5% of the work at our sequence length). It is why attention
remains on the host in this implementation and why we are unapologetic about
that.

**AMD + Cornell** ([2606.07586](https://arxiv.org/abs/2606.07586)) — a deployment
methodology paper on our SKU. Its host-side checklist (reuse the XRT context,
recycle buffer objects, skip redundant transfers, choose layouts so consecutive
kernels hand off without a transpose) matched what we had already implemented,
which was a useful signal that we had not missed a category. Its dispatch-merging
result is part of F1 above.

**STEEL** ([2607.09385](https://arxiv.org/abs/2607.09385)) — an open-source
FlashAttention for XDNA2, written in the same IRON bindings we use. Two things we
took: the memory-tile port budget, which explained two of our own dead ends in
retrospect, and the insistence on balancing pipeline stages by *measured* work
rather than conceptual tidiness. If attention ever moves onto the array here,
this is the starting point rather than a blank page.

**Gemma3 / FastFlowLM** ([2602.06063](https://arxiv.org/abs/2602.06063)) — the
paper behind FastFlowLM, and the closest existing codebase to what we are doing.
Its encoder-layer floorplan (dedicate cores to the elementwise ops, keep vectors
in L2 between stages) is a design we costed carefully and then *rejected* on our
own measurements: at hidden 384 the elementwise ops are too small to earn a
dispatch, and moving them to the host was both faster and more accurate. A good
example of a sound idea that does not survive a different shape of model.

**Clemson MSc** (Du, 2026, [Clemson OPEN](https://open.clemson.edu/all_theses/4790/))
— a non-ML fp32 workload on our exact chip, which independently measures fp32 as
compute-bound against its own data movement. That is the same phenomenon our
`tasks/0026` found from the opposite direction, and it is part of why we stopped
trying to optimise fp32 elementwise kernels on the array.

### GEMM and kernel construction

**ICPP '25 / Binder et al.** ([doi](https://doi.org/10.1145/3754598.3754612), CMU)
— the most directly useful paper for the GEMM itself: a first-principles model for
choosing register and local tile sizes on this architecture, and the observation
that the stock IRON matmul uses the weaker of two algorithms. It also corrected
two constants we had been carrying wrong. We have not yet tried the stronger
algorithm; it remains the highest-value unclaimed GEMM experiment here.

**Rösti & Franz** ([2504.03083](https://arxiv.org/abs/2504.03083)) — the closest
match to our setup and the source of the single most transferable idea in this
project: keep one static configuration and vary only the runtime sequence. We
implemented exactly that, and it is what made zero-switch encoding possible
([`tasks/0029`](../tasks/0029-m7-one-xclbin-probe/TASK.md)). Their assembly
no-op inspection is also one half of our two-signal measurement rule.

**Steinert MSc** (Jena, 2025) — the deepest look inside a single core we found:
VLIW packing, hardware loops, bank conflicts, and a demonstration of what near-peak
looks like when every instruction carries a MAC. Two things stuck: an arithmetic-
intensity threshold that predicts when a shape can reach high efficiency, and the
warning that **power mode is a first-class experimental variable**, not a monotonic
knob. We record it with measurements for that reason.

**Estévez** ([blog](https://destevez.net/2026/05/getting-peak-tops-on-a-ryzen-ai-7-350-npu/))
— besides the peak-TOPS control experiment above, it corrected an architectural
constant we would otherwise have designed against wrongly, and explained *why*
interleaved accumulator chains matter. Our own 4-chain kernels were found
empirically first; this is the reason they work.

### Precision and quantisation

**FastTPS** ([2607.11211](https://arxiv.org/abs/2607.11211)) — decode-phase work
that an encoder does not have, but its methodology is worth copying: classify every
operation as compute or memory reorganisation, put them on a roofline, and target
bandwidth utilisation. Its fused-MLP principle — fold the elementwise nonlinearity
into the matmul epilogue so intermediates never leave L1 — is exactly what we later
built and measured
([`tasks/0030`](../tasks/0030-m7-expert-review-tests/TASK.md)).

**TileFuse** ([2606.11357](https://arxiv.org/abs/2606.11357)) — besides F1, the
source of our decision to pre-tile weights offline. Our weights are static, so
there is no reason to pay a runtime layout cost. We later measured pre-tiling as a
*throughput* wash on our shapes, and kept it anyway for a different reason: it lets
the runtime hand mapped bytes straight to a DMA descriptor
([`tasks/0007`](../tasks/0007-m5-pretiled-gemm-on-npu/TASK.md)).

### Tooling, verification and cautionary tales

**NPUEval** ([2507.14403](https://arxiv.org/abs/2507.14403), AMD) — an LLM
kernel-generation benchmark, which is not our problem, but it contains one trap we
would have hit blind: the Chess compiler pragmas that appear throughout AIE example
code are **silently ignored** by the compiler we use. Copied example code can look
tuned and not be. It also reports that state-of-the-art open AIE kernels score low
on vectorisation, which is a useful calibration against despair.

**PEQC-MLIR** ([2605.01124](https://arxiv.org/abs/2605.01124), Georgia Tech,
Colorado State, AMD) — formal verification work whose relevance to us is one
finding: a real, test-invisible race in the generated memory-tile lock protocol, in
the code family our GEMM descends from. We note it because our own failure modes
have been *consistently* wrong results; this one would be *occasionally* wrong, and
we would not find it by testing.

**IRONSmith** ([2607.10944](https://arxiv.org/abs/2607.10944), Arizona State) — a
visual design environment we do not use, kept for its structural-error checklist.
DMA channel oversubscription is the failure we hit most often writing IRON by hand,
and having it named in advance shortened several debugging sessions.

**ARIES** (FPGA '25) — mostly about a different device, but its ablation table is
the template we follow when reporting our own optimisation work: one change per
row, runtime and utilisation both shown, baseline at the bottom. It also warns that
utilisation and runtime are only loosely coupled, which matches our experience that
a bigger tile is not automatically better.

---

## Where we ended up disagreeing

Worth recording, because it is the part a reading list usually omits.

- **The elementwise floorplan** from the FastFlowLM paper is right for their model
  and wrong for ours. At hidden 384 the elementwise work is roughly 1 : 3h of the
  MAC work, so it cannot amortise a design switch. We measured the crossover
  behaviour rather than assuming the published layout transferred
  ([`tasks/0027`](../tasks/0027-m7-width-hypothesis/TASK.md)).
- **"Fuse everything onto the array"** is the direction every source points, and we
  followed it — up to the point where our own measurements said the opposite for
  three specific operations. Moving LayerNorm, softmax and GELU to the host made the
  model faster *and* more accurate. Published guidance is about a workload shape,
  not a law.
- **Our own published claim about an accuracy cliff was wrong.** We predicted from
  simulation that block floating point would be 6× worse on real activations, and
  hardware said 0.85× ([`tasks/0008`](../tasks/0008-m5-bfp16-real-data/TASK.md)).
  The lesson generalises to reading anyone's simulation results, including ours.

## A note on this document

The working repository this was extracted from keeps fuller internal summaries of
each source, written so the team could avoid re-reading the papers. Those are not
republished: they are close enough to the originals to function as substitutes,
which makes them exactly the wrong thing to put in a public repository. What is
here instead is the part that was ours to write — the conclusions, and what
happened when we tested them.
