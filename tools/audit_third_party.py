# NpuEmbeddings -- how much of this repository is somebody else's code?
# SPDX-License-Identifier: Apache-2.0
#
# Several designs and kernels here started life as mlir-aie examples. That is
# fine -- mlir-aie is Apache-2.0 WITH LLVM-exception and this repository is
# Apache-2.0 -- but it is only fine if the attribution is ACCURATE, and
# attribution rots: a file gets rewritten until nothing upstream remains, or a
# new file gets pasted in and nobody adds a header.
#
# So this measures it instead of remembering it, and regenerates THIRD-PARTY.md
# from what it finds.
#
# THE METRIC, AND WHY THE OBVIOUS ONE IS WRONG
# --------------------------------------------
# Counting shared LINES massively over-reports. Every AIE kernel contains
# `#include <aie_api/aie.hpp>`, `event0();`, `using namespace aie;` and a
# handful of `aie::begin_restrict_vector<16>(...)` calls, so two files that
# share nothing but the API score 30-40%. Measured here: our own fp32 probe
# scored 41% against AMD's gelu.cc while having no relationship to it.
#
# What separates copying from idiom is CONTIGUITY. Copied code arrives in
# runs; idiom arrives scattered. The longest common run of non-comment lines
# tells them apart cleanly:
#
#     gelu_control.cc   36 of 41 lines in one run   -> a copy, and says so
#     saxpy.cc          20 of 28                    -> heavily derived, says so
#     gemm_pretiled.py  11 of 574                   -> scaffolding, says so
#     fp32_probe.cc      5 of 54                    -> idiom. Ours.
#     exp2_probe.cc      2 of 20                    -> idiom. Ours.
#
# A file above the threshold without an attribution header FAILS the audit.
#
# Usage:
#   python tools\audit_third_party.py --upstream C:\dev\mlir-aie
#   python tools\audit_third_party.py --upstream ... --write

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

OUR_DIRS = ["experiments", "reference", "tools", "runtime/src", "runtime/include"]
UP_DIRS = ["programming_examples", "aie_kernels", "python"]
EXT = {".py", ".cc", ".cpp", ".h", ".hpp"}

# Lines in the longest shared run. Below this, an overlap is API idiom; above
# it, structure was taken. Calibrated on the cases above, not guessed.
RUN_THRESHOLD = 8

ATTRIB_MARKERS = ("Advanced Micro Devices", "LLVM-exception", "Derived from")


def norm(p: Path) -> list[str]:
    """Comparable form: code only, no comments, no blank lines, no indent."""
    out = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "//", "*")):
            continue
        out.append(re.sub(r"\s+", " ", s))
    return out


def longest_run(a: list[str], b: list[str]) -> int:
    m = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return max((blk.size for blk in m.get_matching_blocks()), default=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=r"C:\dev\mlir-aie",
                    help="an mlir-aie checkout to compare against")
    ap.add_argument("--write", action="store_true",
                    help="regenerate THIRD-PARTY.md")
    ap.add_argument("--threshold", type=int, default=RUN_THRESHOLD)
    args = ap.parse_args()

    up_root = Path(args.upstream)
    if not up_root.exists():
        print(f"upstream not found: {up_root}")
        print("Pass --upstream <path to an mlir-aie checkout>. Without one "
              "this audit cannot run, and its absence is not a pass.")
        return 2

    upstream = []
    for d in UP_DIRS:
        root = up_root / d
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if (f.suffix in EXT and f.is_file()
                    and "ironenv" not in str(f) and "site-packages" not in str(f)):
                lines = norm(f)
                if len(lines) >= 8:
                    upstream.append((f, lines, set(lines)))
    print(f"upstream: {len(upstream)} files under {up_root}")

    findings = []
    for d in OUR_DIRS:
        root = REPO / d
        if not root.exists():
            continue
        for f in sorted(root.rglob("*")):
            if f.suffix not in EXT or not f.is_file():
                continue
            if "build" in f.parts or "artifacts" in f.parts:
                continue
            mine = norm(f)
            if len(mine) < 8:
                continue
            mset = set(mine)
            best = None
            # Prefilter on shared lines, then measure contiguity on the top
            # candidates -- a full pairwise SequenceMatcher over 400 upstream
            # files per source file is minutes, this is seconds.
            cands = sorted(upstream,
                           key=lambda u: -len(mset & u[2]) / max(1, len(mset)))[:8]
            for uf, ulines, _ in cands:
                run = longest_run(mine, ulines)
                if best is None or run > best[1]:
                    best = (uf, run)
            if best is None or best[1] < 3:
                continue
            text = f.read_text(encoding="utf-8", errors="ignore")
            findings.append({
                "file": str(f.relative_to(REPO)).replace("\\", "/"),
                "upstream": str(best[0].relative_to(up_root)).replace("\\", "/"),
                "run": best[1],
                "lines": len(mine),
                "attributed": any(m in text for m in ATTRIB_MARKERS),
            })

    findings.sort(key=lambda r: -r["run"])
    derived = [r for r in findings if r["run"] >= args.threshold]
    missing = [r for r in derived if not r["attributed"]]

    print(f"\n{'run':>4} {'lines':>6}  {'attr':^6}  file")
    for r in findings:
        if r["run"] < 3:
            continue
        mark = "ok" if r["attributed"] else ("MISSING" if r["run"] >= args.threshold
                                             else "-")
        print(f"{r['run']:>4} {r['lines']:>6}  {mark:^6}  {r['file']}")
        if r["run"] >= args.threshold:
            print(f"{'':20}  <- {r['upstream']}")

    if args.write:
        out = REPO / "THIRD-PARTY.md"
        lines = [
            "# Third-party code",
            "",
            "This repository is Apache-2.0. Some of its NPU designs and kernels",
            "began as examples from **[MLIR-AIE](https://github.com/Xilinx/mlir-aie)**,",
            "which is **Apache-2.0 WITH LLVM-exception** — a compatible licence that",
            "adds a permission rather than a restriction. Those files keep their",
            "original copyright headers and say what was changed.",
            "",
            "This file is generated by `tools/audit_third_party.py`, which measures",
            "the longest contiguous run of shared non-comment lines against an",
            "mlir-aie checkout. Shared *line counts* are misleading here: every AIE",
            "kernel includes the same headers and calls the same API, so unrelated",
            "files score 30–40%. Contiguity separates copying from idiom.",
            "",
            f"Files with a shared run of **{args.threshold} lines or more**:",
            "",
            "| file | longest shared run | of our | upstream origin |",
            "|---|---:|---:|---|",
        ]
        for r in derived:
            lines.append(f"| `{r['file']}` | {r['run']} | {r['lines']} | "
                         f"`{r['upstream']}` |")
        lines += [
            "",
            "Everything else in this repository is original work. Files below the",
            "threshold share only API idiom — the same includes, the same",
            "`aie::` calls, the same `event0()`/`event1()` bracketing — which is",
            "what using a library looks like.",
            "",
            "### Copyright",
            "",
            "> Portions Copyright (C) 2024–2026 Advanced Micro Devices, Inc.  ",
            "> SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
            "",
            "See each file's header for the specific origin and the changes made.",
            "",
        ]
        out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        print(f"\nwrote {out.relative_to(REPO)} ({len(derived)} files listed)")

    if missing:
        print(f"\nFAIL -- {len(missing)} file(s) share {args.threshold}+ "
              f"contiguous lines with upstream and carry no attribution:")
        for r in missing:
            print(f"  {r['file']}  ({r['run']} lines from {r['upstream']})")
        print("\nAdd a header naming the origin, its copyright and licence, and "
              "what you changed.")
        return 1

    print(f"\nPASS -- {len(derived)} derived file(s), all attributed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
