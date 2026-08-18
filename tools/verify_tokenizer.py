# NpuEmbeddings -- is the C++ tokenizer the same function as HuggingFace's?
# SPDX-License-Identifier: Apache-2.0
#
# A tokenizer is not "probably right". It is a pure function from bytes to ids,
# so the only honest test is to run both over the same inputs and compare every
# id -- and to choose inputs that attack the parts most likely to diverge
# rather than inputs that all take the happy path.
#
# THE CORPUS IS ADVERSARIAL ON PURPOSE
# ------------------------------------
# Each group targets a specific way BERT tokenizers are known to go wrong:
#   accents      strip_accents inherits do_lower_case; getting the inheritance
#                backwards is invisible on ASCII
#   combining    the one place this implementation deliberately skips a step
#                (HuggingFace's NFC pass) -- so it gets the most cases
#   CJK          Han is padded per codepoint; Hiragana/Katakana/Hangul are NOT
#   punctuation  every punctuation char splits into its own token
#   subword      out-of-vocabulary words must decompose into ## continuations
#   casing       including Greek final sigma, the one context-dependent case
#   degenerate   empty, whitespace-only, control chars, very long words, emoji
#
# Env: .venv-ref (transformers)
# Usage:
#   & ".\.venv-ref\Scripts\python.exe" tools\verify_tokenizer.py
#   ... --extra-file some_corpus.txt      (one text per line)

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CORPUS = [
    # --- plain -----------------------------------------------------------
    "The quick brown fox jumps over the lazy dog.",
    "hello world",
    "NPU embeddings run on the AMD Ryzen AI XDNA2 array.",
    # --- accents ---------------------------------------------------------
    "Café naïve résumé jalapeño",
    "Blåbærsyltetøy fra Sørøya, Nordkjosbotn og Ærøskøbing",
    "MOTÖRHEAD ÄÖÜ ßẞ",
    "Ångström Å å Ǻ ǻ",
    # --- combining sequences: base + mark, the NFC case -------------------
    "é à ö ñ ç",
    "Ǻ multi-mark stack",
    "ṩ ṩ (dot below then above)",
    # --- CJK: Han is split per char; kana and hangul are NOT ---------------
    "机器学习模型",
    "自然言語処理とニューラルネットワーク",
    "한국어 문장 임베딩",
    "混合 English and 中文 in one line",
    # --- punctuation ------------------------------------------------------
    "Hello, world! (test) [brackets] {braces} <tags>",
    "e-mail: user@example.com — dash…ellipsis «quotes» “smart”",
    "1+1=2, 3*4/5, 100%, #hash, $dollar, ~tilde, `tick`",
    # --- subword ----------------------------------------------------------
    "antidisestablishmentarianism pneumonoultramicroscopicsilicovolcanoconiosis",
    "unbelievableness supercalifragilisticexpialidocious",
    "tokenization wordpiece subword embeddings",
    # --- casing, incl. Greek final sigma ----------------------------------
    "ΟΔΥΣΣΕΥΣ ΚΑΙ Η ΟΔΥΣΣΕΙΑ",
    "ΑΣ ΔΟΥΜΕ ΤΟ ΤΕΛΙΚΟ ΣΙΓΜΑ: ΛΟΓΟΣ",
    "İstanbul ISTANBUL istanbul",
    "ǅ ǈ ǋ ǲ titlecase digraphs",
    # --- numbers, mixed ---------------------------------------------------
    "In 2026, 3.14159 and 1,000,000 units at 99.9%",
    "v1.2.3-rc4+build.567",
    # --- degenerate -------------------------------------------------------
    "",
    "   ",
    "\t\n mixed \t whitespace \n",
    "a" * 200,
    "x" * 99,
    "y" * 100,
    "z" * 101,
    "🙂 emoji 🚀 test 👨‍👩‍👧‍👦 family zwj",
    "​ zero width ­ soft hyphen ﻿ bom",
    "control\x00null\x07bell",
    "[CLS] [SEP] [MASK] [PAD] [UNK] literal specials",
    "##already ##prefixed",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--extra-file", default=None,
                    help="additional corpus, one text per line")
    ap.add_argument("--exe", default=str(REPO / "runtime" / "build" / "npuembed.exe"))
    ap.add_argument("--out", default=str(REPO / "tasks" / "0036-m8-tokenizer"
                                         / "verify_tokenizer.json"))
    args = ap.parse_args()

    corpus = list(CORPUS)
    if args.extra_file:
        for line in Path(args.extra_file).read_text(encoding="utf-8").splitlines():
            corpus.append(line)

    # Lines cross to C++ as a file, one text per line, so anything with a
    # newline in it cannot be represented -- strip them and say so.
    flat = [c.replace("\n", " ").replace("\r", " ") for c in corpus]
    n_flattened = sum(1 for a, b in zip(corpus, flat) if a != b)

    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(str(REPO / "models" / "all-MiniLM-L6-v2"))

    with tempfile.TemporaryDirectory(prefix="tokverify_") as td:
        inp = Path(td) / "corpus.txt"
        inp.write_text("\n".join(flat) + "\n", encoding="utf-8")
        r = subprocess.run([args.exe, str(REPO), "--tokenize", str(inp),
                            str(args.max_len)],
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            print(f"npuembed --tokenize failed ({r.returncode}):\n{r.stderr}")
            return 2
        cpp_lines = [ln for ln in r.stdout.splitlines() if ln.strip() != ""]

    if len(cpp_lines) != len(flat):
        print(f"FAIL -- C++ produced {len(cpp_lines)} lines for {len(flat)} texts")
        return 1

    n_ok, mismatches = 0, []
    for text, line in zip(flat, cpp_lines):
        want = hf(text, padding="max_length", truncation=True,
                  max_length=args.max_len)["input_ids"]
        got = [int(x) for x in line.split()]
        if got == list(want):
            n_ok += 1
        else:
            # Report where they first diverge and in which token -- an id list
            # alone does not tell you what went wrong.
            i = next((k for k in range(max(len(got), len(want)))
                      if k >= len(got) or k >= len(want) or got[k] != want[k]),
                     0)
            mismatches.append({
                "text": text[:90],
                "first_diff_index": i,
                "hf": [int(v) for v in want[max(0, i - 2):i + 4]],
                "cpp": got[max(0, i - 2):i + 4],
                "hf_tokens": hf.convert_ids_to_tokens(
                    [int(v) for v in want[max(0, i - 2):i + 4]]),
                "cpp_tokens": hf.convert_ids_to_tokens(got[max(0, i - 2):i + 4]),
            })

    total = len(flat)
    print(f"tokenizer verification: {n_ok}/{total} exact "
          f"({n_ok / total * 100:.2f}%)  max_len {args.max_len}")
    if n_flattened:
        print(f"  ({n_flattened} corpus entries had newlines flattened to spaces)")
    for m in mismatches[:12]:
        print(f"\n  MISMATCH at token {m['first_diff_index']}: {m['text']!r}")
        print(f"    HF  {m['hf']}  {m['hf_tokens']}")
        print(f"    C++ {m['cpp']}  {m['cpp_tokens']}")
    if len(mismatches) > 12:
        print(f"\n  ... and {len(mismatches) - 12} more")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kind": "verification", "task": "0036",
        "max_len": args.max_len, "n_texts": total, "n_exact": n_ok,
        "agreement": n_ok / total,
        "mismatches": mismatches,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    ok = n_ok == total
    print("PASS -- byte-for-byte identical to HuggingFace" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
