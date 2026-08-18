# NpuEmbeddings -- the CPU side of the energy comparison (tasks/0034).
# SPDX-License-Identifier: Apache-2.0
#
# Encodes a fixed batch N times and exits. It exists so the energy harness can
# run the CPU path with the SAME differential method it uses on the NPU path:
# measure at two encode counts and subtract, which removes model load, thread
# pool spin-up and interpreter startup from the figure entirely.
#
# Env: .venv-ref  (torch, sentence-transformers)
# Usage:
#   & ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\energy_cpu_load.py --encodes 20 --batch 128

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encodes", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    sys.path.insert(0, str(REPO / "reference"))
    from corpus import SENTENCES, SEQ_LEN

    st = SentenceTransformer(str(REPO / "models" / "all-MiniLM-L6-v2"),
                             device="cpu")
    st.max_seq_length = SEQ_LEN
    sents = (SENTENCES * ((args.batch // len(SENTENCES)) + 1))[:args.batch]

    st.encode(sents, batch_size=args.batch, convert_to_numpy=True)   # warm
    t0 = time.perf_counter()
    for _ in range(args.encodes):
        st.encode(sents, batch_size=args.batch, convert_to_numpy=True,
                  normalize_embeddings=True)
    el = time.perf_counter() - t0
    print(f"  {args.encodes} encodes of {args.batch} in {el:.2f} s  ->  "
          f"{args.encodes * args.batch / el:.1f} seq/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
