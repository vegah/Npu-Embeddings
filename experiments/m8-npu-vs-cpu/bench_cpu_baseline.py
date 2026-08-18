# NpuEmbeddings -- the CPU baseline a user would actually reach for.
# SPDX-License-Identifier: Apache-2.0
#
# sentence-transformers on CPU, same model, same corpus, same sequence length as
# reference/encode_npu.py. This is the number our NPU path has to beat to be
# worth anything, and docs/04-model's "table stakes" tier is defined against
# exactly this kind of baseline (">250 seq/s matches an i9-13900K").
#
# Wall clock on both sides, which docs/05-measurement permits for end-to-end
# throughput.
#
# Env: .venv-ref  (torch, sentence-transformers)
# Usage:
#   & ".\.venv-ref\Scripts\python.exe" experiments\m8-npu-vs-cpu\bench_cpu_baseline.py

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"
REPO = HERE.parent.parent


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer

    import sys
    sys.path.insert(0, str(REPO / "reference"))
    from corpus import SENTENCES, SEQ_LEN

    model_dir = REPO / "models" / "all-MiniLM-L6-v2"
    st = SentenceTransformer(str(model_dir), device="cpu")
    st.max_seq_length = SEQ_LEN

    threads = torch.get_num_threads()
    print(f"sentence-transformers on CPU, {threads} torch threads")
    print(f"model {model_dir.name}, seq {SEQ_LEN}\n")

    rows = []
    for batch in (4, 32, 128):
        sents = (SENTENCES * ((batch // len(SENTENCES)) + 1))[:batch]
        st.encode(sents, batch_size=batch, convert_to_numpy=True)   # warm
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            st.encode(sents, batch_size=batch, convert_to_numpy=True,
                      normalize_embeddings=True)
            best = min(best, time.perf_counter() - t0)
        rows.append({"batch": batch, "seconds": best, "seq_per_s": batch / best})
        print(f"  batch {batch:>4}: {best * 1e3:8.1f} ms  ->  "
              f"{batch / best:8.1f} seq/s")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "bench_cpu_baseline.json").write_text(json.dumps({
        "kind": "wall clock, end-to-end (docs/05-measurement)",
        "backend": "sentence-transformers, torch CPU",
        "torch_threads": threads, "seq_len": SEQ_LEN, "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {(ARTIFACTS / 'bench_cpu_baseline.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
