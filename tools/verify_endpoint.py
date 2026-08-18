# NpuEmbeddings -- does the endpoint work with the REAL OpenAI client?
# SPDX-License-Identifier: Apache-2.0
#
# "OpenAI-compatible" is a claim about a client that exists, not about a
# response shape that looks about right. So this drives `npuembed --serve`
# with the official `openai` package, in the ways an application actually
# uses it -- including base64, which the client requests by DEFAULT and which
# a hand-rolled server will usually have got wrong.
#
# It then checks the numbers, not just the plumbing: the vectors must match
# the same texts through `--embed`, and the semantics must survive
# (paraphrases close, unrelated texts far).
#
# Env: .venv-ref (openai, numpy, sentence-transformers)
# Usage:
#   npuembed .. --artifacts artifacts_b128il --pipeline 2 --serve 8420   (elsewhere)
#   & ".\.venv-ref\Scripts\python.exe" tools\verify_endpoint.py --port 8420

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
HIDDEN = 384

PAIRS = [
    ("A man is playing a guitar on stage.",
     "Someone plays a guitar at a concert."),
    ("The cat sat on the mat.", "A feline rested on the rug."),
    ("How do I reset my password?", "I forgot my login credentials."),
]
UNRELATED = ("The stock market closed lower on Tuesday.",
             "Blåbærsyltetøy smaker godt på brødskiva.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", default=str(REPO / "tasks" / "0037-m9-tiers-endpoint"
                                         / "verify_endpoint.json"))
    args = ap.parse_args()

    from openai import OpenAI
    base = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(base_url=base, api_key="not-needed")
    results = {}
    ok = True

    # 1. models
    models = client.models.list()
    model_id = models.data[0].id
    print(f"  models.list()            -> {model_id}")
    results["model"] = model_id

    # 2. a single string, the simplest call an app makes
    r = client.embeddings.create(model=model_id, input="hello world")
    v = np.asarray(r.data[0].embedding, dtype=np.float32)
    print(f"  single string            -> dim {v.size}, "
          f"norm {np.linalg.norm(v):.6f}, tokens {r.usage.prompt_tokens}")
    ok &= v.size == HIDDEN and abs(np.linalg.norm(v) - 1.0) < 1e-3
    results["single"] = {"dim": int(v.size),
                         "norm": float(np.linalg.norm(v))}

    # 3. a batch, and the ORDER must be preserved -- an endpoint that
    #    reorders under batching is silently catastrophic for a client.
    texts = [t for p in PAIRS for t in p] + list(UNRELATED)
    r = client.embeddings.create(model=model_id, input=texts)
    idx = [d.index for d in r.data]
    emb = np.stack([np.asarray(d.embedding, dtype=np.float32) for d in r.data])
    print(f"  batch of {len(texts):<3}            -> indices ordered: "
          f"{idx == sorted(idx)}, shape {emb.shape}")
    ok &= idx == list(range(len(texts))) and emb.shape == (len(texts), HIDDEN)

    # 4. base64. Note what steps 2 and 3 already proved: when the caller does
    #    NOT specify encoding_format, the official client asks for base64
    #    itself and decodes it -- so the default path through the real client
    #    is the base64 path, and it produced unit-norm vectors above. Asking
    #    EXPLICITLY returns the raw string by design, so decode it here and
    #    check it is the same numbers.
    import base64 as b64mod
    r64 = client.embeddings.create(model=model_id, input=texts,
                                   encoding_format="base64")
    emb64 = np.stack([np.frombuffer(b64mod.b64decode(d.embedding),
                                    dtype=np.float32) for d in r64.data])
    d64 = float(np.abs(emb64 - emb).max())
    print(f"  base64 (decoded) vs float-> max abs diff {d64:.3e}")
    ok &= d64 == 0.0
    results["base64_max_diff"] = d64

    # 5. semantics: paraphrases close, unrelated far
    print("  semantics:")
    for i, (a, b) in enumerate(PAIRS):
        c = float(emb[2 * i] @ emb[2 * i + 1])
        print(f"    pair {i} cos {c:+.4f}   {a[:34]!r}")
        ok &= c > 0.5
    c_un = float(emb[-1] @ emb[-2])
    print(f"    unrelated cos {c_un:+.4f}")
    ok &= c_un < 0.4
    results["pair_cos"] = [float(emb[2 * i] @ emb[2 * i + 1])
                           for i in range(len(PAIRS))]
    results["unrelated_cos"] = c_un

    # 6. the endpoint must agree with the batch path it is built on
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(prefix="ep_") as td:
        d = Path(td)
        (d / "in.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
        run = subprocess.run(
            [str(REPO / "runtime" / "build" / "npuembed.exe"), "..",
             "--artifacts", "artifacts_b128il", "--threads", "24",
             "--embed", str(d / "in.txt"), str(d / "out.f32")],
            cwd=str(REPO / "runtime"), capture_output=True, text=True)
        if run.returncode != 0:
            print(f"  --embed cross-check FAILED: {run.stderr[:200]}")
            ok = False
        else:
            ref = np.frombuffer((d / "out.f32").read_bytes(),
                                dtype=np.float32).reshape(len(texts), HIDDEN)
            dmax = float(np.abs(ref - emb).max())
            print(f"  vs --embed               -> max abs diff {dmax:.3e}")
            # Not bit-identical: the tier ladder can split a request
            # differently, and a sequence's row position within its chunk
            # changes nothing mathematically but does change summation order
            # in the host pooling. Same magnitude as bf16 rounding.
            ok &= dmax < 1e-3
            results["vs_embed_max_diff"] = dmax

    # 7. errors must be errors
    import urllib.error
    import urllib.request

    def post(payload):
        req = urllib.request.Request(
            f"{base}/embeddings", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status
        except urllib.error.HTTPError as e:
            return e.code

    codes = {"no input": post({"model": model_id}),
             "empty list": post({"model": model_id, "input": []}),
             "bad format": post({"model": model_id, "input": "x",
                                 "encoding_format": "nonsense"}),
             "token ids": post({"model": model_id, "input": [[1, 2, 3]]})}
    print("  errors:")
    for k, v in codes.items():
        print(f"    {k:<12} -> HTTP {v}")
        ok &= v >= 400
    results["error_codes"] = codes

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results["pass"] = bool(ok)
    out.write_text(json.dumps({"kind": "verification", "task": "0037",
                               **results}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print("PASS -- works with the official OpenAI client" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
