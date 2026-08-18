# NpuEmbeddings -- M4 GATE: verify the .npue container.
#
# Four checks, in increasing order of what they would catch:
#
#   A. SPEC       -- header size, magic, version, and 4096-byte alignment of
#                    every tensor. A misaligned tensor is a DMA descriptor that
#                    silently reads the wrong bytes on hardware.
#   B. ROUND-TRIP -- de-tile every pre-tiled operand and compare BIT-EXACTLY
#                    against the bf16 of the fused source. Not "close": the
#                    tiling is a permutation, so anything but zero differing
#                    elements is a bug.
#   C. GUARD      -- a wrong layout must RAISE, not silently produce garbage
#                    embeddings. This is what layout_hash is for.
#   D. GOLDENS    -- run the actual encoder off the packed weights and compare
#                    to the M3 goldens. This is the check the other three exist
#                    to support: it is the only one that would catch a fusion
#                    that round-trips perfectly but is mathematically wrong
#                    (scale folded into K instead of Q, a transpose applied
#                    twice, Q/K/V concatenated in the wrong order).
#
# Env: iron (numpy only)
# Usage:
#   & "C:\Users\vegar\.conda\envs\iron\python.exe" tools\verify_npue.py

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "reference"))

from encoder import MiniLMReference                                # noqa: E402
from precision_study import make_gemm                              # noqa: E402
from npue import (ALIGN, HEADER_SIZE, MAGIC, VERSION, Reader,      # noqa: E402
                  to_bf16_bits, untile_b)
from safetensors_io import load                                    # noqa: E402


def rel_fro(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


# -- A. spec conformance ---------------------------------------------------

def check_spec(path, r):
    print("A. spec conformance")
    problems = []
    with open(path, "rb") as f:
        head = f.read(HEADER_SIZE)
    magic, version, arch, flags, jo, jl, do, dl, reserved = struct.unpack(
        "<4sIII QQQQ 16s", head)
    size = Path(path).stat().st_size

    checks = [
        ("header is exactly 64 bytes", struct.calcsize("<4sIII QQQQ 16s") == 64),
        ("magic == b'NPUE'", magic == MAGIC),
        (f"version == {VERSION}", version == VERSION),
        ("flags has bit0 (pre-tiled)", bool(flags & 1)),
        ("reserved is zero", reserved == b"\0" * 16),
        (f"data_offset {do} is {ALIGN}-aligned", do % ALIGN == 0),
        ("json sits between header and data", jo == HEADER_SIZE and jo + jl <= do),
        ("file size == data_offset + data_length", size == do + dl),
    ]
    misaligned = [e["name"] for e in r.entries.values()
                  if (do + e["offset"]) % ALIGN]
    checks.append((f"all {len(r.entries)} tensors 4096-aligned", not misaligned))
    if misaligned:
        problems.append(f"misaligned: {misaligned[:5]}")

    for label, ok in checks:
        print(f"   {'ok  ' if ok else 'FAIL'}  {label}")
        if not ok:
            problems.append(label)
    overhead = size - sum(e["nbytes"] for e in r.entries.values())
    print(f"         alignment + json overhead: {overhead/1024:.1f} KB "
          f"on {size/1e6:.1f} MB ({overhead/size*100:.3f}%)")
    return problems


# -- B. bit-exact round-trip ----------------------------------------------

def check_roundtrip(r, src, cfg, fold_scale):
    print("\nB. round-trip (de-tile == bf16 of the fused source, bit-exact)")
    L, hidden = cfg["num_layers"], cfg["hidden"]
    scale = 1.0 / math.sqrt(cfg["head_dim"])
    problems, total = [], 0

    def expect_qkv(i):
        sa = f"encoder.layer.{i}.attention.self."
        m = np.ascontiguousarray(np.concatenate(
            [src[sa + n + ".weight"] for n in ("query", "key", "value")], axis=0).T)
        if fold_scale:
            m = m.copy()
            m[:, :hidden] *= scale
        return m

    def expect_t(name):
        return np.ascontiguousarray(src[name].T)

    for i in range(L):
        cases = {
            f"layer.{i}.qkv": expect_qkv(i),
            f"layer.{i}.attn_out":
                expect_t(f"encoder.layer.{i}.attention.output.dense.weight"),
            f"layer.{i}.ffn_up":
                expect_t(f"encoder.layer.{i}.intermediate.dense.weight"),
            f"layer.{i}.ffn_down":
                expect_t(f"encoder.layer.{i}.output.dense.weight"),
        }
        for name, want_f32 in cases.items():
            e = r.entries[name]
            lay = e["layout"]
            K, N = e["padded_shape"]
            got_bits = untile_b(r.raw(name), K, N, lay["tile_k"], lay["tile_n"],
                                lay["mac_s"], lay["mac_t"])
            want_bits = to_bf16_bits(want_f32)
            ndiff = int((got_bits != want_bits).sum())
            total += want_bits.size
            if ndiff:
                problems.append(f"{name}: {ndiff} of {want_bits.size} differ")

    print(f"   {'ok  ' if not problems else 'FAIL'}  "
          f"{4*L} operands, {total:,} bf16 elements, "
          f"{sum(int(p.split(':')[1].split()[0]) for p in problems) if problems else 0}"
          f" differing")
    for p in problems[:5]:
        print(f"   FAIL  {p}")
    return problems


# -- C. the stale-layout guard --------------------------------------------

def check_guard(r):
    print("\nC. layout_hash guard")
    name = "layer.0.qkv"
    good = dict(r.entries[name]["layout"])
    bad = dict(good, tile_n=good["tile_n"] + 16)
    problems = []
    try:
        r.check_layout(name, good)
        print("   ok    matching layout accepted")
    except ValueError as e:
        problems.append(f"correct layout rejected: {e}")
        print(f"   FAIL  matching layout rejected: {e}")
    try:
        r.check_layout(name, bad)
        problems.append("wrong layout accepted -- a stale file would run silently")
        print("   FAIL  wrong layout accepted")
    except ValueError:
        print(f"   ok    tile_n {good['tile_n']} -> {bad['tile_n']} refused")
    return problems


# -- D. the real check: goldens -------------------------------------------

def build_from_npue(r, cfg, gemm=None):
    """Reconstruct an encoder that reads ONLY packed weights.

    Everything comes back out of the container: the fused QKV, the transposed
    projections, the fp32 biases and LayerNorm params. Nothing is re-derived
    from the original checkpoint, so a fusion bug has nowhere to hide.
    """
    L, hidden = cfg["num_layers"], cfg["hidden"]
    w = {
        "embeddings.word_embeddings.weight": r.tensor("embeddings.word"),
        "embeddings.position_embeddings.weight": r.tensor("embeddings.position"),
        "embeddings.token_type_embeddings.weight": r.tensor("embeddings.token_type"),
        "embeddings.LayerNorm.weight": r.tensor("embeddings.ln.weight"),
        "embeddings.LayerNorm.bias": r.tensor("embeddings.ln.bias"),
    }
    qkv_w, qkv_b = {}, {}
    for i in range(L):
        # encoder.py's linear() does weight.T, and .npue stores [K,N], so
        # transposing here hands back exactly the packed values.
        qkv_w[i] = np.ascontiguousarray(r.tensor(f"layer.{i}.qkv").T)
        qkv_b[i] = r.tensor(f"layer.{i}.qkv.bias")
        p = f"encoder.layer.{i}."
        w[p + "attention.output.dense.weight"] = np.ascontiguousarray(
            r.tensor(f"layer.{i}.attn_out").T)
        w[p + "attention.output.dense.bias"] = r.tensor(f"layer.{i}.attn_out.bias")
        w[p + "attention.output.LayerNorm.weight"] = r.tensor(f"layer.{i}.ln1.weight")
        w[p + "attention.output.LayerNorm.bias"] = r.tensor(f"layer.{i}.ln1.bias")
        w[p + "intermediate.dense.weight"] = np.ascontiguousarray(
            r.tensor(f"layer.{i}.ffn_up").T)
        w[p + "intermediate.dense.bias"] = r.tensor(f"layer.{i}.ffn_up.bias")
        w[p + "output.dense.weight"] = np.ascontiguousarray(
            r.tensor(f"layer.{i}.ffn_down").T)
        w[p + "output.dense.bias"] = r.tensor(f"layer.{i}.ffn_down.bias")
        w[p + "output.LayerNorm.weight"] = r.tensor(f"layer.{i}.ln2.weight")
        w[p + "output.LayerNorm.bias"] = r.tensor(f"layer.{i}.ln2.bias")

    folded = cfg["fusions"]["qk_scale_folded_into_q"]
    return MiniLMReference(
        w, num_layers=L, num_heads=cfg["num_heads"], eps=cfg["layer_norm_eps"],
        gemm=gemm, qkv_w=qkv_w, qkv_b=qkv_b,
        qk_scale=1.0 if folded else math.sqrt(cfg["head_dim"]))


def check_goldens(r, cfg, goldens):
    print("\nD. the encoder, running off packed weights, vs the M3 goldens")
    g, meta = load(goldens)
    if meta["source_sha256"] != cfg["source_sha256"]:
        return [f"checkpoint mismatch: goldens {meta['source_sha256'][:16]} vs "
                f".npue {cfg['source_sha256'][:16]}"]

    # Two runs, because they answer different questions and conflating them
    # would flatter the result:
    #
    #  * fp32 activations -- isolates what PACKING costs (bf16 weights + the
    #    fusions), with nothing else in the way.
    #  * bf16 activations -- the actual NPU datapath, and the only run
    #    comparable to M3's end-to-end bf16 figure of 1-cos = 1.271e-05.
    L = cfg["num_layers"]
    names = ["emb.ln"] + [f"L{i}.ln2" for i in range(L)] + ["last_hidden_state"]
    M3_BF16_1MCOS = 1.271e-05

    runs = {}
    for label, gemm in (("fp32 activations", None),
                        ("bf16 activations", make_gemm("bf16"))):
        ref = build_from_npue(r, cfg, gemm=gemm)
        taps = {}
        emb = ref.encode(g["input_ids"], g["attention_mask"], g["token_type_ids"],
                         taps=taps)
        runs[label] = (emb, taps)

    print(f"   {'tensor':<20} {'bf16 weights only':>19} {'+ bf16 activations':>20}")
    for nm in names:
        print(f"   {nm:<20} "
              f"{rel_fro(runs['fp32 activations'][1][nm], g['hf.' + nm]):>19.3e} "
              f"{rel_fro(runs['bf16 activations'][1][nm], g['hf.' + nm]):>20.3e}")

    problems = []
    print()
    for label, (emb, _) in runs.items():
        cos = (emb.astype(np.float64) * g["hf.out.embedding"].astype(np.float64)).sum(1)
        shift = float(np.abs(emb @ emb.T
                             - g["hf.out.embedding"] @ g["hf.out.embedding"].T).max())
        print(f"   {label:<18} 1-cos {1 - cos.min():.3e}   "
              f"similarity shift {shift:.3e}")
        if label == "bf16 activations":
            ratio = (1 - cos.min()) / M3_BF16_1MCOS
            print(f"   {'':18} vs M3's end-to-end bf16 ({M3_BF16_1MCOS:.2e}): "
                  f"{ratio:.2f}x")
            # Packing must not be lossier than bf16 alone. If it were, a fusion
            # is doing damage that the number format does not explain.
            if ratio > 2.0:
                problems.append(f"packing costs {ratio:.2f}x M3's bf16 baseline "
                                f"-- a fusion is lossier than bf16 alone")
    return problems


# -- E. isolate what the scale fold costs ---------------------------------

def check_fold_cost(cfg, goldens, packed_path):
    """Does folding 1/sqrt(32) into Q cost anything BEYOND bf16 rounding?

    It is not obviously free: 1/sqrt(32) is not a power of two, so folding
    changes which bf16 grid point each Q weight rounds to. Runs the identical
    encoder on a --no-fold-scale pack and compares. Anything else would be
    guessing at a question that takes one extra file to answer.
    """
    print("\nE. what the 1/sqrt(head_dim) fold costs, in isolation")
    unfolded = Path(packed_path).with_name("_verify_nofold.npue")
    import subprocess
    cmd = [sys.executable, str(REPO / "tools" / "pack_npue.py"),
           "--no-fold-scale", "--out", str(unfolded)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        print(f"   FAIL  could not pack the unfolded variant:\n{res.stderr[-400:]}")
        return [f"unfolded pack failed: {res.returncode}"]

    g, _ = load(goldens)
    out = {}
    for label, path in (("folded into Q", packed_path), ("applied at runtime", unfolded)):
        # bf16 activations: the fold changes which bf16 grid point each Q weight
        # lands on, so it must be judged on the datapath that actually rounds.
        with Reader(path) as rr:
            ref = build_from_npue(rr, rr.config, gemm=make_gemm("bf16"))
            emb = ref.encode(g["input_ids"], g["attention_mask"], g["token_type_ids"])
        cos = (emb.astype(np.float64) * g["hf.out.embedding"].astype(np.float64)).sum(1)
        out[label] = 1 - cos.min()
    unfolded.unlink(missing_ok=True)

    for label, v in out.items():
        print(f"   1 - cos, scale {label:<20}: {v:.3e}")
    d = out["folded into Q"] / out["applied at runtime"]
    print(f"   folding costs {d:.3f}x -- "
          f"{'free' if 0.5 <= d <= 2.0 else 'NOT free'}")
    return [] if 0.5 <= d <= 2.0 else [f"scale fold costs {d:.2f}x"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npue", default=str(REPO / "models" / "all-MiniLM-L6-v2.npue"))
    ap.add_argument("--model-dir", default=str(REPO / "models" / "all-MiniLM-L6-v2"))
    ap.add_argument("--goldens", default=str(REPO / "reference" / "goldens"
                                             / "minilm_l6_s64_boundary.safetensors"))
    args = ap.parse_args()

    r = Reader(args.npue)
    cfg = r.config
    print(f"{Path(args.npue).name}  "
          f"({Path(args.npue).stat().st_size/1e6:.2f} MB, {len(r.entries)} tensors)")
    print(f"  tile ({cfg['tile_k']}, {cfg['tile_n']}), "
          f"mac (s={cfg['mac_s']}, t={cfg['mac_t']}), "
          f"fusions: {', '.join(k for k, v in cfg['fusions'].items() if v is True)}\n")

    src, _ = load(Path(args.model_dir) / "model.safetensors")

    problems = []
    problems += check_spec(args.npue, r)
    problems += check_roundtrip(r, src, cfg, cfg["fusions"]["qk_scale_folded_into_q"])
    problems += check_guard(r)
    problems += check_goldens(r, cfg, args.goldens)
    problems += check_fold_cost(cfg, args.goldens, args.npue)

    if problems:
        print(f"\nFAIL -- {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nPASS -- spec conformant, round-trip bit-exact, layout guarded, "
          "goldens reproduced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
