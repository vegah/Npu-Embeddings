# NpuEmbeddings -- design a GELU polynomial that needs no transcendental call.
#
# tasks/0014 isolated the error in IRON's GELU to `aie::tanh` itself, which is
# only ~1% accurate on this hardware. No amount of caller-side precision fixes
# that, so GELU has to be evaluated without it.
#
# THE REWRITE THAT MAKES THIS EASY
# --------------------------------
# Fitting GELU directly, or tanh, is awkward: both saturate, so a polynomial
# needs high degree and still misbehaves outside its range. But
#
#     GELU(-x) = -x*Phi(-x) = -x*(1 - Phi(x)) = GELU(x) - x
#
# so defining  c(x) = GELU(x) - max(x, 0)  gives  c(-x) = c(x):  c is EVEN.
# It is also a bump that decays like a Gaussian --  c(0)=0, |c| peaks near 0.17
# around |x|=0.75, and c(5) = -2.9e-06.
#
# So  GELU(x) = max(x, 0) + c(|x|)  with c smooth, even, and vanishing. And
# because c is already ~0 at the edge of the fit range, CLAMPING the argument
# costs nothing -- no branch, no select:
#
#     u = min(|x|, R)        aie::abs, aie::min
#     p = poly(u)            Horner: aie::mul + aie::add only
#     gelu = max(x, 0) + p   aie::max, aie::add
#
# Every one of those is a native vector op. No transcendental, no division,
# nothing from the elementary-function library we just measured as unreliable.
#
# Env: iron env (numpy only). No NPU needed.
# Usage:
#   & "C:\Users\vegar\.conda\envs\iron\python.exe" experiments\m5-eltwise\design_gelu_poly.py

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "reference"))

from encoder import gelu as gelu_exact          # noqa: E402  (exact erf, fp64)
from safetensors_io import load                 # noqa: E402


def c_exact(u):
    """c(u) = GELU(u) - max(u,0), evaluated in fp64 on u >= 0.

    For u >= 0 that is u*(Phi(u) - 1), which is where the cancellation lives,
    so compute it from the exact erf rather than by subtracting two large
    numbers.
    """
    u = np.asarray(u, dtype=np.float64)
    phi = 0.5 * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in u.ravel()])
                 .reshape(u.shape))
    return u * (phi - 1.0)


def fit(degree, R, n=4000):
    """Least-squares fit of c on [0, R] at Chebyshev nodes.

    Chebyshev nodes rather than uniform: they suppress the edge oscillation a
    uniform fit would put exactly where the activations are densest.
    """
    k = np.arange(n)
    u = 0.5 * R * (1.0 - np.cos(np.pi * (k + 0.5) / n))     # Chebyshev on [0,R]
    y = c_exact(u)
    coef = np.polyfit(u, y, degree)
    return coef


def eval_poly_f32(coef, u):
    """Horner in float32 -- the same arithmetic the kernel will do."""
    u = np.asarray(u, dtype=np.float32)
    acc = np.full_like(u, np.float32(coef[0]))
    for c in coef[1:]:
        acc = acc * u + np.float32(c)
    return acc


def gelu_poly(x, coef, R):
    x = np.asarray(x, dtype=np.float32)
    u = np.minimum(np.abs(x), np.float32(R))
    return (np.maximum(x, np.float32(0.0)) + eval_poly_f32(coef, u)).astype(np.float32)


def to_bf16(v):
    u = np.ascontiguousarray(v, dtype=np.float32).view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", default=str(REPO / "reference" / "goldens"))
    ap.add_argument("--out", default=str(HERE / "artifacts" / "gelu_poly.json"))
    args = ap.parse_args()

    taps, _ = load(Path(args.goldens) / "minilm_l6_s64_taps.safetensors")
    x = taps["L0.ffn_up"].reshape(-1)
    want = taps["L0.gelu"].reshape(-1).astype(np.float64)
    x16 = to_bf16(x)                       # what the kernel will actually see

    print(f"real activations: {x.size:,} values, "
          f"range [{x.min():.2f}, {x.max():.2f}], "
          f"|x|>5 is {(np.abs(x) > 5).mean():.3%} of them\n")

    rel = lambda g: float(np.linalg.norm(np.asarray(g, np.float64) - want)
                          / np.linalg.norm(want))

    # The floors we cannot go below, for reference.
    floor_bf16 = rel(to_bf16(gelu_exact(x16)))
    print(f"  {'variant':<40} {'rel_fro':>11} {'max|err|':>11}")
    print(f"  {'exact erf, bf16 output (the floor)':<40} {floor_bf16:>11.3e} "
          f"{np.abs(to_bf16(gelu_exact(x16)) - want).max():>11.3e}")
    print(f"  {'IRON / aie::tanh, measured on hardware':<40} {1.316e-02:>11.3e} "
          f"{6.162e-02:>11.3e}")
    print()

    cands = []
    for R in (4.0, 5.0, 6.0):
        for degree in (6, 8, 10, 12):
            coef = fit(degree, R)
            g = gelu_poly(x16, coef, R)
            r = rel(to_bf16(g))
            maxerr = float(np.abs(to_bf16(g) - want).max())
            tag = f"poly deg {degree}, clamp {R:.0f}"
            print(f"  {tag:<40} {r:>11.3e} {maxerr:>11.3e}")
            cands.append((r, degree, R, coef, maxerr))

    # Pick the CHEAPEST fit within 5% of the best, not the most accurate one.
    # Every extra degree is one more mul+add in the inner loop, and once the
    # polynomial is at the bf16 floor the extra accuracy is unobservable.
    floor_best = min(c[0] for c in cands)
    affordable = [c for c in cands if c[0] <= floor_best * 1.05]
    best = min(affordable, key=lambda c: (c[1], c[2]))
    r, degree, R, coef, maxerr = best
    print(f"\nchosen: degree {degree}, clamp {R:.0f} -> rel_fro {r:.3e} "
          f"({r / floor_bf16:.2f}x the bf16 floor)")

    # Horner order, highest power first, as the kernel will consume them.
    print("\nHorner coefficients (highest power first):")
    for i, cf in enumerate(coef):
        print(f"  c[{i}] = {cf: .10e}f   // u^{degree - i}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "kind": "cpu design",
        "form": "gelu(x) = max(x,0) + poly(min(|x|, R))",
        "degree": degree, "clamp_R": R,
        "coefficients_highest_first": [float(c) for c in coef],
        "rel_fro_vs_exact_erf_golden": r,
        "max_abs_err": maxerr,
        "bf16_output_floor": floor_bf16,
        "ratio_to_floor": r / floor_bf16,
        "hardware_aie_tanh_reference": 1.316e-02,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {Path(args.out).relative_to(REPO)}")


if __name__ == "__main__":
    main()
