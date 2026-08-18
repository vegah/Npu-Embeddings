# NpuEmbeddings -- regenerate parsed Perfetto trace JSON from stored artifacts.
#
# Why this exists
# ---------------
# The parsed `trace_<tag>.json` files are ~100 MB in total and are pure
# derivatives: `parse_trace(trace.txt, physical.mlir)`. Both inputs are stored
# in `artifacts/` (the .txt raw trace and the copied `input_with_addresses.mlir`),
# so the JSON is kept out of git and regenerated on demand. Rule 6 still holds --
# every number points at a stored artifact, one parse step away.
#
# NOTE: this only works for tags that have a matching `mlir_<tag>.mlir`. The
# whole-array runs (`wa2c`/`wa4c`) did NOT copy their physical MLIR, so their
# JSON is NOT regenerable and is committed as-is. See tasks/0004.
#
# No NPU needed -- this is a pure offline re-parse.
#
# Usage (from a shell where C:\dev\mlir-aie\iron_env.ps1 has been dot-sourced):
#   python experiments\m2-bf16-gemm\regen_trace_json.py            # all tags
#   python experiments\m2-bf16-gemm\regen_trace_json.py --check    # verify only
#   python experiments\m2-bf16-gemm\regen_trace_json.py --tag bf16_f32_bfp16_256x384x384_t64x64x32

import argparse
import hashlib
import sys
from pathlib import Path

from aie.utils.trace import TraceConfig

DEFAULT_ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS = DEFAULT_ARTIFACTS

# The --trace-size the runs used (gemm_single_core.py default). read_trace()
# zero-pads back to this, so it must match or the parse can misindex.
TRACE_SIZE = 262144


def tags():
    """Every tag that has both a raw trace and a physical MLIR stored."""
    for mlir in sorted(ARTIFACTS.glob("mlir_*.mlir")):
        tag = mlir.name[len("mlir_") : -len(".mlir")]
        if (ARTIFACTS / f"trace_{tag}.txt").exists():
            yield tag


def regen(tag, check):
    txt = ARTIFACTS / f"trace_{tag}.txt"
    mlir = ARTIFACTS / f"mlir_{tag}.mlir"
    out = ARTIFACTS / f"trace_{tag}.json"

    if txt.stat().st_size == 0:
        print(f"  {tag}: EMPTY raw trace -- refusing")     # rule: assert non-empty
        return False

    before = hashlib.sha256(out.read_bytes()).hexdigest() if out.exists() else None

    cfg = TraceConfig(trace_size=TRACE_SIZE, trace_file=str(txt))
    if check and before is not None:
        tmp = out.with_suffix(".json.check")
        cfg.trace_to_json(str(mlir), str(tmp))
        after = hashlib.sha256(tmp.read_bytes()).hexdigest()
        tmp.unlink()
        ok = after == before
        print(f"  {tag}: {'MATCH' if ok else 'MISMATCH'}  {after[:16]}")
        return ok

    cfg.trace_to_json(str(mlir), str(out))
    after = hashlib.sha256(out.read_bytes()).hexdigest()
    note = "" if before is None else ("  (unchanged)" if after == before else "  (CHANGED)")
    print(f"  {tag}: {out.stat().st_size/1e6:6.1f} MB  {after[:16]}{note}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="single tag; default is all regenerable tags")
    ap.add_argument("--check", action="store_true",
                    help="regenerate to a temp file and compare against the existing JSON")
    ap.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS),
                    help="artifacts directory; M5 stores its traces in "
                         "experiments/m5-pretiled-gemm/artifacts")
    args = ap.parse_args()

    global ARTIFACTS
    ARTIFACTS = Path(args.artifacts)

    todo = [args.tag] if args.tag else list(tags())
    if not todo:
        print("no regenerable tags found")
        return 1

    print(f"{'checking' if args.check else 'regenerating'} {len(todo)} trace JSON(s)")
    return 0 if all([regen(t, args.check) for t in todo]) else 1


if __name__ == "__main__":
    sys.exit(main())
