import argparse
import csv
import math
import os
import sys

from .core import COMPLIANCE, GPUS, MODELS, QUANT_BYTES, bom, configure


def _finite_float(s: str) -> float:
    """argparse type: a float that is not nan/inf, so limits and prices cannot be bypassed."""
    v = float(s)
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"must be a finite number, got {s!r}")
    return v


def build_parser():
    ap = argparse.ArgumentParser(prog="private-ai-box")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("size")
    s.add_argument("--users", type=int, required=True)
    s.add_argument("--concurrency", type=_finite_float, default=0.2, help="share of users generating at once")
    s.add_argument("--model", choices=sorted(MODELS), default="32b")
    s.add_argument("--quant", choices=sorted(QUANT_BYTES), default="int4")
    s.add_argument("--context", type=int, default=8192)
    s.add_argument("--tps", type=_finite_float, default=15, help="target tokens/s per active user")
    s.add_argument("--datacenter", action="store_true")
    s.add_argument("--compliance", nargs="*", default=[], choices=sorted(COMPLIANCE))
    sub.add_parser("models")
    sub.add_parser("gpus")
    return ap


def _run(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.cmd == "models":
        for m in MODELS.values():
            print(f"{m.key:<9} {m.params_b:>5}B total, {m.active_b}B active, {m.layers} layers")
        return 0
    if a.cmd == "gpus":
        for g in GPUS:
            print(f"{g.name:<20} {g.vram_gb:>3} GB {g.bandwidth_gbs:>6.0f} GB/s  ${g.price:,}")
        return 0
    opts = configure(a.model, a.quant, a.users, a.concurrency, a.context, a.tps, a.datacenter)
    if not opts:
        print("No single-node config meets the target. Try int4, smaller context, or a smaller model.")
        return 1
    print(f"{'option':<28}{'VRAM':>12}{'tok/s/user':>12}{'GPU cost':>12}")
    for o in opts:
        print(
            f"{o['count']} x {o['gpu']:<24}{o['vram_total']:>6}/{o['vram_needed']:<5}{o['tps_per_user']:>12}{o['gpu_cost']:>12,}"
        )
    b = bom(opts[0], a.compliance)
    print(f"\nRecommended BOM ({opts[0]['sequences']} concurrent sequences):")
    for item, price in b["lines"]:
        print(f"  {item:<48} ${price:>8,}")
    print(f"  {'Total cost':<48} ${b['total']:>8,}\n  {'Suggested price':<48} ${b['suggested_price']:>8,.0f}")
    for c, items in b["compliance"].items():
        print(f"\n{c} pack:")
        for i in items:
            print(f"  [ ] {i}")
    return 0


def main(argv=None):
    """Run the CLI; a bad input file or argument exits 2 with a one-line error instead of a traceback.

    Set AGENT_ECONOMY_DEBUG=1 to see the full traceback.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        return _run(argv)
    except (OSError, ValueError, KeyError, csv.Error) as e:
        if os.environ.get("AGENT_ECONOMY_DEBUG"):
            raise
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"error: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
