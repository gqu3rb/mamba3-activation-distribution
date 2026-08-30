"""Report the per-block activation distribution collected by collect_distribution.py.

This file used to be a byte-identical copy of mamba3_compare_nano4/compare.py -- a
two-sided nano-vs-rtx divergence check, which is a different question and already lives in
that directory. It now reads the running statistics that mamba_ssm/utils/tap.py's
accumulate() mode produced during a downstream-task run and answers the question this
directory is named after: what does each Mamba-3 block's activation distribution look
like, and what number format does it need.

Only needs torch, so it runs on the login node.

    python distribution.py tap_stats.pt                       # per-layer summary table
    python distribution.py tap_stats.pt --tap x --detail      # one tap across all layers
    python distribution.py tap_stats.pt --csv stats.csv       # everything, machine-readable
"""

import argparse
import csv
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_key(k):
    """'L07/B_normed' -> (7, 'B_normed'); a bare 'B_normed' -> (None, 'B_normed')."""
    if "/" in k:
        layer, name = k.split("/", 1)
        return int(layer.lstrip("L")), name
    return None, k


def hist_quantile(st, q):
    """Absolute-value quantile read off the log2 histogram.

    The histogram is the only record of the body of the distribution -- absmax alone
    cannot tell a heavy tail from a single stray outlier. `hist_quantile(st, 0.999)` is
    the clipping threshold a percentile-clipped quantizer would use; comparing it against
    `absmax` is what says whether clipping buys anything.
    """
    h = st["hist"].to(torch.float64)
    total = float(h.sum())
    if total <= 0:
        return 0.0
    # Quantile over non-zero magnitudes only (exact zeros are excluded from the histogram
    # by construction), so q is conditioned on |x| > 0.
    target = q * total
    c = torch.cumsum(h, 0)
    i = int(torch.searchsorted(c, torch.tensor(target, dtype=torch.float64)).item())
    i = min(i, st["hist_bins"] - 1)
    width = (st["hist_hi"] - st["hist_lo"]) / st["hist_bins"]
    # upper edge of bin i, in log2 space
    return 2.0 ** (st["hist_lo"] + (i + 1) * width)


def dynamic_range_bits(st, q_lo=0.001, q_hi=0.999):
    """log2(hi/lo) over the central mass: how many *exponent* bits the tensor spans once
    the extreme tails are dropped. This is the number that decides a PoT/shared-exponent
    format's exponent width, and it is not derivable from absmax."""
    hi = hist_quantile(st, q_hi)
    lo = hist_quantile(st, q_lo)
    if lo <= 0 or hi <= 0:
        return float("nan")
    return math.log2(hi / lo)


def int_bits_needed(st, q=0.999, rel_step=None):
    """Bits for a symmetric per-tensor integer format that keeps the q-quantile
    unclipped and resolves `rel_step` of the mean magnitude as one LSB."""
    hi = hist_quantile(st, q)
    step = rel_step if rel_step is not None else st["mean_abs"] / 64.0
    if step <= 0 or hi <= 0:
        return float("nan")
    return math.log2(hi / step) + 1.0   # +1 for the sign


def outlier_ratio(st):
    """absmax / mean|x|. Large values mean a per-tensor scale wastes most of its range on
    a few elements -- the classic argument for per-channel scaling."""
    return st["absmax"] / st["mean_abs"] if st["mean_abs"] > 0 else float("inf")


def channel_spread_bits(st):
    """log2(max channel absmax / median channel absmax). Zero means every channel has the
    same range and one per-tensor scale is enough; several bits means the outliers are
    concentrated in a few channels and per-channel scaling recovers them."""
    ca = st["ch_absmax"].to(torch.float64)
    ca = ca[ca > 0]
    if ca.numel() < 2:
        return 0.0
    return math.log2(float(ca.max()) / float(ca.median()))


def summarize(st):
    return {
        "count": st["count"],
        "min": st["min"],
        "max": st["max"],
        "absmax": st["absmax"],
        "mean": st["mean"],
        "std": st["std"],
        "mean_abs": st["mean_abs"],
        "p999_abs": hist_quantile(st, 0.999),
        "p9999_abs": hist_quantile(st, 0.9999),
        "outlier_ratio": outlier_ratio(st),
        "dyn_bits": dynamic_range_bits(st),
        "int_bits": int_bits_needed(st),
        "ch_spread_bits": channel_spread_bits(st),
        "frac_zero": st["n_zero"] / max(st["count"], 1),
        "n_nonfinite": st["n_nonfinite"],
    }


def print_table(rows, title):
    print(f"\n=== {title} ===")
    header = (f"{'site':<20}{'absmax':>12}{'p99.9':>12}{'mean|x|':>12}"
              f"{'out_ratio':>11}{'dyn_bits':>10}{'int_bits':>10}{'ch_spread':>11}")
    print(header)
    print("-" * len(header))
    for label, s in rows:
        print(f"{label:<20}{s['absmax']:12.4e}{s['p999_abs']:12.4e}{s['mean_abs']:12.4e}"
              f"{s['outlier_ratio']:11.1f}{s['dyn_bits']:10.1f}{s['int_bits']:10.1f}"
              f"{s['ch_spread_bits']:11.2f}")


def print_hist(st, label, width=56):
    """log2|x| histogram, collapsed to whole exponents. Reading the shape matters more
    than any single scalar: a bimodal tensor cannot be quantized like a unimodal one."""
    h = st["hist"]
    bins, lo, hi = st["hist_bins"], st["hist_lo"], st["hist_hi"]
    per_exp = int(bins / (hi - lo))
    h = h.reshape(-1, per_exp).sum(1)
    total = max(int(h.sum()), 1)
    print(f"\n--- log2|x| histogram: {label} "
          f"(zeros: {st['n_zero'] / max(st['count'], 1):.2%}) ---")
    nz = (h > 0).nonzero().flatten()
    if nz.numel() == 0:
        print("  (all zero)")
        return
    peak = int(h.max())
    for e in range(int(nz[0]), int(nz[-1]) + 1):
        c = int(h[e])
        bar = "#" * int(width * c / peak)
        print(f"  2^{int(lo) + e:>4}  {c / total:7.3%} |{bar}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats", nargs="?", default=os.path.join(HERE, "tap_stats.pt"))
    ap.add_argument("--tap", default=None,
                    help="restrict to one tap name (e.g. x, B_normed, out)")
    ap.add_argument("--layer", type=int, default=None, help="restrict to one layer index")
    ap.add_argument("--detail", action="store_true",
                    help="also print the log2 histogram of every selected site")
    ap.add_argument("--top-channels", type=int, default=0,
                    help="list the N widest-range channels of each selected site")
    ap.add_argument("--csv", default=None, help="write every site's summary to this CSV")
    args = ap.parse_args()

    blob = torch.load(args.stats, map_location="cpu")
    stats = blob["stats"]
    print(f"impl={blob.get('impl')}  sites={len(stats)}  meta={blob.get('meta')}")

    sel = {}
    for k, st in stats.items():
        layer, name = parse_key(k)
        if args.tap is not None and name != args.tap:
            continue
        if args.layer is not None and layer != args.layer:
            continue
        sel[k] = (layer, name, st)

    if not sel:
        raise SystemExit("no tap sites matched the filters")

    summaries = {k: summarize(v[2]) for k, v in sel.items()}

    if args.tap is not None or args.layer is not None:
        rows = [(k, summaries[k]) for k in sorted(sel)]
        print_table(rows, f"selected sites ({len(rows)})")
    else:
        # Default view: one table per tap name, layers down the rows. This is the layout
        # that shows how a distribution drifts with depth, which is what decides whether
        # one format can be shared across all blocks or the deep blocks need their own.
        by_name = {}
        for k, (layer, name, _) in sel.items():
            by_name.setdefault(name, []).append((layer, k))
        for name in sorted(by_name):
            rows = [(f"L{l:02d}" if l is not None else "-", summaries[k])
                    for l, k in sorted(by_name[name])]
            print_table(rows, f"tap: {name}")

    if args.detail or args.top_channels:
        for k in sorted(sel):
            st = sel[k][2]
            if args.detail:
                print_hist(st, k)
            if args.top_channels:
                ca = st["ch_absmax"]
                n = min(args.top_channels, ca.numel())
                vals, idx = torch.topk(ca, n)
                med = float(ca.median())
                print(f"\n--- widest channels: {k} (median channel absmax {med:.4e}) ---")
                for v, i in zip(vals.tolist(), idx.tolist()):
                    print(f"  channel {i:>5}  absmax {v:.4e}"
                          f"   {v / med if med > 0 else float('inf'):8.1f}x median")

    if args.csv:
        fields = ["site", "layer", "tap"] + list(next(iter(summaries.values())).keys())
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for k in sorted(sel):
                layer, name, _ = sel[k]
                w.writerow({"site": k, "layer": layer, "tap": name, **summaries[k]})
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
