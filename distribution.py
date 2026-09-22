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
    python distribution.py tap_stats.pt --csv-avg avg.csv     # one row per tap, layers pooled
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


def outlier_stats(st, k=3.0):
    """Count and percentage of elements outside mean +/- k*std -- the 3-sigma rule.

    Read off the sign-split log2 histogram, because the threshold depends on mean and std
    and so is not known while the activations are streaming past. The threshold falls
    inside a bin, so whole bins are counted from the first one whose lower edge clears it;
    at 0.0625 exponent per bin that is a ~4.4% uncertainty on the threshold, not on the
    count of elements far beyond it.

    Mirrors tap.outlier_counts(); duplicated so this file needs only torch, per the module
    docstring. `.pt` files written before hist_neg existed fall back to assuming symmetry,
    which is exact only when mean == 0 -- flagged as `approx` in the returned dict.
    """
    hist = st["hist"].to(torch.float64)
    bins, lo, hi = st["hist_bins"], st["hist_lo"], st["hist_hi"]
    w = (hi - lo) / bins
    mean, sd, n = st["mean"], st["std"], max(st["count"], 1)
    n_zero = st["n_zero"]

    def count_ge(h, t):
        """|x| >= t, splitting the straddling bin proportionally in log space rather than
        discarding it -- dropping it whole undercounts the tail by ~20%."""
        if t <= 0.0:
            return float(h.sum())
        x = (math.log2(t) - lo) / w
        if x <= 0.0:
            return float(h.sum())
        if x >= bins:
            return 0.0
        j = int(math.floor(x))
        return float(h[j + 1:].sum()) + (j + 1 - x) * float(h[j])

    approx = "hist_neg" not in st
    if approx:
        # No sign information: treat the distribution as symmetric about zero, i.e. count
        # |x| beyond the wider of the two bounds.
        t = max(abs(mean + k * sd), abs(mean - k * sd))
        n_out = count_ge(hist, t)
        lo_tail = hi_tail = float("nan")
    else:
        hneg = st["hist_neg"].to(torch.float64)
        hpos = hist - hneg
        up = count_ge
        dn = lambda h, t: float(h.sum()) - up(h, t)
        n_pos, n_neg = float(hpos.sum()), float(hneg.sum())
        U, L = mean + k * sd, mean - k * sd
        if U > 0:      hi_tail = up(hpos, U)
        elif U == 0:   hi_tail = n_pos
        else:          hi_tail = n_pos + n_zero + dn(hneg, -U)
        if L < 0:      lo_tail = up(hneg, -L)
        elif L == 0:   lo_tail = n_neg
        else:          lo_tail = n_neg + n_zero + dn(hpos, L)
        n_out = lo_tail + hi_tail
    return {
        "sigma_k": k,
        "outlier_lo": mean - k * sd,
        "outlier_hi": mean + k * sd,
        "n_outlier_lo": lo_tail,
        "n_outlier_hi": hi_tail,
        "n_outlier": n_out,
        "outlier_pct": 100.0 * n_out / n,
        "outlier_approx": approx,
    }


def _phi(z):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def folded_cdf(t, mu, sigma):
    """P(|X| <= t) for X ~ N(mu, sigma^2). Reduces to erf(t/(sigma*sqrt2)) at mu = 0."""
    if t <= 0.0 or sigma <= 0.0:
        return 0.0
    return _phi((t - mu) / sigma) - _phi((-t - mu) / sigma)


def normal_reference(st):
    """Expected log2|x| histogram if this tap were exactly N(mean, std^2).

    The tap only records |x| in log2 bins, so this compares the *magnitude*
    distribution. It is blind to sign asymmetry -- see normality() for what that costs.
    """
    mu, sigma = st["mean"], st["std"]
    bins, lo, hi = st["hist_bins"], st["hist_lo"], st["hist_hi"]
    w = (hi - lo) / bins
    exp = []
    for i in range(bins):
        a = folded_cdf(2.0 ** (lo + i * w), mu, sigma)
        b = folded_cdf(2.0 ** (lo + (i + 1) * w), mu, sigma)
        exp.append(max(b - a, 0.0))
    return torch.tensor(exp, dtype=torch.float64)


def expected_mean_abs(st):
    """E|X| for X ~ N(mean, std^2). Equals sigma*sqrt(2/pi) when mean = 0.

    Comparing this to the recorded mean_abs is the one normality check that touches no
    histogram at all: both sides are exact moments, so it carries no binning error.
    """
    mu, sigma = st["mean"], st["std"]
    if sigma <= 0:
        return abs(mu)
    return (sigma * math.sqrt(2.0 / math.pi) * math.exp(-mu * mu / (2 * sigma * sigma))
            + mu * (1.0 - 2.0 * _phi(-mu / sigma)))


def normality(st):
    """How far this tap's magnitude distribution is from the best-fit normal.

    Returns effect sizes, not p-values. Every site here pools 1e8-1e9 elements, and at
    that N a chi-square or D'Agostino test rejects normality for a deviation far too
    small to matter for anything downstream -- the test statistic only measures that N
    is large. What matters is *how far off* the fit is, which is what tv_dist and the
    moment ratios report.
    """
    obs = st["hist"].to(torch.float64)
    tot = float(obs.sum())
    if tot <= 0:
        return None
    obs = obs / tot
    exp = normal_reference(st)
    e = float(exp.sum())
    if e <= 0:
        return None
    exp = exp / e                       # renormalize over the represented range
    tv = 0.5 * float((obs - exp).abs().sum())     # total variation distance, in [0, 1]
    ema = expected_mean_abs(st)
    return {
        "tv_dist": tv,
        "max_bin_dev": float((obs - exp).abs().max()),
        # observed / predicted E|X|. Below 1 means the mass sits closer to zero than a
        # normal with the same variance, i.e. a sharper peak paid for by heavier tails.
        "meanabs_ratio": st["mean_abs"] / ema if ema > 0 else float("nan"),
        # |x| exceedance a normal would allow at the recorded absmax, times count. Much
        # less than 1 means the extremes are impossible under a normal.
        "expected_beyond_absmax": st["count"] * (1.0 - folded_cdf(
            st["absmax"], st["mean"], st["std"])),
        "skew": st.get("skew"),
        "excess_kurt": st.get("excess_kurt"),
    }


def print_normality(rows):
    print("\n=== normality of the magnitude distribution vs a fitted N(mean, std^2) ===")
    print("effect sizes, not p-values: at 1e8-1e9 samples every significance test rejects")
    header = (f"{'site':<20}{'tv_dist':>10}{'max_dev':>10}{'E|x| obs/exp':>14}"
              f"{'skew':>10}{'ex_kurt':>10}{'verdict':>12}")
    print(header)
    print("-" * len(header))
    for label, n in rows:
        if n is None:
            continue
        sk = f"{n['skew']:10.2f}" if n["skew"] is not None else f"{'-':>10}"
        ek = f"{n['excess_kurt']:10.2f}" if n["excess_kurt"] is not None else f"{'-':>10}"
        # 0.05 total variation is roughly "indistinguishable by eye on a histogram";
        # 0.20 is a visibly different shape.
        verdict = "normal-ish" if n["tv_dist"] < 0.05 else (
            "skewed/tailed" if n["tv_dist"] < 0.20 else "far from normal")
        print(f"{label:<20}{n['tv_dist']:10.4f}{n['max_bin_dev']:10.4f}"
              f"{n['meanabs_ratio']:14.3f}{sk}{ek}{verdict:>12}")


def summarize(st):
    return {
        "count": st["count"],
        "min": st["min"],
        "max": st["max"],
        "absmax": st["absmax"],
        # .get, not [], so this file still reads .pt files even if the .pt files
        # don't have an "absmin_nonzero" field. It is used to adapt those
        # old .pt files that don't have an "absmin_nonzero" field.
        "absmin_nonzero": st.get("absmin_nonzero", float("nan")),
        "mean": st["mean"],
        "std": st["std"],
        "mean_abs": st["mean_abs"],
        "p999_abs": hist_quantile(st, 0.999),
        "p9999_abs": hist_quantile(st, 0.9999),
        "outlier_ratio": outlier_ratio(st),
        "dyn_bits": dynamic_range_bits(st),
        "int_bits": int_bits_needed(st),
        "ch_spread_bits": channel_spread_bits(st),
        "sigma_k": st.get("sigma_k", 3.0),
        "outlier_lo": outlier_stats(st)["outlier_lo"],
        "outlier_hi": outlier_stats(st)["outlier_hi"],
        "n_outlier": st.get("n_outlier", outlier_stats(st)["n_outlier"]),
        "outlier_pct": st.get("outlier_pct", outlier_stats(st)["outlier_pct"]),
        "tv_dist_normal": (normality(st) or {}).get("tv_dist", float("nan")),
        "meanabs_ratio_normal": (normality(st) or {}).get("meanabs_ratio", float("nan")),
        "skew": st.get("skew", float("nan")),
        "excess_kurt": st.get("excess_kurt", float("nan")),
        "frac_zero": st["n_zero"] / max(st["count"], 1),
        "n_nonfinite": st["n_nonfinite"],
    }


def merge_stats(sts):
    """Pool several per-layer stat dicts into one, as if a single accumulator had seen
    every layer's elements.

    This is what "the average across layers" has to mean for these quantities. Averaging
    the *derived* numbers row by row is wrong for almost every column here: the mean of
    twelve absmaxes is not an absmax, and dyn_bits / int_bits / ch_spread / the 3-sigma
    counts are read off a histogram, so they are not linear in anything that could be
    averaged. Pooling the raw ingredients and re-deriving keeps the whole row internally
    consistent -- every field is a real statistic of one real distribution.

    Exactly poolable: count, n_zero, n_nonfinite, the histograms (elementwise sums), and
    min/max (elementwise extrema). The moments are recovered from what finalize() already
    reduced -- mean/std/skew/excess_kurt back to raw moments, pooled, then reduced again.
    That round trip is algebraically exact; it loses precision only through the usual
    cancellation in m2 - mean^2, which is negligible against std here but is the reason
    skew and excess_kurt are dropped when any input lacks them.
    """
    n_tot = sum(s["count"] for s in sts)
    if n_tot <= 0:
        return None

    # --- raw moments, weighted by each layer's element count -----------------
    s1 = s2 = s3 = s4 = sa = 0.0
    have_hi = all(s.get("skew") is not None and s.get("excess_kurt") is not None
                  and math.isfinite(s.get("skew", float("nan")))
                  and math.isfinite(s.get("excess_kurt", float("nan"))) for s in sts)
    for s in sts:
        n, mean, sd = s["count"], s["mean"], s["std"]
        var = sd * sd
        m2 = var + mean * mean
        s1 += mean * n
        s2 += m2 * n
        sa += s["mean_abs"] * n
        if have_hi:
            mu3 = s["skew"] * sd ** 3
            mu4 = (s["excess_kurt"] + 3.0) * var * var
            m3 = mu3 + 3 * mean * m2 - 2 * mean ** 3
            m4 = mu4 + 4 * mean * m3 - 6 * mean * mean * m2 + 3 * mean ** 4
            s3 += m3 * n
            s4 += m4 * n

    mean = s1 / n_tot
    m2 = s2 / n_tot
    var = max(m2 - mean * mean, 0.0)
    sd = math.sqrt(var)
    if have_hi and sd > 0 and var > 0:
        m3, m4 = s3 / n_tot, s4 / n_tot
        mu3 = m3 - 3 * mean * m2 + 2 * mean ** 3
        mu4 = m4 - 4 * mean * m3 + 6 * mean * mean * m2 - 3 * mean ** 4
        skew, ex_kurt = mu3 / sd ** 3, mu4 / (var * var) - 3.0
    else:
        skew = ex_kurt = float("nan")

    # --- per-channel: same tap, same channel layout in every layer -----------
    widths = {tuple(s["ch_absmax"].shape) for s in sts}
    if len(widths) == 1:
        ch_absmax = torch.stack([s["ch_absmax"].to(torch.float64) for s in sts]).amax(0)
        ch_meanabs = (sum(s["ch_meanabs"].to(torch.float64) * s["count"] for s in sts)
                      / n_tot).to(torch.float32)
        ch_absmax = ch_absmax.to(torch.float32)
    else:
        # Different channel counts across layers means these are not the same site; a
        # pooled per-channel curve would be meaningless, so report none rather than a
        # plausible-looking one.
        ch_absmax = torch.zeros(0, dtype=torch.float32)
        ch_meanabs = torch.zeros(0, dtype=torch.float32)

    out = {
        "dtype": sts[0]["dtype"],
        "shape_tail": sts[0]["shape_tail"],
        "calls": sum(s["calls"] for s in sts),
        "count": n_tot,
        "n_zero": sum(s["n_zero"] for s in sts),
        "n_nonfinite": sum(s["n_nonfinite"] for s in sts),
        "min": min(s["min"] for s in sts),
        "max": max(s["max"] for s in sts),
        "absmax": max(s["absmax"] for s in sts),
        "mean": mean,
        "std": sd,
        "mean_abs": sa / n_tot,
        "skew": skew,
        "excess_kurt": ex_kurt,
        "ch_absmax": ch_absmax,
        "ch_meanabs": ch_meanabs,
        "hist": sum(s["hist"].to(torch.int64) for s in sts),
        "hist_lo": sts[0]["hist_lo"],
        "hist_hi": sts[0]["hist_hi"],
        "hist_bins": sts[0]["hist_bins"],
        "sigma_k": sts[0].get("sigma_k", 3.0),
    }
    # Pools by taking the minimum, same as min/max above. Emitted only when every layer
    # carries it, so a mixed set of old and new .pt files reports nan rather than a
    # figure derived from whichever layers happened to have the field.
    if all("absmin_nonzero" in s for s in sts):
        out["absmin_nonzero"] = min(s["absmin_nonzero"] for s in sts)
    # Only if every layer has it: hist - hist_neg must stay a valid positive histogram.
    if all("hist_neg" in s for s in sts):
        out["hist_neg"] = sum(s["hist_neg"].to(torch.int64) for s in sts)
    # n_outlier / outlier_pct are deliberately NOT carried over. summarize() prefers a
    # stored value and only falls back to outlier_stats(); leaving them out forces the
    # recount against the pooled histogram and the pooled mean/std, which is the only
    # correct answer -- the per-layer counts were taken against per-layer thresholds.
    return out


def average_summaries(rows):
    """Literal column-wise mean of per-layer summary rows: the arithmetic average of
    L00..L11, each layer weighted equally regardless of how many elements it saw.

    Kept as an option because it answers a different question from merge_stats(): "what
    does a typical block look like" rather than "what does the whole stack look like".
    Counts are summed instead of averaged -- a mean element count is not a count.
    """
    summed = {"count", "n_zero", "n_nonfinite", "n_outlier", "calls"}
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        if not isinstance(vals[0], (int, float)):
            out[k] = vals[0]
        elif k in summed:
            out[k] = sum(vals)
        else:
            fin = [v for v in vals if isinstance(v, float) and math.isfinite(v)
                   or isinstance(v, int)]
            out[k] = sum(fin) / len(fin) if fin else float("nan")
    return out


def cross_layer_rows(sel, summaries, mode="pool"):
    """One row per tap name, collapsing every layer of that tap into a single record.

    Returns [(name, summary, extras)], sorted by name. `extras` carries the across-layer
    spread that a single collapsed row would otherwise hide: a tap whose absmax is 4 in
    eleven blocks and 400 in the twelfth averages to something no block ever produced,
    and the min/max columns are what makes that visible in the CSV.
    """
    by_name = {}
    for k, (layer, name, st) in sel.items():
        by_name.setdefault(name, []).append((layer, k, st))

    out = []
    for name in sorted(by_name):
        items = sorted(by_name[name], key=lambda t: (t[0] is None, t[0]))
        layers = [l for l, _, _ in items]
        keys = [k for _, k, _ in items]
        per_layer = [summaries[k] for k in keys]

        if mode == "mean":
            s = average_summaries(per_layer)
        else:
            merged = merge_stats([st for _, _, st in items])
            if merged is None:
                continue
            s = summarize(merged)

        extras = {
            "n_layers": len(items),
            "layers": " ".join("-" if l is None else f"L{l:02d}" for l in layers),
            "absmax_layer_min": min(x["absmax"] for x in per_layer),
            "absmax_layer_max": max(x["absmax"] for x in per_layer),
            "dyn_bits_layer_max": max(x["dyn_bits"] for x in per_layer),
            "int_bits_layer_max": max(x["int_bits"] for x in per_layer),
        }
        out.append((name, s, extras))
    return out


def print_table(rows, title):
    print(f"\n=== {title} ===")
    header = (f"{'site':<20}{'absmax':>12}{'p99.9':>12}{'mean|x|':>12}"
              f"{'out_ratio':>11}{'dyn_bits':>10}{'int_bits':>10}{'ch_spread':>11}"
              f"{'3sig out%':>11}")
    print(header)
    print("-" * len(header))
    for label, s in rows:
        print(f"{label:<20}{s['absmax']:12.4e}{s['p999_abs']:12.4e}{s['mean_abs']:12.4e}"
              f"{s['outlier_ratio']:11.1f}{s['dyn_bits']:10.1f}{s['int_bits']:10.1f}"
              f"{s['ch_spread_bits']:11.2f}{s['outlier_pct']:10.3f}%")


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
    ap.add_argument("--sigma", type=float, default=3.0,
                    help="outlier rule: |x - mean| > SIGMA * std (default 3, the z-score "
                         "convention)")
    ap.add_argument("--outliers", action="store_true",
                    help="per-site outlier counts, both tails, with the bounds used")
    ap.add_argument("--normality", action="store_true",
                    help="report how far each site is from a fitted normal distribution")
    ap.add_argument("--csv", default=None, help="write every site's summary to this CSV")
    ap.add_argument("--csv-avg", default=None,
                    help="write ONE row per tap name, collapsing L00..L11 into a single "
                         "record (see --avg-mode)")
    ap.add_argument("--avg-mode", choices=("pool", "mean"), default="pool",
                    help="pool: re-derive every metric from the summed histograms and "
                         "pooled moments, i.e. the distribution of the whole stack "
                         "(default, and the only self-consistent choice). "
                         "mean: plain column-wise mean of the per-layer rows, every layer "
                         "weighted equally -- what a typical block looks like")
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

    if args.outliers:
        print(f"\n=== outliers: |x - mean| > {args.sigma} * std ===")
        header = (f"{'site':<20}{'range lo':>13}{'range hi':>13}{'below':>14}"
                  f"{'above':>14}{'total':>14}{'pct':>9}")
        print(header)
        print("-" * len(header))
        for k in sorted(sel):
            o = outlier_stats(sel[k][2], args.sigma)
            note = "  (approx: no sign data)" if o["outlier_approx"] else ""
            print(f"{k:<20}{o['outlier_lo']:13.4f}{o['outlier_hi']:13.4f}"
                  f"{o['n_outlier_lo']:14,.0f}{o['n_outlier_hi']:14,.0f}"
                  f"{o['n_outlier']:14,.0f}{o['outlier_pct']:8.3f}%{note}")

    if args.normality:
        print_normality([(k, normality(sel[k][2])) for k in sorted(sel)])

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

    if args.csv_avg:
        avg = cross_layer_rows(sel, summaries, mode=args.avg_mode)
        if not avg:
            raise SystemExit("nothing to average")

        print_table([(name, s) for name, s, _ in avg],
                    f"all layers collapsed per tap ({args.avg_mode})")
        # The pooled row hides how much the layers disagreed, and that spread is exactly
        # what decides whether one shared format can serve every block. Print it next to
        # the collapsed table rather than leaving it in the CSV only.
        print(f"\n{'site':<20}{'layers':>8}{'absmax L-min':>15}{'absmax L-max':>15}"
              f"{'L-max/L-min':>13}")
        for name, _, e in avg:
            lo, hi = e["absmax_layer_min"], e["absmax_layer_max"]
            print(f"{name:<20}{e['n_layers']:>8}{lo:15.4e}{hi:15.4e}"
                  f"{hi / lo if lo > 0 else float('inf'):13.1f}x")

        extra_fields = ["n_layers", "layers", "absmax_layer_min", "absmax_layer_max",
                        "dyn_bits_layer_max", "int_bits_layer_max"]
        # Same leading three columns as --csv, so the two files concatenate and diff
        # cleanly. `layer` is the literal string "all" rather than blank, which keeps a
        # collapsed row from being mistaken for an un-layered site like a bare "out".
        fields = (["site", "layer", "tap", "avg_mode"]
                  + list(avg[0][1].keys()) + extra_fields)
        with open(args.csv_avg, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for name, s, e in avg:
                w.writerow({"site": name, "layer": "all", "tap": name,
                            "avg_mode": args.avg_mode, **s, **e})
        print(f"\nwrote {args.csv_avg}")


if __name__ == "__main__":
    main()
