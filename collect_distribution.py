"""Run a downstream-task evaluation with tap accumulation switched on, and write the
per-block activation statistics to a .pt file for distribution.py.

This is `evals/lm_harness_eval.py` wrapped in one `tap.accumulate()` block. lm_eval is not
modified and not even aware of the taps: importing lm_harness_eval registers the "mamba"
model, and every argument this script does not recognise is forwarded verbatim to the
lm_eval CLI.

    python collect_distribution.py --tap-out stats_lambada.pt --tap-stride 8 -- \
        --model mamba \
        --model_args pretrained=state-spaces/mamba3-mimo-187m,tokenizer=meta-llama/Llama-3.1-8B,dtype=bfloat16,add_bos_token=true \
        --tasks lambada_openai --device cuda --batch_size 16 --limit 256

Why the stats and not the tensors: capture() keeps every tapped tensor for one forward
pass. A full lambada_openai run is ~5k forwards over 24 blocks x 18 taps, which is far
past what any machine can hold. accumulate() folds each tensor into fixed-size running
statistics on the GPU instead, so the cost is constant in the number of documents.
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MAMBA_REPO = os.environ.get(
    "MAMBA_REPO", "/home/m314510193/GithubMamba3Train/RTX6000_mamba3")


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--tap-out", default=os.path.join(HERE, "tap_stats.pt"),
                    help="where to write the accumulated statistics")
    ap.add_argument("--tap-stride", type=int, default=1,
                    help="fold only every N-th call at each tap site (1 = every call)")
    ap.add_argument("--tap-max-rows", type=int, default=None,
                    help="randomly subsample this many token-rows per tapped tensor")
    ap.add_argument("--tap-device", default=None,
                    help="where the accumulators live (default: the tensor's own device)")
    ap.add_argument("--block-taps", action="store_true",
                    help="also tap the residual stream, MLP and norm_f (see block_taps.py)")
    ap.add_argument("-h", "--help", action="store_true")
    mine, rest = ap.parse_known_args()
    if mine.help:
        ap.print_help()
        print("\nAll other arguments are forwarded to the lm_eval CLI.")
        return
    # `--` separates our flags from lm_eval's; argparse leaves it in `rest`.
    rest = [a for a in rest if a != "--"]

    # evals/ is not a package, so it has to go on the path explicitly. Importing it runs
    # @register_model("mamba"), which is the whole reason run_mamba3_lm_eval.sh invokes
    # lm_harness_eval.py instead of the bare `lm_eval` CLI.
    sys.path.insert(0, os.path.join(MAMBA_REPO, "evals"))
    import lm_harness_eval  # noqa: F401  (import for its registration side effect)
    from lm_eval.__main__ import cli_evaluate

    from mamba_ssm.utils import tap

    if mine.block_taps:
        import block_taps
        block_taps.patch_from_pretrained()

    print(f"[tap] mamba_ssm -> {tap.__file__}", flush=True)
    print(f"[tap] impl={tap.IMPL} stride={mine.tap_stride} "
          f"max_rows={mine.tap_max_rows}", flush=True)

    meta = {
        "argv": rest,
        "stride": mine.tap_stride,
        "max_rows": mine.tap_max_rows,
        "block_taps": mine.block_taps,
    }

    argv_backup = sys.argv
    sys.argv = ["lm_eval"] + rest
    try:
        with tap.accumulate(stride=mine.tap_stride,
                            max_rows=mine.tap_max_rows,
                            device=mine.tap_device) as acc:
            try:
                cli_evaluate()
            finally:
                # Write whatever was folded in even if the eval itself dies partway, so a
                # long run is never lost to a crash in lm_eval's reporting code.
                if acc.stats:
                    tap.save(acc, mine.tap_out, meta=meta)
                    print(f"[tap] wrote {len(acc.stats)} tap sites to {mine.tap_out}",
                          flush=True)
                else:
                    print("[tap] WARNING: no taps fired -- is PYTHONPATH pointing at "
                          "the git tree rather than site-packages?", flush=True)
    finally:
        sys.argv = argv_backup


if __name__ == "__main__":
    main()
