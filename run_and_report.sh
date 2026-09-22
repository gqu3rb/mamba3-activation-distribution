#!/bin/bash
# Submit mamba3_run_ref.sh, block until it finishes, then print the accuracy/perplexity
# and the activation-distribution report to THIS terminal.
#
# `sbatch --wait` on its own cannot do that: it blocks and propagates the job's exit code,
# but the job's stdout goes to the file named by #SBATCH --output, never back to the
# submitting shell. This wrapper closes that gap by (a) naming the log itself instead of
# letting SLURM fill in %j, so the path is known before submission, and (b) reading the
# job's artifacts back once --wait returns.
#
#   ./run_and_report.sh
#   TASKS=hellaswag LIMIT=512 ./run_and_report.sh
#   MODEL=state-spaces/mamba3-siso-187m TAG=siso ./run_and_report.sh
#
# Every knob mamba3_run_ref.sh reads (MODEL, TASKS, LIMIT, BATCH_SIZE, TAP_STRIDE, TAG,
# OUT, OUTDIR, METRICS, ...) is forwarded, because --export=ALL passes this shell's env.
#
# The whole body lives in a function and the file never calls `exit` at top level. A bare
# `exit` here would terminate the *calling* shell if anyone ran `. run_and_report.sh`
# instead of `./run_and_report.sh` -- which, over ssh, drops the login session the moment
# the run finishes. Same reason `set -uo pipefail` is scoped and restored below: when a
# file is sourced, shell options leak into the interactive shell and outlive the script.

_rar_main() {
    # Each config variable is seeded from the caller's environment: a bare `local MODEL`
    # would create it EMPTY, so the `${MODEL:-default}` lines below would silently ignore
    # `MODEL=... ./run_and_report.sh` and fall back to the Mamba-3 defaults. The RHS of
    # `local X="$X"` is expanded in the caller's scope, before the local is created, which
    # keeps the override working while still not leaking into a sourcing shell.
    local HERE LOG RAW JOBID RC
    local MODEL="${MODEL-}" TASKS="${TASKS-}" TAG="${TAG-}"
    local OUT="${OUT-}" OUTDIR="${OUTDIR-}" METRICS="${METRICS-}"

    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Recompute the same defaults mamba3_run_ref.sh uses, so this wrapper knows where the
    # job will put its artifacts. Anything already exported wins in both places.
    #MODEL="${MODEL:-state-spaces/mamba2-130m}"
    MODEL="${MODEL:-state-spaces/mamba3-mimo-187m}"
    TASKS="${TASKS:-lambada_openai}"
    #TASKS="${TASKS:-lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa}"
    TAG="${TAG:-$(basename "$MODEL")_${TASKS//,/-}}"
    OUT="${OUT:-$HERE/tap_stats_${TAG}.pt}"
    OUTDIR="${OUTDIR:-$HERE/results_${TAG}}"
    METRICS="${METRICS:-$HERE/metrics_${TAG}.json}"
    TOKENIZER="${TOKENIZER:-meta-llama/Llama-3.1-8B}"
    #TOKENIZER="${TOKENIZER:-EleutherAI/gpt-neox-20b}" # for mamba2
    export MODEL TASKS TAG OUT OUTDIR METRICS TOKENIZER

    # A Mamba-2/Mamba-3 checkpoint paired with the other family's tokenizer is a vocab
    # mismatch: it does not crash, it just returns nonsense (perplexity ~4.6e5, accuracy at
    # chance). Refuse rather than burn a GPU job producing numbers that look like data.
    case "$MODEL" in
        *mamba3*) local want="meta-llama/Llama-3.1-8B" ;;
        *mamba2*|*mamba-2*) local want="EleutherAI/gpt-neox-20b" ;;
        *) local want="" ;;
    esac
    if [ -n "$want" ] && [ -n "${TOKENIZER:-}" ] && [ "${TOKENIZER}" != "$want" ]; then
        echo "ERROR: MODEL=$MODEL expects TOKENIZER=$want but TOKENIZER=$TOKENIZER" >&2
        echo "       (mismatched tokenizers produce plausible-looking garbage, not an error)" >&2
        return 2
    fi

    LOG="$HERE/mamba3_distribution_${TAG}_$(date +%Y%m%d_%H%M%S).log"

    # A stale file from an earlier run must not be mistaken for this run's output.
    rm -f "$METRICS"

    echo "log: $LOG   (tail -f it from another shell to watch progress)"

    # --parsable prints just the job id; --wait blocks until the job ends and exits with
    # the job's own exit code. Together, the id arrives when the job is already finished.
    RAW=$(sbatch --parsable --wait --export=ALL --output="$LOG" "$HERE/mamba3_run_ref.sh")
    # Read the status straight off the assignment. Piping the output through `cut` here
    # would make $? report cut's status and silently discard the job's exit code, which is
    # the entire reason for using --wait.
    RC=$?
    # --parsable emits "jobid;cluster" when a cluster name is configured; keep the id only.
    JOBID="${RAW%%;*}"
    echo "job ${JOBID:-?} finished with exit code $RC"

    if [ ! -f "$METRICS" ]; then
        echo "ERROR: $METRICS was not produced -- last 40 lines of the job log:" >&2
        tail -n 40 "$LOG" >&2 2>/dev/null || echo "(no log at $LOG)" >&2
        [ "$RC" -eq 0 ] && return 1
        return "$RC"
    fi

    # Read the metrics back out of the job's own artifacts.
    python "$HERE/summarize_results.py" "$OUTDIR"
    echo
    echo "metrics json : $METRICS"
    echo "tap stats    : $OUT"
    echo "job log      : $LOG"
    return "$RC"
}

# `return` is only valid in a function or a sourced file, so this succeeds exactly when
# the file is being sourced.
if (return 0 2>/dev/null); then
    _rar_oldopts="$(set +o)"          # snapshot the interactive shell's options
    set -uo pipefail                  # no -e: report on a failed job, don't vanish with it
    _rar_main
    _rar_rc=$?
    eval "$_rar_oldopts"              # restore them; sourcing must not change the session
    unset -f _rar_main
    unset _rar_oldopts
    echo "(sourced: exit code $_rar_rc left in \$_rar_rc, session kept open)"
else
    set -uo pipefail
    _rar_main
    exit $?
fi
