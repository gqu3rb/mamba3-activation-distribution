# Collapse a tap_stats_<TAG>.pt into ONE row per tap name -- every layer of that tap
# pooled into a single record -- and write it as CSV.
#
# Run on LOGIN NODE:
#   ./gen_avg.sh
#
# No --gres here: distribution.py only reads the .pt with torch and never touches a GPU.

set -euo pipefail

HERE=/home/m314510193/GithubMamba3Train/mamba3_distribution

# ---------------------------- edit these ----------------------------------
IN="${IN:-$HERE/tap_stats_mamba3_fineweb_8L_mlp_reg1e-4_lambada_openai_blocktaps.pt}"
OUT="${OUT:-$HERE/tap_stats_mamba3_fineweb_8L_mlp_reg1e-4_lambada_openai_blocktaps_avg.csv}"
# pool: re-derive every metric from the summed histograms and pooled moments, i.e. the
#       distribution of the whole stack. The only self-consistent choice -- the mean of
#       twelve absmaxes is not an absmax.
# mean: plain column-wise mean of the per-layer rows, each layer weighted equally.
AVG_MODE="${AVG_MODE:-pool}"
# --------------------------------------------------------------------------

CONDA_ENV="${CONDA_ENV:-mamba3}"

module load miniconda3/26.1.1
conda activate "$CONDA_ENV"
export PATH="$CONDA_PREFIX/bin:$PATH"

if [ ! -f "$IN" ]; then
    echo "ERROR: input not found: $IN" >&2
    exit 1
fi

echo "in   : $IN"
echo "       collected $(date -r "$IN" '+%F %T')"
echo "out  : $OUT"
echo "mode : $AVG_MODE"

# Regenerating is deterministic given the .pt, so overwriting an existing CSV is normally
# harmless. It is NOT harmless once the .pt has been replaced by a later collection run:
# the numbers then change without the file name saying anything changed.
if [ -f "$OUT" ] && [ "$IN" -nt "$OUT" ]; then
    echo
    echo "WARNING: $(basename "$OUT") was written $(date -r "$OUT" '+%F %T'), but its"
    echo "         input .pt is newer -- it was re-collected since. The values about to"
    echo "         be written come from a different run than the ones being replaced."
    echo "         Set OUT to a new name if you need to keep the old numbers."
    echo
fi

# distribution.py prints the collapsed table and the per-tap layer spread to stdout, so
# the job log is a self-contained record of what was written.
python "$HERE/distribution.py" "$IN" --csv-avg "$OUT" --avg-mode "$AVG_MODE"
