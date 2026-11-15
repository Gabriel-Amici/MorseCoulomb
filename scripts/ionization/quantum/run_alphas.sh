#!/usr/bin/env bash
#
# Run every quantum ionization scan for the saved spectra.
#
# For each non-zero alpha:
#   * amplitude scan      -> MsC
#   * frequency scan      -> MsC
#   * stabilization scan  -> MsC + sC
#
# and for alpha = 0 (Coulomb):
#   * amplitude scan      -> C
#   * frequency scan      -> C
#   * stabilization scan  -> C
#
# sC is intentionally excluded from the amplitude/frequency scans, matching the
# defaults used by those two scripts.
#
# Usage
# -----
#   ./run_alphas.sh                      # outputs under notebooks/.../data/ionization
#   ./run_alphas.sh /path/to/output      # custom output root
#   nohup ./run_alphas.sh > run_alphas.log 2>&1 &
#
# The script keeps going when one run fails and reports every failure at the
# end, so an unattended overnight run is not lost because of a single bad
# spectrum.  A rough estimate for the full set of 7 parameter values is 7-9
# hours on a modern CPU.
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

OUTPUT_ROOT="${1:-$REPO_ROOT/notebooks/plotting/tcc/data/ionization}"
AMPLITUDE_DIR="$OUTPUT_ROOT/quantum_amplitude"
FREQUENCY_DIR="$OUTPUT_ROOT/quantum_frequency"
STABILIZATION_DIR="$OUTPUT_ROOT/quantum_stabilization"

# 0.001, 0.05, 0.1, 0.5, 1.0 and sqrt(2).  The last value is written in full;
# the scripts recognise it and use the "sqrt2" filename tag.
ALPHAS=(0.001 0.05 0.1 0.5 1.0 1.4142135623730951)

if [[ ! -x "$PYTHON" ]]; then
    echo "Python interpreter not found or not executable: $PYTHON" >&2
    echo "Set PYTHON=/path/to/python and retry." >&2
    exit 1
fi

mkdir -p "$AMPLITUDE_DIR" "$FREQUENCY_DIR" "$STABILIZATION_DIR"

FAILURES=()

run_scan() {
    local label="$1"
    shift

    local start
    start=$(date +%s)

    echo
    echo "======================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $label"
    echo "======================================================================"

    if "$@"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished $label in $(( $(date +%s) - start )) s"
    else
        local status=$?
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED $label (exit $status)" >&2
        FAILURES+=("$label")
    fi
}

for alpha in "${ALPHAS[@]}"; do
    run_scan "alpha=$alpha amplitude (MsC)" \
        "$PYTHON" "$SCRIPT_DIR/quantum_ionization_vs_amplitude.py" \
        --potential MsC --alpha "$alpha" \
        --output-dir "$AMPLITUDE_DIR" --no-progress

    run_scan "alpha=$alpha frequency (MsC)" \
        "$PYTHON" "$SCRIPT_DIR/quantum_ionization_vs_frequency.py" \
        --potential MsC --alpha "$alpha" \
        --output-dir "$FREQUENCY_DIR" --no-progress

    run_scan "alpha=$alpha stabilization (MsC + sC)" \
        "$PYTHON" "$SCRIPT_DIR/quantum_ionization_stabilization.py" \
        --potential both --alpha "$alpha" \
        --output-dir "$STABILIZATION_DIR" --no-progress
done

# Coulomb: the special alpha = 0 case.
run_scan "alpha=0 amplitude (C)" \
    "$PYTHON" "$SCRIPT_DIR/quantum_ionization_vs_amplitude.py" \
    --potential C \
    --output-dir "$AMPLITUDE_DIR" --no-progress

run_scan "alpha=0 frequency (C)" \
    "$PYTHON" "$SCRIPT_DIR/quantum_ionization_vs_frequency.py" \
    --potential C \
    --output-dir "$FREQUENCY_DIR" --no-progress

run_scan "alpha=0 stabilization (C)" \
    "$PYTHON" "$SCRIPT_DIR/quantum_ionization_stabilization.py" \
    --potential C \
    --output-dir "$STABILIZATION_DIR" --no-progress

echo
if (( ${#FAILURES[@]} )); then
    echo "Finished with ${#FAILURES[@]} failed run(s):" >&2
    printf '  - %s\n' "${FAILURES[@]}" >&2
    exit 1
fi

echo "All scans completed successfully."
echo "Amplitude:     $AMPLITUDE_DIR"
echo "Frequency:     $FREQUENCY_DIR"
echo "Stabilization: $STABILIZATION_DIR"
