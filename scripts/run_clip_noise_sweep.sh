#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: scripts/run_clip_noise_sweep.sh /path/to/mnist_train.csv [source_run_dir]" >&2
  exit 2
fi

dataset=$1
source_run=${2:-runs/canary_holdout}
python_bin=${PYTHON:-python3}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gaussproof-mpl"
mkdir -p "$MPLCONFIGDIR"

run_one() {
  local name=$1 config=$2 clip=${3:-}
  local output="runs/$name" report="reports/$name"
  if [[ -f "$output/completion.json" ]]; then
    echo "Already complete: $name"
    return
  fi
  if [[ -n "$clip" ]]; then
    "$python_bin" -m gaussproof.clip_noise_sweep --source "$source_run" \
      --data "$dataset" --config "$config" --output "$output" \
      --report "$report" --only-clip "$clip"
  else
    "$python_bin" -m gaussproof.clip_noise_sweep --source "$source_run" \
      --data "$dataset" --config "$config" --output "$output" \
      --report "$report"
  fi
}

run_one clip_noise_sweep configs/clip_noise_sweep.json 1
run_one clip_noise_sweep_C0p1 configs/clip_noise_sweep.json 0.1
run_one clip_noise_sweep_C4 configs/clip_noise_sweep.json 4
run_one clip_noise_low_epsilon_tail configs/clip_noise_low_epsilon_tail.json
run_one clip_noise_weak_clipping_tail configs/clip_noise_weak_clipping_tail.json
run_one clip_noise_utility_rescue_B32 configs/clip_noise_utility_rescue_B32.json
run_one clip_noise_utility_rescue_B64 configs/clip_noise_utility_rescue_B64.json
run_one clip_noise_utility_rescue_B128 configs/clip_noise_utility_rescue_B128.json

"$python_bin" -m scripts.merge_clip_noise_sweep \
  --groups reports/clip_noise_sweep reports/clip_noise_sweep_C0p1 \
           reports/clip_noise_sweep_C4 \
  --tail reports/clip_noise_low_epsilon_tail \
  --weak-clip reports/clip_noise_weak_clipping_tail \
  --utility-rescue reports/clip_noise_utility_rescue_B32 \
                   reports/clip_noise_utility_rescue_B64 \
                   reports/clip_noise_utility_rescue_B128 \
  --output reports/clip_noise_sweep

"$python_bin" -m scripts.compare_clip_noise_scores \
  --runs runs/clip_noise_sweep runs/clip_noise_sweep_C0p1 \
         runs/clip_noise_sweep_C4 runs/clip_noise_low_epsilon_tail \
         runs/clip_noise_weak_clipping_tail \
         runs/clip_noise_utility_rescue_B32 \
         runs/clip_noise_utility_rescue_B64 \
         runs/clip_noise_utility_rescue_B128 \
  --output reports/clip_noise_sweep/paired_differences.csv

"$python_bin" -m scripts.plot_clip_noise_sweep \
  --report reports/clip_noise_sweep
