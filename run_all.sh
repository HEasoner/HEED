#!/usr/bin/env bash
# End-to-end reproduction.  Steps 1-2 need GPUs and the six LLMs; steps 3-6 run from the feature files.
#   bash run_all.sh            # everything
#   SKIP_EXTRACTION=1 bash run_all.sh   # start from existing features/*_balanced.pt
set -euo pipefail
cd "$(dirname "$0")"
MODELS="llama2_7b_chat llama3_8b_instruct qwen2_7b_instruct qwen2_5_7b_instruct gemma_2_9b_it deepseek_7b_chat"
DATASETS="popqa webquestions triviaqa mkqa"

if [[ "${SKIP_EXTRACTION:-0}" != "1" ]]; then
  # 1. data
  python scripts/prepare_datasets.py
  # 2. generation, labelling, features (one greedy answer + one teacher-forced pass per question), balancing
  for m in $MODELS; do
    for d in $DATASETS; do
      python scripts/extract_features.py --dataset "$d" --model "$m"   # add --model-path for a local copy
      python scripts/balance_features.py --dataset "$d" --model "$m"
    done
  done
fi

# 3. HEED + single-evidence ablations (24 settings x seeds 42-46)
python scripts/train_heed.py
# 4. baselines (training-free scores, LR probe, faithful SAPLMA)
python scripts/run_baselines.py
# 5. tables -> outputs/tables/{main,ablation,baselines}.md
python scripts/summarize.py
# 6. analyses and figures
python analysis/score_analysis.py
python analysis/characterize_hall.py
python analysis/obs_multiseed.py
for s in 42 43 44 45 46; do python analysis/knn_robustness.py --seed "$s"; done
python analysis/knn_robustness_plot.py
python analysis/pipeline_figure.py
# GPU-heavy, optional:
#   python analysis/timing.py
#   bash analysis/run_intervention.sh
