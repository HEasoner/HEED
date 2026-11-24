#!/usr/bin/env bash
# Latent-support intervention (steering along the KNOW-HALL mean-difference direction) and the
# projection effect size (Cohen's d).  Needs a GPU and the six chat models; HEED_MODEL_DIR may point to
# local copies (folder names as in analysis/_paths.py), otherwise they are downloaded from the HF hub.
#   bash analysis/run_intervention.sh [device]
set -euo pipefail
cd "$(dirname "$0")"
device="${1:-auto}"

# 1. pilot sweep (Qwen2.5-7B-Instruct / PopQA): seed-42 split, 100 balanced test questions,
#    layer ratios {0.50, 0.65, 0.80, 0.90} x alpha in [-1.5, 1.5]; re-used by the formal qwen25_popqa setting
python intervention_steering.py

# 2. formal runs: sweep + random-direction controls + hidden capture + support + output quality
for setting in qwen25_popqa qwen25_webq llama3_popqa llama3_webq gemma2_popqa gemma2_webq; do
  python intervention_formal.py --setting "$setting" --device "$device"
done

# 3. (optional) primary configuration only (layer ratio 0.8, alpha in {-1, 0}) on the remaining settings
for setting in llama2_popqa llama2_webq qwen2_popqa qwen2_webq deepseek_popqa deepseek_webq \
               llama3_triviaqa llama3_mkqa qwen2_triviaqa qwen2_mkqa qwen25_triviaqa qwen25_mkqa \
               gemma2_triviaqa gemma2_mkqa deepseek_triviaqa deepseek_mkqa; do
  python intervention_formal.py --setting "$setting" --components sweep --layer-ratios 0.8 --alphas=-1,0 \
    --output-root "${HEED_OUTPUT_DIR:-../outputs}/analysis/latent_support_intervention_extended" --device "$device"
done

# 4. statistics (exact McNemar, bootstrap CIs) and the direction case study with Cohen's d
python intervention_analyze.py --bootstrap-resamples 2000 --allow-partial
python intervention_direction_case_study.py
