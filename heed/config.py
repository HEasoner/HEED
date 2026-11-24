"""Experiment grid, paths and the (unified) hyper-parameters of HEED.

Paths can be overridden with environment variables:
  HEED_DATA_DIR      raw QA files                    (default: ./data)
  HEED_FEATURE_DIR   extracted feature files (*.pt)  (default: ./features)
  HEED_OUTPUT_DIR    scores, checkpoints, tables     (default: ./outputs)
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("HEED_DATA_DIR", REPO_ROOT / "data"))
FEATURE_DIR = Path(os.environ.get("HEED_FEATURE_DIR", REPO_ROOT / "features"))
OUTPUT_DIR = Path(os.environ.get("HEED_OUTPUT_DIR", REPO_ROOT / "outputs"))
FIGURE_DIR = Path(os.environ.get("HEED_FIGURE_DIR", REPO_ROOT / "figures"))

DATASETS = ["popqa", "webquestions", "triviaqa", "mkqa"]
DATASET_NAMES = {"popqa": "PopQA", "webquestions": "WebQuestions", "triviaqa": "TriviaQA", "mkqa": "MKQA"}

# model key -> Hugging Face repository id (a local directory can be passed with --model-path instead)
MODELS = {
    "llama2_7b_chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama3_8b_instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen2_7b_instruct": "Qwen/Qwen2-7B-Instruct",
    "qwen2_5_7b_instruct": "Qwen/Qwen2.5-7B-Instruct",
    "gemma_2_9b_it": "google/gemma-2-9b-it",
    "deepseek_7b_chat": "deepseek-ai/deepseek-llm-7b-chat",
}
MODEL_NAMES = {
    "llama2_7b_chat": "Llama-2-7B-Chat",
    "llama3_8b_instruct": "Llama-3-8B-Instruct",
    "qwen2_7b_instruct": "Qwen2-7B-Instruct",
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
    "gemma_2_9b_it": "Gemma-2-9B-it",
    "deepseek_7b_chat": "DeepSeek-LLM-7B-Chat",
}
SEEDS = [42, 43, 44, 45, 46]
SETTINGS = [(d, m) for d in DATASETS for m in MODELS]

# ---------------------------------------------------------------- feature extraction
EXTRACTION = dict(
    prompt_style="auto",          # chat template when the tokenizer has one
    max_new_tokens=32,            # greedy answer length
    target_layers=(-4, -9, -14),  # decoder blocks whose answer-token mean forms h
    max_keywords=12,              # keywords used for H-bar (lowest teacher-forced margin first)
    baseline_samples=5,           # extra sampled answers, used ONLY by the PSA / EigenScore baselines
    sample_temperature=1.0,
    sample_top_p=0.95,
    logit_temperature=0.7,        # only for the temperature-scaled logit baselines
    row_limit=20000,              # rows sampled (seed 42) from each dataset
    popqa_max_subject_pop=1000,   # long-tail PopQA subset
)

# ---------------------------------------------------------------- HEED
# stage 1: e_h, evidential MLP on h
EH = dict(hidden_dim=128, dropout=0.4, lr=1e-3, weight_decay=1e-3, epochs=120, batch_size=256, patience=20,
          kl_weight=0.01, kl_anneal_epochs=20, grad_clip=5.0, monitor="composite")  # 0.5*BAcc + 0.5*AUROC
# stage 2: e_H = softplus(w * z(H-bar) + b), trained with e_h frozen
EHBAR = dict(init_bias=-2.0, lr=1e-2, weight_decay=1e-4, epochs=120, batch_size=256, patience=20,
             kl_weight=0.01, kl_anneal_epochs=20, grad_clip=5.0, monitor="auroc")
HBAR_FEATURE = "mean_entropy"   # name of H-bar in the feature file's scalar_names


def feature_path(dataset: str, model: str, balanced: bool = True) -> Path:
    """Default file layout written by scripts/extract_features.py and scripts/balance_features.py.
    HEED_FEATURE_PATTERN (e.g. '{root}/features_{dataset}_{model}_balanced.pt') overrides it."""
    pattern = os.environ.get("HEED_FEATURE_PATTERN")
    if pattern and balanced:
        return Path(pattern.format(root=FEATURE_DIR, dataset=dataset, model=model))
    return FEATURE_DIR / f"{dataset}_{model}{'_balanced' if balanced else ''}.pt"


def tag(dataset: str, model: str, seed: int) -> str:
    return f"{dataset}_{model}_s{seed}"
