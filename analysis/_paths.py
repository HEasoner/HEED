"""Paths for the analysis scripts (all derived from heed/config.py and its environment variables).

HEED_MODEL_DIR: optional directory holding local model folders (e.g. $HEED_MODEL_DIR/Qwen2.5-7B-Instruct);
otherwise models are loaded from the Hugging Face hub.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from heed import config as C  # noqa: E402,F401

PROJECT_ROOT = REPO_ROOT
PT_DIR = C.FEATURE_DIR
RESULTS_DIR = C.OUTPUT_DIR / "analysis"
FIG_DIR = C.FIGURE_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# local folder name used by the intervention scripts -> Hugging Face id
LOCAL_MODEL_NAMES = {
    "Llama-2-7b-chat-hf": C.MODELS["llama2_7b_chat"],
    "Llama-3-8B-Instruct": C.MODELS["llama3_8b_instruct"],
    "Qwen2-7B-Instruct": C.MODELS["qwen2_7b_instruct"],
    "Qwen2.5-7B-Instruct": C.MODELS["qwen2_5_7b_instruct"],
    "gemma-2-9b-it": C.MODELS["gemma_2_9b_it"],
    "deepseek-llm-7b-chat": C.MODELS["deepseek_7b_chat"],
}


def feature_file(dataset: str, model: str, balanced: bool = True) -> Path:
    return C.feature_path(dataset, model, balanced=balanced)


def feature_from_name(name: str) -> Path:
    """'<dataset>_<model>' -> balanced feature file."""
    dataset, model = Path(str(name)).name.split("_", 1)
    return feature_file(dataset, model)


def local_model_path(name: str) -> str:
    root = os.environ.get("HEED_MODEL_DIR")
    if root and (Path(root) / name).exists():
        return str(Path(root) / name)
    return LOCAL_MODEL_NAMES.get(name, name)


def resolve_input_path(path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    for cand in (Path.cwd() / path, PROJECT_ROOT / path, PT_DIR / path, RESULTS_DIR / path):
        if cand.exists():
            return cand
    return PROJECT_ROOT / path


def resolve_output_path(path, default_dir: Path = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        out = path
    elif path.parent == Path(".") and default_dir is not None:
        out = default_dir / path.name
    else:
        out = PROJECT_ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    return out
