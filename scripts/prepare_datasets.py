"""Download and normalise the four QA datasets into data/.

  popqa.tsv                          PopQA test set (akariasai/PopQA)
  webquestions.jsonl                 WebQuestions train + test (stanfordnlp/web_questions)
  triviaqa_unfiltered_nocontext.jsonl  TriviaQA unfiltered.nocontext train + validation + test
  mkqa_en.jsonl                      MKQA English, short-answer types only

JSONL rows: {"id", "source", "split", "question", "answers": [aliases]}  (WebQuestions keeps "url").
    python scripts/prepare_datasets.py [--only popqa,webquestions,triviaqa,mkqa]
"""
import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from heed.config import DATA_DIR  # noqa: E402

HF = "https://huggingface.co/datasets"
POPQA_URL = f"{HF}/akariasai/PopQA/resolve/main/test.tsv"
WEBQ_URL = f"{HF}/stanfordnlp/web_questions/resolve/main/data/{{split}}-00000-of-00001.parquet"
TRIVIAQA_URL = f"{HF}/mandarjoshi/trivia_qa/resolve/main/unfiltered.nocontext/{{split}}-00000-of-00001.parquet"
MKQA_URL = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"
RAW = DATA_DIR / "raw"
MKQA_TYPES = {0: "entity", 1: "long_answer", 2: "unanswerable", 3: "date", 4: "number", 5: "number_with_unit",
              6: "short_phrase", 7: "binary"}
MKQA_KEEP = {"entity", "date", "number", "number_with_unit", "short_phrase"}


def download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path.exists() and path.stat().st_size > 0):
        print(f">>> downloading {url}", flush=True)
        subprocess.run(["curl", "-L", "--fail", "--retry", "3", "--retry-delay", "5", "-o", str(path), url], check=True)
    return path


def unique_nonempty(values: Iterable[object]) -> List[str]:
    seen, out = set(), []
    for v in values:
        t = str(v).strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def write_jsonl(path: Path, rows) -> None:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f">>> {path}: {n} rows", flush=True)


def popqa():
    download(POPQA_URL, DATA_DIR / "popqa.tsv")


def webquestions():
    def rows():
        for split in ("train", "test"):
            for r in pq.read_table(download(WEBQ_URL.format(split=split), RAW / "webquestions" / f"{split}.parquet")).to_pylist():
                yield {"split": split, "url": r["url"], "question": r["question"], "answers": list(r["answers"])}
    write_jsonl(DATA_DIR / "webquestions.jsonl", rows())


def triviaqa():
    def answers(a):
        vals = [a.get(k) for k in ("value", "normalized_value", "matched_wiki_entity_name",
                                   "normalized_matched_wiki_entity_name") if a.get(k)]
        for k in ("aliases", "normalized_aliases"):
            vals.extend(a.get(k) or [])
        return unique_nonempty(vals)

    def rows():
        for split in ("train", "validation", "test"):
            path = download(TRIVIAQA_URL.format(split=split), RAW / "triviaqa_unfiltered_nocontext" / f"{split}-00000-of-00001.parquet")
            for i, item in enumerate(pq.read_table(path, columns=["question", "question_id", "answer"]).to_pylist()):
                ans = answers(item.get("answer") or {})
                if ans:
                    yield {"id": str(item.get("question_id", f"triviaqa-{split}-{i}")), "source": "triviaqa",
                           "split": split, "question": str(item["question"]).strip(), "answers": ans}
    write_jsonl(DATA_DIR / "triviaqa_unfiltered_nocontext.jsonl", rows())


def mkqa():
    path = download(MKQA_URL, RAW / "mkqa" / "mkqa.jsonl.gz")

    def rows():
        emitted = 0
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                vals = []
                for a in (item.get("answers") or {}).get("en", []):
                    t = a.get("type", "")
                    if (MKQA_TYPES.get(t, str(t)) if isinstance(t, int) else str(t)) in MKQA_KEEP:
                        vals += ([a["text"]] if a.get("text") else []) + list(a.get("aliases") or [])
                ans = unique_nonempty(vals)
                if ans:
                    yield {"id": str(item.get("example_id", f"mkqa-{emitted}")), "source": "mkqa_en", "split": "train",
                           "question": str(item.get("query") or item.get("queries", {}).get("en", "")).strip(),
                           "answers": ans}
                    emitted += 1
    write_jsonl(DATA_DIR / "mkqa_en.jsonl", rows())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="popqa,webquestions,triviaqa,mkqa")
    a = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in a.only.split(","):
        globals()[name]()
