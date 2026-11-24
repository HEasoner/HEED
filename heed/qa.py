"""QA-side utilities: answer parsing and the KNOW / HALL labelling rule.

PopQA uses strict matching (a normalised gold alias appears as a substring of the normalised answer);
WebQuestions, TriviaQA and MKQA use the relaxed rule of answer_match_details(relaxed=True).
(Copied unchanged from the code that produced the paper's feature files.)
"""
import ast
import re
import string
import unicodedata
from typing import Dict, List, Optional, Tuple

import pandas as pd

_PUNCT_TABLE = set(string.punctuation)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "who",
    "what",
    "when",
    "where",
    "which",
    "why",
}

def parse_possible_answers(value) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    except (SyntaxError, ValueError):
        return [text]

def normalize_answer_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def remove_answer_articles(text: str) -> str:
    return " ".join(
        token
        for token in normalize_answer_text(text).split()
        if token not in {"a", "an", "the"}
    )

def answer_token_overlap(generated_answer: str, answer: str) -> Tuple[float, float]:
    generated_tokens = set(remove_answer_articles(generated_answer).split())
    answer_tokens = set(remove_answer_articles(answer).split())
    if not generated_tokens or not answer_tokens:
        return 0.0, 0.0
    recall = len(generated_tokens & answer_tokens) / len(answer_tokens)
    precision = len(generated_tokens & answer_tokens) / len(generated_tokens)
    return recall, precision

def answer_match_details(
    generated_answer: str,
    possible_answers: List[str],
    relaxed: bool = False,
) -> Dict[str, object]:
    normalized_generated = f" {normalize_answer_text(generated_answer)} "
    generated_no_articles = f" {remove_answer_articles(generated_answer)} "
    generated_no_articles_token_count = len(generated_no_articles.split())

    best = {
        "matched": False,
        "match_type": "none",
        "matched_answer": None,
        "token_recall": 0.0,
        "token_precision": 0.0,
    }

    for answer in possible_answers:
        normalized_answer = normalize_answer_text(answer)
        if not normalized_answer:
            continue
        if f" {normalized_answer} " in normalized_generated:
            return {
                "matched": True,
                "match_type": "strict_substring",
                "matched_answer": str(answer),
                "token_recall": 1.0,
                "token_precision": 1.0,
            }

        if not relaxed:
            continue

        answer_no_articles = remove_answer_articles(answer)
        if not answer_no_articles:
            continue

        recall, precision = answer_token_overlap(generated_answer, answer)
        if recall > best["token_recall"]:
            best.update(
                {
                    "matched_answer": str(answer),
                    "token_recall": float(recall),
                    "token_precision": float(precision),
                }
            )

        if f" {answer_no_articles} " in generated_no_articles:
            return {
                "matched": True,
                "match_type": "no_article_substring",
                "matched_answer": str(answer),
                "token_recall": float(recall),
                "token_precision": float(precision),
            }

        # Useful for SimpleQA answers like "multiple sclerosis diagnosis" where
        # the generated answer is the shorter factual span "multiple sclerosis".
        generated_span = generated_no_articles.strip()
        if (
            generated_span
            and generated_no_articles_token_count >= 2
            and f" {generated_span} " in f" {answer_no_articles} "
        ):
            return {
                "matched": True,
                "match_type": "generated_span_in_gold",
                "matched_answer": str(answer),
                "token_recall": float(recall),
                "token_precision": float(precision),
            }

        # Allows order-insensitive short factual answers such as dates
        # ("25 November 2018" vs "November 25, 2018") and reversed names.
        answer_token_count = len(answer_no_articles.split())
        if answer_token_count <= 8 and recall >= 0.8 and precision >= 0.25:
            return {
                "matched": True,
                "match_type": "token_recall",
                "matched_answer": str(answer),
                "token_recall": float(recall),
                "token_precision": float(precision),
            }

    return best

def generated_contains_answer(generated_answer: str, possible_answers: List[str]) -> bool:
    return bool(
        answer_match_details(
            generated_answer,
            possible_answers,
            relaxed=False,
        )["matched"]
    )

def generated_matches_answer_relaxed(
    generated_answer: str,
    possible_answers: List[str],
) -> Dict[str, object]:
    return answer_match_details(
        generated_answer,
        possible_answers,
        relaxed=True,
    )

def is_content_token(token_text: str, prompt_terms: Optional[set] = None) -> bool:
    stripped = token_text.strip()
    if not stripped:
        return False
    if all(ch in _PUNCT_TABLE for ch in stripped):
        return False
    if not any(ch.isalnum() for ch in stripped):
        return False

    normalized = normalize_answer_text(stripped)
    if not normalized:
        return False
    pieces = normalized.split()
    if any(piece in _STOPWORDS for piece in pieces):
        return False

    if prompt_terms:
        for piece in pieces:
            if len(piece) <= 2:
                return False
            if piece in prompt_terms:
                return False
            if any(len(piece) >= 3 and piece in term for term in prompt_terms):
                return False
    return True


DATA_FILES = {"popqa": "popqa.tsv", "webquestions": "webquestions.jsonl",
              "triviaqa": "triviaqa_unfiltered_nocontext.jsonl", "mkqa": "mkqa_en.jsonl"}


def load_questions(dataset: str, path, limit: Optional[int] = 20000, seed: int = 42,
                   popqa_max_subject_pop: Optional[int] = 1000) -> pd.DataFrame:
    """Rows used in the paper: PopQA long tail (s_pop <= 1000); then at most `limit` rows sampled with `seed`."""
    if dataset == "popqa":
        df = pd.read_csv(path, sep="\t")
        if popqa_max_subject_pop is not None:
            df = df[df["s_pop"].fillna(10**18).astype(float) <= popqa_max_subject_pop]
    else:
        df = pd.read_json(path, lines=True)
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=seed)
    return df


def question_and_answers(dataset: str, row) -> Tuple[str, List[str], Dict[str, object]]:
    if dataset == "popqa":
        answers = parse_possible_answers(row["possible_answers"])
        source = {"possible_answers": answers,
                  "s_pop": int(row["s_pop"]) if pd.notna(row.get("s_pop", None)) else None,
                  "id": int(row["id"]) if pd.notna(row.get("id", None)) else None}
    else:
        answers = parse_possible_answers(row["answers"])
        source = {"answers": answers, "split": row.get("split"), "source": row.get("source"), "id": row.get("id")}
        if dataset == "webquestions":
            source["url"] = row.get("url")
    return row["question"], answers, source


def label_answer(dataset: str, generated: str, answers: List[str]) -> Tuple[int, Dict[str, object]]:
    """0 = KNOW (the answer recalls a gold alias), 1 = HALL (false recall)."""
    match = answer_match_details(generated, answers, relaxed=dataset != "popqa")
    return (0 if match["matched"] else 1), match
