"""Characterize HALL answers as specific / type-consistent / abstention.

Uses only stored generations (all HALL answers of the unbalanced v4.1 files; no regeneration).
Categories are mutually exclusive in this order:
  abstention        explicit refusal / "I don't know" / no-information answers
  premise_rejection answer denies the question premise ("X did not appear in ...", "there is no ...")
  hedged            commits only with hedges ("possibly", "probably", "I think", ...)
  specific          none of the above and contains a concrete content token (noun / proper noun /
                    number) that is not copied from the question
Type consistency (computed for every HALL answer; reported over specific answers and over all):
  PopQA  relation template: the answer must contain an entity of the relation's answer type taken
         from the PopQA table itself (objects + aliases of all rows of the same relation group).
         Closed-world, so it is a lower bound.  Relaxed variant also accepts a proper-noun phrase
         not copied from the question for person/location relations.
  Others expected type from the question form (who / where / when / how many / what-X head noun),
         checked by rules (year/month/number patterns, proper nouns).  With --spacy-model the
         person/location/org checks use spaCy NER instead.  Untyped questions are excluded from the
         type-consistency denominator, and the typed fraction is reported.
PopQA popularity: for type-consistent HALL answers whose entity is in the PopQA table, compare its
Wikipedia popularity (PopQA s_pop/o_pop) with the gold object's o_pop, against a same-relation
random-object null.
"""
import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from _paths import C, PROJECT_ROOT, PT_DIR, RESULTS_DIR, feature_file

MODELS = ["llama2_7b_chat", "llama3_8b_instruct", "qwen2_7b_instruct",
          "qwen2_5_7b_instruct", "gemma_2_9b_it", "deepseek_7b_chat"]
DATASETS = ["popqa", "webquestions", "triviaqa", "mkqa"]
OUT = RESULTS_DIR / "characterization"

PERSON_RELS = {"father", "mother", "director", "producer", "screenwriter", "composer", "author"}
LOCATION_RELS = {"place of birth", "country", "capital of", "capital"}
CATEGORICAL_RELS = {"occupation", "genre", "color", "religion", "sport"}

ABSTAIN = re.compile(
    r"\b(i (do not|don't|cannot|can't|could not|couldn't|am unable to|am not able to|have no|do not have|don't have)"
    r" (know|find|provide|answer|determine|say|confirm|verify|be sure|recall|access|information|enough|any|have)"
    r"|i'?m (not (sure|certain|aware|able)|unable|sorry)|i am (not (sure|certain|aware|able)|unable|sorry)"
    r"|i apologi[sz]e|sorry, |no (reliable |specific |available |public )?(information|record|data|answer|details)"
    r"|not enough information|insufficient information|(is|are|remains) (unknown|not known|unclear)"
    r"|unable to (find|determine|provide|answer|verify)|cannot be determined|not (publicly )?available"
    r"|not specified|not mentioned|not provided|without more (context|information)|could you (please )?(clarify|provide))"
    r"|^\W*unknown\W*$|not (explicitly |publicly |widely |well[- ])?(mentioned|known|documented|recognized|recognised)"
    r"|impossible to (determine|say|know|answer)|not a recognized|trick question|not the capital of any"
    r"|(question|title) (seems|is|appears) (to be )?(incomplete|unclear|ambiguous|too broad)"
    r"|could refer to (many|multiple|several|various)|(please|can you|could you) (provide|specify|clarify)"
    r"|need more (context|information)",
    re.I)
PREMISE = re.compile(
    r"\b(did not|didn't|does not|doesn't|do not|don't|is not|isn't|was not|wasn't|were not|has not|hasn't|have not|"
    r"had not|never) (appear|exist|play|star|write|direct|produce|compose|have|win|marry|act|make|record|sing|"
    r"release|serve|live|die|hold|become|work|attend|own|found|invent|discover|belong|participate|take part)"
    r"|there (is|are|was|were|has been|have been) no\b|no such\b|does not exist|doesn't exist|not a real|fictional|"
    r"is not known for|is not associated|has no (known )?(father|mother|children|capital)",
    re.I)
HEDGE = re.compile(
    r"\b(possibly|probably|perhaps|maybe|likely|might be|may be|could be|i think|i believe|i guess|it is possible|"
    r"it's possible|uncertain|unclear|some sources|either\b.+\bor)\b",
    re.I)
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})s?\b|\b[0-9]{1,2}(st|nd|rd|th) century\b|\bcentury\b"
                  r"|\b[0-9][0-9,]*\s*(bce|bc|ad|ce)\b", re.I)
MONTH = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I)
NUMBER = re.compile(r"\b[0-9][0-9,\.]*\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                    r"twenty|thirty|forty|fifty|hundred|thousand|million|billion|dozen)\b", re.I)
STOP = set("a an and are as at be by for from he her his i in is it its of on or she the their this to was were who "
           "what when where which why how whom whose with that they them there these those has have had did do does "
           "not no yes also known best famous one first".split())
TOKEN = re.compile(r"[A-Za-z0-9À-ɏ](?:[A-Za-z0-9À-ɏ'\-]|\.(?=[A-Za-z0-9]))*")

# WordNet hypernym targets for categorical PopQA relations (local nltk WordNet data).
WN_TARGETS = {
    "occupation": ["person.n.01"],
    "sport": ["sport.n.01", "athletic_game.n.01", "athlete.n.01"],
    "genre": ["music_genre.n.01", "genre.n.01", "genre.n.02", "genre.n.03", "literary_composition.n.01",
              "dramatic_composition.n.01", "music.n.01", "fiction.n.01", "film.n.01"],
    "religion": ["religion.n.01", "religious_person.n.01", "religious_movement.n.01"],
    "color": ["color.n.01", "chromatic_color.n.01"],
}
_WN_CACHE = {}


def wordnet_ok(prop: str, answer: str, question: str) -> bool:
    from nltk.corpus import wordnet as wn
    targets = WN_TARGETS.get(prop)
    if not targets:
        return False
    qn = set(norm(question).split())
    words = [w for w in norm(answer).split() if w not in STOP and w not in qn and len(w) > 2]
    grams = words + ["_".join(words[i:i + 2]) for i in range(len(words) - 1)]
    for g in grams:
        key = (prop, g)
        if key not in _WN_CACHE:
            tset = {wn.synset(t) for t in targets}
            hit = False
            for s in wn.synsets(g, pos=wn.NOUN):
                closure = set(s.closure(lambda x: x.hypernyms() + x.instance_hypernyms())) | {s}
                if closure & tset:
                    hit = True
                    break
            _WN_CACHE[key] = hit
        if _WN_CACHE[key]:
            return True
    return False

PERSON_HEADS = {"actor", "actress", "president", "author", "writer", "singer", "king", "queen", "player", "man",
                "woman", "person", "artist", "composer", "director", "poet", "painter", "leader", "minister",
                "character", "wife", "husband", "son", "daughter", "father", "mother", "emperor", "prince",
                "princess", "scientist", "inventor", "coach", "manager", "presenter", "musician", "band member",
                "footballer", "general", "pope", "saint", "god", "goddess", "explorer", "architect", "designer"}
LOC_HEADS = {"country", "city", "state", "county", "continent", "island", "river", "mountain", "town", "capital",
             "region", "province", "nation", "village", "lake", "sea", "ocean", "desert", "district", "place"}
ORG_HEADS = {"team", "company", "band", "university", "college", "party", "organization", "organisation", "club",
             "newspaper", "airline", "network", "school", "group", "brand", "firm", "league"}
DATE_HEADS = {"year", "date", "day", "month", "century", "decade", "era"}


def fires(pattern, answer: str, question: str) -> bool:
    qn = " " + norm(question).replace("'", "") + " "
    return any(" " + norm(m.group(0)).replace("'", "") + " " not in qn for m in pattern.finditer(answer))


def norm(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def ngrams(words, nmax=6):
    for n in range(min(nmax, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            yield " ".join(words[i:i + n])


def load_popqa():
    rows = {}
    group_vocab = defaultdict(dict)    # group -> name -> pop
    global_pop = {}
    path = C.DATA_DIR / "popqa.tsv"
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            prop = r["prop"]
            group = "person" if prop in PERSON_RELS else "location" if prop in LOCATION_RELS else prop
            names_o = [r["obj"]] + json.loads(r["o_aliases"] or "[]")
            names_s = [r["subj"]] + json.loads(r["s_aliases"] or "[]")
            o_pop, s_pop = int(r["o_pop"]), int(r["s_pop"])
            for n in names_o:
                k = norm(n)
                if len(k) >= 3:
                    group_vocab[group][k] = max(group_vocab[group].get(k, 0), o_pop)
                    global_pop[k] = max(global_pop.get(k, 0), o_pop)
            for n in names_s:
                k = norm(n)
                if len(k) >= 3:
                    global_pop[k] = max(global_pop.get(k, 0), s_pop)
            rows[int(r["id"])] = dict(prop=prop, group=group, obj=r["obj"], o_pop=o_pop)
    return rows, group_vocab, global_pop


def content_tokens(answer: str, question: str, tagger):
    qset = {w.lower() for w in TOKEN.findall(question)}
    toks = TOKEN.findall(answer)
    tags = tagger(toks) if toks else []
    keep = []
    for (w, t) in tags:
        lw = w.lower().strip(".'-")
        if lw in STOP or lw in qset or len(lw) < 2 and not lw.isdigit():
            continue
        keep.append((w, t))
    return keep


def proper_phrase(cands):
    # capitalised content token not copied from the question (POS tags are unreliable on names)
    return any(w[:1].isupper() for w, _ in cands)


WH_RULES = [
    (r"\b(what year|which year|in what year|what date|what century|which century|what decade|when)\b", "DATE"),
    (r"\b(how many|how much|how long|how old|how tall|how far|how big|how deep|how high|what percentage|"
     r"what percent|what number|how fast)\b", "NUMBER"),
    (r"\b(who|whom|whose)\b", "PERSON"),
    (r"\bwhere\b", "LOCATION"),
]


def expected_type(question: str) -> str:
    q = norm(question)
    best = (10 ** 9, "UNTYPED")
    for pat, typ in WH_RULES:
        m = re.search(pat, q)
        if m and m.start() < best[0]:
            best = (m.start(), typ)
    m = re.search(r"\b(what|which)(?: \w+)?? (\w+)\b", q)
    if m and m.start() < best[0]:
        head = m.group(2)
        if head.endswith("s") and head[:-1] in PERSON_HEADS | LOC_HEADS | ORG_HEADS | DATE_HEADS:
            head = head[:-1]
        for heads, typ in ((DATE_HEADS, "DATE"), (LOC_HEADS, "LOCATION"), (PERSON_HEADS, "PERSON"), (ORG_HEADS, "ORG")):
            if head in heads:
                return typ
    return best[1]


def type_ok_rule(etype: str, answer: str, cands, ents=None) -> bool:
    if etype == "DATE":
        return bool(YEAR.search(answer) or MONTH.search(answer)) or bool(ents and ents & {"DATE"})
    if etype == "NUMBER":
        return bool(NUMBER.search(answer)) or bool(ents and ents & {"CARDINAL", "QUANTITY", "MONEY", "PERCENT"})
    if ents is not None:
        want = {"PERSON": {"PERSON"}, "LOCATION": {"GPE", "LOC", "FAC"}, "ORG": {"ORG"}}[etype]
        return bool(ents & want)
    return proper_phrase(cands)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacy-model", default=None, help="path to an approved spaCy model (optional)")
    ap.add_argument("--tag", default="rules")
    a = ap.parse_args()
    import nltk
    tagger = nltk.pos_tag
    nlp = None
    if a.spacy_model:
        import spacy
        nlp = spacy.load(a.spacy_model, disable=["lemmatizer"])
    rows, group_vocab, global_pop = load_popqa()
    rng = np.random.default_rng(42)
    rel_objects = defaultdict(list)
    for r in rows.values():
        rel_objects[r["prop"]].append(r["o_pop"])
    OUT.mkdir(parents=True, exist_ok=True)
    per_answer = []
    summary = []
    for dataset in DATASETS:
        for model in MODELS:
            data = torch.load(str(feature_file(dataset, model, balanced=False)),
                              map_location="cpu", mmap=True)
            recs = [r for r in data["records"] if r["label"] == "HALLUCINATION"]
            del data
            answers = [str(r["generated_answer"]) for r in recs]
            docs = list(nlp.pipe(answers, batch_size=256)) if nlp else [None] * len(recs)
            cnt = defaultdict(int)
            pop_rows = []
            for r, ans, doc in zip(recs, answers, docs):
                q = str(r["question"])
                cands = content_tokens(ans, q, tagger)
                if fires(ABSTAIN, ans, q):
                    cat = "abstention"
                elif fires(PREMISE, ans, q):
                    cat = "premise_rejection"
                elif fires(HEDGE, ans, q):
                    cat = "hedged"
                elif cands:
                    cat = "specific"
                else:
                    cat = "other"
                cnt[cat] += 1
                ents = None
                if doc is not None:
                    qn = norm(q)
                    ents = {e.label_ for e in doc.ents if norm(e.text) and norm(e.text) not in qn}
                qnorm = norm(q)
                typed, ok_strict, ok_relaxed, wrong_pop, gold_pop, prop = True, False, False, None, None, ""
                if dataset == "popqa":
                    src = r["label_source"]
                    src = ast.literal_eval(src) if isinstance(src, str) else src
                    row = rows[int(src["id"])]
                    prop, group, gold_pop = row["prop"], row["group"], row["o_pop"]
                    vocab = group_vocab[group]
                    hit = None
                    for g in ngrams(norm(ans).split()):
                        if g in vocab and g not in qnorm and g not in STOP:
                            hit = g
                            break
                    ok_strict = hit is not None
                    if group in ("person", "location"):
                        if ents is not None:
                            want = {"PERSON"} if group == "person" else {"GPE", "LOC", "FAC"}
                            ok_relaxed = ok_strict or bool(ents & want)
                        else:
                            ok_relaxed = ok_strict or proper_phrase(cands)
                    else:
                        ok_relaxed = ok_strict or wordnet_ok(prop, ans, q)
                    if ok_strict:
                        wrong_pop = vocab.get(hit, global_pop.get(hit))
                        if wrong_pop is not None and norm(row["obj"]) != hit:
                            null = rng.choice(rel_objects[prop], size=20)
                            pop_rows.append((wrong_pop, gold_pop, float(np.mean(null > gold_pop))))
                else:
                    etype = expected_type(q)
                    typed = etype != "UNTYPED"
                    if typed:
                        ok_strict = ok_relaxed = type_ok_rule(etype, ans, cands, ents)
                    prop = etype
                cnt["typed"] += typed
                cnt["type_ok_strict"] += typed and ok_strict
                cnt["type_ok_relaxed"] += typed and ok_relaxed
                if cat == "specific":
                    cnt["spec_typed"] += typed
                    cnt["spec_type_ok_strict"] += typed and ok_strict
                    cnt["spec_type_ok_relaxed"] += typed and ok_relaxed
                per_answer.append(dict(dataset=dataset, model=model, question=q, answer=ans, category=cat,
                                       expected=prop, typed=typed, type_ok_strict=ok_strict,
                                       type_ok_relaxed=ok_relaxed, wrong_pop=wrong_pop, gold_pop=gold_pop))
            n = len(recs)
            row = dict(dataset=dataset, model=model, n_hall=n,
                       specific=cnt["specific"] / n, hedged=cnt["hedged"] / n, abstention=cnt["abstention"] / n,
                       premise_rejection=cnt["premise_rejection"] / n, other=cnt["other"] / n,
                       typed_frac=cnt["typed"] / n,
                       type_ok_strict_all=cnt["type_ok_strict"] / max(cnt["typed"], 1),
                       type_ok_relaxed_all=cnt["type_ok_relaxed"] / max(cnt["typed"], 1),
                       type_ok_strict_specific=cnt["spec_type_ok_strict"] / max(cnt["spec_typed"], 1),
                       type_ok_relaxed_specific=cnt["spec_type_ok_relaxed"] / max(cnt["spec_typed"], 1))
            if pop_rows:
                w, g, nullp = map(np.asarray, zip(*pop_rows))
                pos = (w > 0) & (g > 0)
                row.update(pop_n=len(w), pop_more_popular=float(np.mean(w > g)),
                           pop_null_more_popular=float(np.mean(nullp)),
                           pop_median_log10_ratio=float(np.median(np.log10(w[pos] / g[pos]))))
            summary.append(row)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    with (OUT / f"summary_{a.tag}.csv").open("w", newline="") as fh:
        keys = sorted({k for r in summary for k in r}, key=lambda k: list(summary[0]).index(k) if k in summary[0] else 99)
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    with (OUT / f"per_answer_{a.tag}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_answer[0]))
        w.writeheader()
        w.writerows(per_answer)


if __name__ == "__main__":
    main()
