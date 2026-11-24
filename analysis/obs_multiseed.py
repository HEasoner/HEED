"""Re-run Obs.1 (relative factual support, PCA-128, k=10, all 24 settings) and Obs.3 (support-deficient region
ER, PopQA x 6 models, k=10, Q0.15 / Q0.05) on the seed splits 42-46, re-using
(obs1_relative_support.compute_case_density, obs3_support_deficient_region.analyze_case).
No figures are written; results -> outputs/analysis/obs_seeds.json.
"""
import argparse
import json
from types import SimpleNamespace

import obs1_relative_support as OBS1
import obs3_support_deficient_region as OBS3
from _paths import RESULTS_DIR

p = argparse.ArgumentParser()
p.add_argument("--seeds", default="42,43,44,45,46")
p.add_argument("--device", default="cuda")
a = p.parse_args()
out = RESULTS_DIR / "obs_seeds.json"
res = json.loads(out.read_text()) if out.exists() else {"obs1": {}, "obs3": {}}
for s in [int(x) for x in a.seeds.split(",")]:
    a1 = SimpleNamespace(field=OBS1.DEFAULT_FIELD, seed=s, max_train_reference_per_class=0, max_test_per_class=0,
                         pca_dim=128, k=10, eps=1e-8, device=a.device, test_batch_size=128, reference_batch_size=2048)
    for m, _ in OBS1.MODELS:
        for d, _ in OBS1.DATASETS:
            key = f"{d}/{m}/s{s}"
            if key in res["obs1"]:
                continue
            meta = OBS1.compute_case_density(d, m, a1)["meta"]
            rs = meta["relative_factual_support"]
            res["obs1"][key] = dict(know_mean=rs["KNOW"]["mean"], hall_mean=rs["HALLUCINATION"]["mean"],
                                    know_median=rs["KNOW"]["median"], hall_median=rs["HALLUCINATION"]["median"],
                                    void_auroc=meta["metrics"]["relative_factual_support_void_score"].get("auroc"),
                                    test_n=meta["split_sizes"]["test"])
            out.write_text(json.dumps(res, indent=1))
    a3 = SimpleNamespace(field="probe_target_hidden_mean_answer", seed=s, k=10, eps=1e-8, know_support_quantile=0.15,
                         overall_density_quantile=0.05, grid_size=60, extent_quantile=0.995, extent_pad=0.08,
                         visual_occupancy_quantile=0.95, device=a.device, query_batch_size=128,
                         reference_batch_size=512)
    for m, _ in OBS3.MODELS:
        key = f"popqa/{m}/s{s}"
        if key in res["obs3"]:
            continue
        c = OBS3.analyze_case("popqa", m, a3)
        res["obs3"][key] = dict(er=c["enrichment_ratio"], hall_entry=c["hall_entry"], know_entry=c["know_entry"],
                                train_candidate_rate=c["train_candidate_rate"], test_n=c["test_n"])
        out.write_text(json.dumps(res, indent=1))
    print(f"seed {s} done", flush=True)
