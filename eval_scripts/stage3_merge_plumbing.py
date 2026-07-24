"""
Stage 3 -- plumbing check only (per task spec step 4): run baselines +
proposed method once at tau=0.4, mu=1.0 across all 224 real locations to
confirm the full pipeline wires together correctly. This is NOT the tuned
experiment (tau/mu sweeps = Experiments C/D, real accuracy = Stage 4, both
need the Llama-3-8B base model + GPU forward passes -- out of scope here).

Reuses:
  - Stage 1 cache (Bc_t/Ac_t/M_t/U_ref/V_ref per location) from
    core_space_cache.pkl -- no SVDs re-run.
  - Stage 2 gate: per_location_cv from contamination_cv_result.json,
    computed earlier directly on the original (r x n) A_t subspaces --
    same definition as the (Tr x Tr) core-space Pi_t (isometric embedding,
    see experiment note), so no recomputation needed.
"""
import json
import os
import pickle
import time

import numpy as np

from core_space_common import (
    ta_merge, ties_merge_np, dare_ties_merge_np, isoc_merge_np, proposed_merge,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(REPO_ROOT, "eval_scripts", "core_space_cache.pkl")
CV_PATH = os.path.join(REPO_ROOT, "eval_scripts", "contamination_cv_result.json")

TAU = 0.4
MU = 1.0


def frob(m):
    return float(np.linalg.norm(m))


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    with open(CV_PATH) as f:
        cv_result = json.load(f)
    per_location_cv = cv_result["per_location_cv"]

    keys = list(cache.keys())
    assert set(keys) == set(per_location_cv.keys()), "Stage1 cache and Stage2 CV keys must match exactly"
    print(f"{len(keys)} locations loaded; tau={TAU}, mu={MU}")

    n_gated = 0
    rows = []
    t0 = time.time()
    for key in keys:
        core = cache[key]
        m_list = core["M_list"]
        cv = per_location_cv[key]
        gated = cv > TAU

        m_ta = ta_merge(m_list)
        m_ties = ties_merge_np(m_list)
        m_dare = dare_ties_merge_np(m_list)
        m_isoc = isoc_merge_np(m_list)

        if gated:
            n_gated += 1
            m_proposed, phi, owner = proposed_merge(core, MU)
        else:
            m_proposed = m_ta  # skipped per Stage-3 gate rule -> plain sum
            phi = np.ones(core["T"] * core["r"])

        # non-destructiveness sanity at the per-location level: gated=False
        # locations must equal TA exactly; for gated=True locations, phi
        # should have moved away from 1 (mu=1.0 actually did something).
        if not gated:
            assert np.allclose(m_proposed, m_ta), f"ungated location {key} deviated from TA"

        rows.append({
            "key": key,
            "cv": cv,
            "gated": gated,
            "phi_mean": float(phi.mean()),
            "phi_std": float(phi.std()),
            "frob_ta": frob(m_ta),
            "frob_ties": frob(m_ties),
            "frob_dare_ties": frob(m_dare),
            "frob_isoc": frob(m_isoc),
            "frob_proposed": frob(m_proposed),
            "proposed_vs_ta_rel_diff": float(np.linalg.norm(m_proposed - m_ta) / max(frob(m_ta), 1e-12)),
        })

        for name, m in [("TA", m_ta), ("TIES", m_ties), ("DARE-TIES", m_dare),
                         ("Iso-C", m_isoc), ("Proposed", m_proposed)]:
            assert np.all(np.isfinite(m)), f"non-finite values in {name} at {key}"

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s")
    print(f"\ngated locations: {n_gated}/{len(keys)} ({100*n_gated/len(keys):.1f}%) at tau={TAU}")

    gated_rows = [r for r in rows if r["gated"]]
    ungated_rows = [r for r in rows if not r["gated"]]

    print(f"\nungated (n={len(ungated_rows)}): proposed==TA exactly, by construction -- verified above.")
    if gated_rows:
        rel_diffs = [r["proposed_vs_ta_rel_diff"] for r in gated_rows]
        phi_stds = [r["phi_std"] for r in gated_rows]
        print(f"gated (n={len(gated_rows)}):")
        print(f"  proposed-vs-TA relative Frobenius diff: mean={np.mean(rel_diffs):.4f} "
              f"min={np.min(rel_diffs):.4f} max={np.max(rel_diffs):.4f}")
        print(f"  phi* std within location: mean={np.mean(phi_stds):.4f} "
              f"(0 would mean mu=1.0 had no effect -- should be >0)")

    # module-type breakdown of the gate, sanity cross-check against the
    # earlier CV diagnostic (down_proj/gate_proj early layers should be
    # heavily gated, k_proj should be almost never gated)
    import re
    from collections import defaultdict
    by_module = defaultdict(lambda: [0, 0])
    for r in rows:
        m = re.search(r"\.(mlp|self_attn)\.(\w+_proj)\.lora_A", r["key"]).group(2)
        by_module[m][0] += int(r["gated"])
        by_module[m][1] += 1
    print("\ngate rate by module type:")
    for m, (g, tot) in sorted(by_module.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
        print(f"  {m:12s} {g:3d}/{tot:3d} ({100*g/tot:.0f}%)")

    out_path = os.path.join(REPO_ROOT, "eval_scripts", "stage3_plumbing_result.json")
    with open(out_path, "w") as f:
        json.dump({"tau": TAU, "mu": MU, "n_gated": n_gated, "n_total": len(keys), "rows": rows}, f, indent=2)
    print(f"\nSaved per-location plumbing results -> {out_path}")


if __name__ == "__main__":
    main()
