"""
Section 10-A: mechanism check for the closed-form solve (mu=1.0, tau=0.4).

For every rank-1 unit m in each of the 44 gated locations, compute
    ratio_m = con_m / own_m      (contamination-to-own, per unit, in the
                                   Tr x Tr core space -- Sec 6.4 formula,
                                   NOT the coarser per-location CV used for
                                   the gate itself)
and correlate it against (1 - phi_m) -- how much the closed-form solve
suppressed that unit relative to baseline (phi=1).

A strong positive correlation is the direct evidence that mu=1.0 is doing
what the objective claims: units that leak more into other tasks' input
subspaces get suppressed more. Weak/negative correlation would mean the
observed rel_diff (Stage 3 plumbing) is coming from something else (e.g.
G's conditioning) rather than the intended contamination-suppression
mechanism -- and Stage 4 should not proceed until that's understood.

Reuses core_space_cache.pkl (Stage 1) and stage3_plumbing_result.json
(Stage 3, for the list of gated keys) -- no SVDs re-run, CPU/numpy only.
"""
import json
import os
import pickle

import numpy as np

from core_space_common import rank1_units, core_input_subspaces, proposed_merge

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(REPO_ROOT, "eval_scripts", "core_space_cache.pkl")
PLUMBING_PATH = os.path.join(REPO_ROOT, "eval_scripts", "stage3_plumbing_result.json")

MU = 1.0


def per_unit_ratios(core):
    units, owner = rank1_units(core)
    q_list = core_input_subspaces(core)
    pis = [q @ q.T for q in q_list]  # (Tr, Tr) projectors

    ratios = np.empty(len(units))
    for i, u in enumerate(units):
        t = owner[i]
        own_m = np.linalg.norm(u @ pis[t])
        con_m = sum(np.linalg.norm(u @ pis[s]) for s in range(len(pis)) if s != t)
        ratios[i] = con_m / own_m if own_m > 0 else np.nan
    return ratios, np.array(owner)


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    with open(PLUMBING_PATH) as f:
        plumbing = json.load(f)
    gated_keys = [r["key"] for r in plumbing["rows"] if r["gated"]]
    print(f"{len(gated_keys)} gated locations (tau={plumbing['tau']}, mu={plumbing['mu']})")

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)

    all_ratio, all_one_minus_phi = [], []
    per_location_corr = []

    for key in gated_keys:
        core = cache[key]
        ratios, owner = per_unit_ratios(core)
        _, phi, _ = proposed_merge(core, MU)
        one_minus_phi = 1.0 - phi

        r = pearson(ratios, one_minus_phi)
        per_location_corr.append((r, key))

        all_ratio.extend(ratios.tolist())
        all_one_minus_phi.extend(one_minus_phi.tolist())

    pooled_r = pearson(all_ratio, all_one_minus_phi)
    per_loc_vals = np.array([r for r, _ in per_location_corr if np.isfinite(r)])

    print(f"\npooled correlation across {len(all_ratio)} units (44 locations x 96 units): r = {pooled_r:.3f}")
    print(f"per-location correlation: mean={per_loc_vals.mean():.3f} median={np.median(per_loc_vals):.3f} "
          f"std={per_loc_vals.std():.3f}  (n={len(per_loc_vals)} locations)")
    print(f"per-location correlation range: [{per_loc_vals.min():.3f}, {per_loc_vals.max():.3f}]")
    n_negative = int((per_loc_vals < 0).sum())
    print(f"locations with negative correlation: {n_negative}/{len(per_loc_vals)}")

    per_location_corr.sort()
    print("\nweakest/most-negative 5 locations:")
    for r, key in per_location_corr[:5]:
        print(f"  r={r:+.3f}  {key}")
    print("strongest 5 locations:")
    for r, key in per_location_corr[-5:]:
        print(f"  r={r:+.3f}  {key}")

    result = {
        "mu": MU,
        "n_gated_locations": len(gated_keys),
        "pooled_correlation": pooled_r,
        "per_location_correlation_mean": float(per_loc_vals.mean()),
        "per_location_correlation_median": float(np.median(per_loc_vals)),
        "per_location_correlation_std": float(per_loc_vals.std()),
        "n_negative_locations": n_negative,
        "per_location": [{"key": k, "correlation": r} for r, k in per_location_corr],
    }
    out_path = os.path.join(REPO_ROOT, "eval_scripts", "stage10a_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
