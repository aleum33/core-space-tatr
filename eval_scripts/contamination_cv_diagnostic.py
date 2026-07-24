"""
Section 5 go/no-go diagnostic for the contamination-based closed-form scaling idea.

For each task t, the LoRA A_t matrix (r x n) spans an r-dimensional input
subspace of the layer's input space. Q_t = orth(A_t^T) is an orthonormal
basis for that subspace (obtained directly from the SVD of A_t, since A_t is
already exactly rank r: Q_t = Vh_t^T where A_t = U_t S_t Vh_t).

For each rank-1 direction u_m (a column of some task's Q_own):
    own_m = || Pi_own u_m ||_2      (projection onto its own task's subspace)
    con_m = sum_{s != own} || Pi_s u_m ||_2   (leakage into every other task's subspace)
    ratio_m = con_m / own_m

For orthonormal Q, ||Q Q^T u|| == ||Q^T u||, so both norms are computed as
column norms of small (r x r) matrices Q_s^T @ Q_own -- no n x n projector is
ever formed.

Decision rule on the coefficient of variation (std/mean) of ratio_m, pooled
across every rank-1 unit in every layer/module/task:
    CV > 0.4      -> heterogeneous contamination -> method is worth pursuing
    0.2 <= CV <= 0.4 -> ambiguous -> needs mu-tuning / per-layer separation
    CV < 0.2      -> indistinguishable from uniform scaling -> pivot method

Needs only numpy + safetensors (no torch/peft) -- runs on CPU in seconds.
"""
import json
import os

import numpy as np
from safetensors import safe_open

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASK_PATHS = {
    "mnli": "output_mnli_final/adapter_model.safetensors",
    "qnli": "output_qnli_final/adapter_model.safetensors",
    "rte": "output_rte_final/adapter_model.safetensors",
    "scitail": "output_scitail_final/adapter_model.safetensors",
    "sick": "output_sick_final/adapter_model.safetensors",
    "snli": "output_snli_final/output_snli_final/adapter_model.safetensors",
}


def load_lora_a(handles):
    """Return {location_key: {task: A (r x n) float64 array}}."""
    ref_task = next(iter(handles))
    lora_a_keys = sorted(k for k in handles[ref_task].keys() if "lora_A" in k)
    for task, f in handles.items():
        assert set(k for k in f.keys() if "lora_A" in k) == set(lora_a_keys), \
            f"key mismatch for task {task}"

    per_location = {}
    for key in lora_a_keys:
        per_location[key] = {
            task: f.get_tensor(key).astype(np.float64) for task, f in handles.items()
        }
    return per_location


def orth_basis(a):
    """Q = orth(A^T): (n, r) matrix with orthonormal columns spanning row(A)."""
    _, _, vh = np.linalg.svd(a, full_matrices=False)  # vh: (r, n), rows orthonormal
    return vh.T  # (n, r)


def location_ratios(a_by_task):
    tasks = list(a_by_task.keys())
    q = {t: orth_basis(a_by_task[t]) for t in tasks}  # each (n, r)

    ratios = []
    per_task_ratios = {t: [] for t in tasks}
    for own in tasks:
        q_own = q[own]  # (n, r)
        r = q_own.shape[1]
        own_norm = np.ones(r)  # by construction: ||Q_own^T u_m|| == 1 for u_m a column of Q_own
        con = np.zeros(r)
        for s in tasks:
            if s == own:
                continue
            cross = q[s].T @ q_own  # (r_s, r_own)
            con += np.linalg.norm(cross, axis=0)  # column norms -> ||Q_s^T u_m|| per m
        ratio_m = con / own_norm
        ratios.extend(ratio_m.tolist())
        per_task_ratios[own].extend(ratio_m.tolist())
    return ratios, per_task_ratios


def coefficient_of_variation(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean()
    std = values.std()
    return std / mean if mean != 0 else float("nan")


def main():
    handles = {}
    for task, rel_path in TASK_PATHS.items():
        path = os.path.join(REPO_ROOT, rel_path)
        handles[task] = safe_open(path, framework="numpy")

    per_location = load_lora_a(handles)

    all_ratios = []
    per_task_all = {t: [] for t in TASK_PATHS}
    per_location_cv = {}

    for key, a_by_task in per_location.items():
        ratios, per_task_ratios = location_ratios(a_by_task)
        all_ratios.extend(ratios)
        for t, vals in per_task_ratios.items():
            per_task_all[t].extend(vals)
        per_location_cv[key] = coefficient_of_variation(ratios)

    global_cv = coefficient_of_variation(all_ratios)
    per_task_cv = {t: coefficient_of_variation(v) for t, v in per_task_all.items()}
    loc_cv_values = np.array(list(per_location_cv.values()))

    if global_cv > 0.4:
        verdict = "PROCEED: contamination is heterogeneous enough to justify a closed-form per-direction coefficient."
    elif global_cv >= 0.2:
        verdict = "AMBIGUOUS: squeeze more via mu-tuning and per-layer separation before deciding."
    else:
        verdict = "PIVOT: contamination is statistically indistinguishable from uniform scaling."

    result = {
        "n_locations": len(per_location),
        "n_tasks": len(TASK_PATHS),
        "rank": next(iter(next(iter(per_location.values())).values())).shape[0],
        "n_units_pooled": len(all_ratios),
        "global_ratio_mean": float(np.mean(all_ratios)),
        "global_ratio_std": float(np.std(all_ratios)),
        "global_cv": float(global_cv),
        "per_task_cv": per_task_cv,
        "per_location_cv_summary": {
            "mean": float(loc_cv_values.mean()),
            "std": float(loc_cv_values.std()),
            "min": float(loc_cv_values.min()),
            "max": float(loc_cv_values.max()),
            "median": float(np.median(loc_cv_values)),
        },
        "per_location_cv": {k: float(v) for k, v in per_location_cv.items()},
        "verdict": verdict,
    }

    out_path = os.path.join(REPO_ROOT, "eval_scripts", "contamination_cv_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    printable = {k: v for k, v in result.items() if k != "per_location_cv"}
    print(json.dumps(printable, indent=2))
    print(f"\nSaved (incl. per-location breakdown) to {out_path}")


if __name__ == "__main__":
    main()
