"""
Stage 1 -- Core Space construction on the real checkpoints + the lossless
reconstruction stopping condition (must pass before Stage 3 is trusted).

Step A: single location smoke test (fast fail if something's off).
Step B: all 224 real attention/MLP locations.

Caches Bc_t/Ac_t/M_t (+ U_ref/V_ref) per location to disk so Stage 3 never
has to redo these SVDs (per the task spec: don't discard them).
"""
import os
import pickle
import time

import numpy as np
from safetensors import safe_open

from core_space_common import build_core_space, check_lossless, check_lossless_fast

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(REPO_ROOT, "eval_scripts", "core_space_cache.pkl")

TASK_PATHS = {
    "mnli": "output_mnli_final/adapter_model.safetensors",
    "qnli": "output_qnli_final/adapter_model.safetensors",
    "rte": "output_rte_final/adapter_model.safetensors",
    "scitail": "output_scitail_final/adapter_model.safetensors",
    "sick": "output_sick_final/adapter_model.safetensors",
    "snli": "output_snli_final/output_snli_final/adapter_model.safetensors",
}
TASKS = list(TASK_PATHS)  # fixed order


def open_handles():
    return {t: safe_open(os.path.join(REPO_ROOT, p), framework="numpy") for t, p in TASK_PATHS.items()}


def location_keys(handles):
    ref = handles[TASKS[0]]
    lora_a_keys = sorted(k for k in ref.keys() if "lora_A" in k)
    for t in TASKS:
        assert set(k for k in handles[t].keys() if "lora_A" in k) == set(lora_a_keys), f"A-key mismatch: {t}"
        assert set(k.replace("lora_A", "lora_B") for k in lora_a_keys) == \
               set(k for k in handles[t].keys() if "lora_B" in k), f"B-key mismatch: {t}"
    return lora_a_keys


def load_ab(handles, a_key):
    b_key = a_key.replace("lora_A", "lora_B")
    a_list = [handles[t].get_tensor(a_key).astype(np.float64) for t in TASKS]
    b_list = [handles[t].get_tensor(b_key).astype(np.float64) for t in TASKS]
    return a_list, b_list


def main():
    handles = open_handles()
    keys = location_keys(handles)
    print(f"{len(keys)} locations, {len(TASKS)} tasks: {TASKS}")

    # --- Step A: single-location smoke test ---
    # Run the expensive literal-formula check (materializes the full m x n
    # delta) once here to empirically confirm the O(n/r)-cheaper
    # check_lossless_fast agrees with it, before relying on the fast path
    # for all 224 locations (the full check is too slow for real (m,n) up
    # to 14336 -- see core_space_common.check_lossless_fast docstring).
    smoke_key = keys[0]
    a_list, b_list = load_ab(handles, smoke_key)
    core = build_core_space(a_list, b_list)
    errs_slow = check_lossless(a_list, b_list, core)
    errs_fast = check_lossless_fast(a_list, b_list, core)
    print(f"\n[Step A] smoke test on {smoke_key}")
    print(f"         per-task recon rel err (literal, full m x n):   {[f'{e:.2e}' for e in errs_slow]}")
    print(f"         per-task recon rel err (fast, projector-ident): {[f'{e:.2e}' for e in errs_fast]}")
    assert max(errs_slow) < 1e-4, "Stage 1 lossless condition failed on smoke test (literal check) -- stop, do not proceed."
    assert max(errs_fast) < 1e-4, "Stage 1 lossless condition failed on smoke test (fast check) -- stop, do not proceed."
    print("         PASS (< 1e-4 threshold, both formulations agree)")

    # --- Step B: all locations (fast check only) ---
    print(f"\n[Step B] running all {len(keys)} locations...")
    t0 = time.time()
    cache = {}
    worst = []
    for key in keys:
        a_list, b_list = load_ab(handles, key)
        core = build_core_space(a_list, b_list)
        errs = check_lossless_fast(a_list, b_list, core)
        worst.append((max(errs), key))
        cache[key] = core
    elapsed = time.time() - t0

    worst.sort(reverse=True)
    max_err = worst[0][0]
    print(f"         done in {elapsed:.1f}s")
    print(f"         worst per-location max-err: {max_err:.2e} at {worst[0][1]}")
    print(f"         top-5 worst locations:")
    for err, key in worst[:5]:
        print(f"           {err:.2e}  {key}")

    n_fail = sum(1 for e, _ in worst if e >= 1e-4)
    if n_fail > 0:
        print(f"\n         FAIL: {n_fail}/{len(keys)} locations exceed the 1e-4 lossless threshold. STOP.")
        raise SystemExit(1)
    print(f"\n         PASS: all {len(keys)} locations satisfy the lossless reconstruction condition.")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"\nCached Bc_t/Ac_t/M_t/U_ref/V_ref for all locations -> {CACHE_PATH}")


if __name__ == "__main__":
    main()
