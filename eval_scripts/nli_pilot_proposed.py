"""
Section 10-B: minimal pilot -- TA (Task Arithmetic, in Core Space) vs the
proposed contamination-suppression closed-form merge, tau=0.4 mu=1.0 fixed,
small eval sample cap per task. NOT the tuned experiment (that's
Experiments A-F, after this pilot shows a directional signal) -- this only
answers "does the direction look right at all" before spending more GPU time.

Reuses, UNMODIFIED:
  - configs/llama8B_r16_tv.py (same model/data/LoRA config the existing TA
    baseline in nli_pertask.py uses)
  - task_merger.py's MatrixPerLayerMerger.get_core_matrices() and
    ._apply_delta() (task_merger.py is untouched -- see experiment note on
    why: the server's copy has independent history, this pilot only adds
    new files)
  - utils.py's prepare_experiment_config / evaluate infra

New (this file + proposed_merge_torch.py):
  - the 'proposed' merge branch (rank-1 decomposition + CV gate + closed
    form solve), computed inline here instead of inside
    MatrixPerLayerMerger.merge(), so nothing shared gets modified.
"""
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)  # task_merger.py / utils.py live at repo root, not eval_scripts/

from task_merger import get_merge_handler
from utils import get_config_from_name, prepare_experiment_config, set_seed
from proposed_merge_torch import proposed_merge_torch, load_location_cv_map

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import transformers
transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

BIGSEED = 420
TAU = 0.4
MU = 1.0
MAX_SAMPLES_PER_TASK = 200  # pilot cap, not the full test set

REPO_ROOT = _REPO_ROOT
CV_JSON_PATH = os.path.join(REPO_ROOT, "eval_scripts", "contamination_cv_result.json")

FINETUNED_LLAMA3_8B = {
    'snli': 92.49796416938111, 'mnli': 90.30820173204279, 'sick': 91.58173664900122,
    'qnli': 94.48512585812358, 'rte': 89.85507246376812, 'scitail': 96.51928504233303,
}


def evaluate_logits_capped(model, loader, device, mask_class, max_samples):
    """Same as utils.evaluate_logits but stops after ~max_samples examples (pilot speed)."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if next(model.parameters()).device.type == 'cpu':
        model = model.to(device)
    model.eval()

    correct, total = 0, 0
    in_device = next(model.parameters()).device
    for batch in loader:
        if total >= max_samples:
            break
        if hasattr(batch, 'to'):
            batch = batch.to(in_device)
        elif isinstance(batch, dict):
            batch = {k: v.to(in_device) if hasattr(v, 'to') else v for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            if mask_class is not None:
                outputs.logits[:, mask_class] = -np.inf
            predictions = outputs.logits.argmax(dim=-1)
            total += batch["labels"].size(0)
            out_device = outputs.logits.device
            correct += (predictions == batch["labels"].to(out_device)).sum().item()

    acc = correct / total if total > 0 else 0.0
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return acc, total


def build_delta_for_method(merge, ftms_params_ab, relevant_ab_keys, all_keys, merge_config, method, cv_map, mu=None):
    """Mirrors task_merger.py's 'core' merge_space loop (lines 676-721), minus TATR
    clipping (no-op at tatr_k_percent=0 anyway), with an added 'proposed' branch.
    Returns a new state dict (deepcopy of pretrained_model) with deltas applied.

    get_core_matrices() caches per-key SVD results on `merge` itself, so calling
    this repeatedly for different mu values (mu sweep) only redoes the cheap
    closed-form solve + reconstruction, not the expensive SVDs.
    """
    if mu is None:
        mu = MU
    new_sd = deepcopy(merge.pretrained_model)
    n_gated = 0
    for key in tqdm(all_keys, desc=f"Merging ({method})"):
        key_base = key.replace('.base_layer', '')
        if key_base not in relevant_ab_keys:
            continue

        m_list, u_b_ref, vh_a_ref = merge.get_core_matrices(ftms_params_ab, key_base, merge_config)

        if method == 'tv':
            m_merged = torch.stack(m_list).sum(dim=0)
        elif method == 'proposed':
            cv = cv_map.get(key_base, 0.0)
            gated = cv > TAU
            if gated:
                n_gated += 1
                a_list, b_list = zip(*[ftm[key_base] for ftm in ftms_params_ab])
                v_ref = vh_a_ref.T
                # get_core_matrices() computes U_B_ref/Vh_A_ref via float64 SVDs
                # internally; raw LoRA A/B tensors are float32, so cast before
                # multiplying (torch, unlike numpy, does not auto-upcast).
                bc_list = [u_b_ref.T @ b.to(u_b_ref.dtype) for b in b_list]
                ac_list = [a.to(v_ref.dtype) @ v_ref for a in a_list]
                r = a_list[0].shape[0]
                m_merged, _phi = proposed_merge_torch(bc_list, ac_list, r, mu)
                m_merged = m_merged.type_as(m_list[0])
            else:
                m_merged = torch.stack(m_list).sum(dim=0)
        else:
            raise ValueError(f"unknown method {method}")

        delta_w = u_b_ref @ m_merged @ vh_a_ref
        merge._apply_delta(new_sd, key, delta_w)

    if method == 'proposed':
        print(f"  gated locations: {n_gated}/{sum(1 for k in all_keys if k.replace('.base_layer', '') in relevant_ab_keys)}")
    return new_sd


def main():
    set_seed(BIGSEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    raw_config = get_config_from_name('llama8B_r16_tv_ours224', device=device)
    config = prepare_experiment_config(raw_config)
    config['task_merge_config']['scaling_coeffs'] = 0.3  # matched across both methods, from the config default

    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    dataloaders = np.array([i for i in config['data']])
    mask_class = np.array([i['mask_class'] for i in config['dataset']])

    task_heads = torch.load("heads.pt")
    cv_map = load_location_cv_map(CV_JSON_PATH)

    lora_state_dicts = np.array([i for i in config['models']['bases']])
    MergeClass = get_merge_handler(config['task_merge_config']['representation'])
    merge = MergeClass(
        lora_state_dicts,
        pretrained_model=config['models']['new'],
        param_handler=config['param_handler'],
        device=device,
        merge_config=config['task_merge_config'],
    )
    merge.set_scaling_coeffs(config['task_merge_config']['scaling_coeffs'])

    ftms_params_ab = [ftm.get_ft_ab_parameters() for ftm in merge.ftms_params]
    relevant_ab_keys = merge.ftms_params[0].get_ft_ab_parameters().keys()
    all_keys = merge.pretrained_model.state_dict().keys()

    def run_one(method, mu=None):
        label = method if mu is None else f"proposed(mu={mu})"
        print(f"\n=== method: {label} ===")
        t0 = time.time()
        new_sd = build_delta_for_method(
            merge, ftms_params_ab, relevant_ab_keys, all_keys,
            config['task_merge_config'], method, cv_map, mu=mu,
        )
        print(f"  merge time: {time.time() - t0:.1f}s")

        new_sd.config.pad_token_id = 128001
        new_sd.config.use_cache = False
        new_sd.config.pretraining_tp = 1

        avg_norm_acc = 0.0
        per_task = {}
        with torch.no_grad():
            for i, loader_dict in enumerate(dataloaders):
                loader = loader_dict['test']['test']
                for name, param in new_sd.named_parameters():
                    if 'modules_to_save' in name:
                        param.copy_(task_heads[dataset_names[i]])

                acc, n = evaluate_logits_capped(new_sd, loader, device, mask_class[i], MAX_SAMPLES_PER_TASK)
                norm_acc = acc * 100 / FINETUNED_LLAMA3_8B[dataset_names[i]] * 100
                per_task[dataset_names[i]] = {"acc": acc * 100, "norm_acc": norm_acc, "n_eval": n}
                print(f"    {dataset_names[i]:10s} acc={acc*100:6.2f}  norm_acc={norm_acc:6.2f}  (n={n})")
                avg_norm_acc += norm_acc
        avg_norm_acc /= len(dataloaders)
        print(f"  S (avg normalized acc) = {avg_norm_acc:.2f}")
        del new_sd
        return {"per_task": per_task, "S": avg_norm_acc}

    # validated 224-location TA baseline (this session's corrected full pilot,
    # after fixing the double-get_peft_model bug + the KnOTS-checkpoint config
    # mixup) -- reused so a proposed-only/mu-sweep rerun doesn't redo tv's eval.
    KNOWN_TV_RESULT = {
        "S": 45.88,
        "per_task": {
            "snli": {"acc": 58.50, "norm_acc": 63.24, "n_eval": 200},
            "mnli": {"acc": 29.00, "norm_acc": 32.11, "n_eval": 200},
            "sick": {"acc": 16.00, "norm_acc": 17.47, "n_eval": 200},
            "qnli": {"acc": 53.00, "norm_acc": 56.09, "n_eval": 200},
            "rte": {"acc": 42.03, "norm_acc": 46.77, "n_eval": 138},
            "scitail": {"acc": 57.50, "norm_acc": 59.57, "n_eval": 200},
        },
    }

    methods = os.environ.get("PILOT_METHODS", "tv,proposed").split(",")
    mu_sweep_str = os.environ.get("MU_SWEEP", "")
    mu_values = [float(x) for x in mu_sweep_str.split(",")] if mu_sweep_str else None

    results = {}
    for method in methods:
        if method == "tv":
            results["tv"] = run_one("tv")
        elif method == "proposed" and mu_values:
            for mu in mu_values:
                results[f"proposed_mu{mu}"] = run_one("proposed", mu=mu)
        elif method == "proposed":
            results["proposed"] = run_one("proposed", mu=MU)

    if "tv" not in results:
        results["tv"] = KNOWN_TV_RESULT

    print("\n=== SUMMARY ===")
    print(f"TA (core space)   S = {results['tv']['S']:.2f}")
    for key, res in results.items():
        if key == "tv":
            continue
        print(f"{key}  S = {res['S']:.2f}  delta = {res['S'] - results['tv']['S']:+.2f}")

    import json
    out_path = os.path.join(REPO_ROOT, "eval_scripts", "nli_pilot_result.json")
    with open(out_path, "w") as f:
        json.dump({"tau": TAU, "mu_sweep": mu_values, "max_samples_per_task": MAX_SAMPLES_PER_TASK, "results": results}, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
