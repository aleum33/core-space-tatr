"""One-off: investigate why 'proposed' saw only 9/128 gated locations instead
of the expected ~44/224 from the CPU-only diagnostic. Prints relevant_ab_keys
count/samples and cross-checks against contamination_cv_result.json's keys."""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from task_merger import get_merge_handler
from utils import get_config_from_name, prepare_experiment_config, set_seed
from proposed_merge_torch import load_location_cv_map

set_seed(420)
device = 'cpu'  # no GPU needed for this key-matching check

raw_config = get_config_from_name('llama8B_r16_tv', device=device)
config = prepare_experiment_config(raw_config)

lora_state_dicts = torch.tensor(0)  # placeholder, replaced below
import numpy as np
lora_state_dicts = np.array([i for i in config['models']['bases']])
MergeClass = get_merge_handler(config['task_merge_config']['representation'])
merge = MergeClass(
    lora_state_dicts,
    pretrained_model=config['models']['new'],
    param_handler=config['param_handler'],
    device=device,
    merge_config=config['task_merge_config'],
)

relevant_ab_keys = list(merge.ftms_params[0].get_ft_ab_parameters().keys())
all_keys = list(merge.pretrained_model.state_dict().keys())

print(f"len(relevant_ab_keys) = {len(relevant_ab_keys)}")
print("sample relevant_ab_keys:")
for k in relevant_ab_keys[:10]:
    print(" ", k)

matched = [k for k in all_keys if k.replace('.base_layer', '') in relevant_ab_keys]
print(f"\nlen(all_keys) = {len(all_keys)}")
print(f"matched (key.replace('.base_layer','') in relevant_ab_keys) count = {len(matched)}")
print("sample matched all_keys:")
for k in matched[:10]:
    print(" ", k)

CV_JSON_PATH = os.path.join(_REPO_ROOT, "eval_scripts", "contamination_cv_result.json")
cv_map = load_location_cv_map(CV_JSON_PATH)
print(f"\nlen(cv_map) = {len(cv_map)}")
print("sample cv_map keys:")
for k in list(cv_map.keys())[:5]:
    print(" ", k)

found = sum(1 for k in relevant_ab_keys if k in cv_map)
print(f"\nrelevant_ab_keys found in cv_map: {found}/{len(relevant_ab_keys)}")

# breakdown by module type to see which modules are missing from relevant_ab_keys
import re
from collections import Counter
mod_counts = Counter()
for k in relevant_ab_keys:
    m = re.search(r"\.(mlp|self_attn)\.(\w+_proj)\.weight$", k)
    if m:
        mod_counts[m.group(2)] += 1
    else:
        mod_counts["UNMATCHED_PATTERN:" + k] += 1
print("\nmodule breakdown of relevant_ab_keys:")
for k, v in mod_counts.items():
    print(f"  {k}: {v}")
