"""Faster version of debug_key_matching.py: skips prepare_experiment_config
entirely (which also builds all 6 dataset loaders, ~15min) since we only
need config['model'] -> prepare_models -> bases/LoRAHandler wrapping."""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import get_config_from_name, prepare_models, prepare_param_handler
from ft_handlers import LoRAHandler

raw_config = get_config_from_name('llama8B_r16_tv_ours224', device='cpu')
models = prepare_models(raw_config['model'], device='cpu')

bases = models['bases']
print(f"len(bases) = {len(bases)}")
raw_sd0 = bases[0]
print(f"raw bases[0] total keys = {len(raw_sd0)}")
raw_lora_a = [k for k in raw_sd0 if 'lora_A' in k]
print(f"raw bases[0] lora_A keys = {len(raw_lora_a)}")

import re
from collections import Counter
c = Counter()
for k in raw_lora_a:
    m = re.search(r"\.(mlp|self_attn)\.(\w+_proj)\.", k)
    c[m.group(2) if m else "NOMATCH:" + k] += 1
print("raw module breakdown:", dict(c))

param_handler = prepare_param_handler(raw_config['model'].get('ft_config', {}))
print(f"param_handler = {param_handler}")
handler0 = param_handler(raw_sd0)
ab0 = handler0.get_ft_ab_parameters()
print(f"\nafter LoRAHandler.get_ft_ab_parameters(): {len(ab0)} entries")
c2 = Counter()
for k in ab0.keys():
    m = re.search(r"\.(mlp|self_attn)\.(\w+_proj)\.weight$", k)
    c2[m.group(2) if m else "NOMATCH:" + k] += 1
print("post-handler module breakdown:", dict(c2))
