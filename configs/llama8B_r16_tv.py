CACHE_DIR = '/home/aleum433/shared/hdd_ext/ssd4000/aleum/data/'          # Path to the cache directory
MODEL_DIR = '/home/aleum433/shared/hdd_ext/ssd4000/aleum/data/'          # Path to the model directory
# INGREDIENTS_PATH = ""   # Path to the ingredients file (If exists)
INGREDIENTS_PATH = ""

PTM_PATH = "/home/aleum433/shared/hdd_ext/ssd4000/aleum/data/"           # Path to the pre-trained model

config = {
    'dataset': [
        {
            'name': 'snli',
            'mask_class': None,
        },

        {
            'name': 'mnli',
            'val_fraction': 0.2,
            'mask_class': None,
        },
        {
            'name': 'sick',
            'mask_class': None,
        },
        {
            'name': 'qnli',
            'val_fraction': 0.2,
            'mask_class': 1,
        },
        {
            'name': 'rte',
            'val_fraction': 0.5,
            'mask_class': 1,
        },
        {
            'name': 'scitail',
            'mask_class': 2,
        }

    ],
    'model': {
        'name': 'meta-llama/Meta-Llama-3-8B-instruct',
        'ptm_path': PTM_PATH,
        'cachedir': CACHE_DIR,
        'bases': [
            # mini train
            # './lora-llama3-8b-snli-full',
            # './lora-llama3-8b-mnli-full',
            # './lora-llama3-8b-sick-full',
            # './lora-llama3-8b-qnli-full',
            # './lora-llama3-8b-rte-full',
            # './lora-llama3-8b-scitail-full'

            # HF models IDs
            # 'hoffman-lab/KnOTS-Llama3_8B_lora_R16_snli',
            # 'hoffman-lab/KnOTS-Llama3_8B_lora_R16_mnli',
            # 'hoffman-lab/KnOTS-Llama3_8B_lora_R16_sick',
            # 'hoffman-lab/KnOTS-Llama3_8B_lora_R16_qnli',
            # 'hoffman-lab/KnOTS-Llama3_8B_lora_R16_rte',
            # 'hoffman-lab/KnOTS-Llama3_8B_lora_R16_scitail',

            #total train
            './output_snli_final',
            './output_mnli_final',
            './output_sick_final',
            './output_qnli_final',
            './output_rte_final',
            './output_scitail_final',

        ],
        'ft_config': {
            'type': 'lora',
            'subtype': 'peft',
        },
        'peft_config': {
            'task_type': "SEQ_CLS",
            'inference_mode': False,
            'r': 16,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"]
            # 'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"]
        },
    },
    'task_merge_config': {
        'ingredients_path': INGREDIENTS_PATH,
        'representation': 'matrix_per_layer',
        'merge_space': 'core',
        'merge_method': 'tv',
        'tatr_k_percent': 0.00,
        'scaling_coeffs': .3,

        'isotropize': False,
    },
    'eval_type': 'logits',
}
