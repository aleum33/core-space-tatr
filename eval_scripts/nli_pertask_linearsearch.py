import os
from copy import deepcopy
import time
import gc
import numpy as np
import torch

from task_merger import get_merge_handler
from utils import evaluate_logits, get_config_from_name, prepare_experiment_config, set_seed, parse_eval_args, merge_args_into_task_merge_config, test_format_collapse
# Set TOKENIZERS_PARALLELISM to true
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import transformers

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)


def run_BIG_function(args):
    EVAL_SPLIT = 'val'
    EVAL_TEST = True
    BIGSEED = 420

    print("Seed : ", BIGSEED)
    set_seed(BIGSEED)
    
    # Get config
    config_name = args.config
    print("Config name : ", config_name)
    EARLY_STOPPING_STEPS = 2

    TASK_HEADS_PATH = "data/llama-3.2-1B/heads.pt" if '1B' in config_name else "heads.pt"
    # TASK_HEADS_PATH = "heads.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    raw_config = get_config_from_name(config_name, device=device)
    print(raw_config['task_merge_config'])
    config = prepare_experiment_config(raw_config)
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    dataloaders = np.array([i for i in config['data']])
    mask_class = np.array([i['mask_class'] for i in config['dataset']])
    print(f"mask_class labels: {mask_class}")

    # transform_listified = [str(i) if k != 'ingredients_path' else os.path.basename(i).replace('.pt', '') for k, i in raw_config['task_merge_config'].items()]
    # transform_listified += [str(v) for k, v in raw_config['model']['ft_config'].items() if k in {'r', 'type', 'lora_alpha'}]

    # Parameters are tuned in the order specified in search_config
    default_params = {
        'scaling_coeffs': 0.3,
        'topK': 70,
        'cart_pruning_rank': 0.04,
        'dare_pruning_coeffs':0.9
    }  # Default config

    order_of_processing_params = [
        'scaling_coeffs',
    ]
    # ===========================================================================================
    search_config = {
        # 'scaling_coeffs': np.arange(0.1, 1.0, step=0.2),
        'scaling_coeffs': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'topK': (np.arange(1, 11, step=1) * 10),
        'dare_pruning_coeffs': [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1e-5][::-1],
        'cart_pruning_rank': [0.04, 0.08, 0.16, 0.32]
    }
    tatr_test_cases = [0.0, 0.05, 0.50, 0.95, 1.00]
    # ===========================================================================================
    print(f"default params: {default_params}")
    print(f"order_of_processing_params: {order_of_processing_params}")

    task_heads = torch.load(TASK_HEADS_PATH)

    finetuned_llama3_8b = {
        'snli': 92.49796416938111, 'mnli': 90.30820173204279, 'sick': 91.58173664900122, 'qnli': 94.48512585812358, 'rte': 89.85507246376812, 'scitail': 96.51928504233303, }

    finetuned_llama32_1b = {"mnli": 84.093, "snli": 88.578, "qnli": 89.725, 'sick': 90.216, 'rte': 78.986, 'scitail': 94.967}

    print("Using Llama fine-tuned acc")
    fine_tuned_acc = finetuned_llama3_8b if '8B' in config_name else finetuned_llama32_1b

    print(f'Finetuned Accs: {fine_tuned_acc}')
    print(search_config)

    def merge_and_eval(Merge, EVAL_SPLIT='val', instance_params=None):
        set_seed(BIGSEED)
        print("EVAL_SPLIT : ", EVAL_SPLIT)
        print(f'Search Run with: {instance_params}')
        all_results = deepcopy(instance_params)
        print('Creating Merge')

        Merge.set_scaling_coeffs(instance_params['scaling_coeffs'])
        config['task_merge_config'].update(instance_params)
        t0 = time.time()
        merged_model = Merge.merge(config['task_merge_config'])
        print(f"Time taken to merge: {time.time() - t0}")

        merged_model.config.pad_token_id = 128001
        merged_model.config.use_cache = False
        merged_model.config.pretraining_tp = 1

        # 평가 점수 출력하기 이전에 출력  "meta-llama/Meta-Llama-3-8B"
        # test_model_generation(merged_model, "meta-llama/Meta-Llama-3-8B")
        # 여기에 넣게 되면 매번 추력되므로 loop끝나고 한 번에

        print('Evaluate Merged Model on Each Dataset')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        avg_accuracy = 0.
        avg_norm_accuracy = 0.
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]
            with torch.no_grad():
                for name, param in merged_model.named_parameters():
                    # Inject task head into model
                    if 'modules_to_save' in name:
                        param.copy_(task_heads[dataset_names[i]])

            acc = evaluate_logits(merged_model, loader, device, mask_class[i])
            print(f"{dataset_names[i]} Normalized accuracy is {np.round((acc * 100)/ fine_tuned_acc[dataset_names[i]] *100, 3)}")
            print(f"{dataset_names[i]} accuracy is {np.round(acc * 100, 3)}")
            all_results[dataset_names[i]] = acc * 100
            all_results[dataset_names[i] + '_norm_acc'] = (acc * 100) / fine_tuned_acc[dataset_names[i]] * 100
            avg_accuracy += acc * 100
            avg_norm_accuracy += (acc * 100) / fine_tuned_acc[dataset_names[i]] * 100
        avg_accuracy /= len(dataloaders)
        avg_norm_accuracy /= len(dataloaders)
        print(f'Average Accuracy is {np.round(avg_accuracy, 3)}')
        print(f'Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}')
        all_results['Average_acc'] = avg_accuracy
        all_results['Average_norm_acc'] = avg_norm_accuracy
        all_results.update(config['task_merge_config'])
        return all_results


    with torch.no_grad():
        # instruct 모델로 변경
        if 'new' in config['models']:
            del config['models']['new']
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        from transformers import AutoModelForSequenceClassification
        instruct_model = AutoModelForSequenceClassification.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            torch_dtype=torch.bfloat16,
            num_labels=3
            # 🔥 여기서 device_map 옵션을 싹 지웠습니다! (안전하게 로드)
        )
        # 이제 프레임워크는 이 '진짜 뇌'를 넘겨받아 덧셈을 시작합니다.
        config['models']['new'] = instruct_model
        lora_state_dicts = np.array([i for i in config['models']['bases']])
        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        Merge = MergeClass(
            lora_state_dicts,
            pretrained_model=config['models']['new'],
            param_handler=config['param_handler'],
            device=device,
            merge_config=config['task_merge_config'],
        )

        if config['task_merge_config']['ingredients_path'] is None or not os.path.exists(
                config['task_merge_config']['ingredients_path']):
            Merge.transform(config['task_merge_config'])


        print(config['task_merge_config'])

        # 🔥 치명적 오류 방지: 매 TATR 루프마다 초기화할 원본 디폴트 파라미터 백업
        original_default_params = deepcopy(default_params)

        # ====================================================================
        # 🌟 대망의 TATR 4연속 자동화 루프 시작
        # ====================================================================
        for tatr_val in tatr_test_cases:
            print(f"\n\n{'=' * 60}")
            print(f"🚀🚀🚀 [실험 시작] TATR Threshold: {tatr_val} 🚀🚀🚀")
            print(f"{'=' * 60}")

            # 1. 현재 루프의 TATR 값을 config에 강제 덮어쓰기
            config['task_merge_config']['tatr_k_percent'] = tatr_val
            # 2. 파라미터 탐색 시작점 초기화 (이전 TATR 루프의 오염 방지)
            current_default_params = deepcopy(original_default_params)

            # ----------------------------------------------------------------
            # [최적의 파라미터 탐색 (Linear Search)]
            # ---------------------------------------------------------------
            for param in order_of_processing_params:
                best_val_results = {'Average_norm_acc': 0.0}
                early_stopping = EARLY_STOPPING_STEPS

                for value in search_config[param]:
                    instance_params = deepcopy(current_default_params)
                    instance_params[param] = value

                    # 평가 진행
                    all_results = merge_and_eval(Merge, EVAL_SPLIT=EVAL_SPLIT, instance_params=instance_params)

                    if (all_results['Average_norm_acc'] >= best_val_results['Average_norm_acc']):
                        best_val_results = deepcopy(all_results)
                        early_stopping = EARLY_STOPPING_STEPS
                    else:
                        early_stopping -= 1
                        if early_stopping <= 0:
                            print(f"Early stopping (Param: {param}, TATR: {tatr_val})")
                            break

                # 최고 성능을 낸 파라미터 업데이트
                current_default_params[param] = best_val_results[param]

            # ----------------------------------------------------------------
            # [최종 Test 셋 평가 및 육안 검사 출력]
            # ----------------------------------------------------------------
            if EVAL_TEST:
                print(f"\n✅ [TATR {tatr_val}] 최적의 파라미터 :", best_val_results)
                for key in search_config.keys():
                    instance_params.update({key: best_val_results[key]})

                # Test 셋으로 최종 병합 및 평가
                test_result = merge_and_eval(Merge, EVAL_SPLIT='test', instance_params=instance_params)
                datasets = ['snli', 'mnli', 'sick', 'qnli', 'rte', 'scitail']
                test_results_str = " & ".join([f"{np.round(test_result[dataset + '_norm_acc'], 2)}" for dataset in
                                               datasets]) + f" & {np.round(test_result['Average_norm_acc'], 2)} \\\\"

                print(f"Normalized Test results [TATR {tatr_val}]: {test_results_str}")
                print(test_result)

                # 🧹 NLI 평가로 더러워진 VRAM 대청소 (OOM 방지)
                gc.collect()
                torch.cuda.empty_cache()
                print("🧹 NLI 평가 완료 후 찌꺼기 VRAM 청소 완료!")

                # 🛑 육안 검사 (Base 모델 환경에서 생성)
                print(f"\n[TATR {tatr_val}] 모델 육안 검사를 시작합니다...")
                test_format_collapse(
                    Merge,
                    config['task_merge_config'],
                    "meta-llama/Meta-Llama-3-8B-instruct",
                    device
                )

                del test_result
                gc.collect()
                torch.cuda.empty_cache()
                print(f" [TATR {tatr_val}] 루프 종료. VRAM 완벽 초기화 완료!")

if __name__ == "__main__":
    args = parse_eval_args()
    run_BIG_function(args)
