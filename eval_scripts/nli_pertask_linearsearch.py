import os
from copy import deepcopy
import time
import gc
import numpy as np
import torch

from task_merger import get_merge_handler
from utils import evaluate_logits, get_config_from_name, prepare_experiment_config, set_seed, parse_eval_args, \
    merge_args_into_task_merge_config

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

    # 🌟 Early Stopping 기준 스텝 수 설정
    EARLY_STOPPING_STEPS = 3

    TASK_HEADS_PATH = "data/llama-3.2-1B/heads.pt" if '1B' in config_name else "heads.pt"

    # ===========================================================================================
    # 🌟 [Grid Search 범위 설정]
    # ===========================================================================================
    env_device = os.environ.get('TARGET_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = env_device

    # 탐색하고 싶으신 TATR 임계값들과 Scaling Coefficient 후보군들
    tatr_test_cases = [0.001,0.003, 0.005, 0.007]
    scaling_coeff_cases = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    print(f"\n🚀 [Grid Search 시작 - NLI 점수 전용] Device: {device}")
    print(f"📊 탐색할 TATR 후보군: {tatr_test_cases}")
    print(f"📊 탐색할 Scaling 후보군: {scaling_coeff_cases}\n")
    # ===========================================================================================

    raw_config = get_config_from_name(config_name, device=device)
    print(raw_config['task_merge_config'])
    config = prepare_experiment_config(raw_config)
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    dataloaders = np.array([i for i in config['data']])
    mask_class = np.array([i['mask_class'] for i in config['dataset']])
    print(f"mask_class labels: {mask_class}")

    default_params = {
        'scaling_coeffs': 0.3,
        'topK': 70,
        'cart_pruning_rank': 0.04,
        'dare_pruning_coeffs': 0.9
    }  # Default config

    order_of_processing_params = [
        'scaling_coeffs',
    ]

    # ===========================================================================================
    # 🌟 [자동화 설정 주입] 지정한 후보군 리스트 적용
    # ===========================================================================================
    search_config = {
        'topK': 70,
        'dare_pruning_coeffs': 0.9,
        'cart_pruning_rank': 0.04,
        'scaling_coeffs': scaling_coeff_cases,
    }

    print(f"default params: {default_params}")
    print(f"order_of_processing_params: {order_of_processing_params}")

    task_heads = torch.load(TASK_HEADS_PATH)

    finetuned_llama3_8b = {
        'snli': 92.49796416938111, 'mnli': 90.30820173204279, 'sick': 91.58173664900122, 'qnli': 94.48512585812358,
        'rte': 89.85507246376812, 'scitail': 96.51928504233303, }

    finetuned_llama32_1b = {"mnli": 84.093, "snli": 88.578, "qnli": 89.725, 'sick': 90.216, 'rte': 78.986,
                            'scitail': 94.967}

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

        print('Evaluate Merged Model on Each Dataset')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        avg_accuracy = 0.
        avg_norm_accuracy = 0.
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]
            with torch.no_grad():
                for name, param in merged_model.named_parameters():
                    if 'modules_to_save' in name:
                        param.copy_(task_heads[dataset_names[i]])

            acc = evaluate_logits(merged_model, loader, device, mask_class[i])
            print(
                f"{dataset_names[i]} Normalized accuracy is {np.round((acc * 100) / fine_tuned_acc[dataset_names[i]] * 100, 3)}")
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
        lora_state_dicts = np.array([i for i in config['models']['bases']])
        original_default_params = deepcopy(default_params)

        # Outer Loop: TATR 수치들을 순회
        for tatr_val in tatr_test_cases:
            print(f"\n{'=' * 60}")
            print(f"🚀 [격자 탐색 중] TATR Threshold: {tatr_val} 🚀")
            print(f"{'=' * 60}")

            config['task_merge_config']['tatr_k_percent'] = tatr_val
            current_default_params = deepcopy(original_default_params)

            MergeClass = get_merge_handler(config['task_merge_config']['representation'])
            Merge = MergeClass(
                lora_state_dicts,
                pretrained_model=config['models']['new'],
                param_handler=config['param_handler'],
                device=device,
                merge_config=config['task_merge_config'],
            )

            # Mask 갱신을 위해 transform 수행
            Merge.transform(config['task_merge_config'])

            print("✅ 적용된 Merge Config:", config['task_merge_config'])

            for param in order_of_processing_params:
                best_val_results = {'Average_norm_acc': 0.0}
                early_stopping = EARLY_STOPPING_STEPS  # 각 파라미터 검색 시작 시 카운터 초기화

                # Inner Loop: Scaling Coeffs 수치들을 순회
                for value in search_config[param]:
                    instance_params = deepcopy(current_default_params)
                    instance_params[param] = value

                    all_results = merge_and_eval(Merge, EVAL_SPLIT=EVAL_SPLIT, instance_params=instance_params)

                    # 성능이 향상되거나 같으면 최고점 갱신 후 early stopping 카운터 복구
                    if (all_results['Average_norm_acc'] >= best_val_results['Average_norm_acc']):
                        best_val_results = deepcopy(all_results)
                        early_stopping = EARLY_STOPPING_STEPS
                    else:
                        # 성능이 꺾이기 시작하면 카운터 차감
                        early_stopping -= 1
                        print(f"📉 성능 하락 감지! Early Stopping 카운트다운: {early_stopping}/{EARLY_STOPPING_STEPS}")
                        if early_stopping <= 0:
                            print(f"⚠️ [조기 종료] Param: {param} | TATR: {tatr_val} 에서 {EARLY_STOPPING_STEPS}회 연속 성능 하락으로 루프 탈출!")
                            break

                current_default_params[param] = best_val_results[param]

            if EVAL_TEST:
                print(f"\n🏆 [TATR {tatr_val}] 최적 검증점수 매칭 완료. 최종 파라미터:", best_val_results)

                for key in search_config.keys():
                    instance_params.update({key: best_val_results[key]})

                test_result = merge_and_eval(Merge, EVAL_SPLIT='test', instance_params=instance_params)
                datasets = ['snli', 'mnli', 'sick', 'qnli', 'rte', 'scitail']
                test_results_str = " & ".join([f"{np.round(test_result[dataset + '_norm_acc'], 2)}" for dataset in
                                               datasets]) + f" & {np.round(test_result['Average_norm_acc'], 2)} \\\\"

                print(f"📊 Normalized Test results [TATR {tatr_val}]: {test_results_str}")

                del test_result
                gc.collect()
                torch.cuda.empty_cache()
                print(f"✅ [TATR {tatr_val}] 루프 종료 및 VRAM 완벽 초기화 완료!\n")


if __name__ == "__main__":
    args = parse_eval_args()
    run_BIG_function(args)