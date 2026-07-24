import os
from copy import deepcopy
import time
import gc
import numpy as np
import torch

from task_merger import get_merge_handler
from utils import evaluate_logits, get_config_from_name, prepare_experiment_config, set_seed, parse_eval_args, \
    merge_args_into_task_merge_config, test_format_collapse

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

    EARLY_STOPPING_STEPS = 3

    TASK_HEADS_PATH = "heads_mlp.pt"
    # ===========================================================================================
    # [자동화 주입] 터미널에서 주입한 변수들 가져오기
    # 터미널에서 변수를 안 주면 기본값으로 cuda, TATR 0.0, Scale 1.0이 들어갑니다.
    # ===========================================================================================
    env_device = os.environ.get('TARGET_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = env_device

    # env_tatr = float(os.environ.get('TATR_VAL', 0.0))
    # env_scale = float(os.environ.get('SCALE_VAL', 1.0))

    # print(f"\n🚀 [실행 환경 세팅] Device: {device} | TATR: {env_tatr} | Scaling: {env_scale} 🚀\n")
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
    }

    order_of_processing_params = [
        'scaling_coeffs',
    ]

    # ===========================================================================================
    # 🌟 [자동화 단일] 하나의 좌표만 확인
    # ===========================================================================================

    # search_config = {
    #     'topK': 70,
    #     'dare_pruning_coeffs': 0.9,
    #     'cart_pruning_rank': 0.04,
    #     'scaling_coeffs': [env_scale],
    # }
    # tatr_test_cases = [env_tatr]
    # ===========================================================================================
    # 🌟 [자동화 루프] 최적의 scaling 값을 찾아 육안 검사
    # ===========================================================================================

    # print(f"\n🚀 [실행 환경 세팅] Device: {device} (자동화 루프 실행) 🚀\n")
    #
    search_config = {
        'topK': 70,
        'dare_pruning_coeffs': 0.9,
        'cart_pruning_rank': 0.04,
        'scaling_coeffs': [0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5],
    }
    tatr_test_cases = [0.0, 0.01, 0.05, 0.1]

    # ===========================================================================================
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

        for tatr_val in tatr_test_cases:
            print(f"\n{'=' * 60}")
            print(f"🚀 [실험 시작] TATR Threshold: {tatr_val} 🚀")
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

            if config['task_merge_config']['ingredients_path'] is None or not os.path.exists(
                    config['task_merge_config']['ingredients_path']):
                Merge.transform(config['task_merge_config'])

            print("✅ 적용된 Merge Config:", config['task_merge_config'])

            for param in order_of_processing_params:
                best_val_results = {'Average_norm_acc': 0.0}
                early_stopping = EARLY_STOPPING_STEPS

                for value in search_config[param]:
                    instance_params = deepcopy(current_default_params)
                    instance_params[param] = value

                    all_results = merge_and_eval(Merge, EVAL_SPLIT=EVAL_SPLIT, instance_params=instance_params)

                    if (all_results['Average_norm_acc'] >= best_val_results['Average_norm_acc']):
                        best_val_results = deepcopy(all_results)
                        early_stopping = EARLY_STOPPING_STEPS
                    else:
                        early_stopping -= 1
                        if early_stopping <= 0:
                            print(f"⚠️ Early stopping 발생 (Param: {param}, TATR: {tatr_val}에서 조기 종료 조치)")
                            break

                current_default_params[param] = best_val_results[param]

            if EVAL_TEST:
                print(f"\n🏆 [TATR {tatr_val}] 평가 완료. 최종 파라미터:", best_val_results)

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
                print("🧹 NLI Test 평가 완료. VRAM 캐시 정리 완료!")

                print(f"\n👀 [TATR {tatr_val}] 모델 텍스트 생성 육안 검사 및 벤치마크를 시작합니다...")

                stress_config = deepcopy(config['task_merge_config'])
                stress_config.update(instance_params)

                optimal_scaling = stress_config['scaling_coeffs']
                print(f"👉 적용된 최적 Scaling 값: {optimal_scaling}")

                test_format_collapse(
                    Merge,
                    stress_config,
                    "meta-llama/Meta-Llama-3-8B-instruct",
                    device
                )

                gc.collect()
                torch.cuda.empty_cache()
                print(f"✅ [TATR {tatr_val}] 루프 및 육안 검사 종료. VRAM 완벽 초기화 완료!")


if __name__ == "__main__":
    args = parse_eval_args()
    run_BIG_function(args)