import os
import gc
import numpy as np
import torch
from copy import deepcopy
from collections import defaultdict

from task_merger import get_merge_handler
from utils import evaluate_logits, get_config_from_name, prepare_experiment_config, set_seed, parse_eval_args, \
    merge_args_into_task_merge_config

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import transformers

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)


def run_heatmap_filler(args):
    EVAL_SPLIT = 'val'
    BIGSEED = 420
    set_seed(BIGSEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_name = args.config
    raw_config = get_config_from_name(config_name, device=device)
    config = prepare_experiment_config(raw_config)
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)

    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    dataloaders = np.array([i for i in config['data']])
    mask_class = np.array([i['mask_class'] for i in config['dataset']])

    TASK_HEADS_PATH = "heads_mlp.pt"
    # TASK_HEADS_PATH = "data/llama-3.2-1B/heads.pt" if '1B' in config_name else "heads.pt"
    task_heads = torch.load(TASK_HEADS_PATH)

    finetuned_llama3_8b = {'snli': 92.49, 'mnli': 90.30, 'sick': 91.58, 'qnli': 94.48, 'rte': 89.85, 'scitail': 96.51}
    fine_tuned_acc = finetuned_llama3_8b  # 8B 기준 고정

    default_params = {
        'topK': 70,
        'cart_pruning_rank': 0.04,
        'dare_pruning_coeffs': 0.9
    }

    # =====================================================================
    target_coordinates = [
        (0.03, 2.1), (0.03, 2.2),
    ]
    grouped_coords = defaultdict(list)
    for t, s in target_coordinates:
        grouped_coords[t].append(s)

    # 최종 결과를 모아둘 딕셔너리
    final_results = {}

    with torch.no_grad():
        lora_state_dicts = np.array([i for i in config['models']['bases']])

        print(f"\n🚀 [MLP 극한 효율 모드] 성능 표 빈칸 채우기 시작 🚀\n")

        for tatr_val, scales in grouped_coords.items():
            print(f"\n{'=' * 50}\n🔍 TATR: {tatr_val} 그룹 분석 시작 (대상 스케일: {scales})\n{'=' * 50}")

            # TATR 세팅 및 SVD 코어 스페이스 추출 (TATR당 1번만 실행됨)
            config['task_merge_config']['tatr_k_percent'] = tatr_val
            MergeClass = get_merge_handler(config['task_merge_config']['representation'])
            Merge = MergeClass(
                lora_state_dicts, pretrained_model=config['models']['new'],
                param_handler=config['param_handler'], device=device, merge_config=config['task_merge_config']
            )

            if config['task_merge_config']['ingredients_path'] is None or not os.path.exists(
                    config['task_merge_config']['ingredients_path']):
                Merge.transform(config['task_merge_config'])

            for scale_val in scales:
                print(f"   ➤ 테스트 진행 중: TATR {tatr_val} | Scaling {scale_val} ...", end="", flush=True)

                instance_params = deepcopy(default_params)
                instance_params['scaling_coeffs'] = scale_val

                Merge.set_scaling_coeffs(instance_params['scaling_coeffs'])
                merged_model = Merge.merge(config['task_merge_config'])

                merged_model.config.pad_token_id = 128001
                merged_model.config.use_cache = False

                avg_norm_accuracy = 0.

                for i, loader_dict in enumerate(dataloaders):
                    loader = loader_dict['test'][EVAL_SPLIT]

                    # MLP 헤드 씌우기
                    for name, param in merged_model.named_parameters():
                        if 'modules_to_save' in name:
                            param.copy_(task_heads[dataset_names[i]])

                    acc = evaluate_logits(merged_model, loader, device, mask_class[i])
                    avg_norm_accuracy += (acc * 100) / fine_tuned_acc[dataset_names[i]] * 100

                avg_norm_accuracy /= len(dataloaders)
                final_score = np.round(avg_norm_accuracy, 2)

                # 딕셔너리에 저장 및 화면 출력
                final_results[(tatr_val, scale_val)] = final_score
                print(f" 완료! [Score: {final_score}%]")

                del merged_model
                gc.collect()
                torch.cuda.empty_cache()

    print("\n" + "🔥" * 30)
    print("🏆 [최종 결과 요약] 엑셀 복붙용 데이터")
    print("🔥" * 30)
    print("TATR\tScaling\tNorm_Acc(%)")
    for (t, s), score in final_results.items():
        print(f"{t}\t{s}\t{score}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_eval_args()
    run_heatmap_filler(args)