import os
import gc
import numpy as np
import torch
from copy import deepcopy
from collections import defaultdict

from task_merger import get_merge_handler
from utils import get_config_from_name, prepare_experiment_config, set_seed, parse_eval_args, \
    merge_args_into_task_merge_config, test_format_collapse

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import transformers

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)


def run_fis_grid(args):
    BIGSEED = 420
    set_seed(BIGSEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_name = args.config
    raw_config = get_config_from_name(config_name, device=device)
    config = prepare_experiment_config(raw_config)
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)

    default_params = {
        'topK': 70,
        'cart_pruning_rank': 0.04,
        'dare_pruning_coeffs': 0.9
    }

    # =====================================================================
    # 🔥 [수정 포인트] FIS 벤치마크를 진행할 (TATR, Scaling) 좌표 리스트
    # 테스트하고 싶은 지점들을 자유롭게 추가하세요!
    # =====================================================================
    target_coordinates = [
        (0.0, 0.5), (0.0, 0.6),(0.0, 0.8), (0.0, 1.0), (0.0, 2.0),(0.01, 3.0),
        (0.01, 0.7), (0.01, 1.0),(0.01, 1.5), (0.01, 2.0),(0.01, 0.5),(0.01, 1.4),(0.01, 1.6),(0.01, 2.5),(0.01, 3.0),(0.01, 4.0),(0.01, 5.0),
        (0.1, 1.0), (0.1, 3.0), (0.1, 4.0)
    ]

    # SVD 중복 연산 방지를 위해 TATR 단위로 그룹화
    grouped_coords = defaultdict(list)
    for t, s in target_coordinates:
        grouped_coords[t].append(s)

    final_fis_results = {}
    # Llama-3 Instruct 토크나이저 로드를 위한 모델 이름
    model_name_for_tokenizer = "meta-llama/Meta-Llama-3-8B-instruct"

    with torch.no_grad():
        lora_state_dicts = np.array([i for i in config['models']['bases']])

        print(f"\n🚀 [자동화 모드] FIS (Format Integrity Score) Grid Search 시작 🚀\n")

        for tatr_val, scales in grouped_coords.items():
            print(f"\n{'=' * 50}\n🔍 TATR: {tatr_val} 그룹 벤치마크 시작\n{'=' * 50}")

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
                print(f"\n➤ [테스트 좌표] TATR: {tatr_val} | Scaling: {scale_val}")

                instance_params = deepcopy(default_params)
                instance_params['scaling_coeffs'] = scale_val

                # 텍스트 생성을 위해 파라미터 업데이트
                stress_config = deepcopy(config['task_merge_config'])
                stress_config.update(instance_params)

                Merge.set_scaling_coeffs(instance_params['scaling_coeffs'])
                merged_model = Merge.merge(config['task_merge_config'])

                merged_model.config.pad_token_id = 128001
                merged_model.config.use_cache = False

                # 🔥 핵심: NLI 평가 생략 후 바로 FIS(text generation) 실행
                success_count = test_format_collapse(Merge, stress_config, model_name_for_tokenizer, device)

                # 요약 표를 위한 점수 수집 (test_format_collapse가 성공 횟수를 리턴한다고 가정)
                if success_count is not None:
                    fis_score = (success_count / 50.0) * 100
                    final_fis_results[(tatr_val, scale_val)] = fis_score
                    print(f"🎯 좌표 완료! FIS Score: {fis_score:.1f}%")

                del merged_model
                gc.collect()
                torch.cuda.empty_cache()

    # =====================================================================
    # 📊 최종 FIS 결과 요약 출력 (엑셀 복붙용)
    # =====================================================================
    print("\n" + "🔥" * 30)
    print("🏆 [최종 FIS 결과 요약] 엑셀 복붙용 데이터")
    print("🔥" * 30)
    print("TATR\tScaling\tFIS_Score(%)")
    for (t, s), score in final_fis_results.items():
        print(f"{t}\t{s}\t{score}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_eval_args()
    run_fis_grid(args)