import os
import sys
import subprocess

# 🔥 자식 프로세스로 실행될 때만 무거운 모듈들을 불러옵니다. (메모리 절약)
if os.environ.get("CHILD_RUN") == "1":
    import gc
    import numpy as np
    import torch
    from copy import deepcopy

    from task_merger import get_merge_handler
    from utils import get_config_from_name, prepare_experiment_config, set_seed, parse_eval_args, \
        merge_args_into_task_merge_config, test_format_collapse

    import transformers

    transformers.utils.logging.set_verbosity(transformers.logging.ERROR)


    def run_single_visual_check():
        args = parse_eval_args()
        tatr_val = float(os.environ.get("CHILD_TATR"))
        scale_val = float(os.environ.get("CHILD_SCALE"))

        BIGSEED = 420
        set_seed(BIGSEED)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"\n" + "=" * 70)
        print(f"👀 [완전 격리 실행] TATR: {tatr_val} | Scaling: {scale_val}")
        print("=" * 70)

        raw_config = get_config_from_name(args.config, device=device)
        config = prepare_experiment_config(raw_config)
        config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)
        config['task_merge_config']['tatr_k_percent'] = tatr_val

        default_params = {'topK': 70, 'cart_pruning_rank': 0.04, 'dare_pruning_coeffs': 0.9}
        instance_params = deepcopy(default_params)
        instance_params['scaling_coeffs'] = scale_val

        stress_config = deepcopy(config['task_merge_config'])
        stress_config.update(instance_params)

        lora_state_dicts = np.array([i for i in config['models']['bases']])

        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        Merge = MergeClass(
            lora_state_dicts,
            pretrained_model=config['models']['new'],
            param_handler=config['param_handler'],
            device=device,
            merge_config=config['task_merge_config'],
        )

        Merge.transform(config['task_merge_config'])

        # 텍스트 생성 테스트 실행 (포맷 붕괴 여부 확인)
        test_format_collapse(
            Merge,
            stress_config,
            "meta-llama/Meta-Llama-3-8B-Instruct",
            device
        )
        print(f"✅ [TATR: {tatr_val} | Scale: {scale_val}] 육안 검사 완료!\n")

if __name__ == "__main__":
    # 환경 변수를 통해 부모/자식 프로세스를 구분합니다.
    if os.environ.get("CHILD_RUN") == "1":
        # 자식 프로세스: 진짜로 모델을 로드하고 평가하는 역할
        run_single_visual_check()
    else:
        # 부모 프로세스 (Master 지휘관): 파이썬을 4번 껐다 켜는 역할
        print(f"\n🚀 [Track 2 - 서브프로세스 완전 격리 모드] 4단계 육안 검사 시작 🚀\n")

        visual_coordinates = [
            (0.0, 0.3),  # 🟢 Stage 1: 정상 (Baseline Peak)
            (0.0, 1.0),  # 🟡 Stage 2: 붕괴 전조 (Onset of Collapse)
            # 🔴 Stage 3: 완전 붕괴 (Catastrophic Collapse)
            (0.0, 4.0),
            (0.0, 3.0),
            (0.1, 5.0),  # 🔵 Stage 4: 완벽 구출 (TATR Rescue)
            (0.1, 4.0),
            (0.1, 3.0),
        ]

        for tatr, scale in visual_coordinates:
            # 자식 프로세스에게 넘겨줄 지시서(환경 변수) 세팅
            env = os.environ.copy()
            env["CHILD_RUN"] = "1"
            env["CHILD_TATR"] = str(tatr)
            env["CHILD_SCALE"] = str(scale)

            # 동일한 명령어와 인자를 그대로 넘겨 완전히 독립된 파이썬 프로세스 생성
            cmd = [sys.executable] + sys.argv
            subprocess.run(cmd, env=env)

        print("\n🎉 모든 지정 좌표에 대한 독립 육안 검사가 완벽하게 완료되었습니다!")