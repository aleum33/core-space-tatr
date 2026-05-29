import torch
import os

try:
    from safetensors.torch import load_file
except ImportError:
    pass

datasets = ['snli', 'mnli', 'sick', 'qnli', 'rte', 'scitail']

model_dirs = {
    'snli': "./output_snli_final",
    'mnli': "./output_mnli_final",
    'sick': "./output_sick_final",
    'qnli': "./output_qnli_final",
    'rte': "./output_rte_final",
    'scitail': "./output_scitail_final"
}

new_heads = {}

print("🚀 최신 MLP 모델들로부터 분류 헤드(Score weight) 추출을 시작합니다...\n")

for ds in datasets:
    path = model_dirs[ds]

    # PEFT 가중치 파일 경로 찾기
    sf_path = os.path.join(path, "adapter_model.safetensors")
    bin_path = os.path.join(path, "adapter_model.bin")

    if os.path.exists(sf_path):
        sd = load_file(sf_path)
    elif os.path.exists(bin_path):
        sd = torch.load(bin_path, map_location='cpu')
    else:
        print(f"❌ [{ds}] 폴더에 가중치 파일이 없습니다: {path}")
        continue

    score_key = next((k for k in sd.keys() if 'score' in k and 'weight' in k), None)

    if score_key:
        new_heads[ds] = sd[score_key].clone()
        print(f"✅ [{ds}] 성공적으로 헤드를 추출했습니다! (Key: {score_key}, Shape: {new_heads[ds].shape})")
    else:
        print(f"⚠️ [{ds}] score 레이어를 찾을 수 없습니다. (학습 시 modules_to_save=['score'] 설정 확인 필요)")

torch.save(new_heads, "heads_mlp.pt")
print("\n🎉 모든 추출 완료! 새로운 [heads_mlp.pt] 파일이 생성되었습니다.")