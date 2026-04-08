from huggingface_hub import login, snapshot_download

print("🔑 Hugging Face 로그인을 시작합니다.")
login()

print("📦 모델 다운로드를 시작합니다 (목적지: SSD data 폴더)")
# ignore_patterns 옵션으로 중복되는 16GB짜리 불필요한 파일을 걸러냅니다!
snapshot_download(
    repo_id="meta-llama/Meta-Llama-3-8B",
    cache_dir="/home/aleum433/shared/hdd_ext/ssd4000/aleum/data",
    ignore_patterns=["*.pth", "original/*"]
)
print("✅ 다운로드 100% 완료!")