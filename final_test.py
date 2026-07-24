import os
from datasets import load_dataset
# 캐시 경로를 아예 새로 지정
os.environ["HF_DATASETS_CACHE"] = os.path.expanduser("~/final_retry_cache")
try:
    # 스크립트(.py)를 쓰지 않도록 trust_remote_code=False 설정
    ds = load_dataset("sick", trust_remote_code=False)
    print("🎉 [최종] 드디어 데이터셋 로드에 성공했습니다!")
except Exception as e:
    print(f"❌ 여전한 에러:\n{e}")
