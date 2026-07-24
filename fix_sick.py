import os
import gzip
import shutil

# 우리가 의심하는 모든 캐시 경로 총망라
cache_dirs = [
    os.path.expanduser("~/.cache/huggingface/modules/datasets_modules/datasets/sick"),
    os.path.expanduser("~/.cache/huggingface/datasets/sick"),
    "./my_hf_cache/sick"
]

print("🔍 GZIP(0x8b) 에러의 진짜 범인 탐색 시작...")
found = False

for d in cache_dirs:
    if os.path.exists(d):
        for root, _, files in os.walk(d):
            for f in files:
                filepath = os.path.join(root, f)
                try:
                    # 파일을 바이너리(rb)로 열어서 첫 2바이트만 확인
                    with open(filepath, 'rb') as f_in:
                        header = f_in.read(2)

                        # 0x1f 0x8b가 바로 GZIP 압축 파일의 시그니처!
                        if header == b'\x1f\x8b':
                            print(f"\n🚨 [범인 검거] 압축 해제 안 된 파일 발견!: {filepath}")
                            found = True

                            f_in.seek(0)
                            temp_path = filepath + ".tmp"

                            # 파이썬 내장 라이브러리로 강제 압축 해제해서 임시 파일로 저장
                            with gzip.open(filepath, 'rb') as gz:
                                with open(temp_path, 'wb') as f_out:
                                    shutil.copyfileobj(gz, f_out)

                            # 압축 풀린 텍스트 파일로 원본 덮어쓰기!
                            os.replace(temp_path, filepath)
                            print(f"✅ 강제 텍스트 변환(압축 해제) 완료!")
                except Exception as e:
                    pass

if not found:
    print("\n🤷‍♂️ 캐시 폴더 내에 0x8b 파일이 없습니다. 허깅페이스 다운로드 자체의 문제일 수 있습니다.")
else:
    print("\n🎉 모든 조치가 완료되었습니다! 메인 코드를 다시 실행해 보세요.")