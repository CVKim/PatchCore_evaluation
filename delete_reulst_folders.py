import os
import shutil

def delete_result_folders(directory):
    # 지정된 디렉토리 내부를 탐색
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in dirs:
            # 폴더 이름이 '_result'를 포함하면 삭제
            if '_result' in name:
                folder_path = os.path.join(root, name)
                shutil.rmtree(folder_path)
                print(f"Deleted: {folder_path}")

# 사용 예제
# 주의: 실제 사용 전에 경로를 확인하세요.

directory_path = "F:/Dataset/mvtec_anomaly_detection"
delete_result_folders(directory_path)
