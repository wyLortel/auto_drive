import os
import pandas as pd

USER_NAME = "YJU"
DATASET_FOLDER = "dataset" # 바탕화면의 폴더 이름

# 바탕화면 절대 경로 설정
BASE_DIR = f"C:\\Users\\{USER_NAME}\\Desktop"

# 데이터셋 루트 폴더 (이미지 파일이 들어있는 실제 위치)
dataset_root = os.path.join(BASE_DIR, DATASET_FOLDER) # C:\Users\YJU\Desktop\dataset

# 기존 CSV 파일 경로 (수정할 파일)
# CSV 파일이 dataset_root 폴더 안에 있다고 가정
csv_file = os.path.join(dataset_root, "data_labels.csv") 

# 새로운 CSV 파일 경로 (클리닝된 결과 파일)
new_csv_file = os.path.join(dataset_root, "data_labels_updated.csv")
# --- 경로 설정 끝 ---

# CSV 파일 읽기
df = pd.read_csv(csv_file)

# 이미지 파일 존재 여부 체크 함수
def file_exists(filename):
    file_path = os.path.join(dataset_root, filename)
    return os.path.exists(file_path)

# 존재하는 파일만 필터링하여 새로운 CSV 파일 생성
valid_rows = []

for _, row in df.iterrows():
    img_path = row['image_path']  # 'image_path' 컬럼으로 수정
    if file_exists(img_path):
        valid_rows.append(row)
    else:
        print(f"[WARN] Missing file: {img_path}")

# 새로운 CSV 파일 생성
valid_df = pd.DataFrame(valid_rows)
valid_df.to_csv(new_csv_file, index=False)

# 출력: 전체 파일 수 및 각 각도별 이미지 수와 비율
total_images = len(valid_df)
print(f"[INFO] Total valid images: {total_images}")

# 각 각도별 이미지 수와 비율 계산
angle_counts = valid_df['servo_angle'].value_counts()
angle_percentages = (angle_counts / total_images) * 100

print("\n[INFO] Image counts and percentages by angle:")
for angle, count in angle_counts.items():
    percentage = angle_percentages[angle]
    print(f"Angle: {angle}, Count: {count}, Percentage: {percentage:.2f}%")
