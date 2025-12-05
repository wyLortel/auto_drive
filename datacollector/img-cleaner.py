import os
import pandas as pd


# -----------------------------
# 1) 사용자 이름 / 기본 경로 설정
# -----------------------------
USER_NAME = "YJU"
DATASET_FOLDER = "dataset"  # 바탕화면에 있는 폴더 이름

# 바탕화면 절대 경로
BASE_DIR = f"C:\\Users\\{USER_NAME}\\Desktop"

# -----------------------------
# 2) 데이터셋 폴더 경로 설정
# -----------------------------
dataset_root = os.path.join(BASE_DIR, DATASET_FOLDER)

# -----------------------------
# 3) CSV 파일 경로 설정
# -----------------------------
csv_file = os.path.join(dataset_root, "data_labels.csv")
new_csv_file = os.path.join(dataset_root, "data_labels_updated.csv")

print("CSV 파일 경로:", csv_file)
print("파일 존재 여부:", os.path.exists(csv_file))

# -----------------------------
# 4) CSV 로드
# -----------------------------
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_file}")

df = pd.read_csv(csv_file)
print("CSV 로드 성공! 데이터 크기:", df.shape)
print("CSV 컬럼:", df.columns)

# -----------------------------
# 5) 이미지 파일 존재 여부 체크
# -----------------------------
def file_exists(filename):
    file_path = os.path.join(dataset_root, filename)
    return os.path.exists(file_path)

# -----------------------------
# 6) 존재하는 파일만 필터링
# -----------------------------
valid_rows = []

for _, row in df.iterrows():
    img_name = row['image_filename']       # ✔ 실제 컬럼명으로 수정
    if file_exists(img_name):
        valid_rows.append(row)
    else:
        print(f"[WARN] Missing file: {img_name}")

valid_df = pd.DataFrame(valid_rows)
valid_df.to_csv(new_csv_file, index=False)

# -----------------------------
# 7) 통계 출력
# -----------------------------
total_images = len(valid_df)
print(f"[INFO] Total valid images: {total_images}")

angle_counts = valid_df['steering_angle'].value_counts()  # ✔ 컬럼명 수정
angle_percentages = (angle_counts / total_images) * 100

print("\n[INFO] Image counts and percentages by angle:")
for angle, count in angle_counts.items():
    percentage = angle_percentages[angle]
    print(f"Angle: {angle}, Count: {count}, Percentage: {percentage:.2f}%")

