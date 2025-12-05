import pandas as pd
import os

# 1. 목표 샘플 수를 정의합니다.
TARGET_SIZE = 1813 # ✅ 최종 유효 데이터의 최소값으로 수정 (150도: 1813개)

# --- 파일 경로 설정 ---
# 이전에 성공했던 경로 논리를 그대로 사용합니다.
DATASET_FOLDER = "dataset" 
BASE_DIR = os.path.join(os.path.expanduser('~'), "Desktop")
dataset_root = os.path.join(BASE_DIR, DATASET_FOLDER)
file_path = os.path.join(dataset_root, "data_labels.csv") # 원본 CSV 파일 경로


try:
    df = pd.read_csv(file_path) 
    print(f"✅ 'data_labels.csv' 파일을 성공적으로 로드했습니다.")
except FileNotFoundError:
    print(f"\n🚨🚨 오류: 파일 경로를 다시 확인해주세요.")
    exit()


# 2. 데이터 균등화 실행 (가장 안정적인 필터링 & 합치기 방식 사용)
sampled_data = []
unique_angles = df['servo_angle'].unique()

print("\n데이터 균등화 작업 중...")

for angle in unique_angles:
    # 1. 해당 각도(클래스)의 데이터만 필터링
    subset = df[df['servo_angle'] == angle]
    
    # 2. 목표 크기(1813개)만큼 샘플링 (random_state로 재현성 보장)
    if len(subset) >= TARGET_SIZE:
        sampled_subset = subset.sample(n=TARGET_SIZE, random_state=42)
    else:
        sampled_subset = subset # 데이터가 부족하면 전체를 사용
        
    sampled_data.append(sampled_subset)

# 3. 샘플링된 모든 데이터프레임을 하나로 합칩니다.
df_balanced = pd.concat(sampled_data).reset_index(drop=True)


# 4. 균등화된 데이터셋의 정보를 확인합니다.
print("\n### 최종 균등화 결과 확인 (TARGET: 1,813) ###")
print(f"총 데이터 수: {len(df_balanced)}개")
print("각도별 최종 개수:")
print(df_balanced['servo_angle'].value_counts().sort_index())

# 5. 균등화된 데이터를 새 CSV 파일로 저장합니다.
output_file_name = f'data_labels_balanced_{TARGET_SIZE}.csv'
output_file_path = os.path.join(dataset_root, output_file_name)
df_balanced.to_csv(output_file_path, index=False)

# 최종 파일 저장 위치를 출력하여 확인합니다.
print(f"\n✅ 최종 파일 저장 성공: 균등화된 데이터가 '{output_file_path}'에 저장되었습니다.")