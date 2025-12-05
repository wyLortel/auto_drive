import pandas as pd
import os
import cv2
import sys

# =======================================================
# 1. 설정 (train_pilotnet.py와 동일해야 합니다)
# =======================================================
CSV_FILENAME = "data_labels_balanced_1813"
DATASET_ROOT = "C:/Users/YJU/Desktop/dataset"
IMAGE_FOLDER = "" # 이미지 파일이 'dataset' 폴더 바로 아래에 있으면 빈 문자열 ("") 유지
                  # 만약 'dataset/images/' 안에 있다면 "images/"로 수정


def check_dataset():
    csv_path = os.path.join(DATASET_ROOT, f"{CSV_FILENAME}.csv")
    
    # ----------------------------
    # 2. CSV 로드
    # ----------------------------
    if not os.path.exists(csv_path):
        print(f"🚨 오류: CSV 파일을 찾을 수 없습니다. 경로를 확인하세요: {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    print(f"[INFO] CSV 로드 성공. 총 {len(df)}개 샘플 확인.")
    
    # ----------------------------
    # 3. 이미지 파일 검증
    # ----------------------------
    missing_files = []
    unreadable_files = []
    
    print("\n[INFO] 이미지 파일 존재 및 읽기 가능성 검사 중...")
    
    for index, row in df.iterrows():
        # image_path 컬럼에 있는 파일 이름을 가져옵니다.
        filename = str(row['image_path']).replace("\\", "/")
        
        # 파일 경로 조합 (루트 + 이미지 폴더 + 파일 이름)
        image_full_path = os.path.join(DATASET_ROOT, IMAGE_FOLDER, filename)
        image_full_path = image_full_path.replace("\\", "/")

        if not os.path.exists(image_full_path):
            # 파일이 디스크에 아예 없는 경우
            missing_files.append(filename)
        else:
            # 파일은 있지만, cv2.imread가 읽지 못하는 경우 (손상 또는 권한 문제)
            img = cv2.imread(image_full_path)
            if img is None:
                unreadable_files.append(filename)

        # 진행 상황 표시
        if (index + 1) % 1000 == 0:
            print(f"  > {index + 1} / {len(df)}개 파일 검사 완료.")


    # ----------------------------
    # 4. 결과 출력
    # ----------------------------
    print("\n" + "="*40)
    print("      ✅ 데이터셋 최종 검증 결과")
    print("="*40)
    
    if not missing_files and not unreadable_files:
        print("🎉 축하합니다! 모든 파일이 존재하며 읽기 가능합니다!")
        print("  -> 이제 train_pilotnet.py를 실행하시면 됩니다.")
    else:
        print("🚨 오류 파일이 발견되었습니다. 목록을 확인하고 CSV에서 제거해야 합니다.")
        
        if missing_files:
            print(f"\n[❌ 누락된 파일 (CSV에 있지만 디스크에 없음) - {len(missing_files)}개]")
            for f in missing_files[:5]: # 최대 5개만 출력
                print(f"  - {f}")
            if len(missing_files) > 5:
                print(f"  ...외 {len(missing_files) - 5}개")

        if unreadable_files:
            print(f"\n[⚠️ 읽기 불가능한 파일 (디스크에 있지만 손상) - {len(unreadable_files)}개]")
            for f in unreadable_files[:5]: # 최대 5개만 출력
                print(f"  - {f}")
            if len(unreadable_files) > 5:
                print(f"  ...외 {len(unreadable_files) - 5}개")
                
        # 문제 파일 제거를 위한 코드 예시 (선택적)
        all_bad_files = set(missing_files + unreadable_files)
        df_clean = df[~df['image_path'].isin(all_bad_files)]
        
        clean_csv_path = os.path.join(DATASET_ROOT, "data_labels_clean.csv")
        df_clean.to_csv(clean_csv_path, index=False)
        
        print(f"\n[INFO] 문제 파일이 제거된 CSV 파일이 '{clean_csv_path}'로 저장되었습니다.")
        print(f"  -> 이 파일(data_labels_clean.csv)을 사용해 학습을 재시도하세요.")


if __name__ == "__main__":
    check_dataset()