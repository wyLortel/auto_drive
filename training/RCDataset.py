import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from preprocessor.RCPreprocessor import RCPreprocessor
from preprocessor.RCAugmentor import RCAugmentor


class RCDataset(Dataset):
    """
    RC 자율주행용 Dataset
    """

    def __init__(
        self,
        csv_filename,
        root,
        preprocessor: RCPreprocessor,
        augmentor: RCAugmentor = None,
        split: str = "train",
        split_ratio: float = 0.8,
        shuffle: bool = True,
        random_seed: int = 42
    ):
        
        # ----------------------------
        # 0) 경로 정리
        # ----------------------------
        root = root.replace("\\", "/")
        self.image_root = root.rstrip("/")

        self.preprocessor = preprocessor
        self.augmentor = augmentor
        self.split = split

        # CSV 파일 경로
        csv_path = f"{self.image_root}/{csv_filename}.csv"
        self.df_full = pd.read_csv(csv_path)

        # ============================
        # CSV 컬럼명 검증 (수정 완료)
        # ============================
        required_cols = ["image_path", "servo_angle"]
        for col in required_cols:
            if col not in self.df_full.columns:
                raise ValueError(f"[ERROR] CSV must contain column '{col}'")

        # -------------------------------
        # 1) split 컬럼 존재 시 사용
        # -------------------------------
        if "split" in self.df_full.columns:
            print("[RCDataset] Using existing 'split' column.")
            self.df = (
                self.df_full[self.df_full["split"] == split]
                .reset_index(drop=True)
            )

        # -------------------------------
        # 2) split 없으면 stratified split (수정 완료)
        # -------------------------------
        else:
            print("[RCDataset] Performing stratified split...")

            if shuffle:
                self.df_full = self.df_full.sample(frac=1.0, random_state=random_seed)

            df_list = []
            for angle, df_group in self.df_full.groupby("servo_angle"):
                n = len(df_group)
                n_train = int(n * split_ratio)

                if split == "train":
                    df_split = df_group.iloc[:n_train]
                else:
                    df_split = df_group.iloc[n_train:]

                df_list.append(df_split)

            self.df = pd.concat(df_list).reset_index(drop=True)

        # -------------------------------
        # 3) angle → class index 매핑 (수정 완료)
        # -------------------------------
        self.angles = sorted(self.df["servo_angle"].unique().tolist())
        self.angle_to_idx = {a: i for i, a in enumerate(self.angles)}

        print(f"[RCDataset:{split}] samples={len(self.df)}")
        print(self.df["servo_angle"].value_counts().sort_index())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --------------------------------------
        # 1) 이미지 경로 생성 (최종 수정: 하위 폴더 제거)
        # --------------------------------------
        filename = str(row["image_path"]).replace("\\", "/")
        
        # 🚨 코드 최종 확인: CSV와 이미지 파일이 'dataset' 폴더 바로 아래에 있다고 가정
        img_path = f"{self.image_root}/{filename}" 
        
        img_bgr = cv2.imread(img_path)
        
        if img_bgr is None:
            print(f"[DEBUG] Attempted path: {img_path}") 
            raise RuntimeError(f"[ERROR] Failed to read image: {img_path}. 파일이 'dataset' 폴더 바로 아래에 있는지 확인해주세요.")

        # --------------------------------------
        # 2) servo_angle 가져오기 (수정 완료)
        # --------------------------------------
        angle = int(row["servo_angle"])

        # --------------------------------------
        # 3) augmentation (train only)
        # --------------------------------------
        if self.split == "train" and self.augmentor is not None:
            img_bgr, angle = self.augmentor(img_bgr, angle)

        # --------------------------------------
        # 4) 전처리 → CHW float32 tensor
        # --------------------------------------
        img_chw = self.preprocessor(img_bgr)
        img_tensor = torch.from_numpy(img_chw).float()

        # --------------------------------------
        # 5) angle → class index 변환
        # --------------------------------------
        label = self.angle_to_idx[angle]

        return img_tensor, label