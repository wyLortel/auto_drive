# training/RCDataset.py

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

    def __init__(self,
                 csv_filename,
                 root,
                 preprocessor: RCPreprocessor,
                 augmentor: RCAugmentor = None,
                 split: str = "train",
                 split_ratio: float = 0.8,
                 shuffle: bool = True,
                 random_seed: int = 42):

        # ----------------------------
        # 0) 경로 자동 정리
        # ----------------------------
        # \ → / 로 통일
        root = root.replace("\\", "/")
        self.image_root = root.rstrip("/")

        self.preprocessor = preprocessor
        self.augmentor = augmentor
        self.split = split

        # CSV 파일 경로도 정리
        csv_path = os.path.join(self.image_root, csv_filename).replace("\\", "/")

        self.df_full = pd.read_csv(csv_path)

        # -------------------------------
        # 1) split 컬럼 존재 시 사용
        # -------------------------------
        if "split" in self.df_full.columns:
            print("[RCDataset] Using existing 'split' column.")
            self.df = self.df_full[self.df_full["split"] == split].reset_index(drop=True)

        # -------------------------------
        # 2) split 없으면 stratified split
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
        # 3) angle → 클래스로 매핑
        # -------------------------------
        self.angles = sorted(self.df["servo_angle"].unique().tolist())
        self.angle_to_idx = {a: i for i, a in enumerate(self.angles)}

        print(f"[RCDataset:{split}] samples={len(self.df)}")
        print(self.df["servo_angle"].value_counts().sort_index())


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # CSV 내부 image_path 정리
        rel_path = str(row["image_path"]).replace("\\", "/")

        # dataset/ 중복 제거
        if rel_path.startswith("dataset/"):
            rel_path = rel_path[len("dataset/"):]

        # 절대 경로 생성
        img_path = f"{self.image_root}/{rel_path}"

        # --------------------------
        # 이미지 로드
        # --------------------------
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            raise RuntimeError(f"[ERROR] Failed to read image: {img_path}")

        # --------------------------
        # 증강
        # --------------------------
        angle = int(row["servo_angle"])

        if self.split == "train" and self.augmentor is not None:
            img_bgr, angle = self.augmentor(img_bgr, angle)

        # --------------------------
        # 전처리 (CHW float32)
        # --------------------------
        img_chw = self.preprocessor(img_bgr)
        img_tensor = torch.from_numpy(img_chw).float()

        # --------------------------
        # 라벨 인덱스 변환
        # --------------------------
        label = self.angle_to_idx[angle]

        return img_tensor, label
