import os
import pandas as pd
from torch.utils.data import Dataset,DataLoader, Subset
from collections import Counter
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    RandFlipd,
    RandRotated,
    RandZoomd,
    EnsureTyped
)

# 定义数据类
class ADNI(Dataset):
    def __init__(self, label_file, mri_dir, pet_dir,task='ADCN', augment=False):
        # self.label = pd.read_csv(label_file)
        self.label = pd.read_csv(label_file, encoding='ISO-8859-1')
        self.mri_dir = mri_dir
        self.pet_dir = pet_dir
        self.task = task
        self.augment = augment

        self._process_labels()
        self._build_data_dict()
        self._print_class_counts()

    def _process_labels(self):
        """根据指定的任务从标签 CSV 文件中提取数据标签"""
        if self.task == 'ADCN':
            self.labels = self.label[(self.label['Group'] == 'AD') | (self.label['Group'] == 'CN')]
            self.label_dict = {'CN': 0, 'AD': 1}
        if self.task == 'SMCIPMCI':
            self.labels = self.label[(self.label['Group'] == 'SMCI') | (self.label['Group'] == 'PMCI')]
            self.label_dict = {'SMCI': 0, 'PMCI': 1}

    def _build_data_dict(self):
        subject_list = self.labels['Subject_ID'].tolist()
        label_list = self.labels['Group'].tolist()
        self.data_dict = [
            {
                'MRI': os.path.join(self.mri_dir, f'{subject}.nii'),
                'PET': os.path.join(self.pet_dir, f'{subject}.nii'),
                'label': self.label_dict[group],
                'Subject': subject
            } for subject, group in zip(subject_list, label_list)
        ]

    def _print_class_counts(self):
        """打印当前 data_dict 里每个 label 的样本数量。"""
        inv = {v: k for k, v in self.label_dict.items()}
        cnt = Counter(sample['label'] for sample in self.data_dict)
        print(f"\n[ADNI Dataset: {self.task}] 样本分布：")
        for lbl_value, num in cnt.items():
            print(f"  {inv[lbl_value]} ({lbl_value}): {num}")
        print()

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        """仅返回MRI图像和标签"""
        sample = self.data_dict[idx]
        label = sample['label']

        # 加载MRI图像
        mri_img = LoadImaged(keys=['MRI'])({'MRI': sample['MRI']})['MRI']
        pet_img = LoadImaged(keys=['PET'])({'PET': sample['PET']})['PET']
        return mri_img, pet_img, label

    def print_dataset_info(self, start=0, end=None):
        print(f"\nDataset Structure:\n{'=' * 40}")
        print(f"Total Samples: {len(self)}")
        print(f"Task: {self.task}")

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        end = end or len(self)
        df = pd.DataFrame(
            [
                [s['MRI'], s['label'], s['Subject']]
                for s in self.data_dict[start:end]
            ],
            columns=["MRI", "Label", "Subject"]
        )
        print(df)
        print(f"{'=' * 40}\n")

# 修改预处理函数，仅处理MRI数据
def ADNI_transform(augment=False):
    keys = ['MRI','PET']  # 统一管理数据键

    base_transforms = [
        LoadImaged(keys=keys),              #加载 
        EnsureChannelFirstd(keys=keys),     #通道维度
        ScaleIntensityd(keys=keys),         #强度归一化
        EnsureTyped(keys=keys)              #
    ]

    if augment:
        base_transforms.insert(2, RandFlipd(keys=keys, prob=0.3, spatial_axis=0))
        base_transforms.insert(3, RandRotated(keys=keys, prob=0.3, range_x=0.05))
        base_transforms.insert(4, RandZoomd(keys=keys, prob=0.3, min_zoom=0.95, max_zoom=1))

    train_transform = Compose(base_transforms)
    test_transform = Compose(base_transforms[:4])  # 基础预处理（无增强）

    return train_transform, test_transform


def main():
    # ------------- 基本路径与任务 -------------
    label_filename  = rf'C:\Users\dongzj\Desktop\Multimodal_AD\adni_dataset\ADNI_902.csv'
    mri_dir         = rf'C:\Users\dongzj\Desktop\Multimodal_AD\adni_dataset\MRI'
    pet_dir         = rf'C:\Users\dongzj\Desktop\Multimodal_AD\adni_dataset\PET'
    task            = 'ADCN'        # 四分类

    # ------------- 1) 只创建一次完整数据集 -------------
    full_dataset = ADNI(
        label_file=label_filename,
        mri_dir=mri_dir,
        pet_dir=pet_dir,
        task=task
    )   # 这里会自动打印一次样本分布

    # ------------- 2) 分层划分索引 -------------
    indices = list(range(len(full_dataset)))
    labels  = [full_dataset.data_dict[i]['label'] for i in indices]

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=labels           # 保证四个类别比例一致
    )

    # ------------- 3) 构建子集 -------------
    train_dataset = Subset(full_dataset, train_idx)
    test_dataset  = Subset(full_dataset, test_idx)

    # ------------- 4) 数据验证 -------------
    sample_mri, sample_pet, sample_label = train_dataset[0]
    print(f"Sample MRI shape: {sample_mri.shape}, Label: {sample_label}")

    # 如仍需查看部分样本，可在 Subset 上迭代索引：
    def preview(ds, name, k=5):
        print(f"\n{name} preview (前 {k} 条):")
        for i in range(k):
            subj = full_dataset.data_dict[ds.indices[i]]['Subject']
            lbl  = full_dataset.data_dict[ds.indices[i]]['label']
            print(f"  idx={ds.indices[i]:>4}  Subject={subj}  Label={lbl}")
    preview(train_dataset, "Train", 20)
    preview(test_dataset,  "Test",  5)

    # ------------- 5) 预处理流程 -------------
    train_transform, _ = ADNI_transform(augment=False)
    print("\nTransforms pipeline:")
    for i, t in enumerate(train_transform.transforms):
        print(f"  {i+1:>2}. {t.__class__.__name__}")


if __name__ == '__main__':
    main()
    # run_5fold_cv()
    