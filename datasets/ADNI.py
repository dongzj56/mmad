import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
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
    """
    用于处理ADNI数据集的类，仅处理MRI数据和标签
    """

    def __init__(self, label_file, mri_dir, task='ADCN', augment=False):
        """
        初始化ADNI数据集类，读取数据和标签文件，并生成数据字典

        :param label_file: 标签文件路径（包含 Group 和 Subject ID 等信息）
        :param mri_dir: MRI图像所在目录
        :param task: 任务类型，用于选择不同标签类别
        :param augment: 是否进行数据增强
        """
        self.label = pd.read_csv(label_file)
        self.mri_dir = mri_dir
        self.task = task
        self.augment = augment

        self._process_labels()
        self._build_data_dict()

    def _process_labels(self):
        """根据指定的任务从标签 CSV 文件中提取数据标签"""
        if self.task == 'ADCN':
            self.labels = self.label[(self.label['Group'] == 'AD') | (self.label['Group'] == 'CN')]
            self.label_dict = {'CN': 0, 'AD': 1}
        if self.task == 'CNEMCI':
            self.labels = self.label[(self.label['Group'] == 'CN') | (self.label['Group'] == 'EMCI')]
            self.label_dict = {'CN': 0, 'EMCI': 1}
        if self.task == 'LMCIAD':
            self.labels = self.label[(self.label['Group'] == 'LMCI') | (self.label['Group'] == 'AD')]
            self.label_dict = {'LMCI': 0, 'AD': 1}
        if self.task == 'EMCILMCI':
            self.labels = self.label[(self.label['Group'] == 'EMCI') | (self.label['Group'] == 'LMCI')]
            self.label_dict = {'EMCI': 0, 'LMCI': 1}

    def _build_data_dict(self):
        subject_list = self.labels['Subject ID'].tolist()
        label_list = self.labels['Group'].tolist()
        self.data_dict = [
            {
                'MRI': os.path.join(self.mri_dir, f'{subject}.nii'),
                'label': self.label_dict[group],
                'Subject': subject
            } for subject, group in zip(subject_list, label_list)
        ]

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        """仅返回MRI图像和标签"""
        sample = self.data_dict[idx]
        label = sample['label']

        # 加载MRI图像
        mri_img = LoadImaged(keys=['MRI'])({'MRI': sample['MRI']})['MRI']
        return mri_img, label

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
    keys = ['MRI']  # 统一管理数据键

    base_transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ScaleIntensityd(keys=keys),
        EnsureTyped(keys=keys)
    ]

    if augment:
        base_transforms.insert(2, RandFlipd(keys=keys, prob=0.3, spatial_axis=0))
        base_transforms.insert(3, RandRotated(keys=keys, prob=0.3, range_x=0.05))
        base_transforms.insert(4, RandZoomd(keys=keys, prob=0.3, min_zoom=0.95, max_zoom=1))

    train_transform = Compose(base_transforms)
    test_transform = Compose(base_transforms[:4])  # 基础预处理（无增强）

    return train_transform, test_transform


def main():
    dataroot = rf'C:\Users\dongz\Desktop\adni_dataset'
    label_filename = rf'C:\Users\dongz\Desktop\adni_dataset\ADNI.csv'
    mri_dir = os.path.join(dataroot, 'MRI-GM')
    task = 'ADCN'

    # 创建数据集对象
    adni_dataset = ADNI(
        label_file=label_filename,
        mri_dir=mri_dir,
        task=task
    )

    # 数据集拆分
    train_data, test_data = train_test_split(
        adni_dataset.data_dict,
        test_size=0.2,
        random_state=42
    )

    # 创建子数据集
    train_dataset = ADNI(label_filename, mri_dir, task)
    train_dataset.data_dict = train_data

    test_dataset = ADNI(label_filename, mri_dir, task)
    test_dataset.data_dict = test_data

    # 验证数据加载
    sample_mri, sample_label = train_dataset[0]
    print(f"Sample MRI shape: {sample_mri.shape}, Label: {sample_label}")

    # 打印数据集信息
    train_dataset.print_dataset_info(end=3)
    test_dataset.print_dataset_info(end=2)

    # 验证预处理流程
    train_transform, _ = ADNI_transform(augment=False)
    print("\nAugmented Transforms:")
    for i, t in enumerate(train_transform.transforms):
        print(f"{i + 1}. {t.__class__.__name__}")


if __name__ == '__main__':
    main()
    