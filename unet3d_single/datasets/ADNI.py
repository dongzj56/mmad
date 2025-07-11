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
    """
    用于处理ADNI数据集的类，仅处理MRI数据和标签
    """

    def __init__(self, label_file, mri_dir, task='ADCN', augment=False):
        """
        初始化ADNI数据集类，读取数据和标签文件，并生成数据字典

        :param label_file: 标签文件路径（包含 Group 和 Subject_ID 等信息）
        :param mri_dir: MRI图像所在目录
        :param task: 任务类型，用于选择不同标签类别
        :param augment: 是否进行数据增强
        """
        # self.label = pd.read_csv(label_file)
        self.label = pd.read_csv(label_file, encoding='ISO-8859-1')
        self.mri_dir = mri_dir
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
        # if self.task == 'CNEMCI':
        #     self.labels = self.label[(self.label['Group'] == 'CN') | (self.label['Group'] == 'EMCI')]
        #     self.label_dict = {'CN': 0, 'EMCI': 1}
        # if self.task == 'LMCIAD':
        #     self.labels = self.label[(self.label['Group'] == 'LMCI') | (self.label['Group'] == 'AD')]
        #     self.label_dict = {'LMCI': 0, 'AD': 1}
        # if self.task == 'EMCILMCI':
        #     self.labels = self.label[(self.label['Group'] == 'EMCI') | (self.label['Group'] == 'LMCI')]
        #     self.label_dict = {'EMCI': 0, 'LMCI': 1}
        if self.task == 'SMCIPMCI':
            self.labels = self.label[(self.label['Group'] == 'SMCI') | (self.label['Group'] == 'PMCI')]
            self.label_dict = {'SMCI': 0, 'PMCI': 1}

    # def _process_labels(self):
    #     """根据指定任务生成 self.labels 和 self.label_dict"""
    #     t = self.task.upper()

    #     if t == 'ADCN':
    #         groups = ['AD', 'CN']
    #     elif t == 'SMCIPMCI':
    #         groups = ['SMCI', 'PMCI']
    #     elif t == 'ADCNSMCIPMCI':
    #         # ---- 新增四分类 ----
    #         groups = ['CN', 'SMCI', 'PMCI', 'AD']
    #     else:
    #         raise ValueError(f'Unsupported task: {self.task}')

    #     # 1) 选出需要的行
    #     self.labels = self.label[self.label['Group'].isin(groups)].copy()

    #     # 2) 建立标签映射，确保数值从 0 开始连续
    #     self.label_dict = {g: i for i, g in enumerate(groups)}

    def _build_data_dict(self):
        subject_list = self.labels['Subject_ID'].tolist()
        label_list = self.labels['Group'].tolist()
        self.data_dict = [
            {
                'MRI': os.path.join(self.mri_dir, f'{subject}.nii'),
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
    # ------------- 基本路径与任务 -------------
    dataroot        = r'C:\Users\dongzj\Desktop\adni_dataset\MRI_GM_112_136_112'
    label_filename  = r'C:\Users\dongzj\Desktop\adni_dataset\ADNI_902.csv'
    mri_dir         = dataroot              # 已在 ADNI 内部拼 path
    task            = 'ADCN'        # 四分类

    # ------------- 1) 只创建一次完整数据集 -------------
    full_dataset = ADNI(
        label_file=label_filename,
        mri_dir=mri_dir,
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
    sample_mri, sample_label = train_dataset[0]
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


def run_5fold_cv():
    # ---------------- 路径与任务 ----------------
    dataroot       = r'C:\Users\dongz\Desktop\adni_dataset\MRI_GM_112_136_112'
    label_filename = r'C:\Users\dongz\Desktop\adni_dataset\ADNI_902.csv'
    task           = 'ADCNSMCIPMCI'

    # 1) 只实例化一次完整数据集
    full_dataset = ADNI(label_filename, dataroot, task)

    # 2) 取出标签，用于分层划分
    y = [d["label"] for d in full_dataset.data_dict]

    # 3) StratifiedKFold 确保四类比例在每折中大致一致
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(torch.arange(len(full_dataset)), y), 1):
        print(f"\n======== Fold {fold} ========")

        # 4) 构建子集
        train_set = Subset(full_dataset, train_idx)   # 训练
        val_set   = Subset(full_dataset, val_idx)     # 验证

        # 5) 如有需要，可为训练/验证分别指定 DataLoader
        train_loader = DataLoader(train_set, batch_size=8, shuffle=True,  num_workers=4)
        val_loader   = DataLoader(val_set,   batch_size=8, shuffle=False, num_workers=4)

        # ---- 这里放你的训练 / 验证代码 ----
        # for epoch in range(num_epochs):
        #     train_one_epoch(train_loader, ...)
        #     validate(val_loader, ...)

        # 示例：打印每折样本数
        print(f"Train: {len(train_set)} , Val: {len(val_set)}")
        # 若想查看类别分布，可自行统计 train_set.dataset.data_dict[i]["label"]


if __name__ == '__main__':
    main()
    # run_5fold_cv()
    