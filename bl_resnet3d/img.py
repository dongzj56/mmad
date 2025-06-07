import os
import pandas as pd
import matplotlib.pyplot as plt

# 尝试多个可能的路径
possible_paths = [
    'baseline_MRI_3D/training_log1.csv',
    '/root/shared-nvme/baseline_MRI_3D/training_log1.csv'
]
csv_path = None
for p in possible_paths:
    if os.path.exists(p):
        csv_path = p
        break

if csv_path is None:
    raise FileNotFoundError(f"未找到 training_log1.csv，请检查路径。已尝试路径: {possible_paths}")

# 读取 CSV 文件
df = pd.read_csv(csv_path)

# 筛选所有要绘制的指标列（除 'epoch' 之外）
metrics = [col for col in df.columns if col != 'epoch']

# 针对每个指标绘制折线图
for metric in metrics:
    plt.figure()
    plt.plot(df['epoch'], df[metric])
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f'{metric} over Epochs')
    plt.show()
