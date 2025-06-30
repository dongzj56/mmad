import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, auc  # Add these imports
)
import json
import matplotlib.pyplot as plt
from models import resnet
from datasets.ADNI import ADNI, ADNI_transform
from sklearn.model_selection import train_test_split
from monai.data import Dataset
from thop import profile


def load_config(path="/root/shared-nvme/MM_AD_code/config/config.json"):
    with open(path) as f:
        return json.load(f)

class Config:
    def __init__(self, d):
        for k, v in d.items(): 
            setattr(self, k, v)
        self.print_config()

    def print_config(self):
        print("Configuration Parameters:\n" + "="*40)
        for k, v in vars(self).items():
            print(f"{k}: {v}")
        print("="*40)

def generate_model(model_type='resnet', model_depth=50,
                   input_W=224, input_H=224, input_D=224,
                   resnet_shortcut='B',
                   pretrain_path='config/pretrain/resnet_18_23dataset.pth',
                   nb_class=2,  # 修改为2分类输出
                   dropout_rate=0.5,
                   device=torch.device('cpu')):
    assert model_type == 'resnet'
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    fn = {
        10: resnet.resnet10, 18: resnet.resnet18, 34: resnet.resnet34,
        50: resnet.resnet50, 101: resnet.resnet101,
        152: resnet.resnet152, 200: resnet.resnet200
    }[model_depth]

    net = fn(
        sample_input_W=input_W, sample_input_H=input_H, sample_input_D=input_D,
        shortcut_type=resnet_shortcut, no_cuda=True, num_seg_classes=1
    )

    fc_in = {10: 256, 18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048, 200: 2048}[model_depth]
    net.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(fc_in, nb_class)  # 输出维度与nb_class=2匹配
    )

    net.to(device)
    sd = net.state_dict()
    if os.path.isfile(pretrain_path):
        ckpt = torch.load(pretrain_path, map_location=device)
        state = ckpt.get('state_dict', ckpt)
        pd = {k: v for k, v in state.items() if k in sd}
        sd.update(pd)
        net.load_state_dict(sd)
        print("Loaded pretrained weights.")
    else:
        print(f"[Warning] no pretrained file at {pretrain_path}")
    return net

# 计算指标
def calculate_metrics(y_true, y_pred, y_score):
    """
    返回字典中键的排列顺序：ACC → PRE → SEN → SPE → F1 → AUC → MCC
    """
    # 混淆矩阵：[[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 7 个指标
    acc  = accuracy_score(y_true, y_pred)                       # ACC
    pre  = precision_score(y_true, y_pred, zero_division=0)     # PRE
    sen  = recall_score(y_true, y_pred, zero_division=0)        # SEN (=TPR)
    spe  = tn / (tn + fp + 1e-8)                                # SPE (=TNR)
    f1   = f1_score(y_true, y_pred, zero_division=0)            # F1
    auc  = roc_auc_score(y_true, y_score)                       # AUC
    mcc  = matthews_corrcoef(y_true, y_pred)                    # MCC

    return {
        'ACC': acc, 'PRE': pre, 'SEN': sen, 'SPE': spe,
        'F1': f1, 'AUC': auc, 'MCC': mcc,
        'cm': np.array([[tn, fp],
                        [fn, tp]])    
    }

def load_test_data(cfg):
    """加载测试数据（与训练时完全相同的分割方式）"""
    dataset = ADNI(cfg.label_file, cfg.mri_dir, cfg.task, cfg.augment).data_dict
    _, test_data = train_test_split(
        dataset, 
        test_size=0.2, 
        random_state=42,
        stratify=[d['label'] for d in dataset]
    )
    return test_data

def test_models(checkpoint_dir, test_data):
    """主测试函数"""
    cfg = Config(load_config())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 数据准备
    test_transforms = ADNI_transform(augment=False)[1]
    test_ds = Dataset(data=test_data, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # 存储结果
    all_metrics = []
    all_fold_probs = []
    all_fold_labels = []

    # 打开结果文件
    results_path = os.path.join(checkpoint_dir, 'test_result.txt')
    with open(results_path, 'w') as result_file:
        result_file.write("=== Test Results by Fold ===\n\n")

        plt.figure(figsize=(10, 8))
        for fold in range(1, cfg.n_splits + 1):
            # 加载模型
            model = generate_model(
                model_type=cfg.model_type,
                model_depth=cfg.model_depth,
                input_W=cfg.input_W,
                input_H=cfg.input_H,
                input_D=cfg.input_D,
                resnet_shortcut=cfg.resnet_shortcut,
                nb_class=2,
                dropout_rate=cfg.dropout_rate,
                device=device
            )
            ckpt_path = os.path.join(checkpoint_dir, f"model_fold{fold}.pth")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # —— 新增：计算 Params, FLOPs, Memory —— 
            dummy_input = torch.randn(1, 1, cfg.input_W, cfg.input_H, cfg.input_D).to(device)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            # 参数量与内存开销
            total_params = params
            mem_bytes = total_params * 4  # 假设 float32，每个参数 4 字节
            mem_mb = mem_bytes / (1024**2)
            flops_g = flops / 1e9  # 转为 GFLOPs

            info_line = (
                f"Fold {fold} Model Complexity:\n"
                f"  Params: {total_params:,} ({mem_mb:.2f} MB)\n"
                f"  FLOPs: {flops_g:.3f} GFLOPs\n\n"
            )
            print(info_line, end='')
            result_file.write(info_line)
            # — end 新增 —

            # 预测
            fold_probs = []
            y_true = []
            with torch.no_grad():
                for batch in test_loader:
                    x = batch['MRI'].to(device)
                    y = batch['label'].squeeze().cpu().numpy()
                    y_true.extend(y)
                    out = model(x)
                    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                    fold_probs.extend(probs)

            all_fold_probs.extend(fold_probs)
            all_fold_labels.extend(y_true)

            # 计算指标
            y_pred = (np.array(fold_probs) > 0.5).astype(int)
            metrics = calculate_metrics(y_true, y_pred, fold_probs)
            all_metrics.append(metrics)

            # ROC 曲线
            fpr, tpr, _ = roc_curve(y_true, fold_probs)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold} (AUC={auc(fpr, tpr):.2f})')

            # 打印 & 写入文件：本折结果
            header = f"--- Fold {fold} Test Metrics ---\n"
            print(header, end='')
            result_file.write(header)
            for k in ['ACC','PRE','SEN','SPE','F1','AUC','MCC']:
                line = f"{k}: {metrics[k]:.4f}\n"
                print(line, end='')
                result_file.write(line)
            cm_line = f"Confusion Matrix:\n{metrics['cm']}\n\n"
            print(cm_line, end='')
            result_file.write(cm_line)

        # 绘制与保存合并 ROC
        fpr, tpr, _ = roc_curve(all_fold_labels, all_fold_probs)
        roc_auc = auc(fpr, tpr)
        mean_fpr = np.linspace(0, 1, 100)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        plt.plot(mean_fpr, interp_tpr, 'b-', lw=2, label=f'Mean ROC (AUC={roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Test ROC Curves')
        plt.legend(loc="lower right")
        roc_path = os.path.join(checkpoint_dir, 'test_roc_curves.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nROC曲线已保存至: {roc_path}")

        # 计算总体指标
        result_file.write("=== Overall Test Results ===\n\n")
        print("\n=== Overall Test Results ===")
        for k in ['ACC','PRE','SEN','SPE','F1','AUC','MCC']:
            vals = [m[k] for m in all_metrics]
            mean, std = np.mean(vals), np.std(vals)
            line = f"{k}: {mean:.4f} ± {std:.4f}\n"
            print(line, end='')
            result_file.write(line)

    print(f"\n测试结果已保存至: {results_path}")
    return (
        {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0] if k != 'cm'},
        {k: np.std([m[k] for m in all_metrics])  for k in all_metrics[0] if k != 'cm'}
    )

if __name__ == '__main__':
    # 加载配置
    cfg = Config(load_config())
    
    # 确保使用与训练时相同的测试集
    test_data = load_test_data(cfg)
    
    # 指定checkpoint目录
    checkpoint_dir = "/root/shared-nvme/MM_AD_code/checkpoints/MRI"
    
    # 运行测试
    test_models(checkpoint_dir, test_data)