import os
import json
import time
import csv
import numpy as np
from multiprocessing import freeze_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from monai.data import Dataset
from models import resnet
from datasets.ADNI import ADNI, ADNI_transform
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import logging  # 新增：导入 logging

def load_config(path="config/config.json"):
    with open(path) as f:
        return json.load(f)


class Config:
    def __init__(self, d):
        for k, v in d.items(): setattr(self, k, v)
        self.weight_decay = getattr(self, 'weight_decay', 1e-4)
        self.dropout_rate = getattr(self, 'dropout_rate', 0.5)
        self.n_splits = getattr(self, 'n_splits', 5)
        self.print_config()

    def print_config(self):
        print("Configuration Parameters:\n" + "=" * 40)
        for k, v in vars(self).items():
            print(f"{k}: {v}")
        print("=" * 40)


def generate_model(model_type='resnet', model_depth=50,
                   input_W=112, input_H=136, input_D=112,
                   resnet_shortcut='B',
                   pretrain_path='config/pretrain/resnet_50_23dataset.pth',
                   nb_class=2,
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
        nn.Linear(fc_in, nb_class)
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


def calculate_metrics(y_true, y_pred, y_score):
    return {
        'acc': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_score),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'cm': confusion_matrix(y_true, y_pred)
    }


def train():
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = Config(load_config())

    # 配置 logging
    log_path = os.path.join(cfg.checkpoint_dir, 'log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_path,
        filemode='w'
    )
    # 同时输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # 加载数据
    dataset = ADNI(cfg.label_file, cfg.mri_dir, cfg.task, cfg.augment).data_dict
    tr_val, test_data = train_test_split(dataset, test_size=0.2, random_state=42,
                                         stratify=[d['label'] for d in dataset])
    labels = [d['label'] for d in tr_val]

    # 初始化 TensorBoard 和 CSV
    writer = SummaryWriter(cfg.checkpoint_dir)
    csv_path = os.path.join(cfg.checkpoint_dir, 'cv_results.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['fold', 'epoch',
             'tr_acc', 'tr_auc', 'tr_loss',
             'vl_acc', 'vl_auc', 'vl_loss', 'lr']
        )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")  # 可选：用 logging 记录设备信息

    kf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(tr_val, labels), 1):
        print(f"\n=== Fold {fold}/{cfg.n_splits} ===")
        train_data = [tr_val[i] for i in train_idx]
        val_data   = [tr_val[i] for i in val_idx]

        tf_tr, tf_vt = ADNI_transform(augment=cfg.augment)
        ds_tr = Dataset(data=train_data, transform=tf_tr)
        ds_vl = Dataset(data=val_data,   transform=tf_vt)
        loader_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        loader_vl = DataLoader(ds_vl, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = generate_model(
            model_type=cfg.model_type, model_depth=cfg.model_depth,
            input_W=cfg.input_W, input_H=cfg.input_H, input_D=cfg.input_D,
            resnet_shortcut=cfg.resnet_shortcut,
            pretrain_path=cfg.pretrain_path,
            nb_class=2,
            dropout_rate=cfg.dropout_rate,
            device=device
        )
        
        class_counts = np.bincount([d['label'] for d in train_data])
        class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )

        warmup_epochs = max(1, min(10, int(cfg.num_epochs * 0.1)))
        total_epochs  = cfg.num_epochs
        cosine_epochs = total_epochs - warmup_epochs
        min_lr        = cfg.lr * 1e-4

        warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=min_lr, last_epoch=-1)
        scheduler    = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

        best_metric = -np.inf

        for epoch in range(1, cfg.num_epochs + 1):
            t0 = time.time()
            model.train()
            loss_sum = 0
            y_true, y_pred, y_score = [], [], []

            for batch in loader_tr:
                x = batch['MRI'].to(device)
                y = batch['label'].to(device).squeeze().long()
                out = model(x)
                loss = criterion(out, y)
                loss_sum += loss.item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                probs = torch.softmax(out, 1)[:, 1].detach().cpu().numpy()
                preds = out.argmax(1).detach().cpu().numpy()
                y_true.extend(y.cpu().numpy())
                y_score.extend(probs)
                y_pred.extend(preds)

            tr_metrics = calculate_metrics(y_true, y_pred, y_score)
            tr_loss = loss_sum / len(loader_tr)

            model.eval()
            v_true, v_pred, v_score = [], [], []
            vl_loss = 0.0
            with torch.no_grad():
                for batch in loader_vl:
                    x = batch['MRI'].to(device)
                    y = batch['label'].to(device).squeeze().long()
                    out = model(x)
                    loss = nn.CrossEntropyLoss()(out, y)
                    vl_loss += loss.item()
                    probs = torch.softmax(out, 1)
                    v_score.extend(probs[:, 1].cpu().numpy())
                    v_pred.extend(out.argmax(1).cpu().numpy())
                    v_true.extend(y.cpu().numpy())

            vl_metrics = calculate_metrics(v_true, v_pred, v_score)
            vl_loss = vl_loss / len(loader_vl)

            lr_now = scheduler.get_last_lr()[0]
            scheduler.step()

            writer.add_scalar(f'fold{fold}/train/acc', tr_metrics['acc'], epoch)
            writer.add_scalar(f'fold{fold}/val/acc', vl_metrics['acc'], epoch)
            writer.add_scalar(f'fold{fold}/train/loss', tr_loss, epoch)
            writer.add_scalar(f'fold{fold}/val/loss', vl_loss, epoch)
            writer.add_scalar(f'fold{fold}/lr', lr_now, epoch)

            # 使用 logging 记录每轮信息
            epoch_time = time.time() - t0
            logging.info(
                f"Fold{fold} Ep{epoch:03d} | "
                f"tr_acc={tr_metrics['acc']:.6f}(AUC={tr_metrics['auc']:.6f}) "
                f"vl_acc={vl_metrics['acc']:.6f}(AUC={vl_metrics['auc']:.6f}) "
                f"lr={lr_now:.7f} time={epoch_time:.3f}s"
            )

            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    fold, epoch,
                    f"{tr_metrics['acc']:.6f}", f"{tr_metrics['auc']:.6f}", f"{tr_loss:.6f}",
                    f"{vl_metrics['acc']:.6f}", f"{vl_metrics['auc']:.6f}", f"{vl_loss:.6f}",
                    f"{lr_now:.6f}"
                ])

            current_metric = 0.3 * vl_metrics['auc'] + 0.7 * vl_metrics['acc']
            if current_metric > best_metric:
                best_metric = current_metric
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics': {
                        'train_auc': tr_metrics['auc'],
                        'val_auc': vl_metrics['auc'],
                        'val_loss': vl_loss,
                        'current_metric': current_metric
                    },
                    'config': vars(cfg)
                }, os.path.join(cfg.checkpoint_dir, f"best_fold{fold}.pth"))

        torch.save({
            'epoch': cfg.num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'final_metrics': {
                'train_auc': tr_metrics['auc'],
                'val_auc': vl_metrics['auc'],
                'val_loss': vl_loss
            }
        }, os.path.join(cfg.checkpoint_dir, f"model_fold{fold}_final.pth"))

    print("\n=== CV complete ===")
    
    test_models(cfg.checkpoint_dir, test_data)
    writer.close()

# rest of the code (test_models, if __name__ == '__main__', etc.) 保持不变
