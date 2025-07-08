#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p_value_compare.py

比较两个实验在同一 5 折交叉验证上的指标差异，并输出 p 值
作者：ChatGPT
"""

import numpy as np
from scipy import stats

def compute_p_values(arr1, arr2):
    """
    计算成对 t 检验与 Wilcoxon 检验的 p 值

    参数
    ----
    arr1, arr2 : array-like, shape (n_folds,)
        两个模型（或实验）在每一折交叉验证上的评估指标（如 AUC、Accuracy …）

    返回
    ----
    dict : {'t-test': p_t, 'wilcoxon': p_w}
    """
    # 转成 numpy 并做基本检查
    x = np.asarray(arr1, dtype=float)
    y = np.asarray(arr2, dtype=float)
    if x.shape != y.shape:
        raise ValueError("两组结果长度必须一致！")

    # 成对（配对）t 检验
    t_stat, p_t = stats.ttest_rel(x, y)

    # Wilcoxon 符号秩检验（非参数，要求无并列零差值或使用 correction）
    try:
        w_stat, p_w = stats.wilcoxon(x, y, correction=True)
    except ValueError as e:
        # 如果全是零差值会报错；视情况处理
        p_w = np.nan
        print("Wilcoxon 检验无法执行：", e)

    return {"t-test": p_t, "wilcoxon": p_w}

if __name__ == "__main__":
    # ======= 示例：手动输入 5 折 Accuracy =========
    # 模型 A 五折结果
    model_a = [0.9152, 0.8830, 0.9218, 0.9340, 0.9418]
    # 模型 B 五折结果
    model_b = [0.9867, 0.9767, 0.9806, 0.9845, 0.9751]

    p_vals = compute_p_values(model_a, model_b)
    print("成对 t-test p 值：", p_vals["t-test"])
    print("Wilcoxon p 值 ：", p_vals["wilcoxon"])

    # ======= 如果想改为读取 CSV / Excel =========
    # import pandas as pd
    # df = pd.read_csv("cv_results.csv")  # 或 read_excel(...)
    # model_a = df["ResNet50"].values
    # model_b = df["ProposedNet"].values
    # p_vals = compute_p_values(model_a, model_b)
