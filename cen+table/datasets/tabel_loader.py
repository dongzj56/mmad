import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

def load_tabular(csv_path: str,
                            label_col: str,
                            classes: List[str],
                            feature_cols: Optional[List[str]] = None,
                            start_col: Optional[int] = None,
                            dropna: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    通用 CSV → (X, y) 加载函数，支持二分类 / 多分类。

    Parameters
    ----------
    csv_path : str
        CSV 文件路径
    label_col : str
        标签列名
    classes : List[str]
        要纳入的类别列表；映射为 0..n-1
    feature_cols : List[str], optional
        指定特征列名列表；如果为 None，则使用 start_col 起始的所有列
    start_col : int, optional
        当 feature_cols=None 时，从此索引开始到末列均视为特征列
    dropna : bool, default True
        是否删除在 label 或特征列上包含 NaN 的样本

    Returns
    -------
    X : np.ndarray  (dtype=float32, shape=(n_samples, n_features))
    y : np.ndarray  (dtype=int64,   shape=(n_samples,))
    """
    df = pd.read_csv(csv_path)

    # ---------- 1. 选择特征列 ----------
    if feature_cols is None:
        if start_col is None:
            raise ValueError("必须提供 feature_cols 或 start_col 其中之一")
        all_cols = list(df.columns)
        if len(all_cols) <= start_col:
            raise ValueError(f"start_col={start_col} 已超出列数范围")
        feature_cols = [c for c in all_cols[start_col:] if c != label_col]

    # 检查列存在
    missing = [c for c in feature_cols + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # ---------- 2. 仅保留所需类别 ----------
    df = df[df[label_col].isin(classes)].copy()
    if df.empty:
        raise ValueError(f"No samples belonging to {classes} in '{label_col}'")

    # ---------- 3. 标签映射 ----------
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    df[label_col] = df[label_col].map(mapping).astype("int64")

    # ---------- 4. 类别特征整数编码 ----------
    cat_cols = [
        c for c in feature_cols
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes.astype("int16")

    # ---------- 5. 删除缺失 ----------
    if dropna:
        df = df.dropna(subset=[label_col] + feature_cols)

    # ---------- 6. 构建 X, y ----------
    X = df[feature_cols].astype("float32").values
    y = df[label_col].values

    return X, y

