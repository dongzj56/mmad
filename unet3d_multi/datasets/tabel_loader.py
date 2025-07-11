# tabel_loader.py

import pandas as pd


def load_adni_data_binary(csv_path: str,
                          start_col: int,
                          label_col: str = "Group",
                          class0: str = "CN",
                          class1: str = "AD") -> tuple:
    """
    从 CSV 文件加载 ADNI 表格数据，进行二分类：
      - 筛选 label_col 列中只包含 class0 或 class1 的行
      - 将 class0 → 0，class1 → 1
      - 从第 start_col 列（索引从 0 开始）开始，到末列的所有列作为特征，排除 label_col
      - 对类别型特征列进行整数编码
      - 删除标签列为 NaN 的行
    返回：
      X: numpy.ndarray, dtype=float32, 形状 (n_samples, n_features)
      y: numpy.ndarray, dtype=int64,  形状 (n_samples,)
    """
    df = pd.read_csv(csv_path)

    # 动态选特征列：从第 start_col 列开始到末列，排除 label_col
    all_cols = list(df.columns)
    if len(all_cols) <= start_col:
        raise ValueError(f"CSV 列数不足 {start_col+1} 列，无法按索引 {start_col} 取特征")
    feature_cols = [c for c in all_cols[start_col:] if c != label_col]

    # 检查必需列存在
    required_cols = feature_cols + [label_col]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # 仅保留两类
    df2 = df[df[label_col].isin([class0, class1])].copy()
    if df2.empty:
        raise ValueError(f"No samples for classes '{class0}' or '{class1}' in column '{label_col}'")

    # 标签映射
    df2[label_col] = df2[label_col].map({class0: 0, class1: 1}).astype("int64")

    # 类别特征整数编码
    cat_cols = [
        c for c in feature_cols
        if df2[c].dtype == "object" or str(df2[c].dtype).startswith("category")
    ]
    for col in cat_cols:
        df2[col] = pd.Categorical(df2[col]).codes.astype("int16")

    # 删除标签缺失行
    df2 = df2.dropna(subset=[label_col])

    # 提取 X, y
    X = df2[feature_cols].astype("float32").values
    y = df2[label_col].values
    return X, y


def load_adni_data_triclass(csv_path: str,
                            start_col: int,
                            label_col: str = "Group",
                            class0: str = "CN",
                            class1: str = "MCI",
                            class2: str = "AD") -> tuple:
    """
    从 CSV 文件加载 ADNI 表格数据，进行三分类：
      - 筛选 label_col 列中只包含 class0、class1 或 class2 的行
      - 将 class0 → 0，class1 → 1，class2 → 2
      - 从第 start_col 列开始，到末列的所有列作为特征，排除 label_col
      - 对类别型特征列进行整数编码
      - 删除标签列为 NaN 的行
    返回：
      X: numpy.ndarray, dtype=float32, 形状 (n_samples, n_features)
      y: numpy.ndarray, dtype=int64,  形状 (n_samples,)
    """
    df = pd.read_csv(csv_path)

    # 动态选特征列
    all_cols = list(df.columns)
    if len(all_cols) <= start_col:
        raise ValueError(f"CSV 列数不足 {start_col+1} 列，无法按索引 {start_col} 取特征")
    feature_cols = [c for c in all_cols[start_col:] if c != label_col]

    # 检查必需列存在
    required_cols = feature_cols + [label_col]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # 仅保留三类
    df2 = df[df[label_col].isin([class0, class1, class2])].copy()
    if df2.empty:
        raise ValueError(f"No samples for classes '{class0}', '{class1}', or '{class2}' in column '{label_col}'")

    # 标签映射
    mapping = {class0: 0, class1: 1, class2: 2}
    df2[label_col] = df2[label_col].map(mapping).astype("int64")

    # 类别特征整数编码
    cat_cols = [
        c for c in feature_cols
        if df2[c].dtype == "object" or str(df2[c].dtype).startswith("category")
    ]
    for col in cat_cols:
        df2[col] = pd.Categorical(df2[col]).codes.astype("int16")

    # 删除标签缺失行
    df2 = df2.dropna(subset=[label_col])

    # 提取 X, y
    X = df2[feature_cols].astype("float32").values
    y = df2[label_col].values
    return X, y


def load_adni_data_quadclass(csv_path: str,
                             start_col: int,
                             label_col: str = "Group",
                             class0: str = "CN",
                             class1: str = "SMCI",
                             class2: str = "PMCI",
                             class3: str = "AD") -> tuple:
    """
    从 CSV 文件加载 ADNI 表格数据，进行四分类：
      - 筛选 label_col 列中只包含 class0、class1、class2 或 class3 的行
      - 将 class0 → 0，class1 → 1，class2 → 2，class3 → 3
      - 从第 start_col 列开始，到末列的所有列作为特征，排除 label_col
      - 对类别型特征列进行整数编码
      - 删除标签列为 NaN 的行
    返回：
      X: numpy.ndarray, dtype=float32, 形状 (n_samples, n_features)
      y: numpy.ndarray, dtype=int64,  形状 (n_samples,)
    """
    df = pd.read_csv(csv_path)

    # 动态选特征列
    all_cols = list(df.columns)
    if len(all_cols) <= start_col:
        raise ValueError(f"CSV 列数不足 {start_col+1} 列，无法按索引 {start_col} 取特征")
    feature_cols = [c for c in all_cols[start_col:] if c != label_col]

    # 检查必需列存在
    required_cols = feature_cols + [label_col]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # 仅保留四类
    df2 = df[df[label_col].isin([class0, class1, class2, class3])].copy()
    if df2.empty:
        raise ValueError(
            f"No samples for classes '{class0}', '{class1}', '{class2}', or '{class3}' in column '{label_col}'"
        )

    # 标签映射
    mapping = {class0: 0, class1: 1, class2: 2, class3: 3}
    df2[label_col] = df2[label_col].map(mapping).astype("int64")

    # 类别特征整数编码
    cat_cols = [
        c for c in feature_cols
        if df2[c].dtype == "object" or str(df2[c].dtype).startswith("category")
    ]
    for col in cat_cols:
        df2[col] = pd.Categorical(df2[col]).codes.astype("int16")

    # 删除标签缺失行
    df2 = df2.dropna(subset=[label_col])

    # 提取 X, y
    X = df2[feature_cols].astype("float32").values
    y = df2[label_col].values
    return X, y
