import torch, pandas as pd, numpy as np
from typing import List, Optional, Tuple
from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding
from sklearn.model_selection import train_test_split
import os

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"▶ Using device: {DEVICE}")

def tabular_encoder_classifier(
    csv_path: str,
    label_col: str,
    classes: List[str],
    feature_cols: Optional[List[str]] = None,
    start_col: Optional[int] = None,
    n_fold: int = 0,                  # 0 = Vanilla, >0 = K‑Fold OoF
    dropna: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    train_out: str = "train_embeddings.csv",
    test_out: str  = "test_embeddings.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    读取 CSV → (X,y) → TabPFN 嵌入 → 保存/返回 train_df, test_df
    """
    # ---------- 1. 读 & 预处理 ----------
    df = pd.read_csv(csv_path)

    # 1.1 选特征列
    if feature_cols is None:
        if start_col is None:
            raise ValueError("必须提供 feature_cols 或 start_col")
        if len(df.columns) <= start_col:
            raise ValueError(f"start_col={start_col} 超出列数")
        feature_cols = [c for c in df.columns[start_col:] if c != label_col]

    # 1.2 过滤类别
    df = df[df[label_col].isin(classes)].copy()
    if df.empty:
        raise ValueError(f"'{label_col}' 列中找不到 {classes}")

    # 1.3 标签映射
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    df[label_col] = df[label_col].map(mapping).astype("int64")

    # 1.4 类别特征整数编码
    cat_cols = [c for c in feature_cols
                if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes.astype("int16")

    # 1.5 删除缺失
    if dropna:
        df = df.dropna(subset=[label_col] + feature_cols)
    if df.empty:
        raise ValueError("清洗后样本为空")

    # ---------- 2. 构建 X, y & 划分 ----------
    X = df[feature_cols].astype("float32").values
    y = df[label_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ---------- 3. TabPFN 嵌入 ----------
    clf      = TabPFNClassifier(device=DEVICE)
    embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)

    train_emb = embedder.get_embeddings(X_tr, y_tr, X_te, data_source="train")[0]
    test_emb  = embedder.get_embeddings(X_tr, y_tr, X_te, data_source="test")[0]
    print(f"✓ train_emb {train_emb.shape} · test_emb {test_emb.shape}")

    # ---------- 4. 封装 & 保存 ----------
    train_df = pd.DataFrame(train_emb); train_df.insert(0, "label", y_tr)
    test_df  = pd.DataFrame(test_emb);  test_df.insert(0, "label", y_te)

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out,  index=False)
    print(f"已保存到: {train_out}  /  {test_out}")

    return train_df, test_df

def tabular_encoder(
    csv_path: str,
    label_col: str,
    classes: List[str],
    feature_cols: Optional[List[str]] = None,
    start_col: Optional[int] = None,
    n_fold: int = 0,
    dropna: bool = False,
    out_csv: str = "tabular_embeddings.csv",
) -> pd.DataFrame:
    """
    读取 CSV -> 预处理 -> 一次性生成 TabPFN 嵌入

    除了嵌入向量，还会保留原始表格第一列（如 Subject_ID）方便后续对齐。

    Returns
    -------
    df_emb : pd.DataFrame  包含 'id', 'label', 及嵌入向量列
    """
    # --- 1. 读入 CSV ---
    df = pd.read_csv(csv_path)

    # 保留原始首列作为 id
    id_col_name = df.columns[0]
    ids = df[id_col_name].astype(str)

    # --- 2. 预处理 ---
    # 2.1 确定特征列
    if feature_cols is None:
        if start_col is None:
            raise ValueError("请提供 feature_cols 或 start_col")
        feature_cols = [c for c in df.columns[start_col:] if c != label_col]

    # 2.2 过滤类别
    df = df[df[label_col].isin(classes)].copy()
    ids = ids[df.index]  # 同步 ids

    # 2.3 标签映射
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    df[label_col] = df[label_col].map(mapping).astype("int64")

    # 2.4 类别型特征编码
    cat_cols = [c for c in feature_cols
                if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes.astype("int16")

    # 2.5 删除缺失
    if dropna:
        mask = df[feature_cols + [label_col]].notna().all(axis=1)
        df = df[mask]
        ids = ids[mask]

    # --- 3. 构建 X, y ---
    X = df[feature_cols].astype("float32").values
    y = df[label_col].values

    # --- 4. 生成嵌入 ---
    clf = TabPFNClassifier(device=DEVICE)
    embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)
    emb = embedder.get_embeddings(X, y, X, data_source="train")[0]
    print(f"✓ full embedding shape = {emb.shape}")

    # --- 5. 封装并保存 ---
    df_emb = pd.DataFrame(emb)
    df_emb.insert(0, 'label', y)
    df_emb.insert(0, id_col_name, ids.values)
    df_emb.to_csv(out_csv, index=False)
    print(f"→ 保存到 {out_csv}")

    return df_emb

def _prepare_tabular_data(
    csv_path: str,
    label_col: str,
    classes: List[str],
    feature_cols: Optional[List[str]] = None,
    start_col: Optional[int] = None,
    dropna: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    读取并清洗 DataFrame，返回 df 和最终用到的 feature_cols 列表。
    """
    df = pd.read_csv(csv_path)

    # 选特征列
    if feature_cols is None:
        if start_col is None:
            raise ValueError("必须提供 feature_cols 或 start_col")
        feature_cols = [c for c in df.columns[start_col:] if c != label_col]

    # 过滤类别、映射标签
    df = df[df[label_col].isin(classes)].copy()
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    df[label_col] = df[label_col].map(mapping).astype("int64")

    # 类别特征编码
    cat_cols = [
        c for c in feature_cols
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]
    for c in cat_cols:
        df[c] = pd.Categorical(df[c]).codes.astype("int16")

    # 删除缺失
    if dropna:
        df = df.dropna(subset=[label_col] + feature_cols)
    if df.empty:
        raise ValueError("清洗后样本为空")

    return df, feature_cols

def tabular_encoder_train(
    csv_path: str,
    label_col: str,
    classes: List[str],
    feature_cols: Optional[List[str]] = None,
    start_col: Optional[int] = None,
    n_fold: int = 0,                  # 0 = Vanilla, >0 = K‑Fold OoF
    dropna: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    train_out: str = "train_embeddings.csv",
) -> pd.DataFrame:
    """
    生成并保存训练集的 TabPFN 嵌入。
    """
    # 1. 预处理
    df, feature_cols = _prepare_tabular_data(
        csv_path, label_col, classes, feature_cols, start_col, dropna
    )

    # 2. 划分
    X = df[feature_cols].astype("float32").values
    y = df[label_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size,
        stratify=y, random_state=random_state
    )

    # 3. 嵌入
    clf      = TabPFNClassifier(device=DEVICE)
    embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)
    train_emb = embedder.get_embeddings(
        X_tr, y_tr, X_te, data_source="train"
    )[0]
    print(f"✓ train_emb shape: {train_emb.shape}")

    # 4. 保存
    train_df = pd.DataFrame(train_emb)
    train_df.insert(0, "label", y_tr)
    train_df.to_csv(train_out, index=False)
    print(f"已保存训练集嵌入 → {train_out}")

    return train_df

def tabular_encoder_test(
    csv_path: str,
    label_col: str,
    classes: List[str],
    feature_cols: Optional[List[str]] = None,
    start_col: Optional[int] = None,
    n_fold: int = 0,
    dropna: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    test_out: str  = "test_embeddings.csv",
) -> pd.DataFrame:
    """
    生成并保存测试集的 TabPFN 嵌入（使用同一份划分）。
    """
    # 1. 预处理
    df, feature_cols = _prepare_tabular_data(
        csv_path, label_col, classes, feature_cols, start_col, dropna
    )

    # 2. 同样划分
    X = df[feature_cols].astype("float32").values
    y = df[label_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size,
        stratify=y, random_state=random_state
    )

    # 3. 嵌入
    clf      = TabPFNClassifier(device=DEVICE)
    embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)
    test_emb = embedder.get_embeddings(
        X_tr, y_tr, X_te, data_source="test"
    )[0]
    print(f"✓ test_emb shape: {test_emb.shape}")

    # 4. 保存
    test_df = pd.DataFrame(test_emb)
    test_df.insert(0, "label", y_te)
    test_df.to_csv(test_out, index=False)
    print(f"已保存测试集嵌入 → {test_out}")

    return test_df


def tabular_encoder_fold(
    fold_dir: str,
    label_col: str,
    classes: List[str],
    feature_cols: Optional[List[str]] = None,
    start_col: Optional[int] = None,
    device: str = "cpu",
    n_fold: int = 0,
    dropna: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    对一个 fold 文件夹下的 train.csv / val.csv / test.csv 进行 TabPFN 编码，
    并保存 train_emb.csv、val_emb.csv、test_emb.csv。

    Parameters
    ----------
    fold_dir : str
        包含 train.csv, val.csv, test.csv 的文件夹路径。
    label_col : str
        CSV 中的标签列名（如 "Group"）。
    classes : List[str]
        标签的所有可能类别列表。
    feature_cols : List[str], optional
        明确的特征列列表；若为 None，则使用 start_col。
    start_col : int, optional
        若 feature_cols=None，则从第几列开始当作特征（0-base）。
    n_fold : int
        OOF fold 数，若>0 则使用 K‑Fold OOF 嵌入；否则 0。
    dropna : bool
        是否删除缺失。

    Returns
    -------
    train_df, val_df, test_df : Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        三个嵌入后的 DataFrame，均已在第一列插入 "Subject_ID"，第二列插入 "label"。
    """
    # 1. 读取三份原始 CSV
    paths = {split: os.path.join(fold_dir, f"{split}.csv")
             for split in ("train", "val", "test")}
    dfs   = {split: pd.read_csv(p) for split, p in paths.items()}

    subjects = {
        split: dfs[split]["Subject_ID"].values
        for split in ("train", "val", "test")
    }

    # 2. 确定 feature_cols
    if feature_cols is None:
        if start_col is None:
            raise ValueError("必须提供 feature_cols 或 start_col")
        cols = dfs["train"].columns.tolist()
        if start_col >= len(cols):
            raise ValueError(f"start_col={start_col} 超出 train.csv 列数")
        feature_cols = [c for c in cols[start_col:] if c != label_col]

    # 3. 统一预处理：过滤类别、映射标签、类别特征编码、dropna
    for split, df in dfs.items():
        dfs[split] = df[df[label_col].isin(classes)].copy()
        mapping = {cls: idx for idx, cls in enumerate(classes)}
        dfs[split][label_col] = dfs[split][label_col].map(mapping).astype("int64")
        cat_cols = [
            c for c in feature_cols
            if dfs[split][c].dtype == "object" or str(dfs[split][c].dtype).startswith("category")
        ]
        for c in cat_cols:
            dfs[split][c] = pd.Categorical(dfs[split][c]).codes.astype("int16")
        if dropna:
            dfs[split] = dfs[split].dropna(subset=[label_col] + feature_cols)
        if dfs[split].empty:
            raise ValueError(f"{split}.csv 清洗后样本为空")

    # 4. 构造训练和测试输入
    X_tr = dfs["train"][feature_cols].astype("float32").values
    y_tr = dfs["train"][label_col].values

    X_val = dfs["val"][feature_cols].astype("float32").values
    y_val = dfs["val"][label_col].values

    X_te = dfs["test"][feature_cols].astype("float32").values
    y_te = dfs["test"][label_col].values

    # 5. TabPFN 嵌入
    clf      = TabPFNClassifier(device=device)
    embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)

    train_emb = embedder.get_embeddings(X_tr, y_tr, X_tr, data_source="train")[0]
    val_emb   = embedder.get_embeddings(X_tr, y_tr, X_val, data_source="test")[0]
    test_emb  = embedder.get_embeddings(X_tr, y_tr, X_te,  data_source="test")[0]

    # 6. 封装结果并保存 (第一列 Subject_ID，第二列 label)
    def make_df(emb, labels, out_name):
        split = out_name.split("_", 1)[0]      # 'train' / 'val' / 'test'
        df_emb = pd.DataFrame(emb)
        df_emb.insert(0, "label", labels)
        df_emb.insert(0, "Subject_ID", subjects[split])
        path = os.path.join(fold_dir, out_name)
        df_emb.to_csv(path, index=False)
        print(f"✓ Saved {out_name} ({df_emb.shape})")
        return df_emb

    train_df = make_df(train_emb, y_tr, "train_emb.csv")
    val_df   = make_df(val_emb,   y_val, "val_emb.csv")
    test_df  = make_df(test_emb,  y_te,  "test_emb.csv")

    return train_df, val_df, test_df


if __name__ == "__main__":
    # 示例参数
    csv_path   = rf"C:\Users\dongzj\Desktop\Multimodal_AD\adni_dataset\ADNI_Tabel.csv"
    label_col  = "Group"
    classes    = ["CN", "AD"]
    start_col  = 4        # 假设从第 21 列开始是表格特征
    n_fold     = 0         # 不做 K‑Fold
    dropna     = False
    out_csv    = "tabular_embeddings.csv"

    # 1) 运行编码器
    df_emb = tabular_encoder(
        csv_path    = csv_path,
        label_col   = label_col,
        classes     = classes,
        feature_cols= None,      # 使用 start_col
        start_col   = start_col,
        n_fold       = n_fold,
        dropna       = dropna,
        out_csv      = out_csv,
    )

    # 2) 打印前几行和形状
    print("\n--- Embedding Preview ---")
    print(df_emb.head())
    print(f"\nEmbedding shape: {df_emb.shape}")  # (n_samples, embed_dim+1)
