# -*- coding: utf-8 -*-
# NOTE: non-research module for this project (kept for archival reference).
"""
Hull Tactical Market Prediction
淇鐗堬細鏃堕棿搴忓垪瀹夊叏鐗瑰緛宸ョ▼ + 闄嶇淮/褰掍竴鍖?+ 鍘婚珮鐩稿叧 + Stacking/Blending

鏍稿績淇锛?
1) 淇 stock_id_freq 鍏?NaN 鐨勭紪鐮?bug
2) 淇 SimpleImputer 鍦ㄦ煇涓?fold 涓涪鍒楋紝瀵艰嚧 DataFrame shape mismatch 鐨?bug
3) 澧炲姞甯搁噺鍒?鍏ㄧ┖鍒楁竻鐞嗭紝鎻愬崌绋冲畾鎬?
4) 浜屽眰 stacking 浣跨敤鏃堕棿搴忓垪 OOF锛岃€屼笉鏄洿鎺ュ湪鍏?OOF 涓婂洖褰掞紝閬垮厤璇勪及鍋忎箰瑙?

杩愯鍓嶅畨瑁咃細
pip install pandas numpy scikit-learn xgboost lightgbm

杩愯鏂瑰紡锛?
python hull_tactical_stacking_fixed.py

榛樿浼氳嚜鍔ㄥ湪浠ヤ笅浣嶇疆鎵炬暟鎹細
- 褰撳墠鐩綍
- ./data
- ~/Desktop 鎴?~/妗岄潰
- 鑴氭湰鎵€鍦ㄧ洰褰?
- /mnt/data
- /kaggle/input/...锛圞aggle Notebook锛?
"""

import os
import gc
import json
import random
import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

CONFIG = {
    "seed": SEED,
    "n_splits": 5,
    "corr_threshold": 0.985,
    "winsor_low": 0.005,
    "winsor_high": 0.995,
    "pca_explained_var": 0.95,
    "pca_max_components": 12,
    "blend_trials": 300,
    "n_lag_base_cols": 4,
    "verbose": True,
}


def log(msg: str):
    if CONFIG["verbose"]:
        print(msg)


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_file(filename: str) -> Path:
    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    home = Path.home()
    candidates = [
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
        script_dir / filename,
        script_dir / "data" / filename,
        home / "Desktop" / filename,
        home / "妗岄潰" / filename,
        Path("/mnt/data") / filename,
        Path("/kaggle/input") / filename,
    ]

    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        for sub in kaggle_root.iterdir():
            if sub.is_dir():
                candidates.append(sub / filename)

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(f"鎵句笉鍒版枃浠? {filename}")


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def safe_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except Exception:
        return np.nan


def safe_logloss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    try:
        return log_loss(y_true, y_pred)
    except Exception:
        return np.nan


def load_data():
    train_path = find_file("train.csv")
    test_path = find_file("test.csv")
    sub_path = find_file("sample_submission.csv")

    log(f"[INFO] train path: {train_path}")
    log(f"[INFO] test path : {test_path}")
    log(f"[INFO] sample submission path: {sub_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sub_path)

    log(f"[INFO] train shape: {train.shape}")
    log(f"[INFO] test shape : {test.shape}")
    log(f"[INFO] sub shape  : {sample_sub.shape}")
    return train, test, sample_sub


def identify_columns(train: pd.DataFrame, test: pd.DataFrame):
    target_col = "target" if "target" in train.columns else train.columns[-1]

    id_col = None
    for c in ["id", "ID", "row_id"]:
        if c in train.columns and c in test.columns:
            id_col = c
            break

    group_col = None
    for c in ["stock_id", "asset_id", "symbol_id", "ticker_id"]:
        if c in train.columns and c in test.columns:
            group_col = c
            break

    return id_col, target_col, group_col


def sort_by_time(train: pd.DataFrame, test: pd.DataFrame, id_col: str | None):
    if id_col is not None:
        train = train.sort_values(id_col).reset_index(drop=True)
        test = test.sort_values(id_col).reset_index(drop=True)
    return train, test


def encode_group_feature(train: pd.DataFrame, test: pd.DataFrame, group_col: str | None):
    """
    淇鐗堬細
    - 棰戠巼缂栫爜濮嬬粓鍩轰簬鍘熷瀛楃涓插垎缁勫€?
    - 鏁板€肩紪鐮佸拰棰戠巼缂栫爜鍒嗗紑鍋氾紝闃叉 key 绫诲瀷涓嶄竴鑷村鑷村叏 NaN
    """
    if group_col is None:
        return train, test

    train_group_raw = train[group_col].astype(str).copy()
    test_group_raw = test[group_col].astype(str).copy()
    full_group_raw = pd.concat([train_group_raw, test_group_raw], axis=0, ignore_index=True)

    # 棰戠巼缂栫爜锛堝師濮嬫爣绛撅級
    freq_map = full_group_raw.value_counts(dropna=False).to_dict()
    train[f"{group_col}_freq"] = train_group_raw.map(freq_map).astype("float32")
    test[f"{group_col}_freq"] = test_group_raw.map(freq_map).astype("float32")

    # 鏁板€肩紪鐮侊紙鍘熷鏍囩 -> 鏁存暟锛?
    uniq = pd.Index(full_group_raw.unique())
    mapping = {v: i for i, v in enumerate(uniq)}
    train[group_col] = train_group_raw.map(mapping).astype("int32")
    test[group_col] = test_group_raw.map(mapping).astype("int32")

    return train, test


def add_row_statistics(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    exclude_cols = exclude_cols or []
    num_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return df

    X = df[num_cols]
    df["row_mean"] = X.mean(axis=1).astype("float32")
    df["row_std"] = X.std(axis=1).astype("float32")
    df["row_min"] = X.min(axis=1).astype("float32")
    df["row_max"] = X.max(axis=1).astype("float32")
    df["row_median"] = X.median(axis=1).astype("float32")
    df["row_na_count"] = X.isna().sum(axis=1).astype("int16")
    df["row_abs_mean"] = X.abs().mean(axis=1).astype("float32")
    return df


def select_high_variance_cols(train: pd.DataFrame, feature_cols, top_k=4, exclude_cols=None):
    exclude_cols = set(exclude_cols or [])
    numeric_cols = [
        c for c in feature_cols
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(train[c])
    ]
    if not numeric_cols:
        return []

    var_series = train[numeric_cols].astype("float32").var().sort_values(ascending=False)
    return var_series.index[:top_k].tolist()


def add_interaction_features(df: pd.DataFrame, cols):
    cols = list(cols)
    if len(cols) < 2:
        return df

    eps = 1e-6
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            df[f"{c1}_plus_{c2}"] = (df[c1] + df[c2]).astype("float32")
            df[f"{c1}_minus_{c2}"] = (df[c1] - df[c2]).astype("float32")
            df[f"{c1}_mul_{c2}"] = (df[c1] * df[c2]).astype("float32")
            df[f"{c1}_div_{c2}"] = (df[c1] / (df[c2].abs() + eps)).astype("float32")
    return df


def add_group_time_features(df: pd.DataFrame, group_col: str | None, base_cols, id_col: str | None = None):
    if group_col is None or group_col not in df.columns:
        return df

    if id_col is not None and id_col in df.columns:
        df = df.sort_values(id_col).reset_index(drop=True)

    base_cols = [c for c in base_cols if c in df.columns]
    if not base_cols:
        return df

    for c in base_cols:
        shifted_1 = df.groupby(group_col, sort=False)[c].shift(1)
        shifted_2 = df.groupby(group_col, sort=False)[c].shift(2)

        df[f"{c}_lag1"] = shifted_1.astype("float32")
        df[f"{c}_lag2"] = shifted_2.astype("float32")

        roll3_mean = (
            shifted_1.groupby(df[group_col])
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        roll5_mean = (
            shifted_1.groupby(df[group_col])
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        roll5_std = (
            shifted_1.groupby(df[group_col])
            .rolling(window=5, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )

        df[f"{c}_roll3_mean"] = roll3_mean.astype("float32")
        df[f"{c}_roll5_mean"] = roll5_mean.astype("float32")
        df[f"{c}_roll5_std"] = roll5_std.astype("float32")
        df[f"{c}_delta_lag1"] = (df[c] - shifted_1).astype("float32")
        df[f"{c}_delta_roll3"] = (df[c] - roll3_mean).astype("float32")

    return df


def clean_inf_nan(train: pd.DataFrame, test: pd.DataFrame):
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    return train, test


def drop_useless_features(train_x: pd.DataFrame, test_x: pd.DataFrame):
    all_nan_cols = [c for c in train_x.columns if train_x[c].isna().all()]
    nunique = train_x.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()

    drop_cols = sorted(set(all_nan_cols + constant_cols))
    if drop_cols:
        log(f"[INFO] 鍒犻櫎鍏ㄧ┖/甯搁噺鐗瑰緛鏁? {len(drop_cols)}")
        train_x = train_x.drop(columns=drop_cols, errors="ignore")
        test_x = test_x.drop(columns=drop_cols, errors="ignore")
    return train_x, test_x, drop_cols


def drop_highly_correlated_features(train_x: pd.DataFrame, test_x: pd.DataFrame, threshold=0.985):
    corr = train_x.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if (upper[col] > threshold).any()]

    if drop_cols:
        log(f"[INFO] 鍒犻櫎楂樼浉鍏崇壒寰佹暟: {len(drop_cols)}")

    train_x = train_x.drop(columns=drop_cols, errors="ignore")
    test_x = test_x.drop(columns=drop_cols, errors="ignore")
    return train_x, test_x, drop_cols


class FoldPreprocessor:
    """
    瀹夊叏鎶樺唴棰勫鐞嗭細
    - 姘镐笉涓㈠垪
    - 璁粌鎶樹腑浣嶆暟濉厖锛堝叏绌哄垪鍥為€€鍒?0锛?
    - 鍒嗕綅鏁拌鍓?
    - 鏍囧噯鍖?+ PCA 浣滀负琛ュ厖鐗瑰緛
    """

    def __init__(self, low_q=0.005, high_q=0.995, pca_explained_var=0.95, pca_max_components=12):
        self.low_q = low_q
        self.high_q = high_q
        self.pca_explained_var = pca_explained_var
        self.pca_max_components = pca_max_components

        self.feature_cols = None
        self.fill_values = None
        self.lower_bounds = None
        self.upper_bounds = None
        self.scaler = None
        self.pca = None
        self.n_components_ = 0

    def fit(self, X: pd.DataFrame):
        self.feature_cols = list(X.columns)
        X = X[self.feature_cols].copy()

        self.fill_values = X.median(axis=0, numeric_only=True)
        self.fill_values = self.fill_values.reindex(self.feature_cols)
        self.fill_values = self.fill_values.fillna(0.0).astype("float32")

        X_imp = X.fillna(self.fill_values)
        self.lower_bounds = X_imp.quantile(self.low_q).astype("float32")
        self.upper_bounds = X_imp.quantile(self.high_q).astype("float32")
        X_clip = X_imp.clip(self.lower_bounds, self.upper_bounds, axis=1).astype("float32")

        max_comp = min(self.pca_max_components, X_clip.shape[1])
        if max_comp >= 2:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_clip)

            pca_full = PCA(n_components=max_comp, random_state=SEED)
            pca_full.fit(X_scaled)
            cum_var = np.cumsum(pca_full.explained_variance_ratio_)
            n_comp = int(np.searchsorted(cum_var, self.pca_explained_var) + 1)
            n_comp = max(2, min(n_comp, max_comp))

            self.n_components_ = n_comp
            self.pca = PCA(n_components=n_comp, random_state=SEED)
            self.pca.fit(X_scaled)
        else:
            self.scaler = None
            self.pca = None
            self.n_components_ = 0

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.feature_cols].copy()
        X_imp = X.fillna(self.fill_values)
        X_clip = X_imp.clip(self.lower_bounds, self.upper_bounds, axis=1).astype("float32")

        if self.pca is None or self.n_components_ == 0:
            return X_clip

        X_scaled = self.scaler.transform(X_clip)
        X_pca = self.pca.transform(X_scaled)
        X_pca = pd.DataFrame(
            X_pca,
            columns=[f"pca_{i + 1}" for i in range(self.n_components_)],
            index=X.index,
        ).astype("float32")

        X_out = pd.concat([X_clip, X_pca], axis=1)
        return X_out


def get_models(seed=SEED):
    models = OrderedDict()

    if xgb is not None:
        models["xgb"] = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.80,
            min_child_weight=3,
            reg_alpha=0.10,
            reg_lambda=1.20,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        log("[WARN] xgboost 鏈畨瑁咃紝璺宠繃 XGBClassifier")

    if lgb is not None:
        models["lgb"] = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.80,
            min_child_samples=50,
            reg_alpha=0.10,
            reg_lambda=1.00,
            objective="binary",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        log("[WARN] lightgbm 鏈畨瑁咃紝璺宠繃 LGBMClassifier")

    models["rf"] = RandomForestClassifier(
        n_estimators=320,
        max_depth=14,
        min_samples_leaf=18,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )

    models["gbdt"] = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_depth=8,
        max_iter=320,
        min_samples_leaf=40,
        l2_regularization=0.10,
        random_state=seed,
    )

    if len(models) < 2:
        raise RuntimeError("鍙敤妯″瀷杩囧皯锛岃鑷冲皯瀹夎 lightgbm / xgboost 涓殑涓€涓?)

    return models


def fit_predict_proba(model, X_train, y_train, X_valid, X_test):
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        valid_pred = model.predict_proba(X_valid)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]
    else:
        valid_pred = model.predict(X_valid)
        test_pred = model.predict(X_test)

    return valid_pred, test_pred


def optimize_blend_weights(oof_pred_dict, y_true, n_trials=300, seed=SEED):
    names = list(oof_pred_dict.keys())
    pred_matrix = np.column_stack([oof_pred_dict[n] for n in names])

    best_w = np.ones(len(names), dtype=float) / len(names)
    best_pred = pred_matrix @ best_w
    best_auc = safe_auc(y_true, best_pred)

    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        w = rng.dirichlet(np.ones(len(names)))
        pred = pred_matrix @ w
        auc = safe_auc(y_true, pred)
        if auc > best_auc:
            best_auc = auc
            best_w = w
            best_pred = pred

    return names, best_w, best_pred, best_auc


def build_meta_oof(stack_train: pd.DataFrame, y: np.ndarray, stack_test: pd.DataFrame, n_splits=5, seed=SEED):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    meta_oof = np.zeros(len(stack_train), dtype=np.float32)
    meta_test = np.zeros(len(stack_test), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(stack_train), start=1):
        meta_model = LogisticRegression(C=0.5, max_iter=3000, random_state=seed)
        meta_model.fit(stack_train.iloc[tr_idx], y[tr_idx])
        meta_oof[va_idx] = meta_model.predict_proba(stack_train.iloc[va_idx])[:, 1].astype(np.float32)
        meta_test += meta_model.predict_proba(stack_test)[:, 1].astype(np.float32) / n_splits
        log(f"[META FOLD {fold}] done")

    return meta_oof, meta_test


def prepare_features(train: pd.DataFrame, test: pd.DataFrame, id_col: str | None, target_col: str, group_col: str | None):
    train, test = encode_group_feature(train, test, group_col)

    base_feature_cols = [c for c in train.columns if c != target_col]
    if id_col is not None and id_col in base_feature_cols:
        base_feature_cols.remove(id_col)

    exclude_for_var = {group_col, f"{group_col}_freq"} if group_col is not None else set()
    top_var_cols = select_high_variance_cols(
        train,
        base_feature_cols,
        top_k=CONFIG["n_lag_base_cols"],
        exclude_cols=exclude_for_var,
    )
    log(f"[INFO] 鐢ㄤ簬浜ゅ弶/lag 鐨勯珮鏂瑰樊杩炵画鍒? {top_var_cols}")

    train["__is_train__"] = 1
    test["__is_train__"] = 0
    full = pd.concat([train, test], axis=0, ignore_index=True)

    full = add_row_statistics(full, exclude_cols=[target_col, "__is_train__"])
    full = add_interaction_features(full, top_var_cols)
    full = add_group_time_features(full, group_col=group_col, base_cols=top_var_cols, id_col=id_col)

    train_fe = full[full["__is_train__"] == 1].drop(columns=["__is_train__"]).reset_index(drop=True)
    test_fe = full[full["__is_train__"] == 0].drop(columns=["__is_train__"]).reset_index(drop=True)

    del full
    gc.collect()

    train_fe, test_fe = clean_inf_nan(train_fe, test_fe)

    feature_cols = [c for c in train_fe.columns if c != target_col]
    if id_col is not None and id_col in feature_cols:
        feature_cols.remove(id_col)

    train_x = train_fe[feature_cols].copy()
    test_x = test_fe[feature_cols].copy()

    train_x, test_x, dropped_useless_cols = drop_useless_features(train_x, test_x)
    train_x, test_x, dropped_corr_cols = drop_highly_correlated_features(
        train_x, test_x, threshold=CONFIG["corr_threshold"]
    )

    y = train_fe[target_col].astype("int8").values

    train_x = reduce_memory(train_x)
    test_x = reduce_memory(test_x)

    return train_fe, test_fe, train_x, test_x, y, dropped_useless_cols, dropped_corr_cols


def main():
    seed_everything(CONFIG["seed"])

    train, test, sample_sub = load_data()
    id_col, target_col, group_col = identify_columns(train, test)

    log(f"[INFO] id_col     = {id_col}")
    log(f"[INFO] target_col = {target_col}")
    log(f"[INFO] group_col  = {group_col}")

    train, test = sort_by_time(train, test, id_col=id_col)

    if target_col in train.columns:
        target_dist = train[target_col].value_counts(normalize=True).sort_index().to_dict()
        log(f"[INFO] target distribution: {target_dist}")

    train_fe, test_fe, train_x, test_x, y, dropped_useless_cols, dropped_corr_cols = prepare_features(
        train=train,
        test=test,
        id_col=id_col,
        target_col=target_col,
        group_col=group_col,
    )

    feature_cols = train_x.columns.tolist()
    log(f"[INFO] 鏈€缁堣缁冪壒寰佹暟: {len(feature_cols)}")

    tscv = TimeSeriesSplit(n_splits=CONFIG["n_splits"])
    models = get_models(seed=CONFIG["seed"])

    oof_pred = {name: np.zeros(len(train_x), dtype=np.float32) for name in models.keys()}
    test_pred = {name: np.zeros(len(test_x), dtype=np.float32) for name in models.keys()}
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train_x), start=1):
        log("=" * 88)
        log(f"[FOLD {fold}] train_size={len(tr_idx)} | valid_size={len(va_idx)}")

        X_tr_raw = train_x.iloc[tr_idx].copy()
        X_va_raw = train_x.iloc[va_idx].copy()
        X_te_raw = test_x.copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        preprocessor = FoldPreprocessor(
            low_q=CONFIG["winsor_low"],
            high_q=CONFIG["winsor_high"],
            pca_explained_var=CONFIG["pca_explained_var"],
            pca_max_components=CONFIG["pca_max_components"],
        )
        preprocessor.fit(X_tr_raw)
        X_tr = preprocessor.transform(X_tr_raw)
        X_va = preprocessor.transform(X_va_raw)
        X_te = preprocessor.transform(X_te_raw)

        log(f"[FOLD {fold}] 棰勫鐞嗗悗鐗瑰緛鏁? {X_tr.shape[1]}")

        fold_result = {}
        for model_name, model in models.items():
            log(f"[FOLD {fold}] Training {model_name} ...")
            valid_pred, test_fold_pred = fit_predict_proba(
                model=model,
                X_train=X_tr,
                y_train=y_tr,
                X_valid=X_va,
                X_test=X_te,
            )

            oof_pred[model_name][va_idx] = valid_pred.astype(np.float32)
            test_pred[model_name] += test_fold_pred.astype(np.float32) / CONFIG["n_splits"]

            fold_auc = safe_auc(y_va, valid_pred)
            fold_ll = safe_logloss(y_va, valid_pred)
            fold_result[model_name] = {
                "auc": float(fold_auc) if not np.isnan(fold_auc) else None,
                "logloss": float(fold_ll) if not np.isnan(fold_ll) else None,
            }
            log(f"[FOLD {fold}] {model_name:<6} -> AUC={fold_auc:.6f}, LogLoss={fold_ll:.6f}")
            gc.collect()

        fold_scores.append(fold_result)
        va_blend = np.mean(np.column_stack([oof_pred[m][va_idx] for m in models.keys()]), axis=1)
        blend_auc = safe_auc(y_va, va_blend)
        blend_ll = safe_logloss(y_va, va_blend)
        log(f"[FOLD {fold}] mean_blend -> AUC={blend_auc:.6f}, LogLoss={blend_ll:.6f}")

        del X_tr_raw, X_va_raw, X_te_raw, X_tr, X_va, X_te
        gc.collect()

    log("=" * 88)
    log("[INFO] Base OOF metrics")
    for model_name in models.keys():
        auc_all = safe_auc(y, oof_pred[model_name])
        ll_all = safe_logloss(y, oof_pred[model_name])
        log(f"{model_name:<8} OOF AUC={auc_all:.6f}, OOF LogLoss={ll_all:.6f}")

    blend_names, blend_weights, blend_oof, blend_auc = optimize_blend_weights(
        oof_pred_dict=oof_pred,
        y_true=y,
        n_trials=CONFIG["blend_trials"],
        seed=CONFIG["seed"],
    )
    blend_test = np.zeros(len(test_x), dtype=np.float32)
    for n, w in zip(blend_names, blend_weights):
        blend_test += (test_pred[n] * w).astype(np.float32)
    blend_ll = safe_logloss(y, blend_oof)

    log("-" * 88)
    log("[INFO] Weighted Blend")
    log(f"models  : {blend_names}")
    log(f"weights : {np.round(blend_weights, 6).tolist()}")
    log(f"OOF AUC : {blend_auc:.6f}")
    log(f"OOF LL  : {blend_ll:.6f}")

    stack_train = pd.DataFrame({f"pred_{k}": v for k, v in oof_pred.items()})
    stack_test = pd.DataFrame({f"pred_{k}": v for k, v in test_pred.items()})
    stack_train["pred_weighted_blend"] = blend_oof
    stack_test["pred_weighted_blend"] = blend_test

    stack_oof, stack_test_pred = build_meta_oof(
        stack_train=stack_train,
        y=y,
        stack_test=stack_test,
        n_splits=CONFIG["n_splits"],
        seed=CONFIG["seed"],
    )
    stack_auc = safe_auc(y, stack_oof)
    stack_ll = safe_logloss(y, stack_oof)

    log("-" * 88)
    log("[INFO] Stacking Meta Model")
    log(f"OOF AUC : {stack_auc:.6f}")
    log(f"OOF LL  : {stack_ll:.6f}")

    hybrid_sources = {"weighted_blend": blend_oof, "stacking": stack_oof}
    hybrid_names, hybrid_weights, hybrid_oof, hybrid_auc = optimize_blend_weights(
        oof_pred_dict=hybrid_sources,
        y_true=y,
        n_trials=120,
        seed=CONFIG["seed"],
    )

    test_source_map = {"weighted_blend": blend_test, "stacking": stack_test_pred}
    hybrid_test = np.zeros(len(test_x), dtype=np.float32)
    for n, w in zip(hybrid_names, hybrid_weights):
        hybrid_test += (test_source_map[n] * w).astype(np.float32)
    hybrid_ll = safe_logloss(y, hybrid_oof)

    log("-" * 88)
    log("[INFO] Hybrid Final")
    log(f"components: {hybrid_names}")
    log(f"weights   : {np.round(hybrid_weights, 6).tolist()}")
    log(f"OOF AUC   : {hybrid_auc:.6f}")
    log(f"OOF LL    : {hybrid_ll:.6f}")

    sub = sample_sub.copy()
    pred_cols = [c for c in sub.columns if c != id_col]
    if not pred_cols:
        raise ValueError("sample_submission 涓湭鎵惧埌棰勬祴鍒?)
    submit_pred_col = pred_cols[0]

    sub_blend = sub.copy()
    sub_stack = sub.copy()
    sub_hybrid = sub.copy()

    sub_blend[submit_pred_col] = np.clip(blend_test, 1e-6, 1 - 1e-6)
    sub_stack[submit_pred_col] = np.clip(stack_test_pred, 1e-6, 1 - 1e-6)
    sub_hybrid[submit_pred_col] = np.clip(hybrid_test, 1e-6, 1 - 1e-6)

    sub_blend.to_csv("submission_weighted_blend.csv", index=False)
    sub_stack.to_csv("submission_stacking.csv", index=False)
    sub_hybrid.to_csv("submission_hybrid_final.csv", index=False)

    run_info = {
        "id_col": id_col,
        "target_col": target_col,
        "group_col": group_col,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_final_features_before_pca": int(len(feature_cols)),
        "dropped_useless_cols_count": int(len(dropped_useless_cols)),
        "dropped_useless_cols": dropped_useless_cols,
        "dropped_corr_cols_count": int(len(dropped_corr_cols)),
        "dropped_corr_cols": dropped_corr_cols[:100],
        "model_names": list(models.keys()),
        "weighted_blend_names": blend_names,
        "weighted_blend_weights": np.round(blend_weights, 8).tolist(),
        "hybrid_names": hybrid_names,
        "hybrid_weights": np.round(hybrid_weights, 8).tolist(),
        "oof_auc": {
            **{k: float(safe_auc(y, oof_pred[k])) for k in models.keys()},
            "weighted_blend": float(blend_auc),
            "stacking": float(stack_auc),
            "hybrid_final": float(hybrid_auc),
        },
        "oof_logloss": {
            **{k: float(safe_logloss(y, oof_pred[k])) for k in models.keys()},
            "weighted_blend": float(blend_ll),
            "stacking": float(stack_ll),
            "hybrid_final": float(hybrid_ll),
        },
        "config": CONFIG,
    }

    with open("run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    log("=" * 88)
    log("[DONE] 宸茬敓鎴愶細")
    log("  - submission_weighted_blend.csv")
    log("  - submission_stacking.csv")
    log("  - submission_hybrid_final.csv")
    log("  - run_info.json")
    log("=" * 88)


if __name__ == "__main__":
    main()

