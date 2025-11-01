import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import warnings, json, os
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.metrics import root_mean_squared_log_error as rmsle

from utils.run_llm_code import run_llm_code
from core.evolve import evolve


# =======================
# == UTILITY FUNCTIONS ==
# =======================

def make_splits(n, folds=5):
    kf = KFold(n_splits=folds, shuffle=True)
    return list(kf.split(np.arange(n)))

def xgb_oof(X, y, splits, gpu=True):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    oof = np.zeros(len(y))
    for tr, va in splits:
        model = XGBRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=8,
            objective='reg:squarederror',
            reg_lambda=3.0,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            device='cuda' if gpu else 'cpu',
            eval_metric='rmse',
            early_stopping_rounds=50,
        )
        model.fit(
            X.iloc[tr],
            y.iloc[tr],
            eval_set=[(X.iloc[va], y.iloc[va])],
            verbose=False
        )
        oof[va] = model.predict(X.iloc[va])
    return oof


# =======================
# ====== OBJECTIVE ======
# =======================

def xgb_gain(feature, df: pd.DataFrame, y_log: pd.Series):
    X = df.copy()
    X['feat0'] = feature
    oof = xgb_oof(X, y_log, splits)
    score = rmsle(np.expm1(y_log), np.expm1(oof)) * 1000
    delta = base_score - score
    return delta

def full_obj(code, df: pd.DataFrame, y_log: pd.Series, jsonl_path="examples/feat_eval/calories_metrics.jsonl"):
    try:
        X = df.copy()
        feat = run_llm_code(code, X)
        res = xgb_gain(feat, X, y_log)

        if not np.isfinite(res):
            raise RuntimeError("Score is NaN")
        
        s = pd.Series(feat)
        ag = dict(
            var=float(np.var(s)),
            missing_rate=float(s.isna().mean()) if isinstance(s, pd.Series) else 0.0,
            pearson_r=float(np.corrcoef(y_log, s)[0,1]) if np.isfinite(s).all() else 0.0,
            spearman_r=float(pd.Series(y_log).corr(pd.Series(s), method='spearman')),
            max_abs_corr_with_existing=float(np.max(np.abs(np.corrcoef(X.values.T, s.values)[0:-1,-1]))),
            skew=float(pd.Series(s).skew()),
            kurtosis=float(pd.Series(s).kurt())
        )

        record = {
            "code": code,
            "dtype": str(s.dtype) if isinstance(s, pd.Series) else "unknown",
            "metrics": {
                "xgb": {
                    "delta_rmsle": float(res),
                },
                "agnostic": ag,
                "sanity": {
                    "nan_rate": float(np.mean(~np.isfinite(np.asarray(s))))
                }
            }
        }

        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return float(res)
    except Exception as e:
        print(e)
        return 0.0


if __name__ == "__main__":

    # smaller size for faster eval
    size = 100_000

    # Load data; minor transform
    df = pd.read_csv("examples/feat_eval/data/calories_train.csv").drop(columns="id")[:size]

    label_enc = LabelEncoder()
    df['Sex'] = label_enc.fit_transform(df['Sex'])
    df = df.astype("float64")
    df['Sex'] = df['Sex'].astype("int8")

    y  = df.pop("Calories").values

    # Compute baseline score
    df0 = df.copy()
    y_log = pd.Series(np.log1p(y))
    splits = make_splits(len(df0), folds=5)
    print("Computing baseline...")
    base_oof = xgb_oof(df0, y_log, splits)
    base_score = rmsle(np.expm1(y_log), np.expm1(base_oof)) * 1000

    # define query
    query = f"""You are mutating a Python function that returns a new feature for a tabular dataset.
            The available columns are: ["Age", "Weight", "Height", "Duration", "Heart_Rate", "Body_Temp", "Sex"].
            "Sex" and "Age" are ints; all other columns are floats. "Sex" is 1 for male; 0 for female.
            The ultimate objective is to predict the "Calories" target; you may therefore not use the target directly.
            An XGBoost model will be used to evaluate your feature; the evaluation will compare the RMSLE of the original dataframe on its own compared to the dataframe plus your feature.
            The data has length {size}. Be aware of this as you manipulate the data; consider the memory requirement of any new arrays you create, such as a ({size}, {size}) array, for example.
            You may use only the columns I said are available to you. You may perform any operation, including numpy or pandas operations, on one or more of the provided columns to construct the new feature.
            When using numpy or pandas operations, ensure compatibility with the latest package versions.
            You will be provided with an existing function. Return a new function without any extraneous text. The function should return a single feature."""
    
    seeds = []
    for i, col in enumerate(df.columns):
        seeds.append(f"def build_feature(df):\n  return df['{col}']")

    # run evolution
    results = evolve(
        query,
        full_obj,
        seeds,
        df,
        y_log,
        K=10,
        C=4,
        GENS=30,
        log_path="examples/feat_eval/calories_log.jsonl"
    )