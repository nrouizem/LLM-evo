import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import warnings, json
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.metrics import root_mean_squared_log_error as rmsle

from utils.run_llm_code import run_llm_code
from core.evolve import evolve


# =======================
# == UTILITY FUNCTIONS ==
# =======================

def make_splits(n, folds=5, seed=1):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
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
            random_state=1,
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

def full_obj(code, df: pd.DataFrame, y_log: pd.Series):
    try:
        X = df.copy()
        code_output = run_llm_code(code, X)
        res = xgb_gain(code_output, X, y_log)
        if not np.isfinite(res):
            raise RuntimeError("Score is NaN")
        return res
    except Exception as e:
        print(e)
        return 0.0


if __name__ == "__main__":

    # Load data; minor transform
    df = pd.read_csv("examples/calories/train.csv").drop(columns="id")

    label_enc = LabelEncoder()
    df['Sex'] = label_enc.fit_transform(df['Sex'])
    df = df.astype("float64")
    df['Sex'] = df['Sex'].astype("int8")

    y  = df.pop("Calories").values

    # Compute baseline score
    df0 = df.copy()
    y_log = pd.Series(np.log1p(y))
    splits = make_splits(len(df0), folds=5, seed=1)
    print("Computing baseline...")
    base_oof = xgb_oof(df0, y_log, splits)
    base_score = rmsle(np.expm1(y_log), np.expm1(base_oof)) * 1000

    # define query
    query = """You are mutating a Python function that returns a new feature for a tabular dataset.
            The available columns are: ["Age", "Weight", "Height", "Duration", "Heart_Rate", "Body_Temp", "Sex"].
            "Sex" and "Age" are ints; all other columns are floats. "Sex" is 1 for male; 0 for female.
            The ultimate objective is to predict the "Calories" target; you may therefore not use the target directly.
            An XGBoost model will be used to evaluate your feature; the evaluation will compare the RMSLE of the original dataframe on its own compared to the dataframe plus your feature.
            The data has length 750,000. Be aware of this as you manipulate the data; don't build a (750000, 750000) array, for example.
            You may use only the columns I said are available to you. You may perform any operation, including numpy or pandas operations, on one or more of the provided columns to construct the new feature.
            When using numpy or pandas operations, ensure compatibility with the latest package versions.
            You will be provided with an existing function. Return a new function without any extraneous text. The function should return a single feature."""
    
    seeds = []
    for i, col in enumerate(["Age", "Weight", "Height", "Duration", "Heart_Rate", "Body_Temp", "Sex"][:2]):
        seeds.append(f"def build_feature(df):\n  return df['{col}']")

    # run evolution
    results = evolve(query, full_obj, seeds, df, y_log, K=8, C=2, GENS=10)
    with open("examples/calories/results.json", 'w') as f:
        json.dump(results, f)