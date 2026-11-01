"""
Train the three plug-in models on 2012-2014 loans and predict on the 2015 pool.

  p_hat(x)  : P(default | x)                    LGBMClassifier, all train rows
  c1_hat(x) : E[c1 | x, Y=1] = ratio * loan_amnt  LGBMRegressor on defaults only
  c0_hat(x) : E[c0 | x, Y=0] = ratio * loan_amnt  LGBMRegressor on good loans only

Outputs (data/processed/):
  pool_preds.parquet  - 2015 pool, row order preserved (split_pool indexes into it)
  val_preds.parquet   - validation set, for calibration diagnostics

Usage (repo root):
    python src/train_models.py
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import calibration_curve
from splits import load_splits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

# ---- Model inputs: 29 features known at origination ----
FEATURES = [
    # Loan terms
    "loan_amnt", "int_rate", "installment", "grade", "sub_grade", "purpose",
    # Application
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "addr_state", "dti",
    # Basic credit history
    "fico_range_low", "credit_age_months", "delinq_2yrs", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "pub_rec_bankruptcies", "revol_bal", "revol_util", "total_acc", "mort_acc",
    "collections_12_mths_ex_med", "acc_now_delinq", "tax_liens",
    "mths_since_last_major_derog",
]

CAT_COLS = ["grade", "sub_grade", "purpose", "home_ownership",
            "verification_status", "addr_state"]

# ---- Must never be model inputs (label, outcomes, identifiers) ----
LEAK_COLS = [
    "y", "loan_status",
    "total_pymnt", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
    "c0_true", "c1_true", "cost_true",
    "id", "issue_d",
]

SAVE_COLS = ["id", "issue_d", "y", "loan_amnt", "c0_true", "c1_true", "cost_true"]

LGB_PARAMS = dict(n_estimators=2000, learning_rate=0.05, num_leaves=31,
                  min_child_samples=100, random_state=0, verbose=-1)
EARLY_STOP = 100


def prepare_X(df, categories):
    """Return df[FEATURES] with CAT_COLS cast to category using TRAIN categories."""
    X = df[FEATURES].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X


def fit_classifier(X_tr, y_tr, X_va, y_va):
    """LGBMClassifier + early stopping on validation logloss."""
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return model


def fit_ratio_regressor(X_tr, r_tr, X_va, r_va):
    """LGBMRegressor (L2 -> conditional mean) + early stopping on validation L2."""
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, r_tr, eval_set=[(X_va, r_va)], eval_metric="l2",
              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return model


def add_predictions(df, X, clf, reg1, reg0):
    """SAVE_COLS + p_hat, c1_hat, c0_hat (ratios clipped, then times loan_amnt)."""
    out = df[SAVE_COLS].copy()
    out["p_hat"] = clf.predict_proba(X)[:, 1]
    r1 = np.clip(reg1.predict(X), 0.0, 1.0)      # loss rate given default, in [0, 1]
    r0 = np.clip(reg0.predict(X), 0.0, None)     # interest rate earned, >= 0
    out["c1_hat"] = r1 * df["loan_amnt"].to_numpy()
    out["c0_hat"] = r0 * df["loan_amnt"].to_numpy()
    return out


if __name__ == "__main__":
    df = pd.read_parquet(DATA / "lc36_2012_2015.parquet")
    train_df, val_df, pool_df = load_splits(df)

    # Leakage guard
    assert not set(FEATURES) & set(LEAK_COLS), set(FEATURES) & set(LEAK_COLS)

    # Category lists come from TRAIN only, reused for val and pool
    categories = {c: sorted(train_df[c].unique()) for c in CAT_COLS}
    X_tr = prepare_X(train_df, categories)
    X_va = prepare_X(val_df, categories)
    X_pool = prepare_X(pool_df, categories)
    y_tr, y_va = train_df["y"].to_numpy(), val_df["y"].to_numpy()

    # Model 1: p_hat on all training rows
    clf = fit_classifier(X_tr, y_tr, X_va, y_va)

    # Model 2: c1 ratio on defaults only
    tr1, va1 = y_tr == 1, y_va == 1
    reg1 = fit_ratio_regressor(
        X_tr[tr1], (train_df["c1_true"] / train_df["loan_amnt"])[tr1],
        X_va[va1], (val_df["c1_true"] / val_df["loan_amnt"])[va1])

    # Model 3: c0 ratio on good loans only
    tr0, va0 = y_tr == 0, y_va == 0
    reg0 = fit_ratio_regressor(
        X_tr[tr0], (train_df["c0_true"] / train_df["loan_amnt"])[tr0],
        X_va[va0], (val_df["c0_true"] / val_df["loan_amnt"])[va0])

    # Predict and save (pool row order preserved)
    val_preds = add_predictions(val_df, X_va, clf, reg1, reg0)
    pool_preds = add_predictions(pool_df, X_pool, clf, reg1, reg0)
    val_preds.to_parquet(DATA / "val_preds.parquet", index=False)
    pool_preds.to_parquet(DATA / "pool_preds.parquet", index=False)

    # ---------- Checks ----------
    print("Trees chosen by early stopping")
    for name, m in [("p_hat", clf), ("c1 ratio", reg1), ("c0 ratio", reg0)]:
        print(f"  {name:9s} {m.best_iteration_}  (cap {LGB_PARAMS['n_estimators']})")

    print("\nClassifier: AUC / logloss / mean p_hat vs actual default rate")
    for name, d in [("val", val_preds), ("pool", pool_preds)]:
        print(f"  {name:4s} AUC {roc_auc_score(d['y'], d['p_hat']):.4f}  "
              f"logloss {log_loss(d['y'], d['p_hat']):.4f}  "
              f"mean p_hat {d['p_hat'].mean():.4f}  actual {d['y'].mean():.4f}")

    print("\nCalibration curve (10 quantile bins): mean p_hat -> actual rate")
    for name, d in [("val", val_preds), ("pool", pool_preds)]:
        frac_pos, mean_pred = calibration_curve(d["y"], d["p_hat"],
                                                n_bins=10, strategy="quantile")
        print(f"  {name}")
        for mp, fp in zip(mean_pred, frac_pos):
            print(f"    {mp:.3f} -> {fp:.3f}")

    print("\nCost models on pool: mean c_hat vs mean c_true (by true label), correlation")
    for lab, hat, true in [(1, "c1_hat", "c1_true"), (0, "c0_hat", "c0_true")]:
        d = pool_preds[pool_preds["y"] == lab]
        corr = np.corrcoef(d[hat], d[true])[0, 1]
        print(f"  y={lab}: mean {hat} {d[hat].mean():,.0f}  mean {true} {d[true].mean():,.0f}  "
              f"corr {corr:.3f}")

    print("\nNo negative cost predictions")
    print(f"  min c1_hat {pool_preds['c1_hat'].min():.2f}   min c0_hat {pool_preds['c0_hat'].min():.2f}")

    print(f"\nSaved: {DATA / 'pool_preds.parquet'}  ({len(pool_preds):,} rows)")
    print(f"Saved: {DATA / 'val_preds.parquet'}  ({len(val_preds):,} rows)")