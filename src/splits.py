import pandas as pd
import numpy as np
from pathlib import Path

def load_splits(df):
    train_issue_d = pd.Timestamp("2015-01-01")

    train_df = df[df["issue_d"] < train_issue_d]
    pool_df = df[df["issue_d"] >= train_issue_d].reset_index(drop=True)

    val_df = train_df.sample(frac=0.1, random_state=42)
    train_df = train_df.drop(val_df.index)

    return train_df, val_df, pool_df

def split_pool(n_pool, seed, cal_frac=0.5):
    rng = np.random.default_rng(seed)
    permu_pool = rng.permutation(n_pool)
    split_idx = int(n_pool * cal_frac)
    cal_idx = permu_pool[:split_idx]
    test_idx = permu_pool[split_idx:]

    return cal_idx, test_idx

if __name__ == "__main__":
    data_path = (Path(__file__).resolve().parents[1]
                    / "data" / "processed" / "lc36_2012_2015.parquet")
    df = pd.read_parquet(data_path)
    
    train_df, val_df, pool_df = load_splits(df)
    cal_idx, test_idx = split_pool(len(pool_df), seed=0)
    cal_df, test_df = pool_df.iloc[cal_idx], pool_df.iloc[test_idx]
 
    sets = {"train": train_df, "val": val_df, "pool": pool_df,
            "cal": cal_df, "test": test_df}
 
    # Check size, default rate, issue date range
    print("size / default rate / issue_d range")
    for name, d in sets.items():
        print(f"  {name:5s} n={len(d):>7,}  default={d['y'].mean():.4f}  "
              f"issue_d {d['issue_d'].min():%Y-%m} ~ {d['issue_d'].max():%Y-%m}")
    assert len(train_df) + len(val_df) + len(pool_df) == len(df)
    assert len(cal_idx) + len(test_idx) == len(pool_df)
 
    # Check no id overlap between train / val / cal / test
    print("\nid overlap (all should be 0)")
    names = ["train", "val", "cal", "test"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            n_overlap = len(set(sets[a]["id"]) & set(sets[b]["id"]))
            print(f"  {a} & {b}: {n_overlap}")
            assert n_overlap == 0
 
    # Check reproducibility
    c2, t2 = split_pool(len(pool_df), seed=0)
    c3, _ = split_pool(len(pool_df), seed=1)
    print("\nreproducibility")
    print(f"  same seed -> identical: {np.array_equal(cal_idx, c2) and np.array_equal(test_idx, t2)}")
    print(f"  different seed -> different: {not np.array_equal(cal_idx, c3)}")
 
    # Check cal vs test default-rate gap across a few seeds
    print("\ncal - test default-rate gap, seeds 0-4")
    for s in range(5):
        c, t = split_pool(len(pool_df), seed=s)
        gap = pool_df["y"].iloc[c].mean() - pool_df["y"].iloc[t].mean()
        print(f"  seed {s}: {gap:+.4f}")
