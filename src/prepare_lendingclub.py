"""
Lending Club preprocessing (Decisions 1-7)

  Decision 1: 36-month loans only
  Decision 2: terminated loans only (Fully Paid -> y=0, Charged Off/Default -> y=1)
  Decision 3: issued on or before 2015-12 (all 36-month loans matured by the
              2018Q4 snapshot, so no selection bias from unfinished loans)
  Decision 6: issued on or after 2012-01 (drops crisis years and the period
              where mort_acc / mths_since_last_major_derog were not collected)
  Decision 4: model inputs = 29 origination-time features; outcome columns
              kept separately for labels / splits / realized dollar loss
  Decision 5: extended bureau features (added ~2012) are NOT used
  Decision 7: realized cost of the wrong action, clipped at 0
              y=1 (default approved):  c1 = max(0, loan_amnt - (total_pymnt - collection_recovery_fee))
              y=0 (good loan declined): c0 = total_rec_int

Usage (repo root):
    python src/prepare_lendingclub.py --raw data/raw/accepted_2007_to_2018Q4.csv
"""
import argparse
import numpy as np
import pandas as pd

# ---- Model inputs: 29 features known at origination ----
FEATURES = [
    # Loan terms
    "loan_amnt", "int_rate", "installment", "grade", "sub_grade", "purpose",
    # Application
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "addr_state", "dti",
    # Basic credit history
    "fico_range_low", "earliest_cr_line", "delinq_2yrs", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "pub_rec_bankruptcies", "revol_bal", "revol_util", "total_acc", "mort_acc",
    "collections_12_mths_ex_med", "acc_now_delinq", "tax_liens",
    "mths_since_last_major_derog",
]
# ---- Kept for labels / splits / realized loss (NOT model inputs) ----
KEEP = [
    "id", "issue_d", "term", "loan_status",
    "total_pymnt", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
]

GOOD = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
BAD = {"Charged Off", "Default",
       "Does not meet the credit policy. Status:Charged Off"}
FIRST_ISSUE = pd.Timestamp("2012-01-01")
LAST_ISSUE = pd.Timestamp("2015-12-01")


def main(raw, out):
    chunks = []
    for ch in pd.read_csv(raw, usecols=FEATURES + KEEP,
                          chunksize=200_000, low_memory=False):
        ch = ch[ch["term"].astype(str).str.strip() == "36 months"]      # Decision 1
        ch = ch[ch["loan_status"].isin(GOOD | BAD)]                      # Decision 2
        ch["issue_d"] = pd.to_datetime(ch["issue_d"], format="%b-%Y")
        ch = ch[ch["issue_d"] <= LAST_ISSUE]                             # Decision 3
        ch = ch[ch["issue_d"] >= FIRST_ISSUE]                            # Decision 6
        chunks.append(ch)
    df = pd.concat(chunks, ignore_index=True)

    # Label
    df["y"] = df["loan_status"].isin(BAD).astype(int)

    # Type conversions
    for c in ["int_rate", "revol_util"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.rstrip("%"), errors="coerce")
    ecl = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_age_months"] = ((df["issue_d"] - ecl).dt.days / 30.44).round()
    df["emp_length"] = (df["emp_length"].astype(str)
                        .str.replace("< 1 year", "0").str.extract(r"(\d+)")[0]
                        .astype(float))
    df = df.drop(columns=["earliest_cr_line", "term"])

    # Realized cost of the wrong action for each loan's true label (Decision 7)
    loss_if_approved = df["loan_amnt"] - (df["total_pymnt"] - df["collection_recovery_fee"])
    df["c1_true"] = np.where(df["y"] == 1, loss_if_approved.clip(lower=0), np.nan)
    df["c0_true"] = np.where(df["y"] == 0, df["total_rec_int"], np.nan)
    df["cost_true"] = df["c1_true"].fillna(df["c0_true"])

    # Store text columns as plain strings (mixed types broke parquet writing)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str)

    # ---------- Diagnostics ----------
    print(f"\n[1] Rows: {len(df):,}   Default rate: {df['y'].mean():.4f}")
    print(f"    Issue months: {df['issue_d'].min():%Y-%m} ~ {df['issue_d'].max():%Y-%m}")

    print("\n[2] Count and default rate by issue year")
    print(df.groupby(df["issue_d"].dt.year)["y"].agg(n="count", default_rate="mean")
          .round(4).to_string())

    print("\n[3] Missing rate of input features by issue year (only features > 5% in some year)")
    feats = [c for c in FEATURES if c != "earliest_cr_line"] + ["credit_age_months"]
    miss = df.groupby(df["issue_d"].dt.year)[feats].apply(lambda g: g.isna().mean())
    miss = miss.loc[:, (miss > 0.05).any()]
    print(miss.round(3).T.to_string() if len(miss.columns) else "    None")

    print("\n[4] Check: total_pymnt - (principal + interest + late fees + recoveries)")
    diff = df["total_pymnt"] - (df["total_rec_prncp"] + df["total_rec_int"]
                                + df["total_rec_late_fee"] + df["recoveries"])
    print(f"    |diff| median {diff.abs().median():.4f}, 99th pct {diff.abs().quantile(.99):.4f}")

    print("\n[5] Realized dollar loss summary")
    for name, col in [("c1 (approve a default, clipped at 0)", "c1_true"),
                      ("c0 (decline a good loan = interest)", "c0_true")]:
        v = df[col].dropna()
        print(f"    {name}: n {len(v):,}, median {v.median():,.0f}, mean {v.mean():,.0f}, "
              f"99th pct {v.quantile(.99):,.0f}, max {v.max():,.0f}, share zero {(v == 0).mean():.4f}")
    print(f"    loan_amnt: min {df['loan_amnt'].min():,.0f}, max {df['loan_amnt'].max():,.0f}")

    try:
        df.to_parquet(out, index=False)
    except ImportError:
        out = out.replace(".parquet", ".csv.gz")
        df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", default="data/processed/lc36_2012_2015.parquet")
    a = p.parse_args()
    main(a.raw, a.out)