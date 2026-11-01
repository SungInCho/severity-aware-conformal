"""
LendingClub experiments of the writing sample (Section 6). Uses data/processed/val_preds.parquet and pool_preds.parquet,
which prepare_lendingclub.py and train_models.py produce.

Commands (repo root)                               output in results/lendingclub/
  python src/lendingclub.py run --n 100            lc_levels.csv, lc_runs.csv, lc_summary.csv, lc_table60.csv,
                                                   fig_lc_frontier.png (Figure 4), fig_lc_weight.png (Figure 3)
  python src/lendingclub.py figs                   redraw Figures 3 and 4 from the saved csv files
  python src/lendingclub.py shift --n 100          lc_shift_levels.csv, lc_shift_summary.csv, fig_lc_shift.png,
                                                   diagnostics of the scores in 2015 (gap paragraph of Section 6)
  python src/lendingclub.py delay --n 100          lc_delay_summary.csv, calibration on loans before 2015 (Discussion)
  python src/lendingclub.py plugin --n 100         lc_plugin_summary.csv, plug-in cut-off without calibration (Section 6)
  python src/lendingclub.py quintiles              default rate and loss per default by loan-amount fifth (Section 6)
Run times: run and shift about 2 minutes each with --n 100.

Rules compared (all meet both constraints)
  ours      multi-risk CRC on h = (theta_hat c_hat / B + 1 - theta_hat) p_hat, theta_hat from validation loans, no labels
  theta0    multi-risk CRC on p_hat
  theta1    multi-risk CRC on p_hat c_hat
  separate  coverage CRC on p_hat and severity CRC on p_hat c_hat, pass only below both (MultiRisk type)
  cncrc     coverage CRC on p_hat and Pr(Y = 1, pass) <= alpha_sev / B by CRC on p_hat (worst-case cost B)
Reference rows: cc_only (class-conditional CP) and sev_only (severity CRC). Hindsight: grid of theta, step 0.05.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from lc_data import B1, CAT_QUANTILE, C_HAT, split_pool

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RES = ROOT / "results" / "lendingclub"
A_COV = 0.10
LEVELS = sorted(set(np.round(np.exp(np.linspace(np.log(5), np.log(100), 20)), 4).tolist() + [60.0]))
MAIN = 60.0
GRID = [round(t, 2) for t in np.arange(0, 1.0001, 0.05)]
RULES = ["ours", "theta0", "theta1", "separate", "cncrc"]
LABELS = {"ours": r"proposed, $\hat\theta$", "theta0": r"$\theta = 0$", "theta1": r"$\theta = 1$",
          "separate": "separate cut-offs", "cncrc": "worst-case cost"}
COLORS = {"ours": "#0b5cad", "theta0": "#7a7a7a", "theta1": "#c4661f", "separate": "#3a9a5b", "cncrc": "#8a5cb8"}
INK, MUTED, GRIDC = "#1f2328", "#6e7781", "#d0d7de"


# ------------------------------------------------------------------ plug-in dual weight (no labels)
def dual_theta(p, c, a_cov, a_usd):
    """theta_hat and the multipliers from the plug-in linear program (no labels used)."""
    n = len(p)
    res = linprog(-np.ones(n), A_ub=np.vstack([p, p * c]), b_ub=[a_cov * p.sum(), n * a_usd],
                  bounds=(0, 1), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    lam = -res.ineqlin.marginals
    if lam.sum() <= 0:                       # neither constraint binds: every case can pass, theta is irrelevant
        return 0.0, lam
    return float(lam[1] * B1 / (lam[0] + lam[1] * B1)), lam


# ------------------------------------------------------------------ data
def load(name):
    df = pd.read_parquet(DATA / name)
    p = df["p_hat"].to_numpy(float)
    c = np.clip(np.nan_to_num(df[C_HAT].to_numpy(float)), 0, B1)
    y = df["y"].to_numpy(int)
    loss = np.where(y == 1, np.nan_to_num(df["c1_true"].to_numpy(float)), 0.0)
    return df, p, c, y, loss


def h(theta, p, c):
    return (theta * c / B1 + 1 - theta) * p


# ------------------------------------------------------------------ conformal risk control
def crc_cut(score, loss, alpha, lbar):
    """sup{t : (sum_i loss_i 1{score_i < t} + lbar) / (n + 1) <= alpha}. The rule passes s < t.
    alpha may be an array (one cut-off per level)."""
    n = len(score)
    o = np.argsort(score, kind="stable")
    s = score[o]
    cum = np.concatenate([[0.0], np.cumsum(loss[o])])
    b = np.atleast_1d(alpha) * (n + 1) - lbar
    k = np.searchsorted(cum, b + 1e-9, side="right") - 1          # cases passed on the calibration sample
    t = np.where(k < 0, -np.inf, np.where(k >= n, np.inf, s[np.clip(k, 0, n - 1)]))
    return t if np.ndim(alpha) else float(t[0])


def cut_cov(score, y, alpha):
    """Coverage cut-off: CRC with loss 1 on the adverse calibration cases only."""
    d = score[y == 1]
    return crc_cut(d, np.ones(len(d)), alpha, 1.0)


def cut_sev(score, loss, alphas):
    return crc_cut(score, loss, alphas, B1)


# ------------------------------------------------------------------ evaluation on the test half
class Test:
    def __init__(self, y, loss, tau):
        self.y, self.loss, self.n = y, loss, len(y)
        self.hl = (y == 1) & (loss >= tau)
        self.nd, self.nhl = y.sum(), self.hl.sum()

    def from_mask(self, passed):
        return dict(miss=(passed & (self.y == 1)).sum() / self.nd,
                    harm=(passed * self.loss).sum() / self.n,
                    hl=100 * (passed & self.hl).sum() / self.nhl,
                    review=100 * (1 - passed.mean()))

    def prefix(self, score):
        """Fast evaluation of cut-offs on one score: returns a function t -> metrics."""
        o = np.argsort(score, kind="stable")
        s = score[o]
        cd = np.concatenate([[0], np.cumsum(self.y[o])])
        cl = np.concatenate([[0.0], np.cumsum(self.loss[o])])
        ch = np.concatenate([[0], np.cumsum(self.hl[o])])

        def ev(t):
            m = np.searchsorted(s, t, side="left")
            return dict(miss=cd[m] / self.nd, harm=cl[m] / self.n, hl=100 * ch[m] / self.nhl,
                        review=100 * (1 - m / self.n))
        return ev


# ------------------------------------------------------------------ main
def run(n_splits):
    """Section 6. Levels, guarantees, frontier (Figure 4), Table 1 and the estimated weight (Figure 3)."""
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    val, pv, cv, yv, lv = load("val_preds.parquet")
    pool, pp, cp, yp, lp = load("pool_preds.parquet")
    tau = np.quantile(lv[yv == 1], CAT_QUANTILE)                    # high-loss threshold, validation labels
    alphas = np.array(LEVELS)

    # ---- R4 inputs: theta_hat per level, unlabeled validation loans --------------------------------------------
    rows = []
    for a in LEVELS:
        th, lam = dual_theta(pv, cv, A_COV, a)
        l1, l2 = float(lam[0]), float(lam[1])
        kappa = l1 / l2 if l2 > 1e-12 else np.inf                 # = B (1 - theta) / theta, dollars per default
        binds = ("both" if l1 > 1e-9 and l2 > 1e-9 else "coverage" if l1 > 1e-9 else
                 "severity" if l2 > 1e-9 else "none")
        rows.append(dict(alpha_sev=a, theta_hat=th, lambda_cov=l1, lambda_sev=l2, kappa=kappa, binds=binds))
    lev = pd.DataFrame(rows)
    lev.to_csv(RES / "lc_levels.csv", index=False)

    # class-conditional CP on validation halves (labels used, for the sentence on the range of levels)
    rng = np.random.default_rng(2026)
    s_cc = []
    for _ in range(20):
        perm = rng.permutation(len(pv))
        a_, b_ = perm[: len(pv) // 2], perm[len(pv) // 2:]
        t = cut_cov(pv[a_], yv[a_], A_COV)
        s_cc.append(((pv[b_] < t) * lv[b_]).sum() / len(b_))
    s_cc = float(np.mean(s_cc))
    print(f"Validation: {len(pv):,} loans. High-loss default: loss >= ${tau:,.0f}. "
          f"Class-conditional CP leaves ${s_cc:.1f} per applicant uncovered.")
    print(lev.round(4).to_string(index=False))
    print(f"[{time.time() - t0:.0f}s]\n", flush=True)

    # ---- splits ------------------------------------------------------------------------------------------------
    recs = []
    for seed in range(n_splits):
        ci, ti = split_pool(len(pool), seed=seed)
        p1, c1, y1, l1 = pp[ci], cp[ci], yp[ci], lp[ci]
        p2, c2, y2, l2 = pp[ti], cp[ti], yp[ti], lp[ti]
        T = Test(y2, l2, tau)

        def add(rule, j, m, theta=np.nan):
            recs.append(dict(seed=seed, alpha_sev=LEVELS[j], rule=rule, theta=theta, **m))

        # theta0, cncrc, cc_only : score p_hat
        ev = T.prefix(p2)
        tc = cut_cov(p1, y1, A_COV)
        ts = cut_sev(p1, l1, alphas)
        tw = crc_cut(p1, (y1 == 1).astype(float), alphas / B1, 1.0)   # Pr(Y = 1, pass) <= alpha_sev / B
        m_cc = ev(tc)
        for j in range(len(LEVELS)):
            add("theta0", j, ev(min(tc, ts[j])))
            add("cncrc", j, ev(min(tc, tw[j])))
            add("cc_only", j, m_cc)

        # theta1, sev_only : score p_hat c_hat
        ev = T.prefix(p2 * c2)
        tc1 = cut_cov(p1 * c1, y1, A_COV)
        ts1 = cut_sev(p1 * c1, l1, alphas)
        for j in range(len(LEVELS)):
            add("theta1", j, ev(min(tc1, ts1[j])))
            add("sev_only", j, ev(ts1[j]))

        # separate cut-offs: p_hat < t_cov (on p_hat) and p_hat c_hat < t_sev (on p_hat c_hat)
        for j in range(len(LEVELS)):
            add("separate", j, T.from_mask((p2 < tc) & (p2 * c2 < ts1[j])))

        # ours: theta_hat differs by level
        for j, th in enumerate(lev.theta_hat):
            s1, s2 = h(th, p1, c1), h(th, p2, c2)
            t = min(cut_cov(s1, y1, A_COV), cut_sev(s1, l1, alphas[j]))
            add("ours", j, T.from_mask(s2 < t), th)

        # hindsight grid
        for th in GRID:
            s1, s2 = h(th, p1, c1), h(th, p2, c2)
            ev = T.prefix(s2)
            tcg, tsg = cut_cov(s1, y1, A_COV), cut_sev(s1, l1, alphas)
            for j in range(len(LEVELS)):
                add("grid", j, ev(min(tcg, tsg[j])), th)

        if (seed + 1) % 10 == 0 or seed == 0:
            print(f"  {seed + 1}/{n_splits} splits  [{time.time() - t0:.0f}s]", flush=True)

    runs = pd.DataFrame(recs)
    runs.to_csv(RES / "lc_runs.csv", index=False)
    summ = summarize(runs, n_splits)
    summ.to_csv(RES / "lc_summary.csv", index=False)
    report(summ, lev)
    figures(summ, lev)
    print(f"\nSaved to {RES}  [{time.time() - t0:.0f}s]")


def summarize(runs, n_splits):
    g = runs.groupby(["alpha_sev", "rule", "theta"], dropna=False)
    s = g[["miss", "harm", "hl", "review"]].mean()
    se = g[["miss", "harm", "hl", "review"]].std() / np.sqrt(n_splits)
    s = s.join(se.add_prefix("se_")).reset_index()
    s["cov_used"] = 100 * s.miss / A_COV
    s["sev_used"] = 100 * s.harm / s.alpha_sev
    return s


def hindsight(summ):
    gr = summ[summ.rule == "grid"]
    return gr.loc[gr.groupby("alpha_sev").review.idxmin()].set_index("alpha_sev")


def report(summ, lev):
    pd.set_option("display.width", 220, "display.max_rows", 500)
    main_rows = summ[summ.rule.isin(RULES)]

    # R1
    m = summ[np.isclose(summ.alpha_sev, MAIN)]
    cc, so = m[m.rule == "cc_only"].iloc[0], m[m.rule == "sev_only"].iloc[0]
    print(f"\nR1 at $60: class-conditional CP leaves ${cc.harm:.1f} per applicant uncovered "
          f"and {100 * cc.miss:.1f}% of defaults (review {cc.review:.2f}%). Severity-only CRC leaves "
          f"{100 * so.miss:.1f}% of defaults and ${so.harm:.1f} per applicant uncovered (review {so.review:.2f}%).")

    # R2
    z_cov = ((main_rows.miss - A_COV) / main_rows.se_miss.clip(lower=1e-12)).max()
    z_sev = ((main_rows.harm - main_rows.alpha_sev) / main_rows.se_harm.clip(lower=1e-12)).max()
    print(f"\nR2 guarantees, {len(LEVELS)} levels x {len(RULES)} rules: largest excess over the level in Monte Carlo SE, "
          f"coverage {z_cov:.2f}, severity {z_sev:.2f} (below 2 means no violation detected).")
    print(f"   uncovered defaults range {main_rows.miss.min():.4f} to {main_rows.miss.max():.4f}; "
          f"severity used at most {main_rows.sev_used.max():.1f}%")

    # R3 (2): table at $60
    tab = m[m.rule.isin(RULES)].set_index("rule").loc[RULES,
          ["miss", "harm", "hl", "review", "cov_used", "sev_used", "se_review", "se_harm", "se_hl"]]
    tab.to_csv(RES / "lc_table60.csv")
    print("\nR3(2) Table 1 at alpha_cov = 0.10, alpha_sev = $60 (means over splits):")
    print(tab.round(4).to_string())

    # R3 (1): review per level and rule
    wide = main_rows.pivot_table(index="alpha_sev", columns="rule", values="review").loc[:, RULES]
    best = hindsight(summ)
    wide["hindsight"] = best.review
    print("\nR3(1) share sent to review (%) per level:")
    print(wide.round(2).to_string())
    for r in RULES[1:]:
        d = wide[r] - wide["ours"]
        print(f"   {r:>9s} minus ours: mean {d.mean():+.2f} pp, min {d.min():+.2f}, max {d.max():+.2f}")

    # R4
    out = lev.set_index("alpha_sev")[["theta_hat", "kappa", "binds"]].copy()
    out["theta_hindsight"] = best.theta
    out["review_ours"] = wide["ours"]
    out["review_hindsight"] = best.review
    out["gap_pp"] = out.review_ours - out.review_hindsight
    print("\nR4 estimated weight (kappa in $ per default, inf where severity does not bind):")
    print(out.round(3).to_string())


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRIDC, lw=0.6)
    ax.tick_params(colors=INK, labelsize=8)


def figures(summ, lev):
    # frontier: review against protection, one curve per rule across levels
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)   # printed at the full text width
    for r in RULES:
        s = summ[summ.rule == r].sort_values("alpha_sev")
        # ours is drawn thick underneath, so that theta = 1 (dashed) stays visible where the two coincide
        kw = dict(color=COLORS[r], marker="o", ms=2.2 if r != "ours" else 3.5, lw=0.9 if r != "ours" else 2.0,
                  ls="--" if r == "theta1" else "-", label=LABELS[r], zorder=2 if r == "ours" else 3)
        axes[0].plot(s.harm, s.review, **kw)
        axes[1].plot(s.hl, s.review, **kw)
    # panel letters go below each plot, under the axis label
    axes[0].set_xlabel("uncovered loss per applicant (\\$)\n(a)", fontsize=9, color=INK, linespacing=1.8)
    axes[1].set_xlabel("high-loss defaults left uncovered (%)\n(b)", fontsize=9, color=INK, linespacing=1.8)
    axes[0].set_ylabel("sent to review (%)", fontsize=9, color=INK)
    for ax in axes:
        style(ax)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "fig_lc_frontier.png", dpi=300)

    # estimated weight and its cost in reviews, against the level
    best = hindsight(summ)
    l = lev.sort_values("alpha_sev")
    ours = summ[summ.rule == "ours"].set_index("alpha_sev").review
    weight_figure(RES / "fig_lc_weight.png", best,
                  [(r"$\hat\theta$", COLORS["ours"], l.alpha_sev.to_numpy(), l.theta_hat.to_numpy(),
                    (ours - best.review).loc[l.alpha_sev].to_numpy())])


def weight_figure(path, best, series):
    """(a) weights against the level, with the best grid weight on the test loans.
    (b) share sent to review with the estimated weight minus that with the best grid weight (pp).
    series: list of (label, color, levels, weights, gaps)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))   # printed at the full text width
    ax = axes[0]
    for lab, col, x, th, _ in series:
        ax.plot(x, th, color=col, marker="o", ms=3, lw=1.2, label=lab)
    # hollow black squares, so that orange keeps meaning theta = 1 as in the frontier figure
    ax.plot(best.index, best.theta, color=INK, marker="s", ms=3.2, mfc="none", mew=0.9, lw=0,
            label="best on the test loans")
    ax.set_ylabel(r"weight $\theta$", fontsize=9, color=INK)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax = axes[1]
    for lab, col, x, _, gap in series:
        ax.plot(x, gap, color=col, marker="o", ms=3, lw=1.2, label=lab)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("extra share sent to review (pp)", fontsize=9, color=INK)
    if len(series) > 1:
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for ax, letter in zip(axes, "ab"):
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 100])
        ax.set_xticklabels(["5", "10", "20", "50", "100"])
        ax.minorticks_off()
        ax.set_xlabel(r"severity level $\alpha_{\mathrm{sev}}$ (\$ per applicant)" + f"\n({letter})",
                      fontsize=9, color=INK, linespacing=1.8)
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)



# ------------------------------------------------------------------ shift between periods
C2015 = "#3a9a5b"


def s_cc_plugin(p, c):
    """Uncovered severity of the cut-off on p_hat at level alpha_cov, computed from p_hat and c_hat only."""
    o = np.argsort(p, kind="stable")
    k = int(np.searchsorted(np.cumsum(p[o]), A_COV * p.sum(), side="right"))
    return float((p[o][:k] * c[o][:k]).sum() / len(p))


def s_cc_actual(p, y, loss, rng, reps=20):
    """Uncovered loss per applicant of class-conditional CP, calibrated on one half and evaluated on the other."""
    out = []
    for _ in range(reps):
        perm = rng.permutation(len(p))
        a, b = perm[: len(p) // 2], perm[len(p) // 2:]
        t = cut_cov(p[a], y[a], A_COV)
        out.append(((p[b] < t) * loss[b]).sum() / len(b))
    return float(np.mean(out))


def shift(n_splits):
    """Section 6, gap to the hindsight weight. Weight from 2015 loans without outcomes, and score diagnostics."""
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    _, pv, cv, yv, lv = load("val_preds.parquet")
    _, pp, cp, yp, lp = load("pool_preds.parquet")
    tau = np.quantile(lv[yv == 1], CAT_QUANTILE)
    alphas = np.array(LEVELS)

    rng = np.random.default_rng(2027)
    perm = rng.permutation(len(pp))
    u, r = perm[: len(pp) // 3], perm[len(pp) // 3:]
    pu, cu = pp[u], cp[u]                                   # unlabeled 2015 sample: scores only
    pr, cr, yr, lr = pp[r], cp[r], yp[r], lp[r]

    # ---- diagnostics (labels used for printing only) ------------------------------------------------------------
    rows = []
    for name, p, c, y, loss in [("validation (pre-2015)", pv, cv, yv, lv), ("2015 unlabeled U", pu, cu, yp[u], lp[u]),
                                ("2015 cal/test pool", pr, cr, yr, lr)]:
        rows.append({"sample": name, "loans": len(p), "mean p_hat": p.mean(), "default rate": y.mean(),
                     "mean p_hat c_hat ($)": (p * c).mean(), "loss per applicant ($)": loss.mean(),
                     "S_cc plug-in ($)": s_cc_plugin(p, c), "S_cc actual ($)": s_cc_actual(p, y, loss, rng)})
    diag = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("Diagnostics (outcomes used only here):")
    print(diag.round(4).to_string(index=False))

    # ---- the two estimates of the weight -----------------------------------------------------------------------
    lev = []
    for a in LEVELS:
        tv, lamv = dual_theta(pv, cv, A_COV, a)
        tu, lamu = dual_theta(pu, cu, A_COV, a)
        lev.append(dict(alpha_sev=a, theta_val=tv, theta_2015=tu,
                        kappa_val=lamv[0] / lamv[1] if lamv[1] > 1e-12 else np.inf,
                        kappa_2015=lamu[0] / lamu[1] if lamu[1] > 1e-12 else np.inf))
    lev = pd.DataFrame(lev)
    lev.to_csv(RES / "lc_shift_levels.csv", index=False)
    print("\nWeights (no labels used):")
    print(lev.round(4).to_string(index=False))
    print(f"[{time.time() - t0:.0f}s]\n", flush=True)

    # ---- splits ------------------------------------------------------------------------------------------------
    recs = []
    for seed in range(n_splits):
        ci, ti = split_pool(len(pr), seed=seed)
        p1, c1, y1, l1 = pr[ci], cr[ci], yr[ci], lr[ci]
        p2, c2 = pr[ti], cr[ti]
        T = Test(yr[ti], lr[ti], tau)

        for j, row in lev.iterrows():
            for rule, th in [("theta_val", row.theta_val), ("theta_2015", row.theta_2015)]:
                s1, s2 = h(th, p1, c1), h(th, p2, c2)
                t = min(cut_cov(s1, y1, A_COV), cut_sev(s1, l1, alphas[j]))
                recs.append(dict(seed=seed, alpha_sev=LEVELS[j], rule=rule, theta=th, **T.from_mask(s2 < t)))
        for th in GRID:
            s1, s2 = h(th, p1, c1), h(th, p2, c2)
            ev = T.prefix(s2)
            tc, ts = cut_cov(s1, y1, A_COV), cut_sev(s1, l1, alphas)
            for j in range(len(LEVELS)):
                recs.append(dict(seed=seed, alpha_sev=LEVELS[j], rule="grid", theta=th, **ev(min(tc, ts[j]))))
        if (seed + 1) % 10 == 0 or seed == 0:
            print(f"  {seed + 1}/{n_splits} splits  [{time.time() - t0:.0f}s]", flush=True)

    runs = pd.DataFrame(recs)
    g = runs.groupby(["alpha_sev", "rule", "theta"])
    summ = g[["miss", "harm", "hl", "review"]].mean()
    summ = summ.join((g[["miss", "harm"]].std() / np.sqrt(n_splits)).add_prefix("se_")).reset_index()
    summ.to_csv(RES / "lc_shift_summary.csv", index=False)

    gr = summ[summ.rule == "grid"]
    best = gr.loc[gr.groupby("alpha_sev").review.idxmin()].set_index("alpha_sev")
    rv = summ[summ.rule == "theta_val"].set_index("alpha_sev")
    ru = summ[summ.rule == "theta_2015"].set_index("alpha_sev")
    out = pd.DataFrame({"theta_val": rv.theta, "theta_2015": ru.theta, "theta_best": best.theta,
                        "review_val": rv.review, "review_2015": ru.review, "review_best": best.review})
    out["gap_val"] = out.review_val - out.review_best
    out["gap_2015"] = out.review_2015 - out.review_best
    print("\nReview (%) and gap to the best grid weight on the test halves (pp):")
    print(out.round(3).to_string())
    mid = out.index[out.theta_best < 1]                      # levels where the best weight is not the end theta = 1
    print(f"\nLevels where the best weight is below 1 ({len(mid)}): mean gap, validation weight "
          f"{out.gap_val[mid].mean():.2f} pp, 2015 unlabeled weight {out.gap_2015[mid].mean():.2f} pp")

    z_cov = ((ru.miss - A_COV) / ru.se_miss.clip(lower=1e-12)).max()
    z_sev = ((ru.harm - ru.index) / ru.se_harm.clip(lower=1e-12)).max()
    print(f"Guarantees with the 2015 weight: largest excess in Monte Carlo SE, coverage {z_cov:.2f}, severity {z_sev:.2f}")

    x = out.index.to_numpy()
    weight_figure(RES / "fig_lc_shift.png", best,
                  [(r"$\hat\theta$, validation loans", COLORS["ours"], x, out.theta_val.to_numpy(), out.gap_val.to_numpy()),
                   (r"$\hat\theta$, 2015 loans without outcomes", C2015, x, out.theta_2015.to_numpy(),
                    out.gap_2015.to_numpy())])
    print(f"\nSaved to {RES}  [{time.time() - t0:.0f}s]")


def quintiles():
    """Default rate and loss per default by fifth of loan amount, on the 2015 loans."""
    pool = pd.read_parquet(DATA / "pool_preds.parquet")
    col = next((c for c in ("loan_amnt", "loan_amount", "funded_amnt", "amount") if c in pool.columns), None)
    if col is None:
        raise SystemExit(f"No loan amount column found. Columns: {list(pool.columns)}")
    q = pd.qcut(pool[col], 5, labels=["1 (smallest)", "2", "3", "4", "5 (largest)"])
    d = pool.y == 1
    tab = pd.DataFrame({"default rate": pool.groupby(q, observed=True).y.mean(),
                        "loss per default ($)": pool[d].groupby(q[d], observed=True).c1_true.mean()})
    print(f"2015 loans: {len(pool):,}, overall default rate {pool.y.mean():.4f}, amount column '{col}'")
    print(tab.round(4).to_string())
    print(f"largest / smallest fifth, loss per default: {tab.iloc[-1, 1] / tab.iloc[0, 1]:.2f}")


def delay(n_splits):
    """Calibration on earlier loans (Discussion). In deployment the calibration loans must have known outcomes and are
    therefore older than the new applicants. Here the calibration sample is drawn from the validation loans (before
    2015) and the rule is evaluated on the 2015 test halves. As a reference, the same rule with the same weight is
    calibrated on the 2015 calibration halves, as in Table 1.

    Each split: the validation loans are cut at random into two halves. Half A gives theta_hat (no labels), half B is
    the earlier calibration sample, so that theta_hat does not depend on the calibration sample (Theorem 2). The 2015
    pool is split by split_pool, as in run. Coverage CRC and severity CRC are applied to the same score.
    """
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    _, pv, cv, yv, lv = load("val_preds.parquet")
    _, pp, cp, yp, lp = load("pool_preds.parquet")
    tau = np.quantile(lv[yv == 1], CAT_QUANTILE)
    alphas = np.array(LEVELS)
    print(f"Validation loans {len(pv):,} (halves of {len(pv) // 2:,}), 2015 loans {len(pp):,}", flush=True)

    recs = []
    for seed in range(n_splits):
        rng = np.random.default_rng(10_000 + seed)
        perm = rng.permutation(len(pv))
        A, Bv = perm[: len(pv) // 2], perm[len(pv) // 2:]
        ci, ti = split_pool(len(pp), seed=seed)
        T = Test(yp[ti], lp[ti], tau)
        p2, c2 = pp[ti], cp[ti]
        cal = {"earlier": (pv[Bv], cv[Bv], yv[Bv], lv[Bv]), "same period": (pp[ci], cp[ci], yp[ci], lp[ci])}
        thetas = [dual_theta(pv[A], cv[A], A_COV, a)[0] for a in LEVELS]

        for period, (p1, c1, y1, l1) in cal.items():
            for j, th in enumerate(thetas):                               # proposed rule
                s1 = h(th, p1, c1)
                t = min(cut_cov(s1, y1, A_COV), cut_sev(s1, l1, alphas[j]))
                recs.append(dict(seed=seed, period=period, rule="ours", alpha_sev=LEVELS[j], theta=th,
                                 **T.from_mask(h(th, p2, c2) < t)))
            for rule, f in [("theta0", lambda p, c: p), ("theta1", lambda p, c: p * c)]:
                s1, ev = f(p1, c1), T.prefix(f(p2, c2))
                tc, ts = cut_cov(s1, y1, A_COV), cut_sev(s1, l1, alphas)
                for j in range(len(LEVELS)):
                    recs.append(dict(seed=seed, period=period, rule=rule, alpha_sev=LEVELS[j], theta=np.nan,
                                     **ev(min(tc, ts[j]))))
        if (seed + 1) % 10 == 0 or seed == 0:
            print(f"  {seed + 1}/{n_splits} splits  [{time.time() - t0:.0f}s]", flush=True)

    runs = pd.DataFrame(recs)
    g = runs.groupby(["period", "rule", "alpha_sev"])
    summ = g[["miss", "harm", "hl", "review"]].mean().join(
        (g[["miss", "harm", "review"]].std() / np.sqrt(n_splits)).add_prefix("se_")).reset_index()
    summ["cov_used"] = 100 * summ.miss / A_COV
    summ["sev_used"] = 100 * summ.harm / summ.alpha_sev
    summ["z_cov"] = (summ.miss - A_COV) / summ.se_miss.clip(lower=1e-12)
    summ["z_sev"] = (summ.harm - summ.alpha_sev) / summ.se_harm.clip(lower=1e-12)
    summ.to_csv(RES / "lc_delay_summary.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["review", "cov_used", "sev_used", "z_cov", "z_sev"]
    print(f"\nAt alpha_sev = ${MAIN:.0f} (coverage and severity used in % of each level, excess in standard errors):")
    print(summ[summ.alpha_sev == MAIN].set_index(["rule", "period"])[cols].round(2).to_string())
    print("\nOver all 21 levels:")
    agg = summ.groupby(["rule", "period"]).agg(
        levels_cov_exceeded=("z_cov", lambda z: int((z > 2).sum())),
        levels_sev_exceeded=("z_sev", lambda z: int((z > 2).sum())),
        max_cov_used=("cov_used", "max"), max_sev_used=("sev_used", "max"),
        mean_cov_used=("cov_used", "mean"), mean_sev_used=("sev_used", "mean"))
    print(agg.round(1).to_string())
    print("(exceeded = mean above the level by more than 2 standard errors)")
    o = summ[summ.rule == "ours"].pivot(index="alpha_sev", columns="period", values=["cov_used", "sev_used", "review"])
    print("\nProposed rule by level:")
    print(o.round(1).to_string())
    print(f"\nSaved lc_delay_summary.csv to {RES}  [{time.time() - t0:.0f}s]")


def plug_cut(score, p, c, a_cov, a_sev):
    """Plug-in cut-off without calibration. Passes the cases with the lowest score for as long as the estimated
    coverage risk sum p_hat / sum p_hat and the estimated severity sum p_hat c_hat / n stay within the levels.
    Uses p_hat and c_hat only, no outcomes. The rule passes score < t."""
    n = len(score)
    o = np.argsort(score, kind="stable")
    ok = (np.cumsum(p[o]) <= a_cov * p.sum() + 1e-12) & (np.cumsum(p[o] * c[o]) <= n * a_sev + 1e-9)
    k = int(np.argmin(ok)) if not ok.all() else n                  # number of cases passed
    return -np.inf if k == 0 else (np.inf if k == n else float(score[o][k]))


def plugin(n_splits):
    """Plug-in rule without conformal calibration (Section 6). Same weight theta_hat as the proposed rule, but the
    cut-off is set from p_hat and c_hat on the 2015 calibration half without its outcomes, as a lender would do who
    trusts the model. Compared with the proposed rule (calibrated with outcomes) on the same test halves."""
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    _, pv, cv, yv, lv = load("val_preds.parquet")
    _, pp, cp, yp, lp = load("pool_preds.parquet")
    tau = np.quantile(lv[yv == 1], CAT_QUANTILE)
    alphas = np.array(LEVELS)
    thetas = [dual_theta(pv, cv, A_COV, a)[0] for a in LEVELS]           # as in run: validation loans, no labels

    recs = []
    for seed in range(n_splits):
        ci, ti = split_pool(len(pp), seed=seed)
        p1, c1, y1, l1 = pp[ci], cp[ci], yp[ci], lp[ci]
        p2, c2 = pp[ti], cp[ti]
        T = Test(yp[ti], lp[ti], tau)
        for j, th in enumerate(thetas):
            s1, s2 = h(th, p1, c1), h(th, p2, c2)
            t_conf = min(cut_cov(s1, y1, A_COV), cut_sev(s1, l1, alphas[j]))
            t_plug = plug_cut(s1, p1, c1, A_COV, alphas[j])
            recs.append(dict(seed=seed, rule="ours", alpha_sev=LEVELS[j], theta=th, **T.from_mask(s2 < t_conf)))
            recs.append(dict(seed=seed, rule="plugin", alpha_sev=LEVELS[j], theta=th, **T.from_mask(s2 < t_plug)))
        if (seed + 1) % 10 == 0 or seed == 0:
            print(f"  {seed + 1}/{n_splits} splits  [{time.time() - t0:.0f}s]", flush=True)

    runs = pd.DataFrame(recs)
    g = runs.groupby(["rule", "alpha_sev"])
    summ = g[["miss", "harm", "hl", "review"]].mean().join(
        (g[["miss", "harm", "review"]].std() / np.sqrt(n_splits)).add_prefix("se_")).reset_index()
    summ["cov_used"] = 100 * summ.miss / A_COV
    summ["sev_used"] = 100 * summ.harm / summ.alpha_sev
    summ["z_cov"] = (summ.miss - A_COV) / summ.se_miss.clip(lower=1e-12)
    summ["z_sev"] = (summ.harm - summ.alpha_sev) / summ.se_harm.clip(lower=1e-12)
    summ.to_csv(RES / "lc_plugin_summary.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["review", "cov_used", "sev_used", "z_cov", "z_sev"]
    print(f"\nAt alpha_sev = ${MAIN:.0f} (level used in %, excess in standard errors):")
    print(summ[summ.alpha_sev == MAIN].set_index("rule")[cols].round(2).to_string())
    print("\nOver all 21 levels:")
    agg = summ.groupby("rule").agg(
        levels_cov_exceeded=("z_cov", lambda z: int((z > 2).sum())),
        levels_sev_exceeded=("z_sev", lambda z: int((z > 2).sum())),
        levels_any_exceeded=("alpha_sev", lambda a: 0),
        max_cov_used=("cov_used", "max"), max_sev_used=("sev_used", "max"))
    any_ex = summ.assign(e=(summ.z_cov > 2) | (summ.z_sev > 2)).groupby("rule").e.sum()
    agg["levels_any_exceeded"] = any_ex.astype(int)
    print(agg.round(1).to_string())
    print("(exceeded = mean above the level by more than 2 standard errors)")
    w = summ.pivot(index="alpha_sev", columns="rule", values=["cov_used", "sev_used", "review"])
    print("\nBy level:")
    print(w.round(1).to_string())
    print(f"\nSaved lc_plugin_summary.csv to {RES}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LendingClub experiments (Section 6)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "shift", "delay", "plugin"):
        sp = sub.add_parser(name)
        sp.add_argument("--n", type=int, default=100, help="number of random calibration/test splits")
    sub.add_parser("figs")
    sub.add_parser("quintiles")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.n)
    elif args.cmd == "shift":
        shift(args.n)
    elif args.cmd == "delay":
        delay(args.n)
    elif args.cmd == "plugin":
        plugin(args.n)
    elif args.cmd == "figs":
        figures(pd.read_csv(RES / "lc_summary.csv"), pd.read_csv(RES / "lc_levels.csv"))
        print(f"Redrew fig_lc_frontier.png and fig_lc_weight.png in {RES}")
    else:
        quintiles()